#pragma once

// Metal port of VKGSRenderTypes.hpp + VKQueryPool.h.
//
// Dropped (Vulkan-only): command_buffer_chunk / command_buffer_chain (mtl::command_context owns the single command
// stream, waits are expressed with sync ids), present/acquire semaphores, data heap snapshots (ring buffers retire
// automatically by sync id) and the VK query pool objects (a single visibility result buffer is used instead).
//
// Added (Metal-only bookkeeping that VK keeps inside its command buffer / render pass objects):
// - mtl::command_buffer_flags: per command buffer flags (vk::command_buffer::flags)
// - mtl::texel_buffer_view: R8Uint texture buffer window over a buffer (vk::buffer_view)
// - mtl::render_encoder_state: cached dynamic encoder state + bindings, re-applied whenever the main pass is reopened
// - mtl::draw_framebuffer: description of the main render pass (vk::framebuffer_holder + render pass key)

// Foundation pulls in CoreServices/CarbonCore MachineExceptions.h, which declares a global 'Vector128' type that collides with
// the Vector128 concept from util/v128.hpp (included by most RSX headers). Rename the Apple type while the frameworks are
// imported (same workaround as MTLCommonDecompiler.h). No-op if the frameworks were already imported by this translation unit.
#pragma push_macro("Vector128")
#undef Vector128
#define Vector128 CarbonCore_Vector128
#include "Emu/RSX/Metal/mtlutils/mtl_common.h"
#undef Vector128
#pragma pop_macro("Vector128")

// Objective-C++ defines __EXCEPTIONS for Objective-C exceptions even though rpcs3 builds with C++ exceptions disabled.
// unordered_dense (pulled in by rsx_cache.h) keys its error handling on that macro; take the no-exceptions path like the
// C++ translation units do (same workaround as MTLProgramBuffer.h).
#if defined(__OBJC__) && defined(__EXCEPTIONS) && !defined(__cpp_exceptions)
#pragma push_macro("__EXCEPTIONS")
#undef __EXCEPTIONS
#include "Emu/RSX/Common/unordered_map.hpp"
#pragma pop_macro("__EXCEPTIONS")
#endif

#include "Emu/RSX/Metal/mtlutils/mtl_common.h"
#include "Emu/RSX/Metal/mtlutils/buffer.h"
#include "Emu/RSX/Metal/mtlutils/command_context.h"
#include "Emu/RSX/Metal/mtlutils/pipeline.h"

#include "Emu/RSX/Common/simple_array.hpp"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/RSX/rsx_cache.h"
#include "Utilities/mutex.h"
#include "util/asm.hpp"

#include <array>
#include <concepts>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <ranges>
#include <tuple>
#include <vector>

// Same meaning as in the VK backend (nanoseconds)
#ifndef FRAME_PRESENT_TIMEOUT
#define FRAME_PRESENT_TIMEOUT 10000000ull // 10 seconds
#endif

#ifndef GENERAL_WAIT_TIMEOUT
#define GENERAL_WAIT_TIMEOUT  2000000ull  // 2 seconds
#endif

namespace mtl
{
	// Initial heap allocation values (MiB). The heaps are growable and will automatically increase in size to accomodate demands.
	// (VK_ATTRIB_RING_BUFFER_SIZE_M and friends; no MTL_ macros to avoid clashing with SDK macro names)
	constexpr u32 attrib_ring_buffer_size_m = 64;
	constexpr u32 texture_upload_ring_buffer_size_m = 64;
	constexpr u32 ubo_ring_buffer_size_m = 16;
	constexpr u32 transform_constants_buffer_size_m = 16;
	constexpr u32 fragment_constants_buffer_size_m = 16;
	constexpr u32 index_ring_buffer_size_m = 16;

	// Number of visibility result slots (8 bytes each). Metal caps visibility result offsets at 256 KiB.
	constexpr u32 occlusion_max_pool_size = 16384;

	// Uniform blocks up to this size are bound with setVertexBytes/setFragmentBytes instead of ring allocations (guide 9.11)
	constexpr u32 max_inline_uniform_block_size = 4096;

	struct program_cache;
	class render_target;

	// MTLGSRender derives from cpu_thread, which has a data member named 'id'. Inside the renderer class (declarations and
	// member function bodies) the Objective-C 'id' type is therefore hidden; spell Metal object types with these aliases.
	using buffer_handle = id<MTLBuffer>;
	using texture_handle = id<MTLTexture>;
	using sampler_state_handle = id<MTLSamplerState>;
	using depth_stencil_state_handle = id<MTLDepthStencilState>;
	using render_pipeline_handle = id<MTLRenderPipelineState>;
	using render_encoder_handle = id<MTLRenderCommandEncoder>;
	using blit_encoder_handle = id<MTLBlitCommandEncoder>;
	using compute_encoder_handle = id<MTLComputeCommandEncoder>;
	using command_buffer_handle = id<MTLCommandBuffer>;
	using drawable_handle = id<CAMetalDrawable>;

	using vertex_cache = rsx::vertex_cache::default_vertex_cache<rsx::vertex_cache::uploaded_range>;
	using weak_vertex_cache = rsx::vertex_cache::weak_vertex_cache;
	using null_vertex_cache = vertex_cache;

	// Identical redeclaration of the alias in MTLProgramBuffer.h (if any) is well-formed
	using shader_cache = rsx::shaders_cache<mtl::pipeline_props, mtl::program_cache>;

	struct vertex_upload_info
	{
		MTLPrimitiveType primitive;
		u32 vertex_draw_count;
		u32 allocated_vertex_count;
		u32 first_vertex;
		u32 vertex_index_base;
		u32 vertex_index_offset;
		u32 persistent_window_offset;
		u32 volatile_window_offset;
		std::optional<std::tuple<u64 /*offset in index ring*/, MTLIndexType>> index_info;
	};

	/**
	 * Per command buffer flags (vk::command_buffer::flags).
	 * mtl::command_context has no flag storage and command buffers can be committed by other modules
	 * (e.g. texture cache readbacks), so the renderer tracks the flags together with the sync id of the command buffer
	 * they belong to. When the context moves to a new command buffer the flags are reset (see MTLGSRender::get_cb_flags).
	 */
	struct command_buffer_flags
	{
		enum : u32
		{
			cb_has_occlusion_task   = 0x01,
			cb_has_blit_transfer    = 0x02,
			cb_has_dma_transfer     = 0x04,
			cb_has_open_query       = 0x08,
			cb_load_occluson_task   = 0x10,
			// 0x20 (cb_has_conditional_render) does not exist: Metal has no hardware conditional rendering
			cb_reload_dynamic_state = 0x40
		};

		rsx::flags32_t flags = 0;
		u64 sync_id = 0;          // Sync id of the command buffer the flags belong to

		// Returns true if cmd is recording a different command buffer than the one the flags were collected for
		bool is_stale(const mtl::command_context& cmd) const
		{
			return sync_id != cmd.current_sync_id();
		}

		// Rebind to the command buffer currently recorded by cmd, starting with the given flags
		void reset(const mtl::command_context& cmd, rsx::flags32_t initial_flags = 0)
		{
			sync_id = cmd.current_sync_id();
			flags = initial_flags;
		}
	};

	/**
	 * Replacement for vk::buffer_view: a MTLTextureTypeTextureBuffer texture aliasing a window of a buffer.
	 * Used for the persistent/volatile vertex attribute streams (usamplerBuffer, R8Uint) and the null stream.
	 * The view keeps the MTLBuffer it was created from; when the attribute ring grows (buffer replaced), the view must be
	 * recreated (buffer != ring.handle()).
	 */
	struct texel_buffer_view
	{
		id<MTLTexture> value = nil;
		id<MTLBuffer> buffer = nil;
		MTLPixelFormat format = MTLPixelFormatInvalid;
		u32 offset = 0;           // Window start (bytes). Must be a multiple of caps.texture_buffer_alignment
		u32 range = 0;            // Window size (bytes)

		// Implemented in MTLVertexBuffers.mm (H2)
		texel_buffer_view(id<MTLBuffer> buffer, MTLPixelFormat format, u32 offset, u32 range);
		~texel_buffer_view() = default;

		texel_buffer_view(const texel_buffer_view&) = delete;
		texel_buffer_view& operator=(const texel_buffer_view&) = delete;

		// Same semantics as vk::buffer_view::in_range
		bool in_range(u32 address, u32 size, u32& offset_out) const
		{
			if (address < offset)
			{
				return false;
			}

			const u32 _offset = address - offset;
			if (range < _offset)
			{
				return false;
			}

			const auto remaining = range - _offset;
			if (size <= remaining)
			{
				offset_out = _offset;
				return true;
			}

			return false;
		}
	};

	/**
	 * Occlusion queries (port of vk::query_pool_manager).
	 * All slots live in one shared-storage visibility result buffer (8 bytes per slot) which is attached to every main
	 * render pass (MTLRenderPassDescriptor.visibilityResultBuffer, visibilityResultType = Accumulate).
	 * - begin_query/end_query only do the slot bookkeeping. Counting is enabled/disabled with
	 *   setVisibilityResultMode:offset: which is encoder state; the renderer keeps it in mtl::render_encoder_state so that
	 *   it survives render pass interruptions (accumulate mode adds the counts of every pass segment).
	 * - A slot is zeroed on the CPU when it is allocated. Freed slots are only recycled once the command buffer that
	 *   was current when they were freed has completed, so no in-flight GPU work can still accumulate into them.
	 * - Like VK, the renderer starts a new slot for every command buffer a query spans (occlusion_data::indices).
	 * - Results are available once the command buffer that ended the query has completed (no partial results).
	 * Implemented in MTLGSRender.mm (H1).
	 */
	class query_pool_manager
	{
		struct query_slot_info
		{
			bool any_passed = false;
			bool active = false;
			bool ready = false;
			u32 data = 0;
			u64 end_sync_id = 0;          // Command buffer that recorded the end of the query (0 = still open)
		};

		std::unique_ptr<mtl::buffer> m_visibility_buffer;
		mtl::command_context* m_cmd = nullptr;

		std::deque<u32> m_available_slots;
		std::vector<query_slot_info> query_slot_status;
		std::vector<u64> m_release_sync_id;  // Per slot: command buffer that was current when the slot was freed

		MTLVisibilityResultMode m_result_mode = MTLVisibilityResultModeBoolean;

		bool poke_query(query_slot_info& query, u32 index);

	public:
		static constexpr u32 slot_size = 8;

		query_pool_manager(mtl::command_context& cmd, u32 num_entries);
		~query_pool_manager();

		query_pool_manager(const query_pool_manager&) = delete;
		query_pool_manager& operator=(const query_pool_manager&) = delete;

		// precise = true: MTLVisibilityResultModeCounting (g_cfg.video.precise_zpass_count), else Boolean
		// (VK: set_control_flags(VK_QUERY_CONTROL_PRECISE_BIT, ...))
		void set_control_flags(bool precise);

		void begin_query(mtl::command_context& cmd, u32 index);
		void end_query(mtl::command_context& cmd, u32 index);
		// Ends the slot in the command buffer with the given sync id. Used when the command buffer that recorded the
		// counting draws was committed before the query could be ended in it (e.g. by a texture cache readback).
		void end_query(u32 index, u64 sync_id);

		bool check_query_status(u32 index);
		u32  get_query_result(u32 index);

		// Returns ~0u when no slot is available (all in use or the oldest free slot may still be referenced by the GPU)
		u32 allocate_query(mtl::command_context& cmd);
		void free_query(mtl::command_context& cmd, u32 index);

		template<typename T>
			requires std::ranges::range<T> && std::same_as<std::ranges::range_value_t<T>, u32> // List of u32
		void free_queries(mtl::command_context& cmd, T& list)
		{
			for (const auto index : list)
			{
				free_query(cmd, index);
			}
		}

		id<MTLBuffer> handle() const { return m_visibility_buffer ? m_visibility_buffer->handle() : nil; }
		MTLVisibilityResultMode result_mode() const { return m_result_mode; }
		static u32 get_slot_offset(u32 index) { return index * slot_size; }
	};

	struct occlusion_data
	{
		rsx::simple_array<u32> indices;
		mtl::command_context* command_buffer_to_wait = nullptr;
		u64 command_buffer_sync_id = 0;

		bool is_current(const mtl::command_context& cmd) const
		{
			return (command_buffer_to_wait == &cmd && command_buffer_sync_id == cmd.current_sync_id());
		}

		void set_sync_command_buffer(mtl::command_context& cmd)
		{
			command_buffer_to_wait = &cmd;
			command_buffer_sync_id = cmd.current_sync_id();
		}

		void sync()
		{
			// VK only flushes the fence here; results are polled afterwards. Metal results become visible on completion,
			// so wait for it (submits the command buffer first if it is still being recorded).
			if (command_buffer_to_wait && !command_buffer_to_wait->is_complete(command_buffer_sync_id))
			{
				command_buffer_to_wait->wait(command_buffer_sync_id);
			}
		}
	};

	struct frame_context_t
	{
		rsx::flags32_t flags = 0;

		// Drawable acquired for this frame, nil if none (VK: present_image index, umax if none).
		// The drawable is presented from the command buffer that renders into it (command_context::present_drawable).
		id<CAMetalDrawable> present_drawable = nil;

		// Sync id of the command buffer that presented this frame, 0 if not queued (VK: swap_command_buffer)
		u64 swap_sync_id = 0;

		u64 last_frame_sync_time = 0;

		// Copy shareable information
		void grab_resources(frame_context_t& other)
		{
			flags = other.flags;
		}

		void tag_frame_end()
		{
			// No heap snapshot: ring buffers retire their allocations by sync id
			last_frame_sync_time = rsx::get_shared_tag();
		}

		void reset_heap_ptrs()
		{
			last_frame_sync_time = 0;
		}
	};

	struct flush_request_task
	{
		atomic_t<bool> pending_state{ false };  //Flush request status; true if rsx::thread is yet to service this request
		atomic_t<int> num_waiters{ 0 };  //Number of threads waiting for this request to be serviced
		bool hard_sync = false;

		flush_request_task() = default;

		void post(bool _hard_sync)
		{
			hard_sync = (hard_sync || _hard_sync);
			pending_state = true;
			num_waiters++;
		}

		void remove_one()
		{
			num_waiters--;
		}

		void clear_pending_flag()
		{
			hard_sync = false;
			pending_state.store(false);
		}

		bool pending() const
		{
			return pending_state.load();
		}

		void consumer_wait() const
		{
			utils::spin_wait(num_waiters, [](auto v)
			{
				return v == 0;
			});
		}

		void producer_wait() const
		{
			utils::spin_wait(pending_state, [](auto v)
			{
				return !v;
			});
		}
	};

	struct present_surface_info
	{
		u32 address;
		u32 format;
		u32 width;
		u32 height;
		u32 pitch;
		u8  eye;
	};

	struct draw_call_t
	{
		u32 subdraw_id;
	};

	/**
	 * Description of the main render pass (replaces vk::framebuffer_holder + the render pass cache key).
	 * Color attachment N is the N-th bound surface (compacted, same as VK's m_fbo_images / m_draw_buffers order).
	 * Filled by MTLGSRender::prepare_rtts (H1); the pass is opened by MTLGSRender::begin_render_pass (H1).
	 */
	struct draw_framebuffer
	{
		enum clear_bits : u32
		{
			clear_bit_color0  = 0x01, // clear_bit_color0 << N for attachment N
			clear_bit_depth   = 0x10,
			clear_bit_stencil = 0x20,
		};

		std::array<mtl::render_target*, 4> color_targets{};   // Compacted, nullptr past color_count
		mtl::render_target* depth_stencil = nullptr;

		// Attachment textures (retained). The pass is opened with these; prepare_rtts compares them to detect surfaces
		// that were recreated at the same address (VK compares the image views of the framebuffer object).
		std::array<id<MTLTexture>, 4> color_textures{};
		id<MTLTexture> depth_stencil_texture = nil;

		u32 color_count = 0;
		u32 width = 0;            // Render area (scaled)
		u32 height = 0;
		u32 layers = 1;
		u8 samples = 1;

		// Framebuffer-less rendering (no attachments): renderTargetWidth/Height/defaultRasterSampleCount are used
		bool no_attachments = false;

		mtl::render_pass_formats formats{};

		// Clears folded into the load actions of the next pass that is opened (consumed by begin_render_pass)
		u32 pending_clear_mask = 0;
		std::array<MTLClearColor, 4> clear_colors{};
		f64 clear_depth = 1.;
		u32 clear_stencil = 0;

		// cmd.render_pass_epoch() right after the main pass was opened (0 = never opened)
		u64 pass_epoch = 0;

		bool is_valid() const
		{
			return no_attachments || color_count || depth_stencil;
		}

		void reset()
		{
			color_targets = {};
			depth_stencil = nullptr;
			color_textures = {};
			depth_stencil_texture = nil;
			color_count = 0;
			width = height = 0;
			layers = 1;
			samples = 1;
			no_attachments = false;
			formats = {};
			pending_clear_mask = 0;
			pass_epoch = 0;
		}
	};

	/**
	 * Per-stage resource bindings with dirty tracking (part of render_encoder_state).
	 * Buffer bindings either reference an MTLBuffer (+ offset) or a CPU block uploaded with set*Bytes. The CPU pointer
	 * must stay valid until the next apply() (the renderer keeps shadow copies of all inline blocks for that reason).
	 */
	template <u32 MaxTextures, u32 MaxSamplers>
	struct stage_bindings
	{
		static_assert(MaxTextures <= 64 && MaxSamplers <= 32);

		static constexpr u32 max_buffers = mtl::binding::max_buffer_slots;
		static constexpr u32 max_textures = MaxTextures;
		static constexpr u32 max_samplers = MaxSamplers;

		struct buffer_binding
		{
			id<MTLBuffer> buffer = nil;
			u32 offset = 0;
			const void* bytes = nullptr;  // set*Bytes payload (buffer == nil)
			u32 length = 0;
		};

		std::array<buffer_binding, max_buffers> buffers{};
		std::array<id<MTLTexture>, MaxTextures> textures{};
		std::array<id<MTLSamplerState>, MaxSamplers> samplers{};

		u32 buffer_dirty = 0;             // Slot needs a full rebind (set*Buffer / set*Bytes)
		u32 buffer_offset_dirty = 0;      // Same buffer, only the offset changed (set*BufferOffset)
		u64 texture_dirty = 0;
		u32 sampler_dirty = 0;

		void set_buffer(u32 slot, id<MTLBuffer> buffer, u32 offset)
		{
			auto& b = buffers[slot];
			if (b.buffer == buffer && buffer != nil && !b.bytes)
			{
				if (b.offset != offset)
				{
					b.offset = offset;
					buffer_offset_dirty |= (1u << slot);
				}
				return;
			}

			b.buffer = buffer;
			b.offset = offset;
			b.bytes = nullptr;
			b.length = 0;
			buffer_dirty |= (1u << slot);
			buffer_offset_dirty &= ~(1u << slot);
		}

		void set_bytes(u32 slot, const void* data, u32 length)
		{
			ensure(length <= mtl::max_inline_uniform_block_size);

			auto& b = buffers[slot];
			b.buffer = nil;
			b.offset = 0;
			b.bytes = data;
			b.length = length;

			// Contents may have changed even if the pointer did not, always re-upload
			buffer_dirty |= (1u << slot);
			buffer_offset_dirty &= ~(1u << slot);
		}

		void set_texture(u32 slot, id<MTLTexture> texture)
		{
			if (textures[slot] != texture)
			{
				textures[slot] = texture;
				texture_dirty |= (1ull << slot);
			}
		}

		void set_sampler(u32 slot, id<MTLSamplerState> sampler)
		{
			if (samplers[slot] != sampler)
			{
				samplers[slot] = sampler;
				sampler_dirty |= (1u << slot);
			}
		}

		// Mark every live binding for re-application (new encoder)
		void invalidate()
		{
			buffer_dirty = 0;
			buffer_offset_dirty = 0;

			for (u32 slot = 0; slot < max_buffers; ++slot)
			{
				if (buffers[slot].buffer != nil || buffers[slot].bytes)
				{
					buffer_dirty |= (1u << slot);
				}
			}

			texture_dirty = (MaxTextures == 64) ? ~0ull : ((1ull << MaxTextures) - 1);
			sampler_dirty = (MaxSamplers == 32) ? ~0u : ((1u << MaxSamplers) - 1);
		}

		// Forget everything (unbound). Nothing is marked dirty.
		void clear()
		{
			buffers = {};
			textures = {};
			samplers = {};
			buffer_dirty = buffer_offset_dirty = 0;
			texture_dirty = 0;
			sampler_dirty = 0;
		}

		bool is_dirty() const
		{
			return buffer_dirty || buffer_offset_dirty || texture_dirty || sampler_dirty;
		}
	};

	/**
	 * Cached render encoder state. The renderer writes the desired state here; apply() pushes the dirty parts into the
	 * render encoder. Whenever the main pass is (re)opened the whole state is invalidated and re-applied (Metal encoder
	 * state does not survive the encoder). In-pass helper passes (attachment_clear_pass / stencil_clear_pass) clobber
	 * pipeline, depth-stencil, viewport and scissor state: call invalidate_helper_clobbered_state() after them.
	 */
	struct render_encoder_state
	{
		enum dirty_bits : u32
		{
			dirty_pipeline       = 0x0001,
			dirty_depth_stencil  = 0x0002,
			dirty_viewport       = 0x0004,
			dirty_scissor        = 0x0008,
			dirty_cull_mode      = 0x0010,
			dirty_front_face     = 0x0020,
			dirty_depth_bias     = 0x0040,
			dirty_stencil_ref    = 0x0080,
			dirty_blend_color    = 0x0100,
			dirty_depth_clip     = 0x0200,
			dirty_fill_mode      = 0x0400,
			dirty_visibility     = 0x0800,

			dirty_all            = 0x0FFF
		};

		// Vertex textures: vtex0-3 + persistent/volatile attribute streams. Fragment: tex0-15, stencil mirrors 16-31, frag_depth 32.
		using vertex_stage_bindings = stage_bindings<mtl::binding::vs_volatile_stream + 1, 4>;
		using fragment_stage_bindings = stage_bindings<mtl::binding::fs_depth_input + 1, 16>;

		id<MTLRenderPipelineState> pipeline = nil;
		id<MTLDepthStencilState> depth_stencil = nil;

		MTLViewport viewport{ 0., 0., 1., 1., 0., 1. };
		MTLScissorRect scissor{ 0, 0, 1, 1 };
		MTLCullMode cull_mode = MTLCullModeNone;
		MTLWinding front_face = MTLWindingCounterClockwise;
		MTLDepthClipMode depth_clip_mode = MTLDepthClipModeClip;
		MTLTriangleFillMode fill_mode = MTLTriangleFillModeFill;

		f32 depth_bias = 0.f;
		f32 depth_bias_slope = 0.f;
		f32 depth_bias_clamp = 0.f;

		u32 stencil_ref_front = 0;
		u32 stencil_ref_back = 0;

		std::array<f32, 4> blend_color{};

		MTLVisibilityResultMode visibility_mode = MTLVisibilityResultModeDisabled;
		u32 visibility_offset = 0;

		vertex_stage_bindings vs{};
		fragment_stage_bindings fs{};

		u32 dirty = dirty_all;

		void set_pipeline(id<MTLRenderPipelineState> value)
		{
			if (pipeline != value) { pipeline = value; dirty |= dirty_pipeline; }
		}

		void set_depth_stencil_state(id<MTLDepthStencilState> value)
		{
			if (depth_stencil != value) { depth_stencil = value; dirty |= dirty_depth_stencil; }
		}

		void set_viewport(const MTLViewport& value)
		{
			if (std::memcmp(&viewport, &value, sizeof(MTLViewport)) != 0) { viewport = value; dirty |= dirty_viewport; }
		}

		void set_scissor(const MTLScissorRect& value)
		{
			if (std::memcmp(&scissor, &value, sizeof(MTLScissorRect)) != 0) { scissor = value; dirty |= dirty_scissor; }
		}

		void set_cull_mode(MTLCullMode value)
		{
			if (cull_mode != value) { cull_mode = value; dirty |= dirty_cull_mode; }
		}

		void set_front_face(MTLWinding value)
		{
			if (front_face != value) { front_face = value; dirty |= dirty_front_face; }
		}

		void set_depth_clip_mode(MTLDepthClipMode value)
		{
			if (depth_clip_mode != value) { depth_clip_mode = value; dirty |= dirty_depth_clip; }
		}

		void set_fill_mode(MTLTriangleFillMode value)
		{
			if (fill_mode != value) { fill_mode = value; dirty |= dirty_fill_mode; }
		}

		void set_depth_bias(f32 constant, f32 slope, f32 clamp)
		{
			if (depth_bias != constant || depth_bias_slope != slope || depth_bias_clamp != clamp)
			{
				depth_bias = constant;
				depth_bias_slope = slope;
				depth_bias_clamp = clamp;
				dirty |= dirty_depth_bias;
			}
		}

		void set_stencil_reference(u32 front, u32 back)
		{
			if (stencil_ref_front != front || stencil_ref_back != back)
			{
				stencil_ref_front = front;
				stencil_ref_back = back;
				dirty |= dirty_stencil_ref;
			}
		}

		void set_blend_color(f32 r, f32 g, f32 b, f32 a)
		{
			const std::array<f32, 4> value = { r, g, b, a };
			if (blend_color != value) { blend_color = value; dirty |= dirty_blend_color; }
		}

		void set_visibility_result_mode(MTLVisibilityResultMode mode, u32 offset)
		{
			if (visibility_mode != mode || visibility_offset != offset)
			{
				visibility_mode = mode;
				visibility_offset = offset;
				dirty |= dirty_visibility;
			}
		}

		// Everything must be re-applied (a new render encoder was created)
		void invalidate()
		{
			dirty = dirty_all;
			vs.invalidate();
			fs.invalidate();
		}

		// State clobbered by in-pass helper draws (guide 9.3). Helpers bind their own parameter block / textures, so the
		// bindings are re-applied as well. NOTE: the caller should disable visibility counting while a helper draws into
		// the main pass (setVisibilityResultMode:Disabled), the visibility mode is re-applied afterwards.
		void invalidate_helper_clobbered_state()
		{
			dirty |= (dirty_pipeline | dirty_depth_stencil | dirty_viewport | dirty_scissor | dirty_cull_mode |
				dirty_front_face | dirty_depth_bias | dirty_fill_mode | dirty_depth_clip | dirty_stencil_ref | dirty_visibility);
			vs.invalidate();
			fs.invalidate();
		}

		bool is_dirty() const
		{
			return dirty || vs.is_dirty() || fs.is_dirty();
		}

		// Forget all bound objects (e.g. on shutdown) without touching the encoder
		void reset()
		{
			pipeline = nil;
			depth_stencil = nil;
			vs.clear();
			fs.clear();
			visibility_mode = MTLVisibilityResultModeDisabled;
			visibility_offset = 0;
			dirty = dirty_all;
		}

		// Push all dirty state into the encoder and clear the dirty masks. Implemented in MTLDraw.mm (H2).
		// The pipeline is only set when non-nil; the depth-stencil state is set when non-nil.
		void apply(id<MTLRenderCommandEncoder> encoder);
	};
}
