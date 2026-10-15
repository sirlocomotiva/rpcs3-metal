#pragma once

// Metal port of VKGSRender.h.
//
// Removed relative to VK (see porting guide section 8): shader interpreter, async compute scheduler / dependency info,
// descriptor pools and binding tables, renderpass/framebuffer caches, hardware conditional rendering
// (begin/end_conditional_rendering use the rsx::thread CPU fallback), host GPU labels (release_GCM_label,
// map_host_object_data, on_guest_texture_read), VMM pools and swapchain objects.
//
// Implementation files (module H):
//   MTLGSRender.mm      (H1) construction/destruction, init/exit, ring setup, render targets and the main render pass,
//                            clears, command submission and sync, local tasks and memory faults, occlusion queries,
//                            program loading and uniform environment, renderctl, misc rsx::thread callbacks
//   MTLDraw.mm          (H2) begin/end/emit_geometry, draw state, textures and samplers, viewport/scissor, encoder state
//   MTLVertexBuffers.mm (H2) vertex/index upload
//   MTLPresent.mm       (H3) flip, presentation, CAMetalLayer setup, present source lookup, screenshots, overlays
//   MTLUpscaler.mm      (H3) mtl::upscaler implementations
//   MTLRenderer.mm      (H1) mtl::create_renderer (MTLRenderer.h)

// Foundation pulls in CoreServices/CarbonCore MachineExceptions.h, which declares a global 'Vector128' type that collides with
// the Vector128 concept from util/v128.hpp (included by most RSX headers). Rename the Apple type while the frameworks are
// imported (same workaround as MTLCommonDecompiler.h). No-op if the frameworks were already imported by this translation unit.
#pragma push_macro("Vector128")
#undef Vector128
#define Vector128 CarbonCore_Vector128
#include "Emu/RSX/Metal/mtlutils/mtl_common.h"
#undef Vector128
#pragma pop_macro("Vector128")

#include "MTLUpscaler.h"
#include "MTLGSRenderTypes.h"
#include "MTLTextureCache.h"
#include "MTLRenderTargets.h"
#include "MTLFormats.h"
#include "MTLHelpers.h"
#include "MTLOverlays.h"
#include "MTLProgramBuffer.h"

#include "mtlutils/buffer.h"
#include "mtlutils/command_context.h"
#include "mtlutils/device.h"
#include "mtlutils/image.h"
#include "mtlutils/pipeline.h"
#include "mtlutils/sampler.h"

#include "Emu/RSX/GSRender.h"
#include "Emu/system_config_types.h"
#include "Utilities/geometry.h"

#include <array>
#include <deque>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

using namespace mtl::upscaling_flags_; // clang workaround (same as VK)

class MTLVertexProgram;
class MTLFragmentProgram;

namespace mtl
{
	/**
	 * Flush request posted by a non-RSX thread (GL's gl::work_item).
	 * Only the RSX thread records into the main command context (guide section 6), so memory faults that need GPU data
	 * written back are handed to the RSX thread, which runs texture_cache::flush_all(m_cmd, section_data) from
	 * do_local_task and marks the item processed. The faulting thread waits with producer_wait().
	 */
	struct work_item
	{
		u32 address_to_flush = 0;
		mtl::texture_cache::thrashed_set section_data;

		atomic_t<bool> processed = false;
		volatile bool result = false;
		atomic_t<bool> received = false;

		void producer_wait()
		{
			utils::spin_wait(processed, [](auto v)
			{
				return v;
			});

			received = true;
		}
	};
}

// NOTE: cpu_thread::id hides the Objective-C 'id' type inside this class (declarations and member function bodies).
// Use the mtl::*_handle aliases from MTLGSRenderTypes.h for Metal object types.
class MTLGSRender : public GSRender, public ::rsx::reports::ZCULL_control
{
private:
	enum frame_context_state : u32
	{
		dirty = 1
	};

	enum flush_queue_state : u32
	{
		ok = 0,
		flushing = 1,
		deadlock = 2
	};

private:
	// Device and the RSX thread's command stream. Declared first so that they are destroyed last.
	std::unique_ptr<mtl::render_device> m_device;
	mtl::command_context m_cmd;

	// Command context handed to texture cache invalidation entry points that run on non-RSX threads
	// (VK: m_secondary_cb_list). Only deferred/unmap invalidation causes are used there, which record no GPU work;
	// flushes always go through the RSX thread (work_item). Guarded like VK's secondary command buffers.
	shared_mutex m_secondary_cb_guard;
	mtl::command_context m_secondary_cmd;

	// Flags of the command buffer currently recorded by m_cmd (vk::command_buffer::flags), see get_cb_flags()
	mtl::command_buffer_flags m_cb_flags{};

	const MTLFragmentProgram* m_fragment_prog = nullptr;
	const MTLVertexProgram* m_vertex_prog = nullptr;
	mtl::program* m_program = nullptr;
	mtl::program* m_prev_program = nullptr;
	mtl::pipeline_props m_pipeline_properties{};

	// Metal keeps depth/stencil state in an immutable object outside of the pipeline state (stencil masks included)
	mtl::depth_stencil_desc m_depth_stencil_properties{};
	mtl::depth_stencil_state_handle m_depth_stencil_state = nil;

	mtl::texture_cache m_texture_cache;
	mtl::surface_cache m_rtts;

	std::unique_ptr<mtl::buffer> null_buffer;
	std::unique_ptr<mtl::texel_buffer_view> null_buffer_view;

	// Placeholder textures bound to texture slots the program references but which have no RSX texture
	// (vk::null_image_view replacement). Key: get_null_texture_key(type, depth). Zero-filled, shared storage.
	std::unordered_map<u32, std::unique_ptr<mtl::viewable_image>> m_null_textures;

	std::unique_ptr<mtl::upscaler> m_upscaler;
	output_scaling_mode m_output_scaling{output_scaling_mode::bilinear};

	shared_mutex m_sampler_mutex;
	atomic_t<bool> m_samplers_dirty = { true };
	std::array<mtl::sampler_state_handle, rsx::limits::fragment_textures_count> fs_sampler_handles{};
	std::array<mtl::sampler_state_handle, rsx::limits::vertex_textures_count> vs_sampler_handles{};

	// Description each handle was created from (VK: vk::sampler::matches). Avoids sampler cache lookups for unchanged state.
	// (fs_sampler_state / vs_sampler_state are rsx::thread members, hence the different names)
	std::array<mtl::sampler_desc, rsx::limits::fragment_textures_count> fs_sampler_descs{};
	std::array<mtl::sampler_desc, rsx::limits::vertex_textures_count> vs_sampler_descs{};

	// Stencil mirrors (fragment textures 16-31) and frag_depth (32) use constexpr samplers (guide section 5):
	// there is no m_stencil_mirror_sampler.

	std::unique_ptr<mtl::texel_buffer_view> m_persistent_attribute_storage;
	std::unique_ptr<mtl::texel_buffer_view> m_volatile_attribute_storage;

public:
	std::unique_ptr<mtl::vertex_cache> m_vertex_cache;
	std::unique_ptr<mtl::shader_cache> m_shaders_cache;

private:
	std::unique_ptr<mtl::program_cache> m_prog_buffer;

	// Presentation (VK: m_swapchain). The layer is attached to the frame's native view (m_frame->handle()).
	CAMetalLayer* m_metal_layer = nil;
	MTLPixelFormat m_present_format = MTLPixelFormatBGRA8Unorm;
	sizeu m_swapchain_dims{};
	bool swapchain_unavailable = false;
	bool should_reinitialize_swapchain = false;

	// Occlusion queries
	std::unique_ptr<mtl::query_pool_manager> m_occlusion_query_manager;
	bool m_occlusion_query_active = false;
	rsx::reports::occlusion_query_info *m_active_query_info = nullptr;
	std::vector<mtl::occlusion_data> m_occlusion_map;

	// Visibility slot of the open query and the main pass epoch it was first counted in. Without
	// MTLVisibilityResultTypeAccumulate every render encoder overwrites the counts of the slots it uses, so a query that
	// spans several passes is moved to a new slot whenever the main pass is reopened (see begin_render_pass).
	u32 m_visibility_slot = umax;
	u64 m_visibility_slot_pass_epoch = 0;

	// Main render pass (VK: m_draw_fbo + m_current_renderpass_key/m_cached_renderpass) and its cached encoder state
	mtl::draw_framebuffer m_draw_fbo{};
	mtl::render_encoder_state m_encoder_state{};

	u64 m_last_heap_sync_time = 0;
	u32 m_texbuffer_view_size = 0;

	mtl::ring_buffer m_attrib_ring_info;                      // Vertex data
	mtl::ring_buffer m_fragment_constants_ring_info;          // Fragment program constants (blocks > max_inline_uniform_block_size)
	mtl::ring_buffer m_transform_constants_ring_info;         // Transform program constants (up to 8 KiB per block)
	mtl::ring_buffer m_index_buffer_ring_info;                // Index data
	mtl::ring_buffer m_texture_upload_buffer_ring_info;       // Texture upload heap
	mtl::ring_buffer m_instancing_buffer_ring_info;           // Instanced rendering data (constants indirection table + instanced constants)

	// Small uniform blocks are bound with setVertexBytes/setFragmentBytes (guide 9.11) instead of the VK
	// fragment env / vertex env / texture params / vertex layout / raster env rings. The CPU copies below are the source
	// of those uploads; they must persist because the encoder state re-uploads them whenever the main pass is reopened.
	// Layouts are the GL ones (GLGSRender::load_program_env / update_vertex_env).
	alignas(16) std::array<u8, 96> m_vertex_env_data{};                  // VertexContextBuffer
	alignas(16) std::array<u8, 128 + 16> m_vertex_layout_data{};         // VertexLayoutBuffer (index base/offset + attribute layout)
	alignas(16) std::array<u8, 48> m_fragment_env_data{};                // FragmentStateBuffer (+ depth_bounds at offset 32)
	alignas(16) std::array<u8, 768> m_fragment_texture_params_data{};    // TextureParametersBuffer
	alignas(16) std::array<u8, 128> m_raster_env_data{};                 // RasterizerHeap (polygon stipple)
	alignas(16) std::array<u8, 32> m_blend_params_data{};                // BlendParameters (programmable blending)
	rsx::simple_array<u8> m_fragment_constants_data;                     // FragmentConstantsBuffer when <= max_inline_uniform_block_size

	u32 m_xform_constants_dynamic_offset = 0;          // Offset of the current transform constants block in m_transform_constants_ring_info
	u32 m_fragment_constants_dynamic_offset = 0;       // Offset of the current fragment constants block in m_fragment_constants_ring_info
	bool m_fragment_constants_inline = true;           // Fragment constants are bound from m_fragment_constants_data
	u32 m_instancing_indirection_offset = 0;
	u32 m_instancing_constants_offset = 0;

	std::vector<mtl::frame_context_t> m_frame_context_storage;
	u32 m_max_async_frames = 0u;
	// Temp frame context to use if the real frame queue is overburdened. Only used for storage
	mtl::frame_context_t m_aux_frame_context;

	u32 m_current_queue_index = 0;
	mtl::frame_context_t* m_current_frame = nullptr;
	std::deque<mtl::frame_context_t*> m_queued_frames;

	MTLViewport m_viewport {};
	MTLScissorRect m_scissor {};

	std::vector<u8> m_draw_buffers;

	shared_mutex m_flush_queue_mutex;
	mtl::flush_request_task m_flush_requests;

	// Pending flushes posted by other threads (see mtl::work_item)
	shared_mutex m_work_queue_guard;
	std::list<mtl::work_item> m_work_queue;

	ullong m_last_cond_render_eval_hint = 0;

	// Offloader thread deadlock recovery
	rsx::atomic_bitmask_t<flush_queue_state> m_queue_status;
	utils::address_range32 m_offloader_fault_range;
	rsx::invalidation_cause m_offloader_fault_cause;

	mtl::draw_call_t m_current_draw {};
	std::vector<mtl::image*> m_fbo_images;

	std::unique_ptr<mtl::viewable_image> m_overlay_recording_img;

	//Vertex layout
	rsx::vertex_input_layout m_vertex_layout;

public:
	u64 get_cycles() final;
	~MTLGSRender() override;

	MTLGSRender(utils::serial* ar) noexcept;
	MTLGSRender() noexcept : MTLGSRender(nullptr) {}

private:
	// ---------------------------------------------------------------------------------------------------------------
	// MTLGSRender.mm (H1): render targets, main render pass, submission
	// ---------------------------------------------------------------------------------------------------------------
	void prepare_rtts(rsx::framebuffer_creation_context context);

	// Ends the main pass, ends the visibility slot of an open occlusion query, optionally attaches a drawable to present,
	// commits m_cmd and returns the sync id of the committed command buffer (VK: fences/semaphores are not needed).
	u64 close_and_submit_command_buffer(mtl::drawable_handle present_drawable = nil);

	void flush_command_queue(bool hard_sync = false, bool do_not_switch = false);

	// Opens the main render pass if it is not the active encoder (!m_cmd.is_render_pass_open() or the epoch changed).
	// The pass is reopened with MTLLoadActionLoad except for the attachments with pending clears; m_encoder_state is
	// invalidated and fully re-applied. Returns the encoder.
	mtl::render_encoder_handle begin_render_pass();
	void close_render_pass();
	// Forces the next begin_render_pass() to start a new pass (attachments / formats changed)
	void invalidate_render_pass();
	// Attachment formats of the main pass (VK: get_render_pass())
	const mtl::render_pass_formats& get_render_pass_formats() const { return m_draw_fbo.formats; }
	// Main pass descriptor for the current m_draw_fbo, consuming the pending clears. Attaches the visibility result buffer.
	MTLRenderPassDescriptor* create_main_pass_descriptor();

	bool is_main_pass_open() const
	{
		return m_cmd.is_render_pass_open() && m_draw_fbo.pass_epoch != 0 && m_cmd.render_pass_epoch() == m_draw_fbo.pass_epoch;
	}

	// ---------------------------------------------------------------------------------------------------------------
	// MTLPresent.mm (H3): presentation
	// ---------------------------------------------------------------------------------------------------------------
	void queue_swap_request();
	void frame_context_cleanup(mtl::frame_context_t *ctx);
	void advance_queued_frames();
	void present(mtl::frame_context_t *ctx);
	// CAMetalLayer (re)initialization: size, pixel format, drawable count, vsync (displaySyncEnabled from m_vsync_mode).
	// framebufferOnly must be NO: drawables are also targets of blit-encoder copies (copy_scaled_image -> copy_image
	// fallback) and sources of screenshot readbacks.
	bool reinitialize_swapchain();
	bool init_metal_layer();

	mtl::viewable_image* get_present_source(mtl::present_surface_info* info, const rsx::avconf& avconfig);

	// ---------------------------------------------------------------------------------------------------------------
	// MTLDraw.mm (H2): draw state
	// ---------------------------------------------------------------------------------------------------------------
	void update_draw_state();

	// MTLPresent.mm (H3)
	void check_present_status();

	// MTLVertexBuffers.mm (H2)
	mtl::vertex_upload_info upload_vertex_data();
	rsx::simple_array<u8> m_scratch_mem;

	// Sync id of the command buffer the vertex cache entries were recorded in (MTLVertexBuffers.mm, H2).
	// Ring buffer space is recycled per command buffer and an enlarged ring only carries over the data of the command buffer
	// being recorded, so cached attribute ranges are only valid within the command buffer that uploaded them.
	u64 m_vertex_cache_sync_id = 0;

	// MTLDraw.mm (H2): ring buffer pressure relief (VK check_heap_status / handle_heap_critical, removed upstream in 268de3cd2).
	// Only call these before anything is allocated for the upcoming draw: a hard flush lets the rings reclaim all space.
	bool check_heap_status(const mtl::ring_buffer& heap);
	void handle_heap_critical();

	// ---------------------------------------------------------------------------------------------------------------
	// MTLGSRender.mm (H1): programs and uniform environment
	// ---------------------------------------------------------------------------------------------------------------
	bool load_program();
	void load_program_env();
	void update_vertex_env(u32 id, const mtl::vertex_upload_info& vertex_info);
	void upload_transform_constants(const rsx::io_buffer& buffer);

	// ---------------------------------------------------------------------------------------------------------------
	// MTLDraw.mm (H2): textures, samplers, encoder state
	// ---------------------------------------------------------------------------------------------------------------
	void load_texture_env();
	bool bind_texture_env();

	static u32 get_null_texture_key(MTLTextureType type, bool depth)
	{
		return static_cast<u32>(type) | (depth ? 0x100u : 0u);
	}

	// Lazily created placeholder (1x1, or 1x1x1 / 6 faces) of the given type. depth = Depth32Float (shadow samplers).
	mtl::image_view* get_null_texture(MTLTextureType type, bool depth = false);

	// Pushes the dirty parts of m_encoder_state into the main pass encoder (begin_render_pass() first)
	void apply_encoder_state();

public:
	// MTLGSRender.mm (H1)
	void init_buffers(rsx::framebuffer_creation_context context, bool skip_reading = false);

	// MTLDraw.mm (H2): compute m_viewport / m_scissor and store them in m_encoder_state
	void set_viewport();
	void set_scissor(bool clip_viewport);
	void bind_viewport();

	// MTLGSRender.mm (H1)
	// Flags of the command buffer m_cmd is currently recording. Resets them when m_cmd moved to a new command buffer
	// (also when another module committed it); in that case an active occlusion query is moved to a new slot on the
	// next draw (cb_load_occluson_task) and visibility counting is disabled until then.
	rsx::flags32_t& get_cb_flags();

	mtl::command_context& get_command_context() { return m_cmd; }

	// Sync (H1)
	void write_barrier(u32 address, u32 range) override;
	void sync_hint(rsx::FIFO::interrupt_hint hint, rsx::reports::sync_hint_payload_t payload) override;

	// Occlusion queries (H1)
	void begin_occlusion_query(rsx::reports::occlusion_query_info* query) override;
	void end_occlusion_query(rsx::reports::occlusion_query_info* query) override;
	bool check_occlusion_query_status(rsx::reports::occlusion_query_info* query) override;
	void get_occlusion_query_result(rsx::reports::occlusion_query_info* query) override;
	void discard_occlusion_query(rsx::reports::occlusion_query_info* query) override;

	// External callback in case we need to suddenly submit a commandlist unexpectedly, e.g in a violation handler (H1)
	void emergency_query_cleanup(mtl::command_context* commands);

	// External callback to handle out of video memory problems (H1)
	bool on_vram_exhausted(rsx::problem_severity severity);

	// GRAPH backend (H1)
	void patch_transform_constants(rsx::context* ctx, u32 index, u32 count) override;

	// Misc (H1). Always false: there is no shader interpreter on Metal.
	bool is_current_program_interpreted() const override;

	// Posts a flush to the RSX thread (H1, GL's post_flush_request). Caller waits on the returned item.
	mtl::work_item& post_flush_request(u32 address, mtl::texture_cache::thrashed_set& flush_data);

protected:
	// H1
	void clear_surface(u32 mask) override;

	// H2 (MTLDraw.mm)
	void begin() override;
	void end() override;
	void emit_geometry(u32 sub_index) override;

	// H1
	void on_init_thread() override;
	void on_exit() override;

	// H3 (MTLPresent.mm)
	void flip(const rsx::display_flip_info_t& info) override;

	// H1
	void renderctl(u32 request_code, void* args) override;

	void do_local_task(rsx::FIFO::state state) override;
	bool scaled_image_from_memory(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate) override;
	void notify_tile_unbound(u32 tile) override;

	bool on_access_violation(u32 address, bool is_writing) override;
	void on_invalidate_memory_range(const utils::address_range32 &range, rsx::invalidation_cause cause) override;
	void on_semaphore_acquire_wait() override;
};
