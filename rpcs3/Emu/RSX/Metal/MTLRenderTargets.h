#pragma once

// Foundation pulls in CoreServices/CarbonCore MachineExceptions.h, which declares a global 'Vector128' type that collides with
// the Vector128 concept from util/v128.hpp (included by most RSX headers). Rename the Apple type while the frameworks are
// imported (same workaround as MTLCommonDecompiler.h). No-op if the frameworks were already imported by this translation unit.
#pragma push_macro("Vector128")
#undef Vector128
#define Vector128 CarbonCore_Vector128
#include "Emu/RSX/Metal/mtlutils/mtl_common.h"
#undef Vector128
#pragma pop_macro("Vector128")

#include "util/types.hpp"
#include "../Common/surface_store.h"

#include "MTLFormats.h"
#include "MTLHelpers.h"
#include "MTLTexture.h"
#include "mtlutils/buffer.h"
#include "mtlutils/command_context.h"
#include "mtlutils/device.h"
#include "mtlutils/image.h"

#include <memory>
#include <vector>

// Port of VKRenderTargets.h
//
// Metal specific notes (applies to the whole file):
// - There are no image layouts and no pipeline barriers. All change_layout/push_layout/pop_layout and
//   insert_*_barrier calls of the VK backend are dropped; Metal tracks hazards between encoders automatically.
// - A "texture barrier" (sampling a surface that is also a bound attachment) ends the currently open render pass so
//   that the sampled data is the stored attachment contents. The renderer reopens its pass with MTLLoadActionLoad.
// - vk::drawable_surface_t is not ported; it only existed to evict cached VkFramebuffer objects on destruction.
//   Render targets and resolve surfaces derive directly from mtl::viewable_image.
// - Objects are destroyed immediately instead of being handed to a resource manager: command buffers retain every
//   Metal object they reference until completion.

namespace mtl
{
	namespace surface_cache_utils
	{
		// VK: hands the buffer to the resource manager. Metal: just delete it (in-flight command buffers retain the MTLBuffer).
		void dispose(mtl::buffer* buf);
	}

	// Implemented in MTLResolveHelper.mm (module E).
	// resolve_image: planar MSAA src -> linear single-sample dst (dst is width*samples_x by height*samples_y).
	// unresolve_image: linear single-sample src -> planar MSAA dst.
	void resolve_image(mtl::command_context& cmd, mtl::viewable_image* dst, mtl::viewable_image* src);
	void unresolve_image(mtl::command_context& cmd, mtl::viewable_image* dst, mtl::viewable_image* src);

	// VK: vkutils/memory.h vk::vmm_determine_memory_load_severity() (defined in VKResourceManager.cpp, which is not ported).
	// Metal: [device currentAllocatedSize] against caps.recommended_working_set_size with VK's thresholds.
	// Used by render_target::spill / surface_cache::can_collapse_surface (F) and MTLGSRender (H: trim, on_vram_exhausted).
	// Forward declaration only: the definition is owned by module A (mtlutils, requested). A redeclaration with this exact
	// signature in mtlutils is fine.
	rsx::problem_severity vmm_determine_memory_load_severity();

	class image_reference_sync_barrier
	{
		u32 m_texture_barrier_count = 0;
		u32 m_draw_barrier_count = 0;
		bool m_allow_skip_barrier = true;

	public:
		void on_insert_texture_barrier()
		{
			m_texture_barrier_count++;
			m_allow_skip_barrier = false;
		}

		void on_insert_draw_barrier()
		{
			// Account for corner case where the same texture can be bound to more than 1 slot
			m_draw_barrier_count = std::max(m_draw_barrier_count + 1, m_texture_barrier_count);
		}

		void allow_skip()
		{
			m_allow_skip_barrier = true;
		}

		void reset()
		{
			m_texture_barrier_count = m_draw_barrier_count = 0u;
			m_allow_skip_barrier = false;
		}

		// Metal: keeps a read-only chain going after a draw that did not write the surface (no post-loop barrier needed)
		void rearm()
		{
			m_texture_barrier_count = m_draw_barrier_count;
		}

		bool can_skip() const
		{
			return m_allow_skip_barrier;
		}

		bool is_enabled() const
		{
			return !!m_texture_barrier_count;
		}

		bool requires_post_loop_barrier() const
		{
			return is_enabled() && m_texture_barrier_count < m_draw_barrier_count;
		}
	};

	class render_target : public viewable_image, public rsx::render_target_descriptor<mtl::viewable_image*>
	{
		// Cyclic reference hazard tracking
		// Metal: there is no GENERAL/FEEDBACK_LOOP layout to test against. A feedback loop is considered active while the
		// tracker is enabled; it is reset by reset_surface_counters() (surface rebound for drawing) and post_texture_barrier().
		image_reference_sync_barrier m_cyclic_ref_tracker;

		// Memory spilling support
		// Shared storage buffer holding the surface contents while the texture is released.
		std::unique_ptr<mtl::buffer> m_spilled_mem;

		// MSAA support:
		// Get the linear resolve target bound to this surface. Initialize if none exists
		// Metal: a single-sample MTLTextureType2D image of (width * samples_x, height * samples_y) with the same pixel format,
		// format class and native_component_map. Usage: surface_cache_traits::get_resolve_target_usage(format(), is_depth_surface()).
		mtl::viewable_image* get_resolve_target_safe(mtl::command_context& cmd);
		// Resolve the planar MSAA data into a linear block (mtl::resolve_image), clears require_resolve
		void resolve(mtl::command_context& cmd);
		// Unresolve the linear data into planar MSAA data (mtl::unresolve_image), clears require_unresolve
		void unresolve(mtl::command_context& cmd);

		// Memory management:
		// Default-initialize memory without loading.
		// Metal: mtl::clear_color_image(cmd, surface, { 0.f, 0.f, 0.f, 1.f }) for color aspects, otherwise
		// mtl::clear_depth_stencil_image(cmd, surface, 1.f, 255) (MTLTexture.h, empty render passes with MTLLoadActionClear).
		void clear_memory(mtl::command_context& cmd, mtl::image* surface);
		// Load memory from cell and use to initialize the surface.
		// Metal: mtl::upload_image into this (or into the resolve target / a typeless helper followed by
		// mtl::copy_scaled_image when resolution scaling or MSAA is active), staged through mtl::get_upload_heap().
		// Tiled regions are detiled with mtl::detile_memory_block and passed as mtl::make_gpu_resident_source() with
		// source_is_gpu_resident (heap_align = width_in_block * bpp), exactly like VK.
		void load_memory(mtl::command_context& cmd);
		// Generic - chooses whether to clear or load.
		void initialize_memory(mtl::command_context& cmd, rsx::surface_access access);

		// Spill helpers
		// Re-initialize using spilled memory (image::create_impl() recreates the texture from 'info')
		void unspill(mtl::command_context& cmd);
		// Build spill transfer descriptors.
		// For combined depth/stencil (Depth32Float_Stencil8) two regions are returned: depth (4 bytes/texel) at offset 0
		// and stencil (1 byte/texel) at offset width * height * 4.
		// The spill/unspill copies are raw blits (VK: raw vkCmdCopyImageToBuffer/BufferToImage) issued on cmd.blit(), using
		// MTLBlitOptionDepthFromDepthStencil / MTLBlitOptionStencilFromDepthStencil for the depth/stencil regions.
		// Do NOT route them through mtl::copy_image_to_buffer/copy_buffer_to_image: those apply D16F/D24S8 packing.
		std::vector<mtl::buffer_image_copy> build_spill_transfer_descriptors(mtl::image* target);

	public:
		u64 frame_tag = 0;              // frame id when invalidated, 0 if not invalid
		u64 last_rw_access_tag = 0;     // timestamp when this object was last used
		u64 spill_request_tag = 0;      // timestamp when spilling was requested
		bool is_bound = false;          // set when the surface is bound for rendering

		using viewable_image::viewable_image;

		mtl::viewable_image* get_surface(rsx::surface_access access_type) override;
		bool is_depth_surface() const override;
		bool matches_dimensions(u16 _width, u16 _height) const;
		void reset_surface_counters();

		// The depth/stencil attachment was cleared in the open render pass: the next draw reading it must end the pass
		void on_depth_stencil_clear() { m_cyclic_ref_tracker.reset(); }

		// Metal replacement for the VK "current_layout == GENERAL / ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT" tests
		// (VKDraw zeta_address_cyclic_barrier and programmable blending checks, VKTextureCache::insert_texture_barrier).
		// True while a texture barrier has been emitted since the surface was last bound for drawing (reset_surface_counters)
		// or since the last post_texture_barrier.
		bool is_feedback_loop_active() const { return m_cyclic_ref_tracker.is_enabled(); }

		// Keep the base get_view(u32 aspect_mask) overload visible
		using viewable_image::get_view;

		// Intercepts MTL_REMAP_VIEW_MULTISAMPLED (returns the identity view of the multisampled image)
		image_view* get_view(const rsx::texture_channel_remap_t& remap,
			u32 mask = image_aspect::color | image_aspect::depth) override;

		// Memory management
		// Copies the contents into m_spilled_mem (shared storage, element size: Depth32Float = 4,
		// Depth32Float_Stencil8 = 5, otherwise mtl::get_format_texel_width()) and releases the texture
		// (value = nil, views cleared; replaces VK's destructive clone()). Any resolve surface is donated to resolve_cache.
		// Where VK queries vk::vmm_determine_memory_load_severity() before allocating a resolve target, Metal uses
		// mtl::vmm_determine_memory_load_severity() (device currentAllocatedSize vs recommendedMaxWorkingSetSize).
		bool spill(mtl::command_context& cmd, std::vector<std::unique_ptr<mtl::viewable_image>>& resolve_cache);

		// Synchronization
		// Metal: ends the open render pass (cmd.end_encoding() if cmd.is_render_pass_open()) so that draws sampling this
		// surface read the stored attachment contents. Skipped for back-to-back read-only depth feedback loops.
		void texture_barrier(mtl::command_context& cmd);
		// Metal: ends the open render pass so that the reads of the previous (feedback loop) draw complete before the
		// next draw writes the attachment. Skipped for read-only depth chains, which stay in the open pass while no draw
		// writes depth or stencil.
		void post_texture_barrier(mtl::command_context& cmd);
		// Same logic as VK (unspill, initialize/load memory, inherit old_contents through mtl::blitter, MSAA resolve/unresolve).
		// Metal: for shader_write with an active feedback loop the VK layout test (GENERAL/FEEDBACK_LOOP) is dropped:
		// m_cyclic_ref_tracker.on_insert_draw_barrier(), then post_texture_barrier(cmd) if requires_post_loop_barrier().
		void memory_barrier(mtl::command_context& cmd, rsx::surface_access access);
		void read_barrier(mtl::command_context& cmd) { memory_barrier(cmd, rsx::surface_access::shader_read); }
		void write_barrier(mtl::command_context& cmd) { memory_barrier(cmd, rsx::surface_access::shader_write); }
	};

	static inline mtl::render_target* as_rtt(mtl::image* t)
	{
		return ensure(dynamic_cast<mtl::render_target*>(t));
	}

	static inline const mtl::render_target* as_rtt(const mtl::image* t)
	{
		return ensure(dynamic_cast<const mtl::render_target*>(t));
	}

	static inline mtl::render_target* try_as_rtt(mtl::image* t)
	{
		return dynamic_cast<mtl::render_target*>(t);
	}

	static inline const mtl::render_target* try_as_rtt(const mtl::image* t)
	{
		return dynamic_cast<const mtl::render_target*>(t);
	}

	static inline bool is_rtt(const mtl::image* t)
	{
		return dynamic_cast<const mtl::render_target*>(t) != nullptr;
	}

	struct surface_cache_traits
	{
		using surface_storage_type = std::unique_ptr<mtl::render_target>;
		using surface_type = mtl::render_target*;
		using buffer_object_storage_type = std::unique_ptr<mtl::buffer>;
		using buffer_object_type = mtl::buffer*;
		using command_list_type = mtl::command_context&;
		using download_buffer_object = void*;
		using barrier_descriptor_t = rsx::deferred_clipped_region<mtl::render_target*>;

		// VK returns {usage, create flags}; Metal only has usage flags.
		// - RenderTarget: color/depth-stencil attachment, also covers framebuffer fetch inputs (VK INPUT_ATTACHMENT).
		// - ShaderRead: sampled by the texture cache and helper passes.
		// - PixelFormatView: replaces VK MUTABLE_FORMAT; required for stencil views (X32_Stencil8) of combined
		//   depth/stencil surfaces and for format-reinterpreting/swizzled views.
		// There is no feedback-loop usage and no vendor specific workaround. ShaderWrite is never set on attachments:
		// MSL cannot write multisampled textures, so the unresolve helpers render into the MSAA image instead.
		static MTLTextureUsage get_attachment_create_flags([[maybe_unused]] MTLPixelFormat format, [[maybe_unused]] u8 samples, [[maybe_unused]] bool depth)
		{
			return MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
		}

		// Metal specific: usage of the single-sample resolve target of an MSAA surface (VK builds this inline in
		// render_target::get_resolve_target_safe from the parent usage).
		// The resolve helpers (MTLResolveHelper.h, module E) only render into it: color as attachment 0, depth via frag depth,
		// stencil via stencil export. No compute kernel writes it, so ShaderWrite is not requested (it would also disable
		// lossless compression on Apple GPUs). Readers: texture cache sampling (ShaderRead), stencil/format views
		// (PixelFormatView), blit_pass / copy_scaled_image targets (RenderTarget).
		static MTLTextureUsage get_resolve_target_usage([[maybe_unused]] MTLPixelFormat format, [[maybe_unused]] bool depth)
		{
			return MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
		}

		// Creates a new color surface.
		// Metal: texture type is MTLTextureType2DMultisample when samples > 1 (else MTLTextureType2D), storage is private,
		// usage = get_attachment_create_flags(...). Pixel format and native component map from mtl::get_compatible_surface_format().
		// No initial layout transition. All rsx descriptor fields (format, aa mode, scaling config, sample layout, usage/state
		// flags, pitches, surface size, queue_tag, add_ref) are set exactly like VK.
		static std::unique_ptr<mtl::render_target> create_new_surface(
			mtl::command_context& cmd,
			u32 address,
			rsx::surface_color_format format,
			usz width, usz height, usz pitch,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& resolution_scaling_config);

		// Creates a new depth/stencil surface.
		// Metal: pixel format from mtl::get_compatible_depth_surface_format(mtl::get_formats_support(), format)
		// (z16 -> Depth16Unorm, z24s8 -> Depth32Float_Stencil8 since Apple GPUs have no D24S8).
		// Native component map is {R, R, R, R}. Texture type/usage/storage as for color surfaces.
		static std::unique_ptr<mtl::render_target> create_new_surface(
			mtl::command_context& cmd,
			u32 address,
			rsx::surface_depth_format2 format,
			usz width, usz height, usz pitch,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& resolution_scaling_config);

		static bool is_reusable_surface(const mtl::render_target* surface, const mtl::render_target* ref)
		{
			// Metal: the texture type replaces VK's create flags (it also encodes the MSAA layout)
			return surface->value != nil && surface->format() == ref->format() &&
				surface->info.usage == ref->info.usage && surface->info.type == ref->info.type;
		}

		static void prepare_for_reuse(mtl::command_context&, mtl::render_target* surface)
		{
			surface->reset_surface_counters();
			surface->last_rw_access_tag = 0;
		}

		// Metal: the sink is created (if needed) with the ref's type, format, sample count, usage and format class.
		// No layout transition; otherwise identical to VK.
		static void clone_surface(
			mtl::command_context& cmd,
			std::unique_ptr<mtl::render_target>& sink, mtl::render_target* ref,
			u32 address, barrier_descriptor_t& prev,
			const rsx::surface_scaling_config_t& scaling_config);

		static std::unique_ptr<mtl::render_target> convert_pitch(
			mtl::command_context& /*cmd*/,
			std::unique_ptr<mtl::render_target>& src,
			usz /*out_pitch*/)
		{
			// Same as VK: pitch conversion is not implemented by any backend, the surface is simply erased.
			src->state_flags = rsx::surface_state_flags::erase_bkgnd;
			return {};
		}

		static bool is_compatible_surface(const mtl::render_target* surface, const mtl::render_target* ref, u16 width, u16 height, u8 sample_count)
		{
			return (surface->format() == ref->format() &&
				surface->get_spp() == sample_count &&
				surface->get_surface_width() == width &&
				surface->get_surface_height() == height);
		}

		static void prepare_surface_for_drawing(mtl::command_context& cmd, mtl::render_target* surface)
		{
			// Special case barrier
			surface->memory_barrier(cmd, rsx::surface_access::gpu_reference);

			// Metal: no attachment layout transition required

			surface->reset_surface_counters();
			surface->memory_usage_flags |= rsx::surface_usage_flags::attachment;
			surface->is_bound = true;
		}

		static void prepare_surface_for_sampling(mtl::command_context& /*cmd*/, mtl::render_target* surface)
		{
			surface->is_bound = false;
		}

		static bool surface_is_pitch_compatible(const std::unique_ptr<mtl::render_target>& surface, usz pitch)
		{
			return surface->rsx_pitch == pitch;
		}

		static void int_invalidate_surface_contents(
			mtl::command_context& /*cmd*/,
			mtl::render_target* surface,
			u32 address,
			usz pitch)
		{
			surface->rsx_pitch = static_cast<u32>(pitch);
			surface->queue_tag(address);
			surface->last_use_tag = 0;
			surface->stencil_init_flags = 0;
			surface->memory_usage_flags = rsx::surface_usage_flags::unknown;
			surface->raster_type = rsx::surface_raster_type::linear;
		}

		// Updates the format, native component layout (mtl::get_compatible_surface_format) and debug name, then
		// calls int_invalidate_surface_contents.
		static void invalidate_surface_contents(
			mtl::command_context& cmd,
			mtl::render_target* surface,
			rsx::surface_color_format format,
			u32 address,
			usz pitch);

		static void invalidate_surface_contents(
			mtl::command_context& cmd,
			mtl::render_target* surface,
			rsx::surface_depth_format2 format,
			u32 address,
			usz pitch)
		{
			surface->set_format(format);
			surface->set_debug_name(fmt::format("DSV @0x%x", address));

			int_invalidate_surface_contents(cmd, surface, address, pitch);
		}

		// Tags the surface with mtl::get_current_frame_id() (MTLHelpers.h), drops deferred writes and releases the ref.
		static void notify_surface_invalidated(const std::unique_ptr<mtl::render_target>& surface);

		static void notify_surface_persist(const std::unique_ptr<mtl::render_target>& /*surface*/)
		{}

		static void notify_surface_reused(const std::unique_ptr<mtl::render_target>& surface)
		{
			surface->state_flags |= rsx::surface_state_flags::erase_bkgnd;
			surface->add_ref();
		}

		static bool int_surface_matches_properties(
			const std::unique_ptr<mtl::render_target>& surface,
			MTLPixelFormat format,
			usz width, usz height,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& scaling_config,
			bool check_refs)
		{
			if (check_refs && surface->has_refs())
			{
				// Surface may still have read refs from data 'copy'
				return false;
			}

			return (surface->info.format == format &&
				surface->get_spp() == get_format_sample_count(antialias) &&
				surface->matches_dimensions(static_cast<u16>(width), static_cast<u16>(height))) &&
				surface->resolution_scaling_config == scaling_config;
		}

		// Resolve the host format with mtl::get_compatible_surface_format / get_compatible_depth_surface_format and
		// forward to int_surface_matches_properties.
		static bool surface_matches_properties(
			const std::unique_ptr<mtl::render_target>& surface,
			rsx::surface_color_format format,
			usz width, usz height,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& scaling_config,
			bool check_refs = false);

		static bool surface_matches_properties(
			const std::unique_ptr<mtl::render_target>& surface,
			rsx::surface_depth_format2 format,
			usz width, usz height,
			rsx::surface_antialiasing antialias,
			const rsx::surface_scaling_config_t& scaling_config,
			bool check_refs = false);

		static void spill_buffer(std::unique_ptr<mtl::buffer>& /*bo*/)
		{
			// Same as VK: DMA buffer objects live in unified memory and are never spilled
		}

		static void unspill_buffer(std::unique_ptr<mtl::buffer>& /*bo*/)
		{
			// Same as VK: DMA buffer objects live in unified memory and are never spilled
		}

		// Writes the surface contents into 'bo' (surface cache DMA block).
		// Metal: scaled surfaces are first downscaled into mtl::get_typeless_helper() with mtl::copy_scaled_image (nearest),
		// the readback uses mtl::copy_image_to_buffer (through mtl::get_scratch_buffer() when the data has to be cropped or
		// is depth) followed by a blit buffer copy into 'bo'. No buffer barriers.
		static void write_render_target_to_memory(
			mtl::command_context& cmd,
			mtl::buffer* bo,
			mtl::render_target* surface,
			u64 dst_offset_in_buffer,
			u64 src_offset_in_buffer,
			u64 max_copy_length);

		template <int BlockSize>
		static mtl::buffer* merge_bo_list(mtl::command_context& cmd, std::vector<mtl::buffer*>& list)
		{
			u32 required_bo_size = 0;
			for (auto& bo : list)
			{
				required_bo_size += (bo ? ::size32(*bo) : BlockSize);
			}

			// Create dst (VK: device local memory)
			auto dst = new mtl::buffer(required_bo_size,
				MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked,
				"Surface cache DMA block");

			// TODO: Initialize the buffer with system RAM contents

			// Copy all the data over from the sub-blocks
			u32 offset = 0;
			for (auto& bo : list)
			{
				if (!bo)
				{
					offset += BlockSize;
					continue;
				}

				const u32 length = ::size32(*bo);
				[cmd.blit() copyFromBuffer:bo->handle()
					sourceOffset:0
					toBuffer:dst->handle()
					destinationOffset:offset
					size:length];

				offset += length;

				// Cleanup
				mtl::surface_cache_utils::dispose(bo);
			}

			return dst;
		}

		template <typename T>
		static T* get(const std::unique_ptr<T>& obj)
		{
			return obj.get();
		}
	};

	class surface_cache : public rsx::surface_store<mtl::surface_cache_traits>
	{
	private:
		u64 get_surface_cache_memory_quota(u64 total_device_memory);

		// Metal: replaces vk::vmm_get_application_pool_usage(VMM_ALLOCATION_POOL_SURFACE_CACHE) since there is no VMM.
		// Sums image::memory_size() of every resident surface texture and resolve surface in m_render_targets_storage,
		// m_depth_stencil_storage and invalidated_resources (spilled surfaces have value == nil and are not counted).
		u64 get_surface_cache_memory_usage();

	public:
		void destroy();
		// Total device memory for the quota is mtl::get_caps().recommended_working_set_size.
		bool spill_unused_memory();
		bool is_overallocated();
		bool can_collapse_surface(const std::unique_ptr<mtl::render_target>& surface, rsx::problem_severity severity) override;
		bool handle_memory_pressure(mtl::command_context& cmd, rsx::problem_severity severity) override;
		// Metal: the run_cleanup_internal pre-task callback is a no-op (command buffers are created on demand); frame ids come
		// from mtl::get_last_completed_frame_id() (MTLHelpers.h); resolve surfaces and surfaces are destroyed directly.
		void trim(mtl::command_context& cmd, rsx::problem_severity memory_pressure);
	};
}
