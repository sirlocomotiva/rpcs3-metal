#pragma once

// Metal texture cache. Port of VKTextureCache.h; implemented in MTLTextureCache.mm.
//
// Metal deviations from the VK version:
// - No layouts/barriers/resource manager: images released by the cache go straight into the reuse pool. Command buffers
//   retain every texture they reference and hazard tracking orders a reuse after all previously encoded accesses, so
//   no deferred disposal is needed (replaces cached_image_reference_t + vk::get_resource_manager()->dispose()).
// - No VKDMA (host memory import / DMA blocks): readbacks (dma_transfer) are written by the GPU into a shared-storage
//   buffer owned by the section which mirrors the guest memory layout. Instead of a vk::event, the section stores the
//   sync id of the command buffer that recorded the transfer; imp_flush waits on it and copies the mapped buffer into
//   guest memory (unified memory, no staging).
// - No async transfer queue / concurrent sharing: texture_create_flags::shareable and do_not_reuse requests coming
//   from the async path never occur; the flags are kept for source compatibility.
// - All entry points taking a mtl::command_context must run on the thread owning it (the RSX thread, guide section 6).

// Foundation pulls in CoreServices/CarbonCore MachineExceptions.h, which declares a global 'Vector128' type that collides with
// the Vector128 concept from util/v128.hpp (texture_cache.h reaches it through ProgramStateCache.h). Rename the Apple type
// while the frameworks are imported (same workaround as MTLCommonDecompiler.h). No-op if the frameworks were already
// imported by this translation unit.
#pragma push_macro("Vector128")
#undef Vector128
#define Vector128 CarbonCore_Vector128
#include "Emu/RSX/Metal/mtlutils/mtl_common.h"
#undef Vector128
#pragma pop_macro("Vector128")

#include "MTLTexture.h"
#include "MTLFormats.h"
#include "MTLHelpers.h"
#include "MTLRenderTargets.h"

#include "mtlutils/mtl_common.h"
#include "mtlutils/buffer.h"
#include "mtlutils/command_context.h"
#include "mtlutils/device.h"
#include "mtlutils/image.h"

#include "../Common/texture_cache.h"
#include "../Common/tiled_dma_copy.hpp"
#include "../Utils/image_utils.hpp"

#include <deque>
#include <memory>
#include <vector>

namespace mtl
{
	class cached_texture_section;
	class texture_cache;

	struct texture_cache_traits
	{
		using commandbuffer_type      = mtl::command_context;
		using section_storage_type    = mtl::cached_texture_section;
		using texture_cache_type      = mtl::texture_cache;
		using texture_cache_base_type = rsx::texture_cache<texture_cache_type, texture_cache_traits>;
		using image_resource_type     = mtl::image*;
		using image_view_type         = mtl::image_view*;
		using image_storage_type      = mtl::image;
		using texture_format          = MTLPixelFormat;
		using viewable_image_type     = mtl::viewable_image*;
	};

	class cached_texture_section : public rsx::cached_texture_section<mtl::cached_texture_section, mtl::texture_cache_traits>
	{
		using baseclass = typename rsx::cached_texture_section<mtl::cached_texture_section, mtl::texture_cache_traits>;
		friend baseclass;

		std::unique_ptr<mtl::viewable_image> managed_texture = nullptr;

		// DMA relevant data
		// Readback target (replaces the VK DMA blocks). Shared storage, mirrors guest memory: byte 0 corresponds to guest
		// address dma_base_address. Reused across transfers; replaced when too small or when the CPU must write into it
		// (load) while a previous transfer may still be in flight.
		std::unique_ptr<mtl::buffer> dma_buffer;
		u32 dma_base_address = 0;

		// Sync id of the command buffer that recorded the last dma_transfer (replaces dma_fence). 0 = no transfer pending.
		u64 dma_sync_id = 0;

		// Sync id of the last command buffer that wrote into dma_buffer. Unlike dma_sync_id it survives dma_abort() and
		// release_dma_resources(): it tells map_dma whether the GPU may still write into the buffer.
		u64 dma_buffer_sync_id = 0;

		// Guest range whose contents in dma_buffer are defined by the last dma_transfer (GPU writes plus CPU loads).
		// imp_flush only writes this part back, which matches the VK host-memory-import DMA path where bytes the GPU
		// does not write are never touched.
		utils::address_range32 dma_written_range{};

		// Command context used for the last transfer, needed by imp_flush to wait on dma_sync_id (replaces m_device).
		mtl::command_context* m_cmd = nullptr;
		mtl::viewable_image* vram_texture = nullptr;

	public:
		using baseclass::cached_texture_section;

		/**
		 * Same logic as VK (in-place image swap notifications through mtl::as_rtt, swizzle state from render targets,
		 * DMA reset when previously synchronized, baseclass::on_section_resources_created()).
		 */
		void create(u16 w, u16 h, u16 depth, u16 mipmaps, mtl::image* image, u32 rsx_pitch, bool managed, u32 gcm_format, bool pack_swap_bytes = false);

		// Framebuffer variant (called from lock_memory_region with the surface as extra argument)
		void create(u16 w, u16 h, u16 depth, u16 mipmaps, mtl::image* image, u32 rsx_pitch, bool managed, const mtl::render_target* surface);

		// Forgets the pending transfer (dma_sync_id = 0). The dma buffer itself is kept for reuse; destroy() frees it.
		void release_dma_resources();

		// Called if a reset occurs, usually via reprotect path after a bad prediction.
		// Discards the pending transfer (ensure synchronized, !flushed and dma_sync_id != 0, then reset dma_sync_id).
		void dma_abort() override;

		// VK logic plus: frees dma_buffer.
		void destroy();

		bool exists() const
		{
			return (vram_texture != nullptr);
		}

		bool is_managed() const
		{
			return !exists() || managed_texture;
		}

		mtl::image_view* get_view(const rsx::texture_channel_remap_t& remap)
		{
			ensure(vram_texture != nullptr);
			return vram_texture->get_view(remap);
		}

		mtl::image_view* get_raw_view()
		{
			ensure(vram_texture != nullptr);
			return vram_texture->get_view(rsx::default_remap_vector);
		}

		mtl::viewable_image* get_raw_texture()
		{
			return managed_texture.get();
		}

		std::unique_ptr<mtl::viewable_image>& get_texture()
		{
			return managed_texture;
		}

		mtl::render_target* get_render_target() const
		{
			return mtl::as_rtt(vram_texture);
		}

		MTLPixelFormat get_format() const
		{
			if (context == rsx::texture_upload_context::dma)
			{
				return MTLPixelFormatR32Uint;
			}

			ensure(vram_texture != nullptr);
			return vram_texture->format();
		}

		bool is_flushed() const
		{
			//This memory section was flushable, but a flush has already removed protection
			return flushed;
		}

		/**
		 * Metal replacement for vk::map_dma() + vk::load_dma().
		 * Makes dma_buffer cover the guest range (reallocating it to cover at least [range.start, range.end] if needed,
		 * dma_base_address = start of the covered range) and returns { buffer handle, byte offset of range.start in it }.
		 * If load is true, the current guest memory contents of the range are copied into the buffer with the CPU
		 * (needed whenever the GPU will not overwrite every byte that imp_flush writes back, exactly where VK calls
		 * load_dma: pitch mismatch, partial tiles, DMA sections created with memory_load).
		 * If load is true and the buffer may still be written by an in-flight transfer (dma_buffer_sync_id not complete),
		 * a new buffer is allocated instead of writing into the busy one.
		 * Newly allocated buffers are 64 KiB aligned, also cover the section range and are initialized from guest
		 * memory (VK dma_block::allocate semantics), so a fresh allocation implies the load.
		 */
		std::pair<id<MTLBuffer>, u32> map_dma(const utils::address_range32& range, bool load);

		/**
		 * Record the readback of src_area of src into dma_buffer with the guest layout (rsx_pitch rows, tiling encode,
		 * byteswap, depth/stencil packing via compute jobs, same as VK). Afterwards:
		 * m_cmd = &cmd, dma_sync_id = cmd.current_sync_id(), synchronized = true, sync_timestamp = rsx::get_shared_tag().
		 */
		void dma_transfer(mtl::command_context& cmd, mtl::image* src, const areai& src_area, const utils::address_range32& valid_range, u32 pitch);

		// Same as VK (uses mtl::get_typeless_helper + mtl::copy_scaled_image for scaled/MSAA surfaces).
		void copy_texture(mtl::command_context& cmd, bool miss);

		/**
		 * Flush
		 * Waits for dma_sync_id (m_cmd->wait(), submits the current command buffer if the transfer was recorded in it),
		 * computes the flush range exactly like VK (section range for framebuffer storage, confirmed range otherwise,
		 * tile-limited length for tiled regions), copies the part of it covered by dma_written_range from dma_buffer into
		 * guest memory (replaces vk::flush_dma) and applies the CPU swizzle for swizzled surfaces.
		 * dma_sync_id is kept after the wait so that repeated flushes of the same transfer (flush_always) work.
		 */
		void imp_flush() override;

		void* map_synchronized(u32, u32)
		{
			// Unused, imp_flush is overridden (same as VK)
			return nullptr;
		}

		void finish_flush()
		{}

		/**
		 * Misc
		 */
		void set_unpack_swap_bytes(bool swap_bytes)
		{
			pack_unpack_swap_bytes = swap_bytes;
		}

		void set_rsx_pitch(u32 pitch)
		{
			ensure(!is_locked());
			rsx_pitch = pitch;
		}

		void sync_surface_memory(const rsx::simple_array<cached_texture_section*>& surfaces);

		bool has_compatible_format(mtl::image* tex) const
		{
			return vram_texture->info.format == tex->info.format;
		}

		bool is_depth_texture() const
		{
			return !!(vram_texture->aspect() & mtl::image_aspect::depth);
		}
	};

	class texture_cache : public rsx::texture_cache<mtl::texture_cache, mtl::texture_cache_traits>
	{
	private:
		using baseclass = rsx::texture_cache<mtl::texture_cache, mtl::texture_cache_traits>;
		friend baseclass;

		struct cached_image_t
		{
			u64 key;
			std::unique_ptr<mtl::viewable_image> data;

			cached_image_t() = default;
			cached_image_t(u64 key_, std::unique_ptr<mtl::viewable_image>& data_) :
				key(key_), data(std::move(data_)) {}
		};

	public:
		enum texture_create_flags : u32
		{
			initialize_image_contents = 1,
			do_not_reuse = 2,
			shareable = 4,      // No-op on Metal (no concurrent queue sharing)
			mutable_format = 8  // Adds MTLTextureUsagePixelFormatView (srgb/snorm reinterpretation views)
		};

		void on_section_destroyed(cached_texture_section& tex) override;

	private:

		// Metal internals
		mtl::render_device* m_device = nullptr;
		mtl::gpu_formats_support m_formats_support{};
		mtl::ring_buffer* m_texture_upload_heap = nullptr;

		// Stuff that has been dereferenced by the GPU goes into these
		const u32 max_cached_image_pool_size = 256;
		std::deque<cached_image_t> m_cached_images;
		atomic_t<u64> m_cached_memory_size = { 0 };
		shared_mutex m_cached_pool_lock;

		// Blocks some operations when exiting
		atomic_t<bool> m_cache_is_exiting = false;

		// Images whose raw pointer is handed out to the caller (upload_image_simple). VK disposes them through the
		// resource manager (deleted when the frame retires); here they are kept alive until the next on_frame_end().
		// Also holds the reuse pool images evicted by on_frame_end()/handle_memory_pressure(). Guarded by m_cached_pool_lock.
		std::vector<std::unique_ptr<mtl::viewable_image>> m_frame_scoped_images;

		void clear();

		/**
		 * Moves an image into the reuse pool (m_cached_images) under m_cached_pool_lock, or destroys it if the cache is
		 * exiting. Replaces VK's cached_image_reference_t destructor (executed immediately, see the note at the top).
		 * The pool key is computed by a hash_image_properties(format, w, h, d, mipmaps, type) helper in the .mm
		 * (MTLPixelFormat needs 10 bits, MTLTextureType 4 bits; cubemaps are MTLTextureTypeCube so no create flags).
		 */
		void add_to_reuse_pool(std::unique_ptr<mtl::viewable_image>& image);

		MTLTextureSwizzleChannels apply_component_mapping_flags(u32 gcm_format, rsx::component_order flags, const rsx::texture_channel_remap_t& remap_vector) const;

		void copy_transfer_regions_impl(mtl::command_context& cmd, mtl::image* dst, const rsx::simple_array<copy_region_descriptor>& sections_to_transfer) const;

		mtl::image* get_template_from_collection_impl(const rsx::simple_array<copy_region_descriptor>& sections_to_transfer) const;

		// Returns a pooled image with matching properties whose usage is a superset of 'usage', or nullptr.
		std::unique_ptr<mtl::viewable_image> find_cached_image(MTLPixelFormat format, u16 w, u16 h, u16 d, u16 mipmaps, MTLTextureType type, MTLTextureUsage usage);

	protected:
		/**
		 * image_type: MTLTextureType2D or MTLTextureType3D (as VK's VkImageType; a 2DMultisample source type must be
		 * passed as 2D). view_type: MTLTextureType2D, MTLTextureType3D or MTLTextureTypeCube (VkImageViewType).
		 * A cube view_type creates an MTLTextureTypeCube texture (6 faces).
		 */
		mtl::image_view* create_temporary_subresource_view_impl(
			mtl::command_context& cmd, mtl::image* source, MTLTextureType image_type, MTLTextureType view_type,
			u32 gcm_format, u16 w, u16 h, u16 d, u8 mips, const rsx::texture_channel_remap_t& remap_vector,
			const copy_region_descriptor* copy = nullptr);

		mtl::image_view* create_temporary_subresource_view(mtl::command_context& cmd, const deferred_subresource& desc) override;

		mtl::image_view* generate_cubemap_from_images(mtl::command_context& cmd, const deferred_subresource& desc) override;

		mtl::image_view* generate_3d_from_2d_images(mtl::command_context& cmd, const deferred_subresource& desc) override;

		mtl::image_view* generate_atlas_from_images(mtl::command_context& cmd, const deferred_subresource& desc) override;

		mtl::image_view* generate_2d_mipmaps_from_images(mtl::command_context& cmd, const deferred_subresource& desc) override;

		void release_temporary_subresource(mtl::image_view* view) override;

		// VK stages this through the auxiliary vk::get_upload_heap(); use *mtl::get_upload_heap() (declared in MTLTexture.h,
		// shared with MTLRenderTargets' load_memory and detile_memory_block).
		void initialize_subresource_from_memory(mtl::command_context& cmd, mtl::image* dst, const deferred_subresource& desc, rsx::texture_dimension_extended type) const;

		void update_image_contents(mtl::command_context& cmd, mtl::image_view* dst_view, const deferred_subresource& desc) override;

		cached_texture_section* create_new_texture(mtl::command_context& cmd, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch,
			u32 gcm_format, rsx::texture_upload_context context, rsx::texture_dimension_extended type, bool swizzled, rsx::component_order swizzle_flags, rsx::flags32_t flags) override;

		cached_texture_section* create_nul_section(mtl::command_context& cmd, const utils::address_range32& rsx_range, const rsx::image_section_attributes_t& attrs,
			const rsx::GCM_tile_reference& tile, bool memory_load) override;

		cached_texture_section* upload_image_from_cpu(mtl::command_context& cmd, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch, u32 gcm_format,
			rsx::texture_upload_context context, const std::vector<rsx::subresource_layout>& subresource_layout, rsx::texture_dimension_extended type, bool swizzled) override;

		void set_component_order(cached_texture_section& section, u32 gcm_format, rsx::component_order expected_flags) override;

		// Feedback loop handling: forwards to mtl::render_target::texture_barrier(cmd) (ends the open render pass)
		void insert_texture_barrier(mtl::command_context& cmd, mtl::image* tex, bool strong_ordering) override;

		bool render_target_format_is_compatible(mtl::image* tex, u32 gcm_format) override;

		// Nothing to prepare (command buffers are created on demand)
		void prepare_for_dma_transfers(mtl::command_context& cmd) override;

		// Submits cmd so the hard-fault transfers start immediately; imp_flush waits on each section's dma_sync_id.
		void cleanup_after_dma_transfers(mtl::command_context& cmd) override;

	public:
		using baseclass::texture_cache;

		// VK: initialize(vk::render_device&, VkQueue submit_queue, vk::data_heap& upload_heap). The submit queue is not
		// needed (submission goes through the command context); the formats support is passed explicitly.
		void initialize(mtl::render_device& device, const mtl::gpu_formats_support& formats_support, mtl::ring_buffer& upload_heap);

		void destroy() override;

		/**
		 * VK: (format_class, format, w, h, d, layers, mips, VkImageType, VkFlags image_flags, VkFlags usage_flags).
		 * image_flags has no Metal equivalent (cube-ness is part of the texture type) and is dropped.
		 * image_type: MTLTextureType2D / MTLTextureType3D / MTLTextureTypeCube (layers must then be 6).
		 * usage_flags is OR'ed with MTLTextureUsageShaderRead, MTLTextureUsagePixelFormatView and, for renderable formats,
		 * MTLTextureUsageRenderTarget (needed for clears and blit_pass scaling). Private storage. Returns nullptr on OOM.
		 */
		std::unique_ptr<mtl::viewable_image> create_temporary_subresource_storage(
			rsx::format_class format_class, MTLPixelFormat format,
			u16 width, u16 height, u16 depth, u16 layers, u8 mips,
			MTLTextureType image_type, MTLTextureUsage usage_flags);

		// Moves the image into the reuse pool. The object stays alive (pool-owned) so raw pointers taken before the call
		// remain valid until the pool is trimmed in on_frame_end() or by handle_memory_pressure().
		void dispose_reusable_image(std::unique_ptr<mtl::viewable_image>& tex);

		bool is_depth_texture(u32 rsx_address, u32 rsx_size) override;

		void on_frame_end() override;

		/**
		 * Uploads a linear guest memory range as a 2D texture (BGRA8Unorm / RGBA8Unorm only, byteswapped like VK).
		 * The returned image is owned by the cache and stays valid until the next on_frame_end().
		 */
		mtl::viewable_image* upload_image_simple(mtl::command_context& cmd, MTLPixelFormat format, u32 address, u32 width, u32 height, u32 pitch);

		bool blit(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate, mtl::surface_cache& m_rtts, mtl::command_context& cmd);

		u32 get_unreleased_textures_count() const override;

		bool handle_memory_pressure(rsx::problem_severity severity) override;

		u64 get_temporary_memory_in_use() const;

		// VK compares the VMM texture cache pool against a quota derived from device-local memory. Metal has no VMM:
		// compare the texture cache footprint against a quota derived from mtl::get_caps().recommended_working_set_size.
		bool is_overallocated() const;
	};
}
