#include "stdafx.h"
#include "MTLTextureCache.h"

// Port of VKTextureCache.cpp (+ the inline parts of VKTextureCache.h).
// See the notes at the top of MTLTextureCache.h for the Metal deviations (no layouts/barriers, immediate reuse pool,
// per-section shared readback buffers instead of the VKDMA blocks, sync ids instead of vk::event).

#include "MTLGSRender.h"
#include "MTLCompute.h"
#include "MTLTexture.h"
#include "MTLFormats.h"
#include "MTLHelpers.h"
#include "MTLRenderTargets.h"

#include "mtlutils/buffer.h"
#include "mtlutils/command_context.h"
#include "mtlutils/device.h"
#include "mtlutils/image.h"

#include "Emu/Memory/vm.h"
#include "Emu/RSX/RSXThread.h"
#include "Emu/system_config.h"

#include "util/asm.hpp"

#include <algorithm>
#include <cstring>

namespace mtl
{
	namespace
	{
		// Granularity of the readback buffers (VK: s_dma_block_length)
		constexpr u32 c_dma_block_alignment = 0x10000;

		u64 hash_image_properties(MTLPixelFormat format, u16 w, u16 h, u16 d, u16 mipmaps, MTLTextureType type)
		{
			/**
			* Key layout:
			* 00-10: Format  (Max 1023)  <- MTLPixelFormat values go beyond 255
			* 10-26: Width   (Max 64K)
			* 26-42: Height  (Max 64K)
			* 42-51: Depth   (Max 512)   <- Encoded as depth - 1 since depth is always in the range [1, 512]
			* 51-56: Mipmaps (Max 31)
			* 56-60: Type    (Max 15)    <- MTLTextureType. Cubemaps are MTLTextureTypeCube, so no create flags are needed.
			* Metal has no sharing modes (VK bit 56).
			*/
			ensure(static_cast<u64>(format) < 0x400);
			ensure(d >= 1 && d <= 512);
			ensure(mipmaps < 32);
			ensure(static_cast<u64>(type) < 16);
			return (static_cast<u64>(format) & 0x3FF) |
				(static_cast<u64>(w) << 10) |
				(static_cast<u64>(h) << 26) |
				(static_cast<u64>(d - 1) << 42) |
				(static_cast<u64>(mipmaps) << 51) |
				(static_cast<u64>(type) << 56);
		}

		// VK: TRANSFER_DST | (TRANSFER_SRC) | SAMPLED. Metal blits need no usage flags. Clears (empty render passes) and the
		// scaled copies (blit_pass) render into the image, so renderable formats also get MTLTextureUsageRenderTarget.
		// PixelFormatView is required for the remapping/reinterpretation views (the image wrapper adds it anyway).
		MTLTextureUsage get_texture_cache_usage(MTLPixelFormat format, MTLTextureUsage base_usage, MTLTextureType type)
		{
			MTLTextureUsage usage = base_usage | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
			if (!mtl::is_compressed_format(format) && type != MTLTextureType1D && type != MTLTextureType1DArray)
			{
				usage |= MTLTextureUsageRenderTarget;
			}

			return usage;
		}

		// Metal only supports depth/stencil pixel formats on 2D, 2D array, cube and cube array textures.
		// Returns a color format with the same texel size used in place of a depth format for 1D/3D textures.
		MTLPixelFormat get_depth_substitute_format(MTLPixelFormat format)
		{
			switch (format)
			{
			case MTLPixelFormatDepth16Unorm:
				return MTLPixelFormatR16Unorm;
			case MTLPixelFormatDepth32Float:
			case MTLPixelFormatDepth32Float_Stencil8:
			case MTLPixelFormatDepth24Unorm_Stencil8:
				return MTLPixelFormatR32Float;
			default:
				return format;
			}
		}

		/**
		 * Applies the Metal texture type restrictions that have no VK counterpart:
		 * - 1D textures cannot be mipmapped: the image gets a single level (the section keeps the RSX mip count, upload
		 *   paths skip the missing levels). The shader decompiler samples 1D textures at level 0 anyway.
		 * - Depth formats are not allowed on 1D/3D textures: a color format of the same size is used instead.
		 * Returns true if the pixel format was substituted.
		 */
		bool apply_image_type_restrictions(MTLTextureType type, MTLPixelFormat& format, rsx::format_class& format_class, u16& mipmaps)
		{
			if (type == MTLTextureType1D || type == MTLTextureType1DArray)
			{
				mipmaps = 1;
			}

			if ((type == MTLTextureType1D || type == MTLTextureType1DArray || type == MTLTextureType3D) &&
				mtl::is_depth_format(format))
			{
				format = get_depth_substitute_format(format);
				format_class = RSX_FORMAT_CLASS_COLOR;
				return true;
			}

			return false;
		}

		// Drops the subresources addressing mip levels the image does not have (1D textures, see above)
		void filter_missing_levels(std::vector<rsx::subresource_layout>& layout, u32 level_count)
		{
			layout.erase(std::remove_if(layout.begin(), layout.end(), [level_count](const rsx::subresource_layout& subres)
			{
				return subres.level >= level_count;
			}), layout.end());
		}

		MTLGSRender* get_metal_renderer()
		{
			return static_cast<MTLGSRender*>(rsx::get_current_renderer());
		}

		// Returns the command buffer flags of the renderer if cmd is its primary command context (VK: cmd.flags)
		rsx::flags32_t* get_cb_flags_for(mtl::command_context& cmd)
		{
			auto renderer = get_metal_renderer();
			if (renderer && &renderer->get_command_context() == &cmd)
			{
				return &renderer->get_cb_flags();
			}

			return nullptr;
		}
	}

	//
	// cached_texture_section
	//

	void cached_texture_section::create(u16 w, u16 h, u16 depth, u16 mipmaps, mtl::image* image, u32 rsx_pitch, bool managed, u32 gcm_format, bool pack_swap_bytes)
	{
		auto new_texture = static_cast<mtl::viewable_image*>(image);
		ensure(!exists() || !is_managed() || vram_texture == new_texture);

		if (vram_texture != new_texture && !managed_texture && get_protection() == utils::protection::no)
		{
			// In-place image swap, still locked. Likely a color buffer that got rebound as depth buffer or vice-versa.
			mtl::as_rtt(vram_texture)->on_swap_out();

			if (!managed)
			{
				// Incoming is also an external resource, reference it immediately
				mtl::as_rtt(image)->on_swap_in(is_locked());
			}
		}

		vram_texture = new_texture;

		ensure(rsx_pitch);

		width = w;
		height = h;
		this->depth = depth;
		this->mipmaps = mipmaps;
		this->rsx_pitch = rsx_pitch;

		this->gcm_format = gcm_format;
		this->pack_unpack_swap_bytes = pack_swap_bytes;

		if (managed)
		{
			managed_texture.reset(vram_texture);
		}

		if (auto rtt = mtl::try_as_rtt(image))
		{
			swizzled = (rtt->raster_type != rsx::surface_raster_type::linear);
		}

		if (synchronized)
		{
			// Even if we are managing the same vram section, we cannot guarantee contents are static
			// The create method is only invoked when a new managed session is required
			release_dma_resources();
			synchronized = false;
			flushed = false;
			sync_timestamp = 0ull;
		}

		// Notify baseclass
		baseclass::on_section_resources_created();
	}

	void cached_texture_section::create(u16 w, u16 h, u16 depth, u16 mipmaps, mtl::image* image, u32 rsx_pitch, bool managed, const mtl::render_target* surface)
	{
		u32 gcm_format;
		bool swap_bytes;

		if (surface->is_depth_surface())
		{
			// NOTE: Same mapping as VK (VKTextureCache.h)
			gcm_format = (surface->get_surface_depth_format() != rsx::surface_depth_format::z16) ? CELL_GCM_TEXTURE_DEPTH16 : CELL_GCM_TEXTURE_DEPTH24_D8;
			swap_bytes = true;
		}
		else
		{
			auto info = rsx::get_compatible_gcm_format(surface->get_surface_color_format());
			gcm_format = info.first;
			swap_bytes = info.second;
		}

		create(w, h, depth, mipmaps, image, rsx_pitch, managed, gcm_format, swap_bytes);
	}

	void cached_texture_section::release_dma_resources()
	{
		// VK disposes the dma_fence. The readback buffer is kept for the next transfer (map_dma checks
		// dma_buffer_sync_id before writing into it), destroy() frees it.
		dma_sync_id = 0;
	}

	void cached_texture_section::dma_abort()
	{
		// Called if a reset occurs, usually via reprotect path after a bad prediction.
		// Discard the sync point, the next sync, if any, will properly recreate this.
		ensure(synchronized);
		ensure(!flushed);
		ensure(dma_sync_id);
		dma_sync_id = 0;
	}

	void cached_texture_section::destroy()
	{
		if (!exists() && context != rsx::texture_upload_context::dma)
			return;

		m_tex_cache->on_section_destroyed(*this);

		vram_texture = nullptr;
		ensure(!managed_texture);
		release_dma_resources();

		// Command buffers retain the buffer if a transfer into it is still in flight
		dma_buffer.reset();
		dma_base_address = 0;
		dma_buffer_sync_id = 0;
		dma_written_range = {};

		baseclass::on_section_resources_destroyed();
	}

	std::pair<id<MTLBuffer>, u32> cached_texture_section::map_dma(const utils::address_range32& range, bool load)
	{
		ensure(range.valid());

		const bool covered = dma_buffer &&
			range.start >= dma_base_address &&
			(static_cast<u64>(range.end) - dma_base_address) < dma_buffer->size();

		// A transfer recorded earlier may not have executed yet. Writing into the buffer with the CPU now would let that
		// transfer overwrite the loaded data afterwards (GPU writes are ordered after this point).
		const bool busy = load && dma_buffer && dma_buffer_sync_id &&
			!(m_cmd && m_cmd->is_complete(dma_buffer_sync_id));

		if (!covered || busy)
		{
			// Also cover the section range so that the following transfers of this section reuse the buffer
			u32 first = range.start;
			u32 last = range.end;

			if (const auto section_range = get_section_range(); section_range.valid())
			{
				first = std::min(first, section_range.start);
				last = std::max(last, section_range.end);
			}

			const u32 base_address = first & ~(c_dma_block_alignment - 1);
			const u64 limit = utils::align<u64>(static_cast<u64>(last) + 1, c_dma_block_alignment);
			const usz size = static_cast<usz>(limit - base_address);

			auto new_buffer = std::make_unique<mtl::buffer>(size, MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked,
				"Texture cache readback buffer");

			// Initialize memory contents (VK: dma_block::allocate).
			// Pre-loading the contents keeps the bytes the GPU does not write consistent with guest memory.
			auto dst = static_cast<u8*>(new_buffer->map(0));
			auto src = vm::get_super_ptr<u8>(base_address);

			if (rsx::get_location(base_address) == CELL_GCM_LOCATION_LOCAL ||
				vm::check_addr(base_address, 0, static_cast<u32>(size))) [[likely]]
			{
				// Linear virtual memory space. Copy all at once.
				std::memcpy(dst, src, size);
			}
			else
			{
				// Some games will have address holes in the range and page in data using faults.
				// Copy page by page. Slow, but we only really have to do this a handful of times.
				// Note that base_address is 64k aligned.
				for (u64 address = base_address; address < limit; address += 4096, src += 4096, dst += 4096)
				{
					if (vm::check_addr(static_cast<u32>(address), 0))
					{
						std::memcpy(dst, src, 4096);
					}
				}
			}

			dma_buffer = std::move(new_buffer);
			dma_base_address = base_address;
			dma_buffer_sync_id = 0;
			dma_written_range = {};
		}
		else if (load)
		{
			// VK: vk::load_dma
			std::memcpy(dma_buffer->map(range.start - dma_base_address), vm::get_super_ptr<u8>(range.start), range.length());
		}

		return { dma_buffer->handle(), range.start - dma_base_address };
	}

	void cached_texture_section::dma_transfer(mtl::command_context& cmd, mtl::image* src, const areai& src_area, const utils::address_range32& valid_range, u32 pitch)
	{
		ensure(src->samples() == 1);

		// NOTE: This can be reached if previously synchronized, or a special path happens.
		// If a hard flush occurred while this surface was flush_always the cache would have reset its protection afterwards.
		// The previous transfer, if any, is superseded by this one (VK disposes the old dma_fence here).
		dma_sync_id = 0;

		if (cmd.is_render_pass_open())
		{
			cmd.end_encoding();
		}

		const auto internal_bpp = mtl::get_format_texel_width(src->format());
		const auto transfer_width = static_cast<u32>(src_area.width());
		const auto transfer_height = static_cast<u32>(src_area.height());
		real_pitch = internal_bpp * transfer_width;
		rsx_pitch = pitch;

		// Row size of the transferred image data (real_pitch changes to the tile pitch below for tiled regions)
		const u32 transfer_pitch = real_pitch;

		const bool require_format_conversion = !!(src->aspect() & mtl::image_aspect::stencil) || src->format() == MTLPixelFormatDepth32Float;
		const auto tiled_region = rsx::get_current_renderer()->get_tiled_memory_region(valid_range);
		const bool require_tiling = !!tiled_region;
		const bool require_gpu_transform = require_format_conversion || pack_unpack_swap_bytes || require_tiling;

		auto dma_sync_region = valid_range;
		std::pair<id<MTLBuffer>, u32> dma_mapping = { nil, 0 };
		utils::address_range32 dma_mapped_range{};
		bool dma_loaded = false;

		auto dma_sync = [&](bool load, bool force = false)
		{
			if (dma_mapping.first && !force)
			{
				return;
			}

			// Metal: the buffer must also cover the last row written by the row-by-row copies below
			// (VK writes into the global DMA blocks which extend past the requested range).
			u32 last_address = dma_sync_region.end;
			if (transfer_height && transfer_pitch)
			{
				const u64 rows_end = static_cast<u64>(dma_sync_region.start) + static_cast<u64>(transfer_height - 1) * rsx_pitch + transfer_pitch - 1;
				last_address = static_cast<u32>(std::max<u64>(last_address, std::min<u64>(rows_end, u32{umax})));
			}

			dma_mapped_range = utils::address_range32::start_end(dma_sync_region.start, last_address);
			dma_mapping = map_dma(dma_mapped_range, load);
			dma_loaded |= load;
		};

		// Guest range whose buffer contents are defined once the copies below have executed
		utils::address_range32 written_range{};

		if (require_gpu_transform)
		{
			const auto task_length = transfer_pitch * src_area.height();
			auto working_buffer_length = mtl::calculate_working_buffer_size(task_length, src->aspect());

#if !DEBUG_DMA_TILING
			if (require_tiling)
			{
				// Safety padding
				working_buffer_length += tiled_region.tile->size;

				// Calculate actual working section for the memory op
				dma_sync_region = tiled_region.tile_align(dma_sync_region);
			}
#endif
			u32 result_offset = 0;

			// Keep a strong reference: the scratch buffer object may be replaced if a later request grows it
			id<MTLBuffer> working_buffer = mtl::get_scratch_buffer(cmd, working_buffer_length)->handle();

			mtl::buffer_image_copy region{};
			region.aspect = src->aspect();
			region.mip_level = 0;
			region.base_layer = 0;
			region.layer_count = 1;
			region.image_offset = MTLOriginMake(src_area.x1, src_area.y1, 0);
			region.image_extent = MTLSizeMake(transfer_width, transfer_height, 1);

			mtl::image_readback_options_t xfer_options{};
			xfer_options.swap_bytes = require_format_conversion && pack_unpack_swap_bytes;
			mtl::copy_image_to_buffer(cmd, src, working_buffer, region, xfer_options);

			// NOTE: For depth/stencil formats, copying to buffer and byteswap are combined into one step above
			if (pack_unpack_swap_bytes && !require_format_conversion)
			{
				const auto texel_layout = mtl::get_format_element_size(src->format());
				const auto elem_size = texel_layout.first;
				mtl::cs_shuffle_base* shuffle_kernel;

				if (elem_size == 2)
				{
					shuffle_kernel = mtl::get_compute_task<mtl::cs_shuffle_16>();
				}
				else if (elem_size == 4)
				{
					shuffle_kernel = mtl::get_compute_task<mtl::cs_shuffle_32>();
				}
				else
				{
					ensure(get_context() == rsx::texture_upload_context::dma);
					shuffle_kernel = nullptr;
				}

				if (shuffle_kernel)
				{
					// Metal: hazard tracking orders the blit -> compute -> blit accesses of the scratch buffer
					shuffle_kernel->run(cmd, working_buffer, task_length);
				}
			}

			if (require_tiling)
			{
#if !DEBUG_DMA_TILING
				// We don't need to calibrate write if two conditions are met:
				// 1. The start offset of our 2D region is a multiple of 64 lines
				// 2. We use the whole pitch.
				// If these conditions are not met, we need to upload the entire tile (or at least the affected tiles wholly)

				// FIXME: There is a 3rd condition - write onto already-persisted range. e.g One transfer copies half the image then the other half is copied later.
				// We don't need to load again for the second copy in that scenario.

				if (valid_range.start != dma_sync_region.start || real_pitch != tiled_region.tile->pitch)
				{
					// Tile indices run to the end of the row (full pitch).
					// Tiles address outside their 64x64 area too, so we need to actually load the whole thing and "fill in" missing blocks.
					// Visualizing "hot" pixels when doing a partial copy is very revealing, there's lots of data from the padding areas to be filled in.

					dma_sync(true);
					ensure(dma_mapping.first);

					// Upload memory to the working buffer
					const auto dst_offset = task_length; // Append to the end of the input
					[cmd.blit() copyFromBuffer:dma_mapping.first
						sourceOffset:dma_mapping.second
						toBuffer:working_buffer
						destinationOffset:dst_offset
						size:dma_sync_region.length()];
				}

				// Prepare payload
				const mtl::RSX_detiler_config config =
				{
					.tile_base_address = tiled_region.base_address,
					.tile_base_offset = valid_range.start - tiled_region.base_address,
					.tile_rw_offset = dma_sync_region.start - tiled_region.base_address,
					.tile_size = tiled_region.tile->size,
					.tile_pitch = tiled_region.tile->pitch,
					.bank = tiled_region.tile->bank,

					.dst = working_buffer,
					.dst_offset = task_length,
					.src = working_buffer,
					.src_offset = 0,

					// TODO: Check interaction with anti-aliasing
					.image_width = static_cast<u16>(transfer_width),
					.image_height = static_cast<u16>(transfer_height),
					.image_pitch = real_pitch,
					.image_bpp = context == rsx::texture_upload_context::dma ? internal_bpp : rsx::get_format_block_size_in_bytes(gcm_format)
				};

				// Execute
				const auto job = mtl::get_compute_task<mtl::cs_tile_memcpy<mtl::RSX_detiler_op::encode>>();
				job->run(cmd, config);

				// Update internal variables
				result_offset = task_length;
				real_pitch = tiled_region.tile->pitch; // We're always copying the full image. In case of partials we're "filling in" blocks, not doing partial 2D copies.

#if VISUALIZE_GPU_TILING
				if (g_cfg.video.renderdoc_compatiblity)
				{
					// Debug write
					auto scratch_img = mtl::get_typeless_helper(MTLPixelFormatBGRA8Unorm, RSX_FORMAT_CLASS_COLOR, tiled_region.tile->pitch / 4, 768);

					mtl::buffer_image_copy dbg_copy{};
					dbg_copy.buffer_offset = config.dst_offset;
					dbg_copy.image_extent = MTLSizeMake(width, height, 1);
					dbg_copy.buffer_row_length = tiled_region.tile->pitch / 4;
					dbg_copy.aspect = mtl::image_aspect::color;
					dbg_copy.mip_level = 0;
					dbg_copy.base_layer = 0;
					dbg_copy.layer_count = 1;
					mtl::copy_buffer_to_image(cmd, working_buffer, scratch_img, dbg_copy);
				}
#endif

#endif
			}

			if (rsx_pitch == real_pitch) [[likely]]
			{
				dma_sync(false);

				// VK copies dma_sync_region.length() bytes unconditionally. Metal validates buffer bounds (an out of
				// range read faults the command buffer), so never read past the end of the working buffer.
				const u64 available_length = (working_buffer.length > result_offset) ? (working_buffer.length - result_offset) : 0;
				const u32 copy_length = static_cast<u32>(std::min<u64>(dma_sync_region.length(), available_length));

				if (copy_length != dma_sync_region.length()) [[unlikely]]
				{
					rsx_log.error("[Metal] DMA transfer of 0x%x bytes at 0x%x exceeds the working buffer, truncated to 0x%x bytes",
						dma_sync_region.length(), dma_sync_region.start, copy_length);
				}

				if (copy_length)
				{
					[cmd.blit() copyFromBuffer:working_buffer
						sourceOffset:result_offset
						toBuffer:dma_mapping.first
						destinationOffset:dma_mapping.second
						size:copy_length];
				}

				if (dma_loaded)
				{
					written_range = dma_mapped_range;
				}
				else if (copy_length)
				{
					written_range = utils::address_range32::start_length(dma_sync_region.start, copy_length);
				}
			}
			else
			{
				dma_sync(true);

				auto blit = cmd.blit();
				u32 dst_offset = dma_mapping.second;
				u32 src_offset = result_offset;

				for (unsigned row = 0; row < transfer_height; ++row)
				{
					[blit copyFromBuffer:working_buffer
						sourceOffset:src_offset
						toBuffer:dma_mapping.first
						destinationOffset:dst_offset
						size:transfer_pitch];

					src_offset += real_pitch;
					dst_offset += rsx_pitch;
				}

				written_range = dma_mapped_range;
			}
		}
		else
		{
			// Metal deviation: VK does not load here and relies on the DMA block (or the imported guest memory) holding
			// the bytes between the rows. Load whenever the copy does not cover the whole mapped range so that imp_flush
			// never writes stale data back (identical to the VK host-memory-import path).
			const u64 rows_end = static_cast<u64>(dma_sync_region.start) + static_cast<u64>(transfer_height) * rsx_pitch - 1;
			const bool partial_write = (rsx_pitch != real_pitch) || (static_cast<u64>(dma_sync_region.end) > rows_end);
			dma_sync(partial_write);

			mtl::buffer_image_copy region{};
			region.buffer_row_length = (rsx_pitch / internal_bpp);
			region.aspect = src->aspect();
			region.mip_level = 0;
			region.base_layer = 0;
			region.layer_count = 1;
			region.image_offset = MTLOriginMake(src_area.x1, src_area.y1, 0);
			region.image_extent = MTLSizeMake(transfer_width, transfer_height, 1);
			region.buffer_offset = dma_mapping.second;

			// VK: raw vkCmdCopyImageToBuffer. No packing conversion applies here (require_format_conversion is false).
			mtl::copy_image_to_buffer(cmd, src, dma_mapping.first, region);

			written_range = dma_mapped_range;
		}

		// Record the sync point of this transfer (VK: dma_fence->signal())
		m_cmd = &cmd;
		dma_sync_id = cmd.current_sync_id();
		dma_buffer_sync_id = dma_sync_id;
		dma_written_range = written_range;

		// Set cb flag for queued dma operations
		if (auto cb_flags = get_cb_flags_for(cmd))
		{
			*cb_flags |= mtl::command_buffer_flags::cb_has_dma_transfer;
		}

		if (get_context() == rsx::texture_upload_context::dma)
		{
			// Save readback hint in case transformation is required later
			switch (internal_bpp)
			{
			case 2:
				gcm_format = CELL_GCM_TEXTURE_R5G6B5;
				break;
			case 4:
			default:
				gcm_format = CELL_GCM_TEXTURE_A8R8G8B8;
				break;
			}
		}

		synchronized = true;
		sync_timestamp = rsx::get_shared_tag();
	}

	void cached_texture_section::copy_texture(mtl::command_context& cmd, bool miss)
	{
		ensure(exists());

		if (!miss) [[likely]]
		{
			ensure(!synchronized);
			baseclass::on_speculative_flush();
		}
		else
		{
			baseclass::on_miss();
		}

		mtl::image* locked_resource = vram_texture;
		u32 transfer_width = width;
		u32 transfer_height = height;
		u32 transfer_x = 0, transfer_y = 0;

		if (context == rsx::texture_upload_context::framebuffer_storage)
		{
			auto surface = mtl::as_rtt(vram_texture);
			surface->memory_barrier(cmd, rsx::surface_access::transfer_read);
			locked_resource = surface->get_surface(rsx::surface_access::transfer_read);
			transfer_width *= surface->samples_x;
			transfer_height *= surface->samples_y;
		}

		mtl::image* target = locked_resource;
		if (transfer_width != locked_resource->width() || transfer_height != locked_resource->height())
		{
			// TODO: Synchronize access to typeles textures
			target = mtl::get_typeless_helper(vram_texture->format(), vram_texture->format_class(), transfer_width, transfer_height);

			// Allow bilinear filtering on color textures where compatibility is likely
			const bool linear_filter = (target->aspect() == mtl::image_aspect::color);

			mtl::copy_scaled_image(cmd, locked_resource, target,
				areai{ 0, 0, static_cast<s32>(locked_resource->width()), static_cast<s32>(locked_resource->height()) },
				areai{ 0, 0, static_cast<s32>(transfer_width), static_cast<s32>(transfer_height) },
				{}, true, linear_filter);
		}

		const auto internal_bpp = mtl::get_format_texel_width(vram_texture->format());
		const auto valid_range = get_confirmed_range();

		if (const auto section_range = get_section_range(); section_range != valid_range)
		{
			if (const auto offset = (valid_range.start - get_section_base()))
			{
				transfer_y = offset / rsx_pitch;
				transfer_x = (offset % rsx_pitch) / internal_bpp;

				ensure(transfer_width >= transfer_x);
				ensure(transfer_height >= transfer_y);
				transfer_width -= transfer_x;
				transfer_height -= transfer_y;
			}

			if (const auto tail = (section_range.end - valid_range.end))
			{
				const auto row_count = tail / rsx_pitch;

				ensure(transfer_height >= row_count);
				transfer_height -= row_count;
			}
		}

		areai src_area;
		src_area.x1 = static_cast<s32>(transfer_x);
		src_area.y1 = static_cast<s32>(transfer_y);
		src_area.x2 = s32(transfer_x + transfer_width);
		src_area.y2 = s32(transfer_y + transfer_height);
		dma_transfer(cmd, target, src_area, valid_range, rsx_pitch);
	}

	void cached_texture_section::imp_flush()
	{
		AUDIT(synchronized);

		if (!dma_sync_id || !dma_buffer) [[unlikely]]
		{
			rsx_log.error("[Metal] Texture cache flush of section 0x%x without a recorded transfer", get_section_base());
			return;
		}

		// Synchronize (VK: vk::wait_for_event(dma_fence)). Submits the command buffer if the transfer was recorded into the
		// one still being built.
		ensure(m_cmd)->wait(dma_sync_id);

		// Calculate smallest range to flush - for framebuffers, the raster region is enough
		const auto range = (context == rsx::texture_upload_context::framebuffer_storage) ? get_section_range() : get_confirmed_range();
		auto flush_length = range.length();

		const auto tiled_region = rsx::get_current_renderer()->get_tiled_memory_region(range);
		if (tiled_region)
		{
			const auto available_tile_size = tiled_region.tile->size - (range.start - tiled_region.base_address);
			const auto max_content_size = tiled_region.tile->pitch * utils::align(height, 64);
			flush_length = std::min(max_content_size, available_tile_size);
		}

		// VK: vk::flush_dma(range.start, flush_length).
		// Only the part defined by the last transfer is written back (see dma_written_range).
		const auto flush_range = utils::address_range32::start_length(range.start, flush_length).get_intersect(dma_written_range);
		if (flush_range.valid())
		{
			const u64 buffer_end = static_cast<u64>(dma_base_address) + dma_buffer->size();
			if (flush_range.start >= dma_base_address && static_cast<u64>(flush_range.end) < buffer_end) [[likely]]
			{
				std::memcpy(vm::get_super_ptr<u8>(flush_range.start), dma_buffer->map(flush_range.start - dma_base_address), flush_range.length());
			}
			else
			{
				rsx_log.error("[Metal] Flush range [0x%x, 0x%x] of section 0x%x is outside of its readback buffer [0x%x, 0x%llx)",
					flush_range.start, flush_range.end, get_section_base(), dma_base_address, buffer_end);
			}
		}

#if DEBUG_DMA_TILING
		// Are we a tiled region?
		if (const auto tiled_region = rsx::get_current_renderer()->get_tiled_memory_region(range))
		{
			auto real_data = vm::get_super_ptr<u8>(range.start);
			auto out_data = rsx::simple_array<u8>(tiled_region.tile->size);
			rsx::tile_texel_data<u32>(
				out_data.data(),
				real_data,
				tiled_region.base_address,
				range.start - tiled_region.base_address,
				tiled_region.tile->size,
				tiled_region.tile->bank,
				tiled_region.tile->pitch,
				width,
				height
			);
			std::memcpy(real_data, out_data.data(), flush_length);
		}
#endif

		if (is_swizzled())
		{
			// This format is completely worthless to CPU processing algorithms where cache lines on die are linear.
			// If this is happening, usually it means it was not a planned readback (e.g shared pages situation)
			rsx_log.trace("[Performance warning] CPU readback of swizzled data");

			// Read-modify-write to avoid corrupting already resident memory outside texture region
			void* data = get_ptr(range.start);
			rsx::simple_array<u8> tmp_data(rsx_pitch * height);
			std::memcpy(tmp_data.data(), data, tmp_data.size());

			switch (gcm_format)
			{
			case CELL_GCM_TEXTURE_A8R8G8B8:
			case CELL_GCM_TEXTURE_DEPTH24_D8:
				rsx::convert_linear_swizzle<u32, false>(tmp_data.data(), data, width, height, rsx_pitch);
				break;
			case CELL_GCM_TEXTURE_R5G6B5:
			case CELL_GCM_TEXTURE_DEPTH16:
				rsx::convert_linear_swizzle<u16, false>(tmp_data.data(), data, width, height, rsx_pitch);
				break;
			default:
				rsx_log.error("Unexpected swizzled texture format 0x%x", gcm_format);
			}
		}
	}

	void cached_texture_section::sync_surface_memory(const rsx::simple_array<cached_texture_section*>& surfaces)
	{
		auto rtt = mtl::as_rtt(vram_texture);
		rtt->sync_tag();

		for (auto& surface : surfaces)
		{
			rtt->inherit_surface_contents(mtl::as_rtt(surface->vram_texture));
		}
	}

	//
	// texture_cache
	//

	void texture_cache::add_to_reuse_pool(std::unique_ptr<mtl::viewable_image>& image)
	{
		ensure(image);

		// Move this object to the cached image pool (VK: ~cached_image_reference_t, executed once the GPU is done with the
		// image). Metal: command buffers retain the texture and hazard tracking orders any reuse after the pending accesses.
		const auto key = hash_image_properties(image->format(), static_cast<u16>(image->width()), static_cast<u16>(image->height()),
			static_cast<u16>(image->depth()), static_cast<u16>(image->mipmaps()), image->type());
		const u64 memory_size = image->memory_size();

		std::lock_guard lock(m_cached_pool_lock);

		if (!m_cache_is_exiting)
		{
			m_cached_memory_size += memory_size;
			m_cached_images.emplace_front(key, image);
		}
		else
		{
			// Destroy if the cache is closed
			image.reset();
		}
	}

	void texture_cache::on_section_destroyed(cached_texture_section& tex)
	{
		if (tex.is_managed() && tex.exists())
		{
			add_to_reuse_pool(tex.get_texture());
		}
	}

	void texture_cache::clear()
	{
		{
			std::lock_guard lock(m_cached_pool_lock);
			m_cache_is_exiting = true;
		}
		baseclass::clear();

		std::deque<cached_image_t> cached_images;
		std::vector<std::unique_ptr<mtl::viewable_image>> frame_scoped_images;
		{
			std::lock_guard lock(m_cached_pool_lock);
			cached_images.swap(m_cached_images);
			frame_scoped_images.swap(m_frame_scoped_images);
			m_cached_memory_size = 0;
		}
	}

	void texture_cache::copy_transfer_regions_impl(mtl::command_context& cmd, mtl::image* dst, const rsx::simple_array<copy_region_descriptor>& sections_to_transfer) const
	{
		const auto dst_aspect = dst->aspect();
		const auto dst_bpp = mtl::get_format_texel_width(dst->format());

		const auto configure_subresource = [&](const copy_region_descriptor& section, coord3i& coord, rsx::image_copy_subresource_layers& mip_layers)
		{
			mip_layers.dst_mip_level = section.level;
			coord.position = { section.dst_x, section.dst_y, 0 };
			coord.size = { section.dst_w, section.dst_h, 1 };

			if (dst->type() == MTLTextureType3D)
			{
				coord.position.z = section.dst_z;
			}
			else
			{
				mip_layers.dst_layer = static_cast<u8>(section.dst_z);
			}
		};

		// Writes data to the final destination (VK: get_output_region + vkCmdCopyImage)
		const auto copy_to_output = [&](const copy_region_descriptor& section, s32 in_x, s32 in_y, u16 w, u16 h, mtl::image* data_src)
		{
			coord3i dst_rect;
			rsx::image_copy_subresource_layers mip_layers;
			configure_subresource(section, dst_rect, mip_layers);
			dst_rect.size = { w, h, 1 };

			const auto src_rect = coord3i{ { in_x, in_y, 0 }, { w, h, 1 } };
			mtl::copy_image(cmd, data_src, dst, src_rect, dst_rect, mip_layers);
		};

		for (const auto& section : sections_to_transfer)
		{
			if (!section.src)
			{
				continue;
			}

			// NOTE: No layouts in Metal (VK pushes TRANSFER_SRC/GENERAL layouts on the inputs here)
			const bool typeless = section.src->aspect() != dst_aspect ||
				!mtl::formats_are_bitcast_compatible(dst, section.src);

			auto src_image = section.src;
			auto src_x = section.src_x;
			auto src_y = section.src_y;
			auto src_w = section.src_w;
			auto src_h = section.src_h;

			rsx::flags32_t transform = section.xform;
			if (section.xform == rsx::surface_transform::coordinate_transform)
			{
				// Dimensions were given in 'dst' space. Work out the real source coordinates
				const auto src_bpp = mtl::get_format_texel_width(section.src->format());
				src_x = (src_x * dst_bpp) / src_bpp;
				src_w = utils::aligned_div<u16>(src_w * dst_bpp, src_bpp);

				transform &= ~(rsx::surface_transform::coordinate_transform);
			}

			if (auto surface = mtl::try_as_rtt(section.src))
			{
				surface->transform_samples_to_pixels(src_x, src_w, src_y, src_h);
			}

			if (typeless) [[unlikely]]
			{
				const auto src_bpp = mtl::get_format_texel_width(section.src->format());
				const u16 convert_w = u16(src_w * src_bpp) / dst_bpp;
				const u16 convert_x = u16(src_x * src_bpp) / dst_bpp;

				if (convert_w == section.dst_w && src_h == section.dst_h &&
					transform == rsx::surface_transform::identity)
				{
					// Optimization to avoid double transfer
					coord3i dst_rect;
					rsx::image_copy_subresource_layers mip_layers;
					configure_subresource(section, dst_rect, mip_layers);

					const auto src_rect = coord3i{ { src_x, src_y, 0 }, { src_w, src_h, 1 } };
					mtl::copy_image_typeless(cmd, section.src, dst, src_rect, dst_rect, mip_layers);
					continue;
				}

				src_image = mtl::get_typeless_helper(dst->format(), dst->format_class(), convert_x + convert_w, src_y + src_h);

				const areai src_rect = coordi{ { src_x, src_y }, { src_w, src_h } };
				const areai dst_rect = coordi{ { 0, 0 }, { convert_w, src_h } };
				mtl::copy_image_typeless(cmd, section.src, src_image, src_rect, dst_rect);

				src_x = 0;
				src_y = 0;
				src_w = convert_w;
			}

			ensure(transform == rsx::surface_transform::identity);

			if (src_w == section.dst_w && src_h == section.dst_h) [[likely]]
			{
				copy_to_output(section, src_x, src_y, src_w, src_h, src_image);
			}
			else
			{
				mtl::image* _dst = dst;

				if (src_image->info.format != dst->info.format) [[unlikely]]
				{
					// Either a bitcast is required or a scale+copy to mipmap level / layer
					const u32 requested_width = dst->width();
					const u32 requested_height = src_y + src_h + section.dst_h; // Accounts for possible typeless ref on the same helper on src
					_dst = mtl::get_typeless_helper(src_image->format(), src_image->format_class(), requested_width, requested_height);
				}

				auto dst_rect = coord3i{ { section.dst_x, section.dst_y, 0 }, { section.dst_w, section.dst_h, 1 } };
				rsx::image_copy_subresource_layers mip_layers{ .dst_mip_level = section.level };

				if (_dst != dst)
				{
					// We place the output after the source to account for the initial typeless-xfer if applicable
					// If src_image == _dst then this is just a write-to-self. Either way, use best-fit placement.
					dst_rect.position.x = 0;
					dst_rect.position.y = src_y + src_h;
					mip_layers = {};
				}
				else if (dst->type() == MTLTextureType3D)
				{
					dst_rect.position.z = section.dst_z;
				}
				else
				{
					mip_layers.dst_layer = static_cast<u8>(section.dst_z);
				}

				mtl::copy_scaled_image(cmd, src_image, _dst,
					coord3i{ { src_x, src_y, 0 }, { src_w, src_h, 1 } },
					dst_rect,
					mip_layers,
					src_image->format() == _dst->format(),
					false); // VK_FILTER_NEAREST

				if (_dst != dst) [[unlikely]]
				{
					// Casting comes after the scaling!
					copy_to_output(section, dst_rect.position.x, dst_rect.position.y, section.dst_w, section.dst_h, _dst);
				}
			}
		}
	}

	MTLTextureSwizzleChannels texture_cache::apply_component_mapping_flags(u32 gcm_format, rsx::component_order flags, const rsx::texture_channel_remap_t& remap_vector) const
	{
		switch (gcm_format)
		{
		case CELL_GCM_TEXTURE_DEPTH24_D8:
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
		case CELL_GCM_TEXTURE_DEPTH16:
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
			// Dont bother letting this propagate
			return { MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed };
		default:
			break;
		}

		// NOTE: VK zero-initializes the mapping (VK_COMPONENT_SWIZZLE_IDENTITY). The Metal equivalent is the identity map,
		// a zero MTLTextureSwizzleChannels would read back zeroes.
		MTLTextureSwizzleChannels mapping = mtl::default_component_map;
		switch (flags)
		{
		case rsx::component_order::default_:
		{
			mapping = mtl::apply_swizzle_remap(mtl::get_component_mapping(gcm_format), remap_vector);
			break;
		}
		case rsx::component_order::native:
		{
			mapping = { MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha };
			break;
		}
		case rsx::component_order::swapped_native:
		{
			mapping = { MTLTextureSwizzleAlpha, MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue };
			break;
		}
		default:
			break;
		}

		return mapping;
	}

	mtl::image* texture_cache::get_template_from_collection_impl(const rsx::simple_array<copy_region_descriptor>& sections_to_transfer) const
	{
		if (sections_to_transfer.size() == 1) [[likely]]
		{
			return sections_to_transfer.front().src;
		}

		mtl::image* result = nullptr;
		for (const auto& section : sections_to_transfer)
		{
			if (!section.src)
				continue;

			if (!result)
			{
				result = section.src;
			}
			else
			{
				if (section.src->native_component_map != result->native_component_map)
				{
					// TODO
					// This requires a far more complex setup as its not always possible to mix and match without compute assistance
					return nullptr;
				}
			}
		}

		return result;
	}

	std::unique_ptr<mtl::viewable_image> texture_cache::find_cached_image(MTLPixelFormat format, u16 w, u16 h, u16 d, u16 mipmaps, MTLTextureType type, MTLTextureUsage usage)
	{
		reader_lock lock(m_cached_pool_lock);

		if (!m_cached_images.empty())
		{
			const u64 desired_key = hash_image_properties(format, w, h, d, mipmaps, type);
			lock.upgrade();

			for (auto it = m_cached_images.begin(); it != m_cached_images.end(); ++it)
			{
				if (it->key == desired_key && (it->data->info.usage & usage) == usage)
				{
					auto ret = std::move(it->data);
					m_cached_images.erase(it);
					m_cached_memory_size -= ret->memory_size();
					return ret;
				}
			}
		}

		return {};
	}

	std::unique_ptr<mtl::viewable_image> texture_cache::create_temporary_subresource_storage(
		rsx::format_class format_class, MTLPixelFormat format,
		u16 width, u16 height, u16 depth, u16 layers, u8 mips,
		MTLTextureType image_type, MTLTextureUsage usage_flags)
	{
		u16 image_mips = mips;
		const MTLPixelFormat requested_format = format;
		if (apply_image_type_restrictions(image_type, format, format_class, image_mips))
		{
			err_once("Depth format 0x%x is not supported on Metal texture type %u, using color format 0x%x instead",
				static_cast<u32>(requested_format), static_cast<u32>(image_type), static_cast<u32>(format));
		}

		if (image_type == MTLTextureTypeCube)
		{
			ensure(layers == 6);
		}

		const MTLTextureUsage usage = get_texture_cache_usage(format, usage_flags, image_type);
		auto image = find_cached_image(format, width, height, depth, image_mips, image_type, usage);

		if (image)
		{
			// The pool is keyed on the host format only
			image->set_format_class(format_class);
		}
		else
		{
			// NOTE: The image wrapper throws if the texture cannot be created, there is no VK_IMAGE_CREATE_ALLOW_NULL equivalent
			image = std::make_unique<mtl::viewable_image>(image_type, format,
				width, height, depth, image_mips, layers, 1, usage, format_class, MTLStorageModePrivate);
		}

		return image;
	}

	void texture_cache::dispose_reusable_image(std::unique_ptr<mtl::viewable_image>& image)
	{
		add_to_reuse_pool(image);
	}

	mtl::image_view* texture_cache::create_temporary_subresource_view_impl(
		mtl::command_context& cmd, mtl::image* source, MTLTextureType image_type, MTLTextureType view_type,
		u32 gcm_format, u16 w, u16 h, u16 d, u8 mips,
		const rsx::texture_channel_remap_t& remap_vector,
		const copy_region_descriptor* copy)
	{
		// Metal: cube-ness is part of the texture type (VK: VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT + cube view)
		const bool is_cubemap = (view_type == MTLTextureTypeCube);
		const MTLTextureType storage_type = is_cubemap ? MTLTextureTypeCube : image_type;
		const MTLTextureUsage usage_flags = MTLTextureUsageShaderRead;
		const MTLPixelFormat dst_format = mtl::get_compatible_sampler_format(m_formats_support, gcm_format);
		const u16 layers = is_cubemap ? 6 : 1;

		// Provision
		auto image = create_temporary_subresource_storage(rsx::classify_format(gcm_format), dst_format, w, h, d, layers, mips, storage_type, usage_flags);

		// OOM?
		if (!image)
		{
			return nullptr;
		}

		// This method is almost exclusively used to work on framebuffer resources
		// Keep the original swizzle layout unless there is data format conversion
		MTLTextureSwizzleChannels view_swizzle;
		if (!source || image->format() != source->info.format)
		{
			// This is a data cast operation
			// Use native mapping for the new type
			// TODO: Also simulate the readback+reupload step (very tricky)
			const auto remap = mtl::get_component_mapping(gcm_format);
			view_swizzle = { remap[1], remap[2], remap[3], remap[0] };
		}
		else
		{
			view_swizzle = source->native_component_map;
		}

		image->set_debug_name(fmt::format("Temp view, fmt=0x%x", gcm_format));
		image->set_native_component_layout(view_swizzle);
		auto view = image->get_view(remap_vector);

		if (copy)
		{
			rsx::simple_array<copy_region_descriptor> region = { *copy };
			copy_transfer_regions_impl(cmd, image.get(), region);
		}

		// TODO: Floating reference. We can do better with some restructuring.
		// The image is owned by its view from here on, release_temporary_subresource() reclaims it.
		image.release();
		return view;
	}

	mtl::image_view* texture_cache::create_temporary_subresource_view(mtl::command_context& cmd, const deferred_subresource& desc)
	{
		ensure(desc.sections_to_copy.size() == 1);
		const auto& section = desc.sections_to_copy.front();

		// VK passes the source image type (2D or 3D) with a 2D view. Metal cannot create 2D views of 3D textures, and the
		// requested depth is 1, so the temporary image is always 2D (slice 0 of a 3D source is copied).
		return create_temporary_subresource_view_impl(cmd, section.src, MTLTextureType2D, MTLTextureType2D,
			desc.gcm_format, desc.width, desc.height, 1, 1, desc.remap, &section);
	}

	mtl::image_view* texture_cache::generate_cubemap_from_images(mtl::command_context& cmd, const deferred_subresource& desc)
	{
		const auto& sections_to_copy = desc.sections_to_copy;
		auto _template = get_template_from_collection_impl(sections_to_copy);
		const u8 mip_count = desc.exact_mip_count();
		auto result = create_temporary_subresource_view_impl(cmd, _template, MTLTextureType2D,
			MTLTextureTypeCube, desc.gcm_format, desc.width, desc.height, 1, mip_count, desc.remap);

		if (!result)
		{
			// Failed to create temporary object, bail
			return nullptr;
		}

		const auto image = result->image();
		const u32 dst_aspect = image->aspect();

		// Whole image: { dst_aspect, 0, mip_count, 0, 6 }
		const mtl::image_subresource_range dst_range = { dst_aspect, 0, mip_count, 0, 6 };

		if (desc.force_bg_load)
		{
			// The memory load covers the whole image, no need to clear it first
			initialize_subresource_from_memory(cmd, image, desc, rsx::texture_dimension_extended::texture_dimension_cubemap);
		}
		else if (!(dst_aspect & mtl::image_aspect::depth))
		{
			mtl::clear_color_image(cmd, image, color4f(0.f, 0.f, 0.f, 0.f), dst_range);
		}
		else
		{
			mtl::clear_depth_stencil_image(cmd, image, 1.f, 0, dst_range);
		}

		// NOTE: The write-after-write ordering barrier is implicit (hazard tracking)
		copy_transfer_regions_impl(cmd, image, sections_to_copy);
		return result;
	}

	mtl::image_view* texture_cache::generate_3d_from_2d_images(mtl::command_context& cmd, const deferred_subresource& desc)
	{
		const auto& sections_to_copy = desc.sections_to_copy;
		auto _template = get_template_from_collection_impl(sections_to_copy);
		const u8 mip_count = desc.exact_mip_count();
		auto result = create_temporary_subresource_view_impl(cmd, _template, MTLTextureType3D,
			MTLTextureType3D, desc.gcm_format, desc.width, desc.height, desc.depth, mip_count, desc.remap);

		if (!result)
		{
			// Failed to create temporary object, bail
			return nullptr;
		}

		const auto image = result->image();
		const u32 dst_aspect = image->aspect();

		// VK: { dst_aspect, 0, mip_count, 0, 1 }, which covers every depth slice of a 3D image.
		// The Metal clear helpers address depth slices through the layer range (0 = all).
		const mtl::image_subresource_range dst_range = { dst_aspect, 0, mip_count, 0, 0 };

		if (desc.force_bg_load)
		{
			// The memory load covers the whole image, no need to clear it first
			initialize_subresource_from_memory(cmd, image, desc, rsx::texture_dimension_extended::texture_dimension_3d);
		}
		else if (!(dst_aspect & mtl::image_aspect::depth))
		{
			mtl::clear_color_image(cmd, image, color4f(0.f, 0.f, 0.f, 0.f), dst_range);
		}
		else
		{
			mtl::clear_depth_stencil_image(cmd, image, 1.f, 0, dst_range);
		}

		copy_transfer_regions_impl(cmd, image, sections_to_copy);
		return result;
	}

	mtl::image_view* texture_cache::generate_atlas_from_images(mtl::command_context& cmd, const deferred_subresource& desc)
	{
		const auto& sections_to_copy = desc.sections_to_copy;
		auto _template = get_template_from_collection_impl(sections_to_copy);
		auto result = create_temporary_subresource_view_impl(cmd, _template, MTLTextureType2D,
			MTLTextureType2D, desc.gcm_format, desc.width, desc.height, 1, 1, desc.remap);

		if (!result)
		{
			// Failed to create temporary object, bail
			return nullptr;
		}

		const auto image = result->image();
		const u32 dst_aspect = image->aspect();
		const mtl::image_subresource_range dst_range = { dst_aspect, 0, 1, 0, 1 };

		if (desc.force_bg_load)
		{
			// The memory load covers the whole image, no need to clear it first
			initialize_subresource_from_memory(cmd, image, desc, rsx::texture_dimension_extended::texture_dimension_2d);
		}
		else if (sections_to_copy[0].dst_w != desc.width || sections_to_copy[0].dst_h != desc.height)
		{
			if (!(dst_aspect & mtl::image_aspect::depth))
			{
				mtl::clear_color_image(cmd, image, color4f(0.f, 0.f, 0.f, 0.f), dst_range);
			}
			else
			{
				mtl::clear_depth_stencil_image(cmd, image, 1.f, 0, dst_range);
			}
		}

		copy_transfer_regions_impl(cmd, image, sections_to_copy);
		return result;
	}

	mtl::image_view* texture_cache::generate_2d_mipmaps_from_images(mtl::command_context& cmd, const deferred_subresource& desc)
	{
		const auto& sections_to_copy = desc.sections_to_copy;
		const auto mipmaps = ::narrow<u8>(sections_to_copy.size());
		auto _template = get_template_from_collection_impl(sections_to_copy);
		auto result = create_temporary_subresource_view_impl(cmd, _template, MTLTextureType2D,
			MTLTextureType2D, desc.gcm_format, desc.width, desc.height, 1, mipmaps, desc.remap);

		if (!result)
		{
			// Failed to create temporary object, bail
			return nullptr;
		}

		const auto image = result->image();
		const u32 dst_aspect = image->aspect();
		const mtl::image_subresource_range dst_range = { dst_aspect, 0, mipmaps, 0, 1 };

		if (desc.force_bg_load)
		{
			// The memory load covers the whole image, no need to clear it first
			initialize_subresource_from_memory(cmd, image, desc, rsx::texture_dimension_extended::texture_dimension_2d);
		}
		else if (!(dst_aspect & mtl::image_aspect::depth))
		{
			mtl::clear_color_image(cmd, image, color4f(0.f, 0.f, 0.f, 0.f), dst_range);
		}
		else
		{
			mtl::clear_depth_stencil_image(cmd, image, 1.f, 0, dst_range);
		}

		copy_transfer_regions_impl(cmd, image, sections_to_copy);
		return result;
	}

	void texture_cache::release_temporary_subresource(mtl::image_view* view)
	{
		auto resource = dynamic_cast<mtl::viewable_image*>(view->image());
		ensure(resource);

		auto image = std::unique_ptr<mtl::viewable_image>(resource);
		add_to_reuse_pool(image);
	}

	void texture_cache::initialize_subresource_from_memory(mtl::command_context& cmd, mtl::image* dst, const deferred_subresource& desc, rsx::texture_dimension_extended type) const
	{
		auto subresources_layout = rsx::get_subresources_layout(desc, type);
		filter_missing_levels(subresources_layout, dst->mipmaps());

		const u16 layer_count = (type == rsx::texture_dimension_extended::texture_dimension_cubemap) ? 6 : 1;
		mtl::upload_image(cmd, dst, subresources_layout, desc.gcm_format, desc.swizzled, layer_count,
			dst->aspect(), *mtl::get_upload_heap(), desc.pitch, mtl::upload_contents_inline);
	}

	void texture_cache::update_image_contents(mtl::command_context& cmd, mtl::image_view* dst_view, const deferred_subresource& desc)
	{
		auto dst = dst_view->image();
		copy_transfer_regions_impl(cmd, dst, desc.sections_to_copy);
	}

	cached_texture_section* texture_cache::create_new_texture(mtl::command_context& cmd, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch,
		u32 gcm_format, rsx::texture_upload_context context, rsx::texture_dimension_extended type, bool swizzled, rsx::component_order swizzle_flags, rsx::flags32_t flags)
	{
		const auto section_depth = depth;

		// Define desirable attributes based on type
		MTLTextureType image_type;
		MTLTextureUsage usage_flags = 0; // ShaderRead/RenderTarget/PixelFormatView are added by get_texture_cache_usage()
		u16 layer = 0;

		switch (type)
		{
		case rsx::texture_dimension_extended::texture_dimension_1d:
			image_type = MTLTextureType1D;
			height = 1;
			depth = 1;
			layer = 1;
			break;
		case rsx::texture_dimension_extended::texture_dimension_2d:
			image_type = MTLTextureType2D;
			depth = 1;
			layer = 1;
			break;
		case rsx::texture_dimension_extended::texture_dimension_cubemap:
			image_type = MTLTextureTypeCube;
			depth = 1;
			layer = 6;
			break;
		case rsx::texture_dimension_extended::texture_dimension_3d:
			image_type = MTLTextureType3D;
			layer = 1;
			break;
		default:
			fmt::throw_exception("Unreachable");
		}

		// Check what actually exists at that address
		const rsx::image_section_attributes_t search_desc = { .gcm_format = gcm_format, .width = width, .height = height, .depth = section_depth, .mipmaps = mipmaps };
		const bool allow_dirty = (context != rsx::texture_upload_context::framebuffer_storage);
		cached_texture_section& region = *find_cached_texture(rsx_range, search_desc, true, true, allow_dirty);
		ensure(!region.is_locked());

		mtl::viewable_image* image = nullptr;
		if (region.exists())
		{
			image = dynamic_cast<mtl::viewable_image*>(region.get_raw_texture());
			bool reusable = true;

			if (flags & texture_create_flags::do_not_reuse)
			{
				reusable = false;
			}
			// texture_create_flags::shareable: no concurrent queue sharing on Metal, every image is shareable

			if (!reusable || !image || region.get_image_type() != type || image->depth() != depth) // TODO
			{
				// Incompatible view/type
				region.destroy();
				image = nullptr;
			}
			else
			{
				ensure(region.is_managed());

				// Reuse
				region.set_rsx_pitch(pitch);

				if (flags & texture_create_flags::initialize_image_contents)
				{
					// Wipe memory (whole image: all levels, layers and depth slices)
					if (image->aspect() & mtl::image_aspect::color)
					{
						mtl::clear_color_image(cmd, image, color4f(0.f, 0.f, 0.f, 1.f));
					}
					else
					{
						mtl::clear_depth_stencil_image(cmd, image, 1.f, 255);
					}
				}
			}
		}

		if (!image)
		{
			MTLPixelFormat mtl_format = mtl::get_compatible_sampler_format(m_formats_support, gcm_format);
			rsx::format_class format_class = rsx::classify_format(gcm_format);
			u16 image_mipmaps = mipmaps;

			if (apply_image_type_restrictions(image_type, mtl_format, format_class, image_mipmaps))
			{
				err_once("Texture format 0x%x is a depth format which Metal does not support on texture type %u. Sampling will return raw data.",
					gcm_format, static_cast<u32>(image_type));
			}

			if (flags & texture_create_flags::mutable_format)
			{
				// VK: VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT with the snorm/srgb view formats
				if (mtl::get_compatible_snorm_format(mtl_format) != MTLPixelFormatInvalid ||
					mtl::get_compatible_srgb_format(mtl_format) != MTLPixelFormatInvalid)
				{
					usage_flags |= MTLTextureUsagePixelFormatView;
				}
			}

			const MTLTextureUsage usage = get_texture_cache_usage(mtl_format, usage_flags, image_type);

			if (auto found = find_cached_image(mtl_format, width, height, depth, image_mipmaps, image_type, usage))
			{
				image = found.release();
				image->set_format_class(format_class);
			}
			else
			{
				image = new mtl::viewable_image(image_type, mtl_format,
					width, height, depth, image_mipmaps, layer, 1, usage, format_class, MTLStorageModePrivate);
			}

			// New section, we must prepare it
			region.reset(rsx_range);
			region.set_gcm_format(gcm_format);
			region.set_image_type(type);
			region.create(width, height, section_depth, mipmaps, image, pitch, true, gcm_format);
		}

		region.set_view_flags(swizzle_flags);
		region.set_context(context);
		region.set_swizzled(swizzled);
		region.set_dirty(false);

		// NOTE: VK assigns native_component_map directly. Use the setter so that views cached on a reused image are rebuilt.
		image->set_native_component_layout(apply_component_mapping_flags(gcm_format, swizzle_flags, rsx::default_remap_vector));

		// Its not necessary to lock blit dst textures as they are just reused as necessary
		switch (context)
		{
		case rsx::texture_upload_context::shader_read:
		case rsx::texture_upload_context::blit_engine_src:
			region.protect(utils::protection::ro);
			read_only_range = region.get_min_max(read_only_range, rsx::section_bounds::locked_range);
			break;
		case rsx::texture_upload_context::blit_engine_dst:
			region.set_unpack_swap_bytes(true);
			no_access_range = region.get_min_max(no_access_range, rsx::section_bounds::locked_range);
			break;
		case rsx::texture_upload_context::dma:
		case rsx::texture_upload_context::framebuffer_storage:
			// Should not be initialized with this method
		default:
			fmt::throw_exception("Unexpected upload context 0x%x", u32(context));
		}

		update_cache_tag();
		return &region;
	}

	cached_texture_section* texture_cache::create_nul_section(
		mtl::command_context& /*cmd*/,
		const utils::address_range32& rsx_range,
		const rsx::image_section_attributes_t& attrs,
		const rsx::GCM_tile_reference& tile,
		bool memory_load)
	{
		auto& region = *find_cached_texture(rsx_range, { .gcm_format = RSX_GCM_FORMAT_IGNORED }, true, false, false);
		ensure(!region.is_locked());

		// Prepare section
		region.reset(rsx_range);
		region.create_dma_only(attrs.width, attrs.height, attrs.pitch);
		region.set_dirty(false);
		region.set_unpack_swap_bytes(true);

		if (memory_load && !tile) // Memory load on DMA tiles will always happen during the actual copy command
		{
			// VK: vk::map_dma + vk::load_dma
			region.map_dma(rsx_range, true);
		}

		no_access_range = region.get_min_max(no_access_range, rsx::section_bounds::locked_range);
		update_cache_tag();
		return &region;
	}

	cached_texture_section* texture_cache::upload_image_from_cpu(mtl::command_context& cmd, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch, u32 gcm_format,
		rsx::texture_upload_context context, const std::vector<rsx::subresource_layout>& subresource_layout, rsx::texture_dimension_extended type, bool swizzled)
	{
		if (context != rsx::texture_upload_context::shader_read)
		{
			if (cmd.is_render_pass_open())
			{
				cmd.end_encoding();
			}
		}

		// NOTE: Metal has no asynchronous upload scheduler, so do_not_reuse/shareable are never requested and uploads
		// are always recorded inline on cmd.
		rsx::flags32_t create_flags = 0;

		if (context == rsx::texture_upload_context::shader_read &&
			!g_cfg.video.disable_hardware_texel_remapping)
		{
			create_flags |= texture_create_flags::mutable_format;
		}

		auto section = create_new_texture(cmd, rsx_range, width, height, depth, mipmaps, pitch, gcm_format, context, type, swizzled,
			rsx::component_order::default_, create_flags);

		auto image = section->get_raw_texture();
		image->set_debug_name(fmt::format("Raw Texture @0x%x", rsx_range.start));

		mtl::enter_uninterruptible();

		bool input_swizzled = swizzled;
		if (context == rsx::texture_upload_context::blit_engine_src)
		{
			// Swizzling is ignored for blit engine copy and emulated using remapping
			input_swizzled = false;
		}

		rsx::flags32_t upload_command_flags = mtl::initialize_image_layout | mtl::upload_contents_inline;

		std::vector<rsx::subresource_layout> tmp;
		auto p_subresource_layout = &subresource_layout;
		u32 heap_align = mtl::upload_heap_align_default;

		if (auto tiled_region = rsx::get_current_renderer()->get_tiled_memory_region(rsx_range);
			context == rsx::texture_upload_context::blit_engine_src && tiled_region)
		{
			if (mipmaps > 1)
			{
				// This really shouldn't happen on framebuffer tiled memory
				rsx_log.error("Tiled decode of mipmapped textures is not supported.");
			}
			else
			{
				const auto bpp = rsx::get_format_block_size_in_bytes(gcm_format);
				const auto [scratch_buf, linear_data_scratch_offset] = mtl::detile_memory_block(cmd, tiled_region, rsx_range, width, height, bpp);

				auto subres = subresource_layout.front();
				// FIXME: !!EVIL!!
				subres.data = mtl::make_gpu_resident_source(scratch_buf, linear_data_scratch_offset);
				subres.pitch_in_block = width;
				upload_command_flags |= mtl::source_is_gpu_resident;
				heap_align = width * bpp;

				tmp.push_back(std::move(subres));
				p_subresource_layout = &tmp;
			}
		}

		if (image->mipmaps() < mipmaps)
		{
			// Mip levels the Metal image does not have (1D textures)
			if (p_subresource_layout != &tmp)
			{
				tmp = subresource_layout;
				p_subresource_layout = &tmp;
			}

			filter_missing_levels(tmp, image->mipmaps());
		}

		const u16 layer_count = (type == rsx::texture_dimension_extended::texture_dimension_cubemap) ? 6 : 1;
		mtl::upload_image(cmd, image, *p_subresource_layout, gcm_format, input_swizzled, layer_count, image->aspect(),
			*m_texture_upload_heap, heap_align, upload_command_flags);

		mtl::leave_uninterruptible();

		// NOTE: VK transitions non shader_read resources to their preferred layout here. Not needed on Metal.

		section->last_write_tag = rsx::get_shared_tag();
		return section;
	}

	void texture_cache::set_component_order(cached_texture_section& section, u32 gcm_format, rsx::component_order expected_flags)
	{
		if (expected_flags == section.get_view_flags())
			return;

		const MTLTextureSwizzleChannels mapping = apply_component_mapping_flags(gcm_format, expected_flags, rsx::default_remap_vector);
		auto image = static_cast<mtl::viewable_image*>(section.get_raw_texture());

		ensure(image);
		image->set_native_component_layout(mapping);

		section.set_view_flags(expected_flags);
	}

	void texture_cache::insert_texture_barrier(mtl::command_context& cmd, mtl::image* tex, bool strong_ordering)
	{
		auto rtt = mtl::as_rtt(tex);

		// VK: tex->current_layout == VK_IMAGE_LAYOUT_GENERAL
		if (!strong_ordering && rtt->is_feedback_loop_active())
		{
			// A previous barrier already exists, do nothing
			return;
		}

		rtt->texture_barrier(cmd);
	}

	bool texture_cache::render_target_format_is_compatible(mtl::image* tex, u32 gcm_format)
	{
		const auto format = tex->info.format;
		switch (gcm_format)
		{
		default:
			//TODO
			warn_once("Format incompatibility detected, reporting failure to force data copy (MTLPixelFormat=0x%X, GCM_FORMAT=0x%X)", static_cast<u32>(format), gcm_format);
			return false;
		case CELL_GCM_TEXTURE_R5G6B5:
			return (format == MTLPixelFormatB5G6R5Unorm);
		case CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT:
			return (format == MTLPixelFormatRGBA16Float);
		case CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT:
			return (format == MTLPixelFormatRGBA32Float);
		case CELL_GCM_TEXTURE_X32_FLOAT:
			return (format == MTLPixelFormatR32Float);
		case CELL_GCM_TEXTURE_A8R8G8B8:
		case CELL_GCM_TEXTURE_D8R8G8B8:
			return (format == MTLPixelFormatBGRA8Unorm || format == MTLPixelFormatDepth24Unorm_Stencil8 || format == MTLPixelFormatDepth32Float_Stencil8);
		case CELL_GCM_TEXTURE_B8:
			return (format == MTLPixelFormatR8Unorm);
		case CELL_GCM_TEXTURE_G8B8:
			return (format == MTLPixelFormatRG8Unorm);
		case CELL_GCM_TEXTURE_DEPTH24_D8:
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
			return (format == MTLPixelFormatDepth24Unorm_Stencil8 || format == MTLPixelFormatDepth32Float_Stencil8);
		case CELL_GCM_TEXTURE_X16:
		case CELL_GCM_TEXTURE_DEPTH16:
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
			return (format == MTLPixelFormatDepth16Unorm || format == MTLPixelFormatDepth32Float);
		}
	}

	void texture_cache::prepare_for_dma_transfers(mtl::command_context& /*cmd*/)
	{
		// Nothing to do, command buffers are created on demand (VK begins recording here)
	}

	void texture_cache::cleanup_after_dma_transfers(mtl::command_context& cmd)
	{
		auto renderer = get_metal_renderer();
		const bool is_primary_context = renderer && (&renderer->get_command_context() == &cmd);

		bool occlusion_query_active = false;
		if (is_primary_context)
		{
			occlusion_query_active = !!(renderer->get_cb_flags() & mtl::command_buffer_flags::cb_has_open_query);
			if (occlusion_query_active)
			{
				// We really stepped in it
				renderer->emergency_query_cleanup(&cmd);
			}
		}

		// End recording and start executing the transfers. No wait here: imp_flush waits on each section's sync id
		// (VK waits on a fence for the secondary command buffers and relies on the dma events otherwise).
		cmd.submit();

		if (renderer)
		{
			renderer->get_stats().submit_count++;
		}

		if (occlusion_query_active)
		{
			renderer->get_cb_flags() |= mtl::command_buffer_flags::cb_load_occluson_task;
		}
	}

	void texture_cache::initialize(mtl::render_device& device, const mtl::gpu_formats_support& formats_support, mtl::ring_buffer& upload_heap)
	{
		m_device = &device;
		m_formats_support = formats_support;
		m_texture_upload_heap = &upload_heap;
	}

	void texture_cache::destroy()
	{
		clear();
	}

	bool texture_cache::is_depth_texture(u32 rsx_address, u32 rsx_size)
	{
		reader_lock lock(m_cache_mutex);

		auto& block = m_storage.block_for(rsx_address);

		if (block.get_locked_count() == 0)
			return false;

		for (auto& tex : block)
		{
			if (tex.is_dirty())
				continue;

			if (!tex.overlaps(rsx_address, rsx::section_bounds::full_range))
				continue;

			if ((rsx_address + rsx_size - tex.get_section_base()) <= tex.get_section_size())
			{
				switch (tex.get_format())
				{
				case MTLPixelFormatDepth16Unorm:
				case MTLPixelFormatDepth32Float:
				case MTLPixelFormatDepth32Float_Stencil8:
				case MTLPixelFormatDepth24Unorm_Stencil8:
					return true;
				default:
					return false;
				}
			}
		}

		//Unreachable; silence compiler warning anyway
		return false;
	}

	bool texture_cache::handle_memory_pressure(rsx::problem_severity severity)
	{
		auto any_released = baseclass::handle_memory_pressure(severity);

		// TODO: This can cause invalidation of in-flight resources
		if (severity <= rsx::problem_severity::low || !m_cached_memory_size)
		{
			// Nothing left to do
			return any_released;
		}

		constexpr u64 _1M = 0x100000;
		if (severity <= rsx::problem_severity::moderate && m_cached_memory_size < (64 * _1M))
		{
			// Some memory is consumed by the temporary resources, but no need to panic just yet
			return any_released;
		}

		std::unique_lock lock(m_cache_mutex, std::defer_lock);
		if (!lock.try_lock())
		{
			rsx_log.warning("Unable to remove temporary resources because we're already in the texture cache!");
			return any_released;
		}

		// Nuke temporary resources. They will still be visible to the GPU.
		// VK hands them to the resource manager (deleted once the frame retires). Here they are kept alive until the next
		// on_frame_end() since views of released temporary subresources may still be referenced by the current frame.
		{
			std::lock_guard pool_lock(m_cached_pool_lock);

			any_released |= !m_cached_images.empty();
			for (auto& img : m_cached_images)
			{
				m_frame_scoped_images.push_back(std::move(img.data));
			}
			m_cached_images.clear();
			m_cached_memory_size = 0;
		}

		any_released |= !m_temporary_subresource_cache.empty();
		for (auto& e : m_temporary_subresource_cache)
		{
			ensure(e.second.second);
			release_temporary_subresource(e.second.second);
		}
		m_temporary_subresource_cache.clear();

		return any_released;
	}

	void texture_cache::on_frame_end()
	{
		trim_sections();

		if (m_storage.m_unreleased_texture_objects >= m_max_zombie_objects)
		{
			purge_unreleased_sections();
		}

		// Release the images handed out during the frame that just ended (upload_image_simple) and the ones evicted at
		// the previous frame end. Command buffers still using them keep the textures alive.
		std::vector<std::unique_ptr<mtl::viewable_image>> expired_images;
		{
			std::lock_guard lock(m_cached_pool_lock);
			expired_images.swap(m_frame_scoped_images);
		}
		expired_images.clear();

		if (m_cached_images.size() > max_cached_image_pool_size ||
			m_cached_memory_size > 256 * 0x100000)
		{
			std::lock_guard lock(m_cached_pool_lock);

			const auto new_size = m_cached_images.size() / 2;
			for (usz i = new_size; i < m_cached_images.size(); ++i)
			{
				m_cached_memory_size -= m_cached_images[i].data->memory_size();

				// VK only pools images the GPU is done with. Ours enter the pool immediately, so keep the evicted ones alive
				// for one more frame in case a released temporary view is still referenced.
				m_frame_scoped_images.push_back(std::move(m_cached_images[i].data));
			}

			m_cached_images.resize(new_size);
		}

		baseclass::on_frame_end();
		reset_frame_statistics();
	}

	mtl::viewable_image* texture_cache::upload_image_simple(mtl::command_context& cmd, MTLPixelFormat format, u32 address, u32 width, u32 height, u32 pitch)
	{
		bool linear_format_supported = false;

		switch (format)
		{
		case MTLPixelFormatBGRA8Unorm:
			linear_format_supported = m_formats_support.bgra8_linear;
			break;
		case MTLPixelFormatRGBA8Unorm:
			linear_format_supported = m_formats_support.argb8_linear;
			break;
		default:
			rsx_log.error("Unsupported MTLPixelFormat 0x%x", static_cast<u32>(format));
			return nullptr;
		}

		if (!linear_format_supported || !width || !height)
		{
			return nullptr;
		}

		// Uploads a linear memory range as a BGRA8 texture
		// Metal: private texture filled with a blit from the upload heap (VK maps a host-visible linear image)
		auto image = std::make_unique<mtl::viewable_image>(MTLTextureType2D, format,
			width, height, 1, 1, 1, 1, MTLTextureUsageShaderRead, RSX_FORMAT_CLASS_COLOR, MTLStorageModePrivate);

		const u32 row_pitch = width * 4;
		const u32 upload_size = row_pitch * height;
		const usz upload_offset = m_texture_upload_heap->alloc<256>(upload_size);

		auto src = vm::_ptr<const char>(address);
		auto dst = static_cast<char*>(m_texture_upload_heap->map(upload_offset));

		// TODO: SSE optimization
		for (u32 row = 0; row < height; ++row)
		{
			auto casted_src = reinterpret_cast<const be_t<u32>*>(src);
			auto casted_dst = reinterpret_cast<u32*>(dst);

			for (u32 col = 0; col < width; ++col)
				casted_dst[col] = casted_src[col];

			src += pitch;
			dst += row_pitch;
		}

		[cmd.blit() copyFromBuffer:m_texture_upload_heap->handle()
			sourceOffset:upload_offset
			sourceBytesPerRow:row_pitch
			sourceBytesPerImage:upload_size
			sourceSize:MTLSizeMake(width, height, 1)
			toTexture:image->value
			destinationSlice:0
			destinationLevel:0
			destinationOrigin:MTLOriginMake(0, 0, 0)];

		// VK disposes the image immediately (deleted once the frame retires). These images aren't really reusable right now.
		auto result = image.get();
		{
			std::lock_guard lock(m_cached_pool_lock);
			m_frame_scoped_images.push_back(std::move(image));
		}

		return result;
	}

	bool texture_cache::blit(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate, mtl::surface_cache& m_rtts, mtl::command_context& cmd)
	{
		mtl::blitter helper;
		auto reply = upload_scaled_image(src, dst, interpolate, cmd, m_rtts, helper);

		if (reply.succeeded)
		{
			if (reply.dst_range.valid())
			{
				flush_if_cache_miss_likely(cmd, reply.dst_range);
			}

			return true;
		}

		return false;
	}

	u32 texture_cache::get_unreleased_textures_count() const
	{
		return baseclass::get_unreleased_textures_count() + ::size32(m_cached_images);
	}

	u64 texture_cache::get_temporary_memory_in_use() const
	{
		// TODO: Technically incorrect, we should have separate metrics for cached evictable resources (this value) and temporary active resources.
		return m_cached_memory_size;
	}

	bool texture_cache::is_overallocated() const
	{
		const auto total_device_memory = mtl::get_caps().recommended_working_set_size / 0x100000;
		u64 quota = 0;

		if (total_device_memory >= 2048)
		{
			quota = std::min<u64>(3072, (total_device_memory * 40) / 100);
		}
		else if (total_device_memory >= 1024)
		{
			quota = std::max<u64>(204, (total_device_memory * 30) / 100);
		}
		else if (total_device_memory >= 768)
		{
			quota = 192;
		}
		else
		{
			quota = std::min<u64>(128, total_device_memory / 2);
		}

		quota *= 0x100000;

		// VK reads the VMM_ALLOCATION_POOL_TEXTURE_CACHE usage. Metal has no VMM: use the memory covered by the cached
		// sections plus the reuse pool (lock-free, this can be called from inside the texture cache).
		if (const u64 texture_cache_pool_usage = get_texture_memory_in_use() + m_cached_memory_size;
			texture_cache_pool_usage > quota)
		{
			rsx_log.warning("Texture cache is using %lluM of memory which exceeds the allocation quota of %lluM",
				texture_cache_pool_usage / 0x100000, quota / 0x100000);
			return true;
		}

		return false;
	}
}
