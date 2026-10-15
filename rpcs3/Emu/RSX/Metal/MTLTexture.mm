#include "stdafx.h"
#include "MTLTexture.h"

#include "MTLCompute.h"
#include "MTLFormats.h"
#include "MTLHelpers.h"
#include "MTLOverlays.h"

#include "mtlutils/buffer.h"
#include "mtlutils/command_context.h"
#include "mtlutils/device.h"
#include "mtlutils/image.h"

#include "Emu/Memory/vm.h"
#include "Emu/RSX/GCM.h"
#include "Emu/RSX/Core/RSXContext.h"
#include "Emu/RSX/Utils/rsx_utils.h"

#include "util/asm.hpp"

#include <array>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

// Port of VKTexture.cpp (plus the typeless helper of vkutils/scratch.cpp).
//
// Metal differences (see MTLTexture.h):
// - No image layouts, queue ownership transfers or pipeline barriers. Metal tracks hazards on all our resources.
// - Obtaining a blit/compute/render encoder from the command context ends the open render pass.
// - Buffer <-> texture copies transfer a single plane per call. Combined depth/stencil images are copied plane by plane
//   with MTLBlitOptionDepthFromDepthStencil / MTLBlitOptionStencilFromDepthStencil (same buffer layout as VK's separate
//   depth and stencil aspect copies).
// - Metal requires buffer offsets of buffer <-> texture copies to be multiples of the texel size and storage buffer
//   bindings to be 4-byte aligned. Misaligned requests are bounced through a private module-local transfer buffer.
// - There are no scaled blits; vkCmdBlitImage is replaced by the overlay mtl::blit_pass.
// - There is no DMA mapping of guest memory (VKDMA is not ported): CPU sourced uploads are always staged through the
//   upload heap, the zero-copy path is only used to describe GPU-resident sources.

namespace mtl
{
	namespace
	{
		// Typeless helpers, keyed by (format class, format) (VK: vkutils/scratch.cpp g_typeless_textures)
		std::unordered_map<u64, std::unique_ptr<mtl::image>> g_typeless_textures;

		// Private intermediate used by image round trips (copy_image between bit-compatible formats that cannot be
		// reinterpreted with a texture view) and to realign transfers whose buffer offsets do not satisfy Metal's alignment
		// rules. It is deliberately separate from the scratch buffer pool so that it never aliases a scratch buffer that
		// the caller is currently using (the pool alternates between two buffers).
		std::unique_ptr<mtl::buffer> g_transfer_buffer;

		id<MTLBuffer> get_transfer_buffer(u64 min_required_size)
		{
			min_required_size = std::max<u64>(min_required_size, 4);

			if (!g_transfer_buffer || g_transfer_buffer->size() < min_required_size)
			{
				// Command buffers retain the previous buffer for as long as they reference it
				const u64 new_size = utils::align<u64>(min_required_size, 0x100000);
				g_transfer_buffer = std::make_unique<mtl::buffer>(new_size,
					MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked, "Transfer buffer");
			}

			return g_transfer_buffer->handle();
		}

		// Byte layout of a buffer <-> image transfer of a single plane
		struct buffer_layout_t
		{
			u32 texel_size = 0;      // Bytes per texel, or per block for block-compressed formats
			u32 bytes_per_row = 0;
			u32 rows_per_image = 0;
			u64 bytes_per_image = 0;
			u32 image_count = 0;     // Number of images in the buffer (array layers or 3D slices)
			u64 span = 0;            // Distance from the first to the last byte accessed by the transfer
		};

		bool is_empty_transfer(const buffer_image_copy& region)
		{
			return !region.image_extent.width || !region.image_extent.height || !region.image_extent.depth || !region.layer_count;
		}

		buffer_layout_t get_buffer_layout(const mtl::image* img, u32 plane_aspect, const buffer_image_copy& region)
		{
			const u32 block_dim = mtl::get_format_block_dim(img->format());
			const u32 width = static_cast<u32>(region.image_extent.width);
			const u32 height = static_cast<u32>(region.image_extent.height);
			const u32 depth = static_cast<u32>(region.image_extent.depth);

			buffer_layout_t result{};
			result.texel_size = mtl::get_aspect_texel_size(img->format(), plane_aspect);
			result.bytes_per_row = utils::aligned_div(region.row_length(), block_dim) * result.texel_size;
			result.rows_per_image = utils::aligned_div(region.image_height(), block_dim);
			result.bytes_per_image = u64{ result.bytes_per_row } * result.rows_per_image;
			result.image_count = (img->type() == MTLTextureType3D) ? std::max(depth, 1u) : std::max(region.layer_count, 1u);

			const u64 last_row_bytes = u64{ utils::aligned_div(width, block_dim) } * result.texel_size;
			const u64 rows_in_last_image = utils::aligned_div(height, block_dim);
			result.span = u64{ result.image_count - 1 } * result.bytes_per_image + (rows_in_last_image - 1) * result.bytes_per_row + last_row_bytes;
			return result;
		}

		// Metal expects bytesPerImage = 0 for anything but 3D and array textures. Non-3D copies are issued one slice at a
		// time, so the value only describes the buffer layout there.
		bool uses_bytes_per_image(MTLTextureType type)
		{
			switch (type)
			{
			case MTLTextureType3D:
			case MTLTextureType1DArray:
			case MTLTextureType2DArray:
			case MTLTextureTypeCubeArray:
				return true;
			default:
				return false;
			}
		}

		MTLBlitOption get_blit_option(MTLPixelFormat format, u32 plane_aspect)
		{
			if (mtl::get_aspect_flags(format) != (image_aspect::depth | image_aspect::stencil))
			{
				return MTLBlitOptionNone;
			}

			switch (plane_aspect)
			{
			case image_aspect::depth:
				return MTLBlitOptionDepthFromDepthStencil;
			case image_aspect::stencil:
				return MTLBlitOptionStencilFromDepthStencil;
			default:
				fmt::throw_exception("Invalid aspect 0x%x for a combined depth/stencil transfer", plane_aspect);
			}
		}

		bool is_single_plane(u32 aspect)
		{
			return aspect == image_aspect::color || aspect == image_aspect::depth || aspect == image_aspect::stencil;
		}

		// Single-plane image -> buffer copy (vkCmdCopyImageToBuffer with one aspect)
		void copy_image_to_buffer_raw(mtl::command_context& cmd, const mtl::image* src, id<MTLBuffer> dst, const buffer_image_copy& region, u32 plane_aspect)
		{
			if (is_empty_transfer(region))
			{
				return;
			}

			ensure(is_single_plane(plane_aspect) && (src->aspect() & plane_aspect));
			ensure(src->samples() == 1); // Multisampled images cannot be copied to buffers (same as VK)

			const auto layout = get_buffer_layout(src, plane_aspect, region);
			ensure((region.buffer_offset + layout.span) <= [dst length]);

			if (region.buffer_offset % layout.texel_size)
			{
				// Metal requires the buffer offset to be a multiple of the texel size. Bounce through the transfer buffer and
				// preserve the bytes between the rows/images that the copy does not write.
				id<MTLBuffer> transfer = get_transfer_buffer(layout.span);
				[cmd.blit() copyFromBuffer:dst sourceOffset:region.buffer_offset toBuffer:transfer destinationOffset:0 size:layout.span];

				buffer_image_copy region2 = region;
				region2.buffer_offset = 0;
				copy_image_to_buffer_raw(cmd, src, transfer, region2, plane_aspect);

				[cmd.blit() copyFromBuffer:transfer sourceOffset:0 toBuffer:dst destinationOffset:region.buffer_offset size:layout.span];
				return;
			}

			const MTLBlitOption options = get_blit_option(src->format(), plane_aspect);
			const NSUInteger bytes_per_image = uses_bytes_per_image(src->type()) ? layout.bytes_per_image : 0;
			auto blit = cmd.blit();

			if (src->type() == MTLTextureType3D)
			{
				[blit copyFromTexture:src->value
					sourceSlice:0
					sourceLevel:region.mip_level
					sourceOrigin:region.image_offset
					sourceSize:region.image_extent
					toBuffer:dst
					destinationOffset:region.buffer_offset
					destinationBytesPerRow:layout.bytes_per_row
					destinationBytesPerImage:bytes_per_image
					options:options];
				return;
			}

			const MTLOrigin origin = MTLOriginMake(region.image_offset.x, region.image_offset.y, 0);
			const MTLSize size = MTLSizeMake(region.image_extent.width, region.image_extent.height, 1);

			for (u32 layer = 0; layer < region.layer_count; ++layer)
			{
				[blit copyFromTexture:src->value
					sourceSlice:(region.base_layer + layer)
					sourceLevel:region.mip_level
					sourceOrigin:origin
					sourceSize:size
					toBuffer:dst
					destinationOffset:(region.buffer_offset + layer * layout.bytes_per_image)
					destinationBytesPerRow:layout.bytes_per_row
					destinationBytesPerImage:bytes_per_image
					options:options];
			}
		}

		// Single-plane buffer -> image copy (vkCmdCopyBufferToImage with one aspect)
		void copy_buffer_to_image_raw(mtl::command_context& cmd, id<MTLBuffer> src, const mtl::image* dst, const buffer_image_copy& region, u32 plane_aspect)
		{
			if (is_empty_transfer(region))
			{
				return;
			}

			ensure(is_single_plane(plane_aspect) && (dst->aspect() & plane_aspect));
			ensure(dst->samples() == 1); // Multisampled images cannot be written from buffers (same as VK)

			const auto layout = get_buffer_layout(dst, plane_aspect, region);
			ensure((region.buffer_offset + layout.span) <= [src length]);

			if (region.buffer_offset % layout.texel_size)
			{
				// Metal requires the buffer offset to be a multiple of the texel size. Realign the data first.
				id<MTLBuffer> transfer = get_transfer_buffer(layout.span);
				[cmd.blit() copyFromBuffer:src sourceOffset:region.buffer_offset toBuffer:transfer destinationOffset:0 size:layout.span];

				buffer_image_copy region2 = region;
				region2.buffer_offset = 0;
				copy_buffer_to_image_raw(cmd, transfer, dst, region2, plane_aspect);
				return;
			}

			const MTLBlitOption options = get_blit_option(dst->format(), plane_aspect);
			const NSUInteger bytes_per_image = uses_bytes_per_image(dst->type()) ? layout.bytes_per_image : 0;
			auto blit = cmd.blit();

			if (dst->type() == MTLTextureType3D)
			{
				[blit copyFromBuffer:src
					sourceOffset:region.buffer_offset
					sourceBytesPerRow:layout.bytes_per_row
					sourceBytesPerImage:bytes_per_image
					sourceSize:region.image_extent
					toTexture:dst->value
					destinationSlice:0
					destinationLevel:region.mip_level
					destinationOrigin:region.image_offset
					options:options];
				return;
			}

			const MTLOrigin origin = MTLOriginMake(region.image_offset.x, region.image_offset.y, 0);
			const MTLSize size = MTLSizeMake(region.image_extent.width, region.image_extent.height, 1);

			for (u32 layer = 0; layer < region.layer_count; ++layer)
			{
				[blit copyFromBuffer:src
					sourceOffset:(region.buffer_offset + layer * layout.bytes_per_image)
					sourceBytesPerRow:layout.bytes_per_row
					sourceBytesPerImage:bytes_per_image
					sourceSize:size
					toTexture:dst->value
					destinationSlice:(region.base_layer + layer)
					destinationLevel:region.mip_level
					destinationOrigin:origin
					options:options];
			}
		}

		// VkImageCopy equivalent used by copy_image
		struct image_copy_region
		{
			u32 src_aspect = 0;
			u32 dst_aspect = 0;
			u32 src_level = 0;
			u32 dst_level = 0;
			u32 src_layer = 0;
			u32 dst_layer = 0;
			u32 layer_count = 1;
			MTLOrigin src_offset = { 0, 0, 0 };
			MTLOrigin dst_offset = { 0, 0, 0 };
			MTLSize extent = { 0, 0, 0 };
		};

		// VkImageBlit equivalent used by copy_scaled_image
		struct image_blit_region
		{
			struct offset3d { s32 x, y, z; };

			offset3d src_offsets[2] = {};
			offset3d dst_offsets[2] = {};
			u32 src_level = 0;
			u32 dst_level = 0;
		};

		// Packed formats cannot be reinterpreted with pixel format views
		bool is_packed_format(MTLPixelFormat format)
		{
			switch (format)
			{
			case MTLPixelFormatB5G6R5Unorm:
			case MTLPixelFormatA1BGR5Unorm:
			case MTLPixelFormatABGR4Unorm:
			case MTLPixelFormatBGR5A1Unorm:
			case MTLPixelFormatRGB10A2Unorm:
			case MTLPixelFormatRGB10A2Uint:
			case MTLPixelFormatBGR10A2Unorm:
			case MTLPixelFormatRG11B10Float:
			case MTLPixelFormatRGB9E5Float:
			case MTLPixelFormatBGR10_XR:
			case MTLPixelFormatBGR10_XR_sRGB:
			case MTLPixelFormatBGRA10_XR:
			case MTLPixelFormatBGRA10_XR_sRGB:
				return true;
			default:
				return false;
			}
		}

		bool is_srgb_pair(MTLPixelFormat a, MTLPixelFormat b)
		{
			const MTLPixelFormat srgb_a = mtl::get_compatible_srgb_format(a);
			const MTLPixelFormat srgb_b = mtl::get_compatible_srgb_format(b);
			return (srgb_a != MTLPixelFormatInvalid && srgb_a == b) || (srgb_b != MTLPixelFormatInvalid && srgb_b == a);
		}

		// Returns a texture that can be used as the source of a texture -> texture blit into a texture of dst_format
		// (the image itself or a pixel format view of it), or nil if no such view can be created.
		id<MTLTexture> get_texture_copy_source(const mtl::image* src, MTLPixelFormat dst_format)
		{
			const MTLPixelFormat src_format = src->format();
			if (src_format == dst_format)
			{
				return src->value;
			}

			// sRGB <-> linear views are always allowed
			if (!is_srgb_pair(src_format, dst_format))
			{
				if (!(src->info.usage & MTLTextureUsagePixelFormatView) ||
					mtl::get_aspect_flags(src_format) != image_aspect::color ||
					mtl::get_aspect_flags(dst_format) != image_aspect::color ||
					mtl::is_compressed_format(src_format) || mtl::is_compressed_format(dst_format) ||
					is_packed_format(src_format) || is_packed_format(dst_format) ||
					mtl::get_format_texel_size(src_format) != mtl::get_format_texel_size(dst_format))
				{
					return nil;
				}
			}

			return [src->value newTextureViewWithPixelFormat:dst_format];
		}

		// vkCmdCopyImage of one region with matching formats (or a pixel format view of the source)
		void copy_texture_region(mtl::command_context& cmd, id<MTLTexture> src_texture, const mtl::image* src, const mtl::image* dst, const image_copy_region& rgn)
		{
			auto blit = cmd.blit();
			const bool src_is_3d = (src->type() == MTLTextureType3D);
			const bool dst_is_3d = (dst->type() == MTLTextureType3D);

			if (src_is_3d && dst_is_3d)
			{
				[blit copyFromTexture:src_texture
					sourceSlice:0
					sourceLevel:rgn.src_level
					sourceOrigin:rgn.src_offset
					sourceSize:rgn.extent
					toTexture:dst->value
					destinationSlice:0
					destinationLevel:rgn.dst_level
					destinationOrigin:rgn.dst_offset];
				return;
			}

			// One slice per copy. Depth slices of a 3D image map to array layers of the other image (VK semantics).
			const u32 count = src_is_3d ? static_cast<u32>(rgn.extent.depth) : rgn.layer_count;
			const MTLSize size = MTLSizeMake(rgn.extent.width, rgn.extent.height, 1);

			for (u32 i = 0; i < count; ++i)
			{
				const MTLOrigin src_origin = MTLOriginMake(rgn.src_offset.x, rgn.src_offset.y, src_is_3d ? (rgn.src_offset.z + i) : 0);
				const MTLOrigin dst_origin = MTLOriginMake(rgn.dst_offset.x, rgn.dst_offset.y, dst_is_3d ? (rgn.dst_offset.z + i) : 0);

				[blit copyFromTexture:src_texture
					sourceSlice:(src_is_3d ? 0 : (rgn.src_layer + i))
					sourceLevel:rgn.src_level
					sourceOrigin:src_origin
					sourceSize:size
					toTexture:dst->value
					destinationSlice:(dst_is_3d ? 0 : (rgn.dst_layer + i))
					destinationLevel:rgn.dst_level
					destinationOrigin:dst_origin];
			}
		}

		// vkCmdCopyImage of one region through the transfer buffer, plane by plane. Used for bit-compatible formats that
		// cannot be reinterpreted with a view and for single-aspect copies of depth/stencil images.
		void copy_image_region_through_buffer(mtl::command_context& cmd, const mtl::image* src, const mtl::image* dst, const image_copy_region& rgn)
		{
			if (src->samples() > 1 || dst->samples() > 1)
			{
				rsx_log.error("Metal: Unsupported multisampled image copy (format 0x%x -> 0x%x, aspect 0x%x -> 0x%x)",
					static_cast<u32>(src->format()), static_cast<u32>(dst->format()), rgn.src_aspect, rgn.dst_aspect);
				return;
			}

			// Pair up the planes
			std::array<std::pair<u32, u32>, 3> planes{};
			u32 plane_count = 0;

			if (rgn.src_aspect == rgn.dst_aspect)
			{
				for (const u32 bit : { u32{ image_aspect::color }, u32{ image_aspect::depth }, u32{ image_aspect::stencil } })
				{
					if (rgn.src_aspect & bit)
					{
						planes[plane_count++] = { bit, bit };
					}
				}
			}
			else if (is_single_plane(rgn.src_aspect) && is_single_plane(rgn.dst_aspect))
			{
				// e.g depth -> color (typeless transport)
				planes[plane_count++] = { rgn.src_aspect, rgn.dst_aspect };
			}

			if (!plane_count)
			{
				rsx_log.error("Metal: Unsupported image copy aspect combination 0x%x -> 0x%x (format 0x%x -> 0x%x)",
					rgn.src_aspect, rgn.dst_aspect, static_cast<u32>(src->format()), static_cast<u32>(dst->format()));
				return;
			}

			const bool src_is_3d = (src->type() == MTLTextureType3D);
			const bool dst_is_3d = (dst->type() == MTLTextureType3D);

			for (u32 i = 0; i < plane_count; ++i)
			{
				const auto [src_plane, dst_plane] = planes[i];

				if (mtl::get_aspect_texel_size(src->format(), src_plane) != mtl::get_aspect_texel_size(dst->format(), dst_plane) ||
					mtl::get_format_block_dim(src->format()) != mtl::get_format_block_dim(dst->format()))
				{
					rsx_log.error("Metal: Image copy between incompatible formats 0x%x (aspect 0x%x) -> 0x%x (aspect 0x%x)",
						static_cast<u32>(src->format()), src_plane, static_cast<u32>(dst->format()), dst_plane);
					continue;
				}

				buffer_image_copy src_copy{};
				src_copy.aspect = src_plane;
				src_copy.mip_level = rgn.src_level;
				src_copy.base_layer = src_is_3d ? 0 : rgn.src_layer;
				src_copy.layer_count = src_is_3d ? 1 : rgn.layer_count;
				src_copy.image_offset = src_is_3d ? rgn.src_offset : MTLOriginMake(rgn.src_offset.x, rgn.src_offset.y, 0);
				src_copy.image_extent = MTLSizeMake(rgn.extent.width, rgn.extent.height, src_is_3d ? rgn.extent.depth : 1);

				buffer_image_copy dst_copy{};
				dst_copy.aspect = dst_plane;
				dst_copy.mip_level = rgn.dst_level;
				dst_copy.base_layer = dst_is_3d ? 0 : rgn.dst_layer;
				dst_copy.layer_count = dst_is_3d ? 1 : (src_is_3d ? static_cast<u32>(rgn.extent.depth) : rgn.layer_count);
				dst_copy.image_offset = dst_is_3d ? rgn.dst_offset : MTLOriginMake(rgn.dst_offset.x, rgn.dst_offset.y, 0);
				dst_copy.image_extent = MTLSizeMake(rgn.extent.width, rgn.extent.height, dst_is_3d ? (src_is_3d ? rgn.extent.depth : rgn.layer_count) : 1);

				if (is_empty_transfer(src_copy) || is_empty_transfer(dst_copy))
				{
					continue;
				}

				const auto layout = get_buffer_layout(src, src_plane, src_copy);
				id<MTLBuffer> transfer = get_transfer_buffer(layout.span);

				copy_image_to_buffer_raw(cmd, src, transfer, src_copy, src_plane);
				copy_buffer_to_image_raw(cmd, transfer, dst, dst_copy, dst_plane);
			}
		}

		void copy_image_region(mtl::command_context& cmd, const mtl::image* src, const mtl::image* dst, const image_copy_region& rgn)
		{
			if (!rgn.extent.width || !rgn.extent.height || !rgn.extent.depth || !rgn.layer_count || !rgn.src_aspect || !rgn.dst_aspect)
			{
				return;
			}

			if (rgn.src_aspect == src->aspect() && rgn.dst_aspect == dst->aspect() && src->samples() == dst->samples())
			{
				// Full copies (all aspects) are done texture to texture
				if (id<MTLTexture> src_texture = get_texture_copy_source(src, dst->format()))
				{
					copy_texture_region(cmd, src_texture, src, dst, rgn);
					return;
				}
			}

			copy_image_region_through_buffer(cmd, src, dst, rgn);
		}

		// Clear helpers
		struct subresource_span
		{
			u32 base_level;
			u32 level_count;
		};

		subresource_span get_level_range(const mtl::image* img, const image_subresource_range& range)
		{
			ensure(range.base_level < img->mipmaps());
			const u32 available = img->mipmaps() - range.base_level;
			return { range.base_level, range.level_count ? std::min(range.level_count, available) : available };
		}

		// Array layers, or depth slices of the given level for 3D images
		std::pair<u32, u32> get_slice_range(const mtl::image* img, const image_subresource_range& range, u32 level)
		{
			const u32 slice_count = (img->type() == MTLTextureType3D) ? std::max(img->depth() >> level, 1u) : img->layers();
			if (range.base_layer >= slice_count)
			{
				return { 0, 0 };
			}

			const u32 available = slice_count - range.base_layer;
			return { range.base_layer, range.layer_count ? std::min(range.layer_count, available) : available };
		}

		// Zero-fills the selected subresources of an image that cannot be used as a render target (e.g block-compressed)
		void zero_fill_image(mtl::command_context& cmd, const mtl::image* img, u32 aspect, const image_subresource_range& range)
		{
			if (img->samples() > 1)
			{
				rsx_log.error("Metal: Cannot clear a multisampled image without render target usage (format=0x%x)", static_cast<u32>(img->format()));
				return;
			}

			const auto [base_level, level_count] = get_level_range(img, range);
			const bool is_3d = (img->type() == MTLTextureType3D);
			const u32 block_dim = mtl::get_format_block_dim(img->format());

			// The same zeroed block is the source of every copy, size it for the largest level
			u64 zero_length = 0;
			for (const u32 plane : { u32{ image_aspect::color }, u32{ image_aspect::depth }, u32{ image_aspect::stencil } })
			{
				if (!(aspect & plane))
				{
					continue;
				}

				const u32 w = std::max(img->width() >> base_level, 1u);
				const u32 h = std::max(img->height() >> base_level, 1u);
				const u32 d = is_3d ? std::max(img->depth() >> base_level, 1u) : 1u;
				const u64 bytes_per_row = u64{ utils::aligned_div(w, block_dim) } * mtl::get_aspect_texel_size(img->format(), plane);
				zero_length = std::max(zero_length, bytes_per_row * utils::aligned_div(h, block_dim) * d);
			}

			if (!zero_length)
			{
				return;
			}

			zero_length = utils::align<u64>(zero_length, 4);
			id<MTLBuffer> zeros = get_transfer_buffer(zero_length);
			[cmd.blit() fillBuffer:zeros range:NSMakeRange(0, zero_length) value:0];

			for (u32 level = base_level; level < base_level + level_count; ++level)
			{
				const u32 w = std::max(img->width() >> level, 1u);
				const u32 h = std::max(img->height() >> level, 1u);
				const auto [first_slice, slice_count] = get_slice_range(img, range, level);

				for (const u32 plane : { u32{ image_aspect::color }, u32{ image_aspect::depth }, u32{ image_aspect::stencil } })
				{
					if (!(aspect & plane) || !slice_count)
					{
						continue;
					}

					buffer_image_copy region{};
					region.aspect = plane;
					region.mip_level = level;

					if (is_3d)
					{
						region.image_offset = MTLOriginMake(0, 0, first_slice);
						region.image_extent = MTLSizeMake(w, h, slice_count);
						copy_buffer_to_image_raw(cmd, zeros, img, region, plane);
						continue;
					}

					region.image_extent = MTLSizeMake(w, h, 1);
					for (u32 slice = first_slice; slice < first_slice + slice_count; ++slice)
					{
						region.base_layer = slice;
						copy_buffer_to_image_raw(cmd, zeros, img, region, plane);
					}
				}
			}
		}

		// Images that can be attached to a render pass (clears, mtl::blit_pass). Metal cannot render into 1D textures.
		bool is_renderable(const mtl::image* img)
		{
			return (img->info.usage & MTLTextureUsageRenderTarget) &&
				!mtl::is_compressed_format(img->format()) &&
				img->type() != MTLTextureType1D &&
				img->type() != MTLTextureType1DArray;
		}
	}

	static void gpu_swap_bytes_impl(mtl::command_context& cmd, id<MTLBuffer> buf, u32 element_size, u32 data_offset, u32 data_length)
	{
		if (element_size == 4)
		{
			mtl::get_compute_task<mtl::cs_shuffle_32>()->run(cmd, buf, data_length, data_offset);
		}
		else if (element_size == 2)
		{
			mtl::get_compute_task<mtl::cs_shuffle_16>()->run(cmd, buf, data_length, data_offset);
		}
		else
		{
			fmt::throw_exception("Unreachable");
		}
	}

	// Max operation on the magnitude of value while preserving the sign
	// Clamp_val must be >= 0
	static int mag_max(int value, int clamp_val)
	{
		if (value >= 0)
		{
			return std::max(value, clamp_val);
		}

		return -std::max(-value, clamp_val);
	}

	static void increment_mip_level(image_blit_region& blit)
	{
		const int mip_src_w = (blit.src_offsets[1].x - blit.src_offsets[0].x) / 2;
		const int mip_src_h = (blit.src_offsets[1].y - blit.src_offsets[0].y) / 2;
		const int mip_src_d = (blit.src_offsets[1].z - blit.src_offsets[0].z) / 2;

		blit.src_offsets[0].x /= 2;
		blit.src_offsets[0].y /= 2;
		blit.src_offsets[0].z /= 2;
		blit.src_offsets[1].x = blit.src_offsets[0].x + mag_max(mip_src_w, 1);
		blit.src_offsets[1].y = blit.src_offsets[0].y + mag_max(mip_src_h, 1);
		blit.src_offsets[1].z = blit.src_offsets[0].z + mag_max(mip_src_d, 1);

		const int mip_dst_w = (blit.dst_offsets[1].x - blit.dst_offsets[0].x) / 2;
		const int mip_dst_h = (blit.dst_offsets[1].y - blit.dst_offsets[0].y) / 2;
		const int mip_dst_d = (blit.dst_offsets[1].z - blit.dst_offsets[0].z) / 2;

		blit.dst_offsets[0].x /= 2;
		blit.dst_offsets[0].y /= 2;
		blit.dst_offsets[0].z /= 2;
		blit.dst_offsets[1].x = blit.dst_offsets[0].x + mag_max(mip_dst_w, 1);
		blit.dst_offsets[1].y = blit.dst_offsets[0].y + mag_max(mip_dst_h, 1);
		blit.dst_offsets[1].z = blit.dst_offsets[0].z + mag_max(mip_dst_d, 1);

		blit.src_level++;
		blit.dst_level++;
	}

	static void increment_mip_level(image_copy_region& copy)
	{
		copy.extent.width = std::max<NSUInteger>(copy.extent.width / 2u, 1u);
		copy.extent.height = std::max<NSUInteger>(copy.extent.height / 2u, 1u);
		copy.extent.depth = std::max<NSUInteger>(copy.extent.depth / 2u, 1u);

		copy.src_offset.x /= 2;
		copy.src_offset.y /= 2;
		copy.src_offset.z /= 2;

		copy.dst_offset.x /= 2;
		copy.dst_offset.y /= 2;
		copy.dst_offset.z /= 2;

		copy.src_level++;
		copy.dst_level++;
	}

	static void increment_mip_level(buffer_image_copy& copy)
	{
		copy.image_offset.x /= 2;
		copy.image_offset.y /= 2;
		copy.image_offset.z /= 2;
		copy.image_extent.width = std::max<NSUInteger>(copy.image_extent.width / 2u, 1u);
		copy.image_extent.height = std::max<NSUInteger>(copy.image_extent.height / 2u, 1u);
		copy.image_extent.depth = std::max<NSUInteger>(copy.image_extent.depth / 2u, 1u);
		copy.mip_level++;
	}

	u64 calculate_working_buffer_size(u64 base_size, u32 aspect)
	{
		if (aspect & (image_aspect::stencil | image_aspect::depth))
		{
			return (base_size * 3);
		}
		else
		{
			return base_size;
		}
	}

	void copy_image_to_buffer(
		mtl::command_context& cmd,
		const mtl::image* src,
		id<MTLBuffer> dst,
		const buffer_image_copy& region,
		const image_readback_options_t& options)
	{
		// NOTE: No layout validation, render pass handling or post-transfer barriers (options.sync_region) are needed.
		// Requesting the blit/compute encoder ends any open render pass and Metal tracks the buffer hazards.

		ensure((region.image_extent.width + region.image_offset.x) <= src->width());
		ensure((region.image_extent.height + region.image_offset.y) <= src->height());

		const u32 aspect = region.aspect ? region.aspect : src->aspect();

		switch (src->format())
		{
		default:
		{
			ensure(!options.swap_bytes); // "Implicit byteswap option not supported for speficied format"
			copy_image_to_buffer_raw(cmd, src, dst, region, aspect);
			break;
		}
		case MTLPixelFormatDepth32Float:
		{
			ensure(aspect == image_aspect::depth);

			const u32 out_w = region.row_length();
			const u32 out_h = region.image_height();
			const u32 packed32_length = out_w * out_h * 4;
			const u32 packed16_length = out_w * out_h * 2;

			const auto data_offset = static_cast<u32>(region.buffer_offset);
			if (data_offset % 4)
			{
				// Metal: storage buffer bindings must be 4-byte aligned. Pack at the start of the transfer buffer instead.
				ensure([dst length] >= (u64{ data_offset } + packed16_length));
				id<MTLBuffer> transfer = get_transfer_buffer(calculate_working_buffer_size(packed16_length, aspect) + 512);

				buffer_image_copy region2 = region;
				region2.buffer_offset = 0;
				copy_image_to_buffer(cmd, src, transfer, region2, options);

				[cmd.blit() copyFromBuffer:transfer sourceOffset:0 toBuffer:dst destinationOffset:data_offset size:packed16_length];
				break;
			}

			const auto z32_offset = utils::align<u32>(data_offset + packed16_length, 256);

			const auto allocation_end = z32_offset + packed32_length;
			ensure([dst length] >= allocation_end);

			// 1. Copy the depth to buffer
			buffer_image_copy region2 = region;
			region2.buffer_offset = z32_offset;
			copy_image_to_buffer_raw(cmd, src, dst, region2, image_aspect::depth);

			// 2. Do conversion with byteswap [D32->D16F]
			if (!options.swap_bytes) [[likely]]
			{
				auto job = mtl::get_compute_task<mtl::cs_fconvert_task<f32, f16>>();
				job->run(cmd, dst, z32_offset, packed32_length, data_offset);
			}
			else
			{
				auto job = mtl::get_compute_task<mtl::cs_fconvert_task<f32, f16, false, true>>();
				job->run(cmd, dst, z32_offset, packed32_length, data_offset);
			}
			break;
		}
		case MTLPixelFormatDepth24Unorm_Stencil8:
		case MTLPixelFormatDepth32Float_Stencil8:
		{
			if (is_single_plane(aspect))
			{
				// Metal: single-aspect readback of a combined image, the plane is copied as-is (no packing)
				ensure(!options.swap_bytes);
				copy_image_to_buffer_raw(cmd, src, dst, region, aspect);
				break;
			}

			ensure(aspect == (image_aspect::depth | image_aspect::stencil));

			if (region.layer_count > 1 || region.image_extent.depth > 1)
			{
				rsx_log.error("Metal: Packed depth/stencil readback only supports a single layer (layers=%u, depth=%u)",
					region.layer_count, static_cast<u32>(region.image_extent.depth));
			}

			const u32 out_w = region.row_length();
			const u32 out_h = region.image_height();
			const u32 packed_length = out_w * out_h * 4;
			const u32 in_depth_size = packed_length;
			const u32 in_stencil_size = out_w * out_h;

			const auto data_offset = static_cast<u32>(region.buffer_offset);
			if (data_offset % 4)
			{
				// Metal: storage buffer bindings must be 4-byte aligned. Pack at the start of the transfer buffer instead.
				ensure([dst length] >= (u64{ data_offset } + packed_length));
				id<MTLBuffer> transfer = get_transfer_buffer(calculate_working_buffer_size(packed_length, aspect) + 512);

				buffer_image_copy region2 = region;
				region2.buffer_offset = 0;
				copy_image_to_buffer(cmd, src, transfer, region2, options);

				[cmd.blit() copyFromBuffer:transfer sourceOffset:0 toBuffer:dst destinationOffset:data_offset size:packed_length];
				break;
			}

			const auto z_offset = utils::align<u32>(data_offset + packed_length, 256);
			const auto s_offset = utils::align<u32>(z_offset + in_depth_size, 256);

			const auto allocation_end = s_offset + in_stencil_size;
			ensure([dst length] >= allocation_end);

			// 1. Copy the depth and stencil blocks to separate banks
			buffer_image_copy sub_regions[2];
			sub_regions[0] = sub_regions[1] = region;
			sub_regions[0].buffer_offset = z_offset;
			sub_regions[0].aspect = image_aspect::depth;
			sub_regions[1].buffer_offset = s_offset;
			sub_regions[1].aspect = image_aspect::stencil;
			copy_image_to_buffer_raw(cmd, src, dst, sub_regions[0], image_aspect::depth);
			copy_image_to_buffer_raw(cmd, src, dst, sub_regions[1], image_aspect::stencil);

			// 2. Interleave the separated data blocks with a compute job
			mtl::cs_interleave_task* job;
			if (!options.swap_bytes) [[likely]]
			{
				if (src->format() == MTLPixelFormatDepth24Unorm_Stencil8)
				{
					job = mtl::get_compute_task<mtl::cs_gather_d24x8<false>>();
				}
				else if (src->format_class() == RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32)
				{
					job = mtl::get_compute_task<mtl::cs_gather_d32x8<false, true>>();
				}
				else
				{
					job = mtl::get_compute_task<mtl::cs_gather_d32x8<false>>();
				}
			}
			else
			{
				if (src->format() == MTLPixelFormatDepth24Unorm_Stencil8)
				{
					job = mtl::get_compute_task<mtl::cs_gather_d24x8<true>>();
				}
				else if (src->format_class() == RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32)
				{
					job = mtl::get_compute_task<mtl::cs_gather_d32x8<true, true>>();
				}
				else
				{
					job = mtl::get_compute_task<mtl::cs_gather_d32x8<true>>();
				}
			}

			job->run(cmd, dst, data_offset, packed_length, z_offset, s_offset);
			break;
		}
		} // switch
	}

	void copy_buffer_to_image(mtl::command_context& cmd, id<MTLBuffer> src, const mtl::image* dst, const buffer_image_copy& region)
	{
		// NOTE: No layout validation, render pass handling or barriers (see copy_image_to_buffer)
		const u32 aspect = region.aspect ? region.aspect : dst->aspect();

		switch (dst->format())
		{
		default:
		{
			copy_buffer_to_image_raw(cmd, src, dst, region, aspect);
			break;
		}
		case MTLPixelFormatDepth32Float:
		{
			ensure(aspect == image_aspect::depth);

			const u32 out_w = region.row_length();
			const u32 out_h = region.image_height();
			const u32 packed32_length = out_w * out_h * 4;
			const u32 packed16_length = out_w * out_h * 2;

			const auto data_offset = static_cast<u32>(region.buffer_offset);
			if (data_offset % 4)
			{
				// Metal: storage buffer bindings must be 4-byte aligned. Move the packed data to the transfer buffer first.
				ensure([src length] >= (u64{ data_offset } + packed16_length));
				id<MTLBuffer> transfer = get_transfer_buffer(calculate_working_buffer_size(packed16_length, aspect) + 512);
				[cmd.blit() copyFromBuffer:src sourceOffset:data_offset toBuffer:transfer destinationOffset:0 size:packed16_length];

				buffer_image_copy region2 = region;
				region2.buffer_offset = 0;
				copy_buffer_to_image(cmd, transfer, dst, region2);
				break;
			}

			const auto z32_offset = utils::align<u32>(data_offset + packed16_length, 256);

			const auto allocation_end = z32_offset + packed32_length;
			ensure([src length] >= allocation_end);

			// 1. Do conversion with byteswap [D16F->D32F]
			auto job = mtl::get_compute_task<mtl::cs_fconvert_task<f16, f32>>();
			job->run(cmd, src, data_offset, packed16_length, z32_offset);

			// 2. Copy the depth data to image
			buffer_image_copy region2 = region;
			region2.buffer_offset = z32_offset;
			copy_buffer_to_image_raw(cmd, src, dst, region2, image_aspect::depth);
			break;
		}
		case MTLPixelFormatDepth24Unorm_Stencil8:
		case MTLPixelFormatDepth32Float_Stencil8:
		{
			if (is_single_plane(aspect))
			{
				// Metal: single-aspect upload into a combined image, the plane is copied as-is (no unpacking)
				copy_buffer_to_image_raw(cmd, src, dst, region, aspect);
				break;
			}

			ensure(aspect == (image_aspect::depth | image_aspect::stencil));

			if (region.layer_count > 1 || region.image_extent.depth > 1)
			{
				rsx_log.error("Metal: Packed depth/stencil upload only supports a single layer (layers=%u, depth=%u)",
					region.layer_count, static_cast<u32>(region.image_extent.depth));
			}

			const u32 out_w = region.row_length();
			const u32 out_h = region.image_height();
			const u32 packed_length = out_w * out_h * 4;
			const u32 in_depth_size = packed_length;
			const u32 in_stencil_size = out_w * out_h;

			const auto data_offset = static_cast<u32>(region.buffer_offset);
			if (data_offset % 4)
			{
				// Metal: storage buffer bindings must be 4-byte aligned. Move the packed data to the transfer buffer first.
				ensure([src length] >= (u64{ data_offset } + packed_length));
				id<MTLBuffer> transfer = get_transfer_buffer(calculate_working_buffer_size(packed_length, aspect) + 512);
				[cmd.blit() copyFromBuffer:src sourceOffset:data_offset toBuffer:transfer destinationOffset:0 size:packed_length];

				buffer_image_copy region2 = region;
				region2.buffer_offset = 0;
				copy_buffer_to_image(cmd, transfer, dst, region2);
				break;
			}

			const auto z_offset = utils::align<u32>(data_offset + packed_length, 256);
			const auto s_offset = utils::align<u32>(z_offset + in_depth_size, 256);

			const auto allocation_end = s_offset + in_stencil_size;
			ensure([src length] >= allocation_end); // "Out of memory (compute heap). Lower your resolution scale setting."

			// Zero out the stencil block
			const u64 stencil_fill_length = std::min<u64>(utils::align(in_stencil_size, 4), [src length] - s_offset);
			[cmd.blit() fillBuffer:src range:NSMakeRange(s_offset, stencil_fill_length) value:0];

			// 1. Scatter the interleaved data into separate depth and stencil blocks
			mtl::cs_interleave_task* job;
			if (dst->format() == MTLPixelFormatDepth24Unorm_Stencil8)
			{
				job = mtl::get_compute_task<mtl::cs_scatter_d24x8>();
			}
			else if (dst->format_class() == RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32)
			{
				job = mtl::get_compute_task<mtl::cs_scatter_d32x8<true>>();
			}
			else
			{
				job = mtl::get_compute_task<mtl::cs_scatter_d32x8<false>>();
			}

			job->run(cmd, src, data_offset, packed_length, z_offset, s_offset);

			// 2. Copy the separated blocks into the target
			buffer_image_copy sub_regions[2];
			sub_regions[0] = sub_regions[1] = region;
			sub_regions[0].buffer_offset = z_offset;
			sub_regions[0].aspect = image_aspect::depth;
			sub_regions[1].buffer_offset = s_offset;
			sub_regions[1].aspect = image_aspect::stencil;
			copy_buffer_to_image_raw(cmd, src, dst, sub_regions[0], image_aspect::depth);
			copy_buffer_to_image_raw(cmd, src, dst, sub_regions[1], image_aspect::stencil);
			break;
		}
		}
	}

	void copy_image_typeless(
		mtl::command_context& cmd,
		mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers,
		u32 src_transfer_mask, u32 dst_transfer_mask)
	{
		ensure(!src_rect.is_flipped() && !dst_rect.is_flipped()); //<- Flip conversion must be handled by the caller.

		if (src->format() == dst->format())
		{
			if (src->format_class() == dst->format_class())
			{
				rsx_log.warning("[Performance warning] Image copy requested incorrectly for matching formats.");
				copy_image(cmd, src, dst, src_rect, dst_rect, mip_layers, src_transfer_mask, dst_transfer_mask);
				return;
			}
			else
			{
				// Should only happen for DEPTH_FLOAT <-> DEPTH_UINT at this time
				const u32 mask = src->format_class() | dst->format_class();
				if (mask != (RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32 | RSX_FORMAT_CLASS_DEPTH24_UNORM_X8_PACK32))
				{
					rsx_log.error("Unexpected (and possibly incorrect) typeless transfer setup.");
				}
			}
		}

		buffer_image_copy src_copy{}, dst_copy{};
		src_copy.image_extent = MTLSizeMake(static_cast<u32>(src_rect.width), static_cast<u32>(src_rect.height), static_cast<u32>(src_rect.depth));
		src_copy.image_offset = MTLOriginMake(static_cast<u32>(src_rect.x), static_cast<u32>(src_rect.y), static_cast<u32>(src_rect.z));
		src_copy.aspect = src->aspect() & src_transfer_mask;
		src_copy.mip_level = mip_layers.src_mip_level;
		src_copy.base_layer = mip_layers.src_layer;
		src_copy.layer_count = mip_layers.layer_count;

		dst_copy.image_extent = MTLSizeMake(static_cast<u32>(dst_rect.width), static_cast<u32>(dst_rect.height), static_cast<u32>(dst_rect.depth));
		dst_copy.image_offset = MTLOriginMake(static_cast<u32>(dst_rect.x), static_cast<u32>(dst_rect.y), static_cast<u32>(dst_rect.z));
		dst_copy.aspect = dst->aspect() & dst_transfer_mask;
		dst_copy.mip_level = mip_layers.dst_mip_level;
		dst_copy.base_layer = mip_layers.dst_layer;
		dst_copy.layer_count = mip_layers.layer_count;

		const auto src_texel_size = mtl::get_format_texel_width(src->info.format);
		mtl::buffer* scratch_buf = nullptr;

		for (u32 remaining_levels = mip_layers.mipmap_count; remaining_levels > 0; remaining_levels--)
		{
			const auto src_length = static_cast<u32>(src_texel_size * src_copy.image_extent.width * src_copy.image_extent.height * src_copy.image_extent.depth * mip_layers.layer_count);
			if (!scratch_buf)
			{
				// Initialize scratch memory
				const auto min_scratch_size = calculate_working_buffer_size(src_length, src->aspect() | dst->aspect());
				scratch_buf = mtl::get_scratch_buffer(cmd, min_scratch_size);
			}

			mtl::copy_image_to_buffer(cmd, src, scratch_buf->handle(), src_copy);

			auto src_convert = mtl::get_format_convert_flags(src->info.format);
			auto dst_convert = mtl::get_format_convert_flags(dst->info.format);

			const bool needs_shuffle = (src_convert.first || dst_convert.first) &&
				(src_convert.first != dst_convert.first || src_convert.second != dst_convert.second);

			if (needs_shuffle)
			{
				mtl::cs_shuffle_base* shuffle_kernel = nullptr;
				if (src_convert.first && dst_convert.first)
				{
					shuffle_kernel = mtl::get_compute_task<mtl::cs_shuffle_32_16>();
				}
				else
				{
					const auto block_size = src_convert.first ? src_convert.second : dst_convert.second;
					if (block_size == 4)
					{
						shuffle_kernel = mtl::get_compute_task<mtl::cs_shuffle_32>();
					}
					else if (block_size == 2)
					{
						shuffle_kernel = mtl::get_compute_task<mtl::cs_shuffle_16>();
					}
					else
					{
						fmt::throw_exception("Unreachable");
					}
				}

				shuffle_kernel->run(cmd, scratch_buf->handle(), src_length);
			}

			mtl::copy_buffer_to_image(cmd, scratch_buf->handle(), dst, dst_copy);

			if (remaining_levels > 1)
			{
				increment_mip_level(src_copy);
				increment_mip_level(dst_copy);
			}
		}
	}

	void copy_image(mtl::command_context& cmd,
		mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers,
		u32 src_transfer_mask, u32 dst_transfer_mask)
	{
		// NOTE: src_aspect should match dst_aspect according to spec but some drivers seem to work just fine with the mismatch
		if (const u32 aspect_bridge = (src->aspect() | dst->aspect());
			(aspect_bridge & image_aspect::color) == 0 &&
			src->format() != dst->format())
		{
			// Copying between two depth formats must match exactly or crashes will happen
			rsx_log.warning("[Performance warning] Image copy was requested incorrectly for mismatched depth formats");
			copy_image_typeless(cmd, src, dst, src_rect, dst_rect, mip_layers);
			return;
		}

		ensure(!src_rect.is_flipped() && !dst_rect.is_flipped()); //<- Flip conversion must be handled by the caller.

		image_copy_region rgn{};
		rgn.src_aspect = src->aspect() & src_transfer_mask;
		rgn.dst_aspect = dst->aspect() & dst_transfer_mask;
		rgn.src_level = mip_layers.src_mip_level;
		rgn.dst_level = mip_layers.dst_mip_level;
		rgn.src_layer = mip_layers.src_layer;
		rgn.dst_layer = mip_layers.dst_layer;
		rgn.layer_count = mip_layers.layer_count;
		rgn.extent = MTLSizeMake(static_cast<u32>(src_rect.width), static_cast<u32>(src_rect.height), static_cast<u32>(src_rect.depth));
		rgn.dst_offset = MTLOriginMake(static_cast<u32>(dst_rect.x), static_cast<u32>(dst_rect.y), static_cast<u32>(dst_rect.z));
		rgn.src_offset = MTLOriginMake(static_cast<u32>(src_rect.x), static_cast<u32>(src_rect.y), static_cast<u32>(src_rect.z));

		ensure(mip_layers.mipmap_count > 0);

		for (u32 remaining_levels = mip_layers.mipmap_count; remaining_levels > 0; --remaining_levels)
		{
			copy_image_region(cmd, src, dst, rgn);

			if (remaining_levels > 1)
			{
				increment_mip_level(rgn);
			}
		}
	}

	void copy_scaled_image(mtl::command_context& cmd,
		mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers,
		bool compatible_formats,
		bool linear_filter)
	{
		// Compatible formats is now optional (false), so we should check the formas again just in case.
		compatible_formats |= (src->format() == dst->format());

		if (compatible_formats && !src_rect.is_flipped() && !dst_rect.is_flipped() &&
			src_rect.width == dst_rect.width && src_rect.height == dst_rect.height && src_rect.depth == dst_rect.depth)
		{
			copy_image(cmd, src, dst, src_rect, dst_rect, mip_layers);
			return;
		}

		if (!is_renderable(dst))
		{
			// Scaled copies are rendered (mtl::blit_pass)
			rsx_log.error("Metal: Scaled copy into a non-renderable image (format=0x%x, type=%u)",
				static_cast<u32>(dst->format()), static_cast<u32>(dst->type()));
			return;
		}

		if (src->aspect() == image_aspect::color)
		{
			// vkCmdBlitImage equivalent
			if ((dst->aspect() & image_aspect::color) == 0)
			{
				rsx_log.error("Metal: Scaled copy from a color image into a depth/stencil image (format 0x%x -> 0x%x)",
					static_cast<u32>(src->format()), static_cast<u32>(dst->format()));
				return;
			}

			if (src->type() == MTLTextureType3D || dst->type() == MTLTextureType3D || src->samples() > 1)
			{
				// mtl::blit_pass samples a single-sampled 2D slice
				rsx_log.error("Metal: Unsupported scaled copy (src type=%u, samples=%u, dst type=%u)",
					static_cast<u32>(src->type()), src->samples(), static_cast<u32>(dst->type()));
				return;
			}

			image_blit_region rgn{};
			rgn.src_offsets[0] = { src_rect.x, src_rect.y, src_rect.z };
			rgn.src_offsets[1] = { src_rect.x + src_rect.width, src_rect.y + src_rect.height, src_rect.z + src_rect.depth };
			rgn.dst_offsets[0] = { dst_rect.x, dst_rect.y, dst_rect.z };
			rgn.dst_offsets[1] = { dst_rect.x + dst_rect.width, dst_rect.y + dst_rect.height, dst_rect.z + dst_rect.depth };
			rgn.src_level = mip_layers.src_mip_level;
			rgn.dst_level = mip_layers.dst_mip_level;

			ensure(mip_layers.mipmap_count > 0);
			auto blitter_pass = mtl::get_overlay_pass<mtl::blit_pass>();

			for (u32 remaining_levels = mip_layers.mipmap_count; remaining_levels > 0; remaining_levels--)
			{
				const areai src_area = { rgn.src_offsets[0].x, rgn.src_offsets[0].y, rgn.src_offsets[1].x, rgn.src_offsets[1].y };
				const areai dst_area = { rgn.dst_offsets[0].x, rgn.dst_offsets[0].y, rgn.dst_offsets[1].x, rgn.dst_offsets[1].y };

				for (u32 layer = 0; layer < mip_layers.layer_count; ++layer)
				{
					blitter_pass->run(cmd, src, dst, src_area, dst_area,
						rgn.src_level, mip_layers.src_layer + layer, rgn.dst_level, mip_layers.dst_layer + layer,
						linear_filter, image_aspect::color);
				}

				if (remaining_levels > 1)
				{
					increment_mip_level(rgn);
				}
			}
			return;
		}

		ensure(mip_layers.layer_count == 1);                //<- Unimplemented, but also not really needed. We don't currently support non-2D typeless scratch.
		ensure(mip_layers.mipmap_count == 1);               //<- Unsupported.
		ensure(dst_rect.depth == 1 && src_rect.depth == 1); //<- Unsupported.

		// Most depth/stencil formats cannot be scaled using hw blit
		if (src->format() != dst->format())
		{
			// Can happen because of depth float mismatch. Format width should be equal RSX-side
			auto typeless = mtl::get_typeless_helper(dst->format(), dst->format_class(), src_rect.abs_width(), src_rect.abs_height());
			copy_image_typeless(cmd, src, typeless, src_rect, src_rect, mip_layers.without_dst());
			copy_scaled_image(cmd, typeless, dst, src_rect, dst_rect, mip_layers.without_src(), true, linear_filter);
			return;
		}

		ensure(!dst_rect.is_flipped());

		// Metal: The VK typeless "stretch" round trips through color intermediates are not needed. mtl::blit_pass writes the
		// depth aspect with fragment depth and the stencil aspect with stencil export (depth/stencil is never interpolated).
		mtl::get_overlay_pass<mtl::blit_pass>()->run(cmd, src, dst, src_rect, dst_rect,
			mip_layers.src_mip_level, mip_layers.src_layer, mip_layers.dst_mip_level, mip_layers.dst_layer,
			linear_filter, src->aspect());
	}

	void clear_color_image(mtl::command_context& cmd, mtl::image* img, const color4f& color, const image_subresource_range& range)
	{
		const u32 aspect = range.aspect ? (range.aspect & img->aspect()) : img->aspect();
		ensure(aspect == image_aspect::color);

		if (!is_renderable(img))
		{
			if (color.r != 0.f || color.g != 0.f || color.b != 0.f || color.a != 0.f)
			{
				rsx_log.warning("Metal: Clear value ignored for a non-renderable image (format=0x%x), zero-filling instead", static_cast<u32>(img->format()));
			}

			zero_fill_image(cmd, img, aspect, range);
			return;
		}

		const auto [base_level, level_count] = get_level_range(img, range);
		const bool is_3d = (img->type() == MTLTextureType3D);

		for (u32 level = base_level; level < base_level + level_count; ++level)
		{
			const auto [first_slice, slice_count] = get_slice_range(img, range, level);

			for (u32 slice = first_slice; slice < first_slice + slice_count; ++slice)
			{
				MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor renderPassDescriptor];
				MTLRenderPassColorAttachmentDescriptor* attachment = desc.colorAttachments[0];
				attachment.texture = img->value;
				attachment.level = level;

				if (is_3d)
				{
					attachment.depthPlane = slice;
				}
				else
				{
					attachment.slice = slice;
				}

				attachment.loadAction = MTLLoadActionClear;
				attachment.storeAction = MTLStoreActionStore;
				attachment.clearColor = MTLClearColorMake(color.r, color.g, color.b, color.a);

				cmd.begin_render_pass(desc, @"Clear color image");
				cmd.end_encoding();
			}
		}
	}

	void clear_depth_stencil_image(mtl::command_context& cmd, mtl::image* img, f32 depth, u8 stencil, const image_subresource_range& range)
	{
		const u32 image_aspect_mask = img->aspect();
		const u32 aspect = range.aspect ? (range.aspect & image_aspect_mask) : image_aspect_mask;
		ensure(aspect && (aspect & image_aspect::color) == 0);

		if (!is_renderable(img))
		{
			rsx_log.warning("Metal: Clear value ignored for a non-renderable depth/stencil image (format=0x%x), zero-filling instead", static_cast<u32>(img->format()));
			zero_fill_image(cmd, img, aspect, range);
			return;
		}

		const auto [base_level, level_count] = get_level_range(img, range);

		for (u32 level = base_level; level < base_level + level_count; ++level)
		{
			const auto [first_slice, slice_count] = get_slice_range(img, range, level);

			for (u32 slice = first_slice; slice < first_slice + slice_count; ++slice)
			{
				MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor renderPassDescriptor];

				// Attach every plane of the image. Planes that are not cleared are preserved.
				if (image_aspect_mask & image_aspect::depth)
				{
					MTLRenderPassDepthAttachmentDescriptor* attachment = desc.depthAttachment;
					attachment.texture = img->value;
					attachment.level = level;
					attachment.slice = slice;
					attachment.loadAction = (aspect & image_aspect::depth) ? MTLLoadActionClear : MTLLoadActionLoad;
					attachment.storeAction = MTLStoreActionStore;
					attachment.clearDepth = depth;
				}

				if (image_aspect_mask & image_aspect::stencil)
				{
					MTLRenderPassStencilAttachmentDescriptor* attachment = desc.stencilAttachment;
					attachment.texture = img->value;
					attachment.level = level;
					attachment.slice = slice;
					attachment.loadAction = (aspect & image_aspect::stencil) ? MTLLoadActionClear : MTLLoadActionLoad;
					attachment.storeAction = MTLStoreActionStore;
					attachment.clearStencil = stencil;
				}

				cmd.begin_render_pass(desc, @"Clear depth/stencil image");
				cmd.end_encoding();
			}
		}
	}

	template <typename BaseType, typename BlockType>
	static cs_deswizzle_base* get_deswizzle_transformation_swapped(bool swap_bytes)
	{
		if (swap_bytes) [[likely]]
		{
			return mtl::get_compute_task<cs_deswizzle_3d<BaseType, BlockType, true>>();
		}

		return mtl::get_compute_task<cs_deswizzle_3d<BaseType, BlockType, false>>();
	}

	template <typename WordType>
	static cs_deswizzle_base* get_deswizzle_transformation(u32 block_size, bool swap_bytes)
	{
		switch (block_size)
		{
		case 1:
			return get_deswizzle_transformation_swapped<u8, u8>(swap_bytes);
		case 2:
			return get_deswizzle_transformation_swapped<u16, WordType>(swap_bytes);
		case 4:
		case 8:
		case 16:
			// Maximum block size on RSX is 4 bytes. Wider blocks are stored as multiple texels.
			return get_deswizzle_transformation_swapped<u32, WordType>(swap_bytes);
		default:
			fmt::throw_exception("Unreachable");
		}
	}

	static void gpu_deswizzle_sections_impl(mtl::command_context& cmd, mtl::buffer* scratch_buf, u32 dst_offset, int word_size, int word_count, bool swap_bytes, std::vector<buffer_image_copy>& sections)
	{
		// NOTE: This has to be done individually for every LOD
		mtl::cs_deswizzle_base* job = nullptr;
		const u32 block_size = (word_size * word_count);
		const u32 scale_x = std::max(block_size / 4u, 1u); // Virtual width multiplier. RSX only does texel sizes upto 32 bits.

		switch (word_size)
		{
		case 1:
			job = get_deswizzle_transformation<u8>(block_size, swap_bytes);
			break;
		case 2:
			job = get_deswizzle_transformation<u16>(block_size, swap_bytes);
			break;
		case 4:
			job = get_deswizzle_transformation<u32>(block_size, swap_bytes);
			break;
		default:
			fmt::throw_exception("Unimplemented deswizzle for format.");
		}

		ensure(job);

		auto next_layer = sections.front().base_layer;
		auto next_level = sections.front().mip_level;
		unsigned base = 0;
		unsigned lods = 0;

		std::vector<std::pair<unsigned, unsigned>> packets;
		for (unsigned i = 0; i < sections.size(); ++i)
		{
			ensure(sections[i].buffer_row_length);

			const auto layer = sections[i].base_layer;
			const auto level = sections[i].mip_level;

			if (layer == next_layer &&
				level == next_level)
			{
				next_level++;
				lods++;
				continue;
			}

			packets.emplace_back(base, lods);
			next_layer = layer;
			next_level = 1;
			base = i;
			lods = 1;
		}

		if (packets.empty() ||
			(packets.back().first + packets.back().second) < sections.size())
		{
			packets.emplace_back(base, lods);
		}

		for (const auto& packet : packets)
		{
			const auto& section = sections[packet.first];
			const auto src_offset = section.buffer_offset;

			// Align output to 128-byte boundary to keep some drivers happy
			dst_offset = utils::align(dst_offset, 128);

			u32 data_length = 0;
			for (unsigned i = 0, j = packet.first; i < packet.second; ++i, ++j)
			{
				const u32 packed_size = static_cast<u32>(sections[j].image_extent.width * sections[j].image_extent.height * sections[j].image_extent.depth * block_size);
				sections[j].buffer_offset = dst_offset;
				dst_offset += packed_size;
				data_length += packed_size;
			}

			const u32 buf_off32 = static_cast<u32>(section.buffer_offset);
			const u32 src_off32 = static_cast<u32>(src_offset);

			job->run(cmd, scratch_buf->handle(), buf_off32, scratch_buf->handle(), src_off32, data_length,
				static_cast<u32>(section.image_extent.width) * scale_x, static_cast<u32>(section.image_extent.height), static_cast<u32>(section.image_extent.depth), packet.second);
		}

		ensure(dst_offset <= scratch_buf->size());
	}

	static std::pair<u32, u32> calculate_upload_pitch(int format, u32 heap_align, const mtl::image* dst_image, const rsx::subresource_layout& layout, const rsx::texture_uploader_capabilities& caps)
	{
		u32 block_in_pixel = rsx::get_format_block_size_in_texel(format);
		u8  block_size_in_bytes = rsx::get_format_block_size_in_bytes(format);

		u32 row_pitch, upload_pitch_in_texel;

		if (!heap_align) [[likely]]
		{
			if (!layout.border) [[likely]]
			{
				row_pitch = (layout.pitch_in_block * block_size_in_bytes);
			}
			else
			{
				// Skip the border texels if possible. Padding is undesirable for GPU deswizzle
				row_pitch = (layout.width_in_block * block_size_in_bytes);
			}

			// We have row_pitch in source coordinates. But some formats have a software decode step which can affect this packing!
			// For such formats, the packed pitch on src does not match packed pitch on dst
			if (!rsx::is_compressed_host_format(caps, format))
			{
				const auto host_texel_width = mtl::get_format_texel_width(dst_image->format());
				const auto host_packed_pitch = host_texel_width * layout.width_in_texel;
				row_pitch = std::max<u32>(row_pitch, host_packed_pitch);
				upload_pitch_in_texel = row_pitch / host_texel_width;
			}
			else
			{
				upload_pitch_in_texel = std::max<u32>(block_in_pixel * row_pitch / block_size_in_bytes, layout.width_in_texel);
			}
		}
		else
		{
			row_pitch = rsx::align2(layout.width_in_block * block_size_in_bytes, heap_align);
			upload_pitch_in_texel = std::max<u32>(block_in_pixel * row_pitch / block_size_in_bytes, layout.width_in_texel);
			ensure(row_pitch == heap_align);
		}

		return { row_pitch, upload_pitch_in_texel };
	}

	void upload_image(mtl::command_context& cmd, mtl::image* dst_image,
		const std::vector<rsx::subresource_layout>& subresource_layout, int format, bool is_swizzled, u16 layer_count,
		u32 aspect, mtl::ring_buffer& upload_heap, u32 heap_align, rsx::flags32_t image_setup_flags)
	{
		// NOTE: There is no async transfer queue (upload_contents_async is ignored) and no image layouts in Metal.
		// The VK prepare_for_transfer step therefore reduces to recording on cmd.

		const bool requires_depth_processing = (dst_image->aspect() & image_aspect::stencil) || (format == CELL_GCM_TEXTURE_DEPTH16_FLOAT);
		rsx::texture_uploader_capabilities caps{ .supports_dxt = mtl::get_caps().supports_bc_texture_compression, .alignment = heap_align };
		rsx::texture_memory_info opt{};
		bool check_hw_caps = !(image_setup_flags & source_is_userptr);

		mtl::buffer* scratch_buf = nullptr;
		u32 scratch_offset = 0;
		u32 image_linear_size = 0;

		id<MTLBuffer> upload_buffer = nil;
		usz offset_in_upload_buffer = 0;

		struct buffer_copy_t
		{
			id<MTLBuffer> src;
			u64 src_offset;
			u64 dst_offset;
			u64 size;
		};

		std::vector<buffer_image_copy> copy_regions;
		std::vector<id<MTLBuffer>> copy_region_sources; // Metal: source buffer of each copy region
		std::vector<buffer_copy_t> buffer_copies;
		copy_regions.reserve(subresource_layout.size());
		copy_region_sources.reserve(subresource_layout.size());

		for (const rsx::subresource_layout& layout : subresource_layout)
		{
			if (layout.level >= dst_image->mipmaps())
			{
				rsx_log.error("Invalid subresource definition for the output texture. Mip level does not exist.");
				continue;
			}

			const auto [row_pitch, upload_pitch_in_texel] = calculate_upload_pitch(format, heap_align, dst_image, layout, caps);
			caps.alignment = row_pitch;

			// Calculate estimated memory utilization for this subresource
			image_linear_size = row_pitch * layout.depth * (rsx::is_compressed_host_format(caps, format) ? layout.height_in_block : layout.height_in_texel);

			// Only do GPU-side conversion if occupancy is good
			if (check_hw_caps)
			{
				caps.supports_byteswap = (image_linear_size >= 1024) || (image_setup_flags & source_is_gpu_resident);
				caps.supports_hw_deswizzle = caps.supports_byteswap;
				// Metal: Guest memory cannot be mapped as a GPU buffer (no DMA layer), so CPU sources are always staged through the
				// upload heap. Zero-copy only describes GPU-resident sources (no CPU access to the data at all).
				caps.supports_zero_copy = !!(image_setup_flags & source_is_gpu_resident);
				caps.supports_vtc_decoding = false;
				check_hw_caps = false;
			}

			auto buf_allocator = [&](usz) -> std::tuple<void*, usz>
			{
				if (image_setup_flags & source_is_gpu_resident)
				{
					// We should never reach here, unless something is very wrong...
					fmt::throw_exception("Cannot allocate CPU memory for GPU-only data");
				}

				// Map with extra padding bytes in case of realignment
				offset_in_upload_buffer = upload_heap.alloc<512>(image_linear_size + 8);
				void* mapped_buffer = upload_heap.map(offset_in_upload_buffer);
				return { mapped_buffer, image_linear_size };
			};

			auto io_buf = rsx::io_buffer(buf_allocator);
			opt = upload_texture_subresource(io_buf, layout, format, is_swizzled, caps);

			if (image_setup_flags & source_is_gpu_resident)
			{
				// Read from GPU buf if the input is already uploaded.
				const auto [iobuf, io_offset] = mtl::get_gpu_resident_source(layout.data);
				upload_buffer = iobuf;
				offset_in_upload_buffer = io_offset;
				// Never upload. Data is already resident.
				opt.require_upload = false;
			}
			else
			{
				// Read from upload buffer
				upload_buffer = upload_heap.handle();
			}

			// Without zero-copy for CPU sources, the deferred (DMA) transfer path of VK never triggers
			ensure(!opt.require_upload);

			copy_regions.push_back({});
			copy_region_sources.push_back(upload_buffer);

			auto& copy_info = copy_regions.back();
			copy_info.buffer_offset = offset_in_upload_buffer;
			copy_info.image_extent = MTLSizeMake(layout.width_in_texel, layout.height_in_texel, layout.depth);
			copy_info.aspect = aspect;
			copy_info.layer_count = 1;
			copy_info.base_layer = layout.layer;
			copy_info.mip_level = layout.level;
			copy_info.buffer_row_length = upload_pitch_in_texel;

			if (!layout.layer && !layout.level)
			{
				// Do not allow mixed transfer modes.
				// This can happen in special cases, e.g mipN having different processing than mip0 as is the case with the last VTC mip
				caps.supports_zero_copy = false;
			}

			if (opt.require_swap || opt.require_deswizzle || requires_depth_processing)
			{
				if (!scratch_buf)
				{
					// Calculate enough scratch memory. We need 2x the size of layer 0 to fit all the mip levels and an extra 128 bytes per level as alignment overhead.
					const u64 layer_size = (image_linear_size + image_linear_size);
					u64 scratch_buf_size = 128u * static_cast<u64>(subresource_layout.size()) + (layer_size * layer_count);
					if (opt.require_deswizzle)
					{
						// Double the memory if hw deswizzle is going to be used.
						// For GPU deswizzle, the memory is not transformed in-place, rather the decoded texture is placed at the end of the uploaded data.
						scratch_buf_size += scratch_buf_size;
					}

					if (requires_depth_processing)
					{
						// D-S aspect requires a load section that can fit a separated block => D(4) + S(1)
						// Due to reverse processing of inputs, only enough space to fit one layer is needed here.
						scratch_buf_size += (image_linear_size * 5) / 4;
					}

					scratch_buf = mtl::get_scratch_buffer(cmd, scratch_buf_size);
					buffer_copies.reserve(subresource_layout.size());
				}

				if (layout.level == 0)
				{
					// Align mip0 on a 128-byte boundary
					scratch_offset = utils::align(scratch_offset, 128);
				}

				// Copy from upload heap to scratch mem
				if (upload_buffer != scratch_buf->handle() || offset_in_upload_buffer != scratch_offset)
				{
					buffer_copies.push_back({ upload_buffer, offset_in_upload_buffer, scratch_offset, image_linear_size });
				}

				// Point data source to scratch mem
				copy_info.buffer_offset = scratch_offset;
				copy_region_sources.back() = scratch_buf->handle();

				scratch_offset += image_linear_size;
				ensure((scratch_offset + image_linear_size) <= scratch_buf->size()); // "Out of scratch memory"
			}
		}

		ensure(upload_buffer);

		if (opt.require_swap || opt.require_deswizzle || requires_depth_processing)
		{
			ensure(scratch_buf);

			auto blit = cmd.blit();
			for (const auto& copy : buffer_copies)
			{
				[blit copyFromBuffer:copy.src sourceOffset:copy.src_offset toBuffer:scratch_buf->handle() destinationOffset:copy.dst_offset size:copy.size];
			}
		}

		// Swap and deswizzle if requested
		if (opt.require_deswizzle)
		{
			gpu_deswizzle_sections_impl(cmd, scratch_buf, scratch_offset, opt.element_size, opt.block_length, opt.require_swap, copy_regions);
		}
		else if (opt.require_swap)
		{
			// NOTE: The shuffle kernels process whole 32-bit words. Round up so that a trailing 16-bit word is swapped too.
			gpu_swap_bytes_impl(cmd, scratch_buf->handle(), opt.element_size, 0, utils::align(scratch_offset, 4));
		}

		// CopyBufferToImage routines
		if (requires_depth_processing)
		{
			// Upload in reverse to avoid polluting data in lower space
			for (auto rIt = copy_regions.crbegin(); rIt != copy_regions.crend(); ++rIt)
			{
				mtl::copy_buffer_to_image(cmd, scratch_buf->handle(), dst_image, *rIt);
			}
		}
		else
		{
			if (scratch_buf)
			{
				ensure(opt.require_deswizzle || opt.require_swap);
			}

			for (usz i = 0; i < copy_regions.size(); ++i)
			{
				const auto& region = copy_regions[i];
				copy_buffer_to_image_raw(cmd, copy_region_sources[i], dst_image, region, region.aspect ? region.aspect : dst_image->aspect());
			}
		}

		// NOTE: VK signals on_guest_texture_read() here for host GPU labels, which the Metal backend does not implement
	}

	std::pair<id<MTLBuffer>, u32> detile_memory_block(mtl::command_context& cmd, const rsx::GCM_tile_reference& tiled_region,
		const utils::address_range32& range, u16 width, u16 height, u8 bpp)
	{
		// Calculate the true length of the usable memory section
		const auto available_tile_size = tiled_region.tile->size - (range.start - tiled_region.base_address);
		const auto max_content_size = tiled_region.tile->pitch * utils::align<u32>(height, 64);
		const auto section_length = std::min(max_content_size, available_tile_size);

		// Stage the guest data (VK: map_dma + load_dma, then a copy into the scratch buffer).
		// The detiler reads whole tile rows; keep those reads inside our allocation.
		auto upload_heap = ensure(mtl::get_upload_heap());
		const u32 staging_length = utils::align<u32>(section_length, tiled_region.tile->pitch);
		const auto tiled_data_offset = static_cast<u32>(upload_heap->alloc<512>(staging_length));
		std::memcpy(upload_heap->map(tiled_data_offset), vm::get_super_ptr<u8>(range.start), section_length);

		// Allocate scratch and prepare for the GPU job
		const auto scratch_buf = mtl::get_scratch_buffer(cmd, section_length * 3); // 0 = linear data, 1 = padding (deswz), 2 = unused (tiled data lives in the upload heap)
		const auto linear_data_scratch_offset = 0u;

		// Schedule the job
		const RSX_detiler_config config =
		{
			.tile_base_address = tiled_region.base_address,
			.tile_base_offset = range.start - tiled_region.base_address,
			.tile_rw_offset = range.start - tiled_region.base_address,   // TODO
			.tile_size = tiled_region.tile->size,
			.tile_pitch = tiled_region.tile->pitch,
			.bank = tiled_region.tile->bank,

			.dst = scratch_buf->handle(),
			.dst_offset = linear_data_scratch_offset,
			.src = upload_heap->handle(),
			.src_offset = tiled_data_offset,

			.image_width = width,
			.image_height = height,
			.image_pitch = static_cast<u32>(width) * bpp,
			.image_bpp = bpp
		};

		// Detile
		mtl::get_compute_task<mtl::cs_tile_memcpy<RSX_detiler_op::decode>>()->run(cmd, config);

		// Return a descriptor pointing to the decrypted data
		return { scratch_buf->handle(), linear_data_scratch_offset };
	}

	mtl::image* get_typeless_helper(MTLPixelFormat format, rsx::format_class format_class, u32 requested_width, u32 requested_height)
	{
		auto create_texture = [&]()
		{
			const u32 new_width = utils::align(requested_width, 256u);
			const u32 new_height = utils::align(requested_height, 256u);

			MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
			if (!mtl::is_compressed_format(format))
			{
				// Target of scaled copies (mtl::blit_pass)
				usage |= MTLTextureUsageRenderTarget;
			}

			return new mtl::image(MTLTextureType2D, format, new_width, new_height, 1, 1, 1, 1, usage, format_class, MTLStorageModePrivate);
		};

		const u64 key = (static_cast<u64>(format_class) << 32u) | static_cast<u64>(format);
		auto& ptr = g_typeless_textures[key];

		if (!ptr || ptr->width() < requested_width || ptr->height() < requested_height)
		{
			if (ptr)
			{
				requested_width = std::max(requested_width, ptr->width());
				requested_height = std::max(requested_height, ptr->height());

				// Command buffers retain the old texture while they reference it
				ptr.reset();
			}

			ptr.reset(create_texture());
			ptr->set_debug_name(fmt::format("Scratch: Format=0x%x", static_cast<u32>(format)));
		}

		return ptr.get();
	}

	void clear_typeless_helpers()
	{
		g_typeless_textures.clear();
		g_transfer_buffer.reset();
	}

	void blitter::scale_image(mtl::command_context& cmd, mtl::image* src, mtl::image* dst, areai src_area, areai dst_area, bool interpolate, const rsx::typeless_xfer& xfer_info)
	{
		mtl::image* real_src = src;
		mtl::image* real_dst = dst;

		// NOTE: VK initializes the layout of lazily created destinations here. Not applicable to Metal.

		// Optimization pass; check for pass-through data transfer
		if (!xfer_info.flip_horizontal && !xfer_info.flip_vertical && src_area.height() == dst_area.height())
		{
			auto src_w = src_area.width();
			auto dst_w = dst_area.width();

			if (xfer_info.src_is_typeless) src_w = static_cast<int>(src_w * xfer_info.src_scaling_hint);
			if (xfer_info.dst_is_typeless) dst_w = static_cast<int>(dst_w * xfer_info.dst_scaling_hint);

			if (src_w == dst_w)
			{
				// Final dimensions are a match
				if (xfer_info.src_is_typeless || xfer_info.dst_is_typeless)
				{
					mtl::copy_image_typeless(cmd, src, dst, src_area, dst_area);
				}
				else
				{
					copy_image(cmd, src, dst, src_area, dst_area);
				}

				return;
			}
		}

		if (xfer_info.src_is_typeless)
		{
			const auto format = xfer_info.src_native_format_override ?
				static_cast<MTLPixelFormat>(xfer_info.src_native_format_override) :
				mtl::get_compatible_sampler_format(mtl::get_formats_support(), xfer_info.src_gcm_format);

			if (format != src->format())
			{
				// Normalize input region (memory optimization)
				const auto old_src_area = src_area;
				src_area.y2 -= src_area.y1;
				src_area.y1 = 0;
				src_area.x2 = static_cast<int>(src_area.width() * xfer_info.src_scaling_hint);
				src_area.x1 = 0;

				// Transfer bits from src to typeless src
				real_src = mtl::get_typeless_helper(format, rsx::classify_format(xfer_info.src_gcm_format), src_area.width(), src_area.height());
				mtl::copy_image_typeless(cmd, src, real_src, old_src_area, src_area);
			}
		}

		// Save output region descriptor
		const auto old_dst_area = dst_area;

		if (xfer_info.dst_is_typeless)
		{
			const auto format = xfer_info.dst_native_format_override ?
				static_cast<MTLPixelFormat>(xfer_info.dst_native_format_override) :
				mtl::get_compatible_sampler_format(mtl::get_formats_support(), xfer_info.dst_gcm_format);

			if (format != dst->format())
			{
				// Normalize output region (memory optimization)
				dst_area.y2 -= dst_area.y1;
				dst_area.y1 = 0;
				dst_area.x2 = static_cast<int>(dst_area.width() * xfer_info.dst_scaling_hint);
				dst_area.x1 = 0;

				// Account for possibility where SRC is typeless and DST is typeless and both map to the same format
				auto required_height = dst_area.height();
				if (real_src != src && real_src->format() == format)
				{
					required_height += src_area.height();

					// Move the dst area just below the src area
					dst_area.y1 += src_area.y2;
					dst_area.y2 += src_area.y2;
				}

				real_dst = mtl::get_typeless_helper(format, rsx::classify_format(xfer_info.dst_gcm_format), dst_area.width(), required_height);
			}
		}

		// Checks
		if (src_area.x2 <= src_area.x1 || src_area.y2 <= src_area.y1 || dst_area.x2 <= dst_area.x1 || dst_area.y2 <= dst_area.y1)
		{
			rsx_log.error("Blit request consists of an empty region descriptor!");
			return;
		}

		if (src_area.x1 < 0 || src_area.x2 > static_cast<s32>(real_src->width()) || src_area.y1 < 0 || src_area.y2 > static_cast<s32>(real_src->height()))
		{
			rsx_log.error("Blit request denied because the source region does not fit!");
			return;
		}

		if (dst_area.x1 < 0 || dst_area.x2 > static_cast<s32>(real_dst->width()) || dst_area.y1 < 0 || dst_area.y2 > static_cast<s32>(real_dst->height()))
		{
			rsx_log.error("Blit request denied because the destination region does not fit!");
			return;
		}

		if (xfer_info.flip_horizontal)
		{
			src_area.flip_horizontal();
		}

		if (xfer_info.flip_vertical)
		{
			src_area.flip_vertical();
		}

		ensure(real_src->aspect() == real_dst->aspect()); // "Incompatible source and destination format!"

		copy_scaled_image(cmd, real_src, real_dst, src_area, dst_area, {},
			mtl::formats_are_bitcast_compatible(real_src, real_dst),
			interpolate);

		if (real_dst != dst)
		{
			mtl::copy_image_typeless(cmd, real_dst, dst, dst_area, old_dst_area);
		}
	}
}
