#pragma once

// Texture helpers of the Metal backend.
// Mirrors the texture section of VKHelpers.h plus everything VKTexture.cpp (and vkutils/scratch.h' typeless helper)
// exposes to other files. Implemented in MTLTexture.mm.
//
// General Metal notes (apply to every function below):
// - There are no image layouts or barriers. Metal tracks hazards on all our resources, so the VK push/pop_layout,
//   change_image_layout and insert_*_barrier calls simply disappear.
// - Every function that records GPU work takes the RSX thread's mtl::command_context and obtains the encoder it needs
//   from it (cmd.blit(), cmd.compute(), cmd.begin_render_pass()). Doing so ends the renderer's open render pass; the
//   renderer reopens it with MTLLoadActionLoad before its next draw (porting guide 9.3). These helpers never restore
//   renderer state.
// - Buffers are passed as id<MTLBuffer> plus an explicit byte offset where VK passes vk::buffer* (porting guide 9.2).

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
#include "Utilities/geometry.h"
#include "Utilities/address_range.h"
#include "Emu/RSX/Common/TextureUtils.h"
#include "Emu/RSX/Utils/rsx_utils.h"

#include <utility>
#include <vector>

namespace rsx
{
	struct GCM_tile_reference;
}

namespace mtl
{
	// Forward declarations
	class command_context;
	class ring_buffer;
	class image;

	struct image_readback_options_t
	{
		bool swap_bytes = false;

		// NOTE: Metal tracks buffer hazards automatically. The sync region is accepted for source compatibility with
		// the VK callers but no barrier is required for it.
		struct
		{
			u64 offset = 0;
			u64 length = 0;

			operator bool() const { return length != 0; }
		} sync_region {};
	};

	/**
	 * VkBufferImageCopy equivalent.
	 * - buffer_row_length / buffer_image_height are expressed in texels (like VK), 0 means tightly packed
	 *   (image_extent.width / image_extent.height). For block-compressed formats they must be multiples of the block size.
	 *   Bytes per row = (row_length / mtl::get_format_block_dim(fmt)) * mtl::get_format_texel_size(fmt).
	 * - aspect is a mtl::image_aspect mask. Combined depth+stencil transfers are split into separate depth and stencil
	 *   copies (MTLBlitOptionDepthFromDepthStencil / MTLBlitOptionStencilFromDepthStencil) by the copy helpers.
	 * - base_layer/layer_count address array slices (cubemap faces). For 3D images, image_offset.z/image_extent.depth
	 *   address depth slices and layer_count must be 1.
	 */
	struct buffer_image_copy
	{
		u64 buffer_offset = 0;
		u32 buffer_row_length = 0;
		u32 buffer_image_height = 0;
		u32 aspect = 0;
		u32 mip_level = 0;
		u32 base_layer = 0;
		u32 layer_count = 1;
		MTLOrigin image_offset = { 0, 0, 0 };
		MTLSize image_extent = { 0, 0, 0 };

		u32 row_length() const
		{
			return buffer_row_length ? buffer_row_length : static_cast<u32>(image_extent.width);
		}

		u32 image_height() const
		{
			return buffer_image_height ? buffer_image_height : static_cast<u32>(image_extent.height);
		}
	};

	/**
	 * VkImageSubresourceRange equivalent used by the clear helpers.
	 * aspect = 0 selects the image's own aspect mask. level_count/layer_count = 0 select all remaining levels/layers.
	 * For 3D images the layer range addresses depth slices.
	 */
	struct image_subresource_range
	{
		u32 aspect = 0;
		u32 base_level = 0;
		u32 level_count = 0;
		u32 base_layer = 0;
		u32 layer_count = 0;
	};

	enum image_upload_options
	{
		upload_contents_async   = 0x0001, // Ignored: Metal has no async transfer queue (VKAsyncScheduler is not ported), uploads are always inline
		initialize_image_layout = 0x0002, // No-op (no layouts in Metal), accepted for source compatibility
		preserve_image_layout   = 0x0004, // No-op (no layouts in Metal), accepted for source compatibility
		source_is_gpu_resident  = 0x0008, // subresource_layout::data holds a GPU buffer reference, see make_gpu_resident_source()
		source_is_userptr       = 0x0010,

		// meta-flags
		upload_contents_inline    = 0,
		upload_heap_align_default = 0
	};

	/**
	 * Encoding of a GPU-resident upload source in rsx::subresource_layout::data, used together with
	 * image_upload_options::source_is_gpu_resident (Metal version of VK's "subres.data = { vk::buffer*, offset }" hack).
	 * The pointer is a non-owning bridged id<MTLBuffer>; the buffer (normally the scratch buffer returned by
	 * detile_memory_block) must stay alive until upload_image has recorded its commands.
	 */
	inline rsx::io_buffer make_gpu_resident_source(id<MTLBuffer> buffer, u32 offset)
	{
		return rsx::io_buffer((__bridge void*)buffer, offset);
	}

	inline std::pair<id<MTLBuffer>, u32> get_gpu_resident_source(const rsx::io_buffer& data)
	{
		const auto [ptr, offset] = data.raw();
		return { (__bridge id<MTLBuffer>)ptr, static_cast<u32>(offset) };
	}

	/**
	 * Upload CPU texture data (or GPU-resident data, see source_is_gpu_resident) into dst_image.
	 * Data is staged through upload_heap; byteswap/deswizzle/depth-stencil conversions run as compute jobs
	 * (mtl::get_compute_task<>) on a scratch buffer exactly like VK. 'aspect' is a mtl::image_aspect mask.
	 * Always inline on cmd (upload_contents_async is ignored).
	 */
	void upload_image(mtl::command_context& cmd, mtl::image* dst_image,
		const std::vector<rsx::subresource_layout>& subresource_layout, int format, bool is_swizzled, u16 layer_count,
		u32 aspect, mtl::ring_buffer& upload_heap, u32 heap_align, rsx::flags32_t image_setup_flags);

	/**
	 * Detile a tiled guest memory region on the GPU (section length computed exactly like VK).
	 * VK reads the guest data through map_dma/load_dma. Metal: the guest range is memcpy'd by the CPU into a CPU-visible
	 * upload allocation (mtl::get_upload_heap(); never into the private scratch buffer, and never into a shared buffer the
	 * GPU may still read for an earlier job of the same command buffer), then the cs_tile_memcpy<decode> compute job
	 * (RSX_detiler_config.src = upload heap handle/offset) writes the linear data to offset 0 of
	 * mtl::get_scratch_buffer(cmd, section_length * 3).
	 * Returns { scratch buffer handle, offset of the linear data } (VK returns { vk::buffer*, offset }).
	 * The returned buffer belongs to the scratch pool and is only valid for immediate use on cmd.
	 */
	std::pair<id<MTLBuffer>, u32> detile_memory_block(
		mtl::command_context& cmd, const rsx::GCM_tile_reference& tiled_region, const utils::address_range32& range,
		u16 width, u16 height, u8 bpp);

	// Other texture management helpers

	/**
	 * Copy an image region into a buffer. D32F (D16F packing) and D32F_S8 (D24S8 packing) sources are converted by
	 * compute jobs exactly like VK, so dst must have calculate_working_buffer_size() bytes of room after region.buffer_offset.
	 */
	void copy_image_to_buffer(mtl::command_context& cmd, const mtl::image* src, id<MTLBuffer> dst, const buffer_image_copy& region, const image_readback_options_t& options = {});

	/**
	 * Copy buffer data into an image region. Packed D16F/D24S8 data destined to D32F/D32F_S8 images is converted by
	 * compute jobs first (in-place in src, VK semantics), so src must have calculate_working_buffer_size() bytes of room.
	 */
	void copy_buffer_to_image(mtl::command_context& cmd, id<MTLBuffer> src, const mtl::image* dst, const buffer_image_copy& region);

	u64  calculate_working_buffer_size(u64 base_size, u32 aspect);

	/**
	 * Bit-cast copy between images of different formats (image -> scratch buffer -> image), with optional byte shuffles.
	 * src_transfer_mask/dst_transfer_mask are mtl::image_aspect masks applied to the images' aspects.
	 */
	void copy_image_typeless(mtl::command_context& cmd, mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers = {},
		u32 src_transfer_mask = 0xFF, u32 dst_transfer_mask = 0xFF);

	/**
	 * Unscaled copy (vkCmdCopyImage equivalent, blit encoder copyFromTexture).
	 * Metal deviations:
	 * - Bit-compatible but different pixel formats are copied through a pixel format view of the source (requires
	 *   MTLTextureUsagePixelFormatView) or a scratch buffer.
	 * - Single-aspect copies of depth/stencil images (transfer masks) go through a scratch buffer using
	 *   MTLBlitOptionDepthFromDepthStencil / MTLBlitOptionStencilFromDepthStencil.
	 * - Mismatched depth formats are routed to copy_image_typeless (same as VK).
	 */
	void copy_image(mtl::command_context& cmd, mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers = {},
		u32 src_transfer_mask = 0xFF, u32 dst_transfer_mask = 0xFF);

	/**
	 * Scaled copy (vkCmdBlitImage equivalent). Falls back to copy_image when no scaling/flip is needed and formats are
	 * compatible. Metal has no scaled blits: scaling (and flips, i.e. negative rect extents) are done with the overlay
	 * mtl::blit_pass (MTLOverlays.h) which renders into dst, so dst must have MTLTextureUsageRenderTarget and src
	 * MTLTextureUsageShaderRead. blit_pass handles color->color (nearest/linear), depth->depth (fragment depth) and
	 * stencil (stencil export) directly, so the VK D/S typeless "stretch" round trips are not needed except for format
	 * mismatches. linear_filter replaces VK's VkFilter (true = VK_FILTER_LINEAR).
	 */
	void copy_scaled_image(mtl::command_context& cmd,
		mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		const rsx::image_copy_subresource_layers& mip_layers = {},
		bool compatible_formats = false, bool linear_filter = true);

	/**
	 * Clear helpers replacing vkCmdClearColorImage / vkCmdClearDepthStencilImage.
	 * Implemented with empty render passes using MTLLoadActionClear (one per mip level and array layer / depth slice),
	 * so the image must have MTLTextureUsageRenderTarget. Non-renderable (block-compressed) images are zero-filled
	 * through a blit from a zeroed scratch buffer instead (the clear value is ignored for them).
	 */
	void clear_color_image(mtl::command_context& cmd, mtl::image* image, const color4f& color, const image_subresource_range& range = {});
	void clear_depth_stencil_image(mtl::command_context& cmd, mtl::image* image, f32 depth, u8 stencil, const image_subresource_range& range = {});

	/**
	 * Scratch 2D image used as an intermediate for typeless transfers (VK: vkutils/scratch.cpp).
	 * One image per (format, format_class); grows (aligned to 256) as required and is never shrunk.
	 * Created with MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget (if renderable) | MTLTextureUsagePixelFormatView,
	 * private storage. Replacing a grown helper is safe at any time: command buffers retain the textures they use.
	 */
	mtl::image* get_typeless_helper(MTLPixelFormat format, rsx::format_class format_class, u32 requested_width, u32 requested_height);

	// Releases all typeless helpers (VK does this inside vk::clear_scratch_resources). Call on renderer shutdown.
	void clear_typeless_helpers();

	/**
	 * Auxiliary CPU-visible upload ring (VK: vkutils/data_heap.h vk::get_upload_heap(), "auxilliary upload heap", 64 MiB).
	 * Used where VK uses it: render_target::load_memory (MTLRenderTargets), texture_cache::initialize_subresource_from_memory
	 * and detile_memory_block (staging of the guest data).
	 * Forward declaration only: the definition is owned by module A (mtlutils, requested; same pattern as
	 * VKRenderTargets.h forward-declaring resolve_image). A redeclaration with this exact signature in mtlutils is fine.
	 */
	mtl::ring_buffer* get_upload_heap();

	struct blitter
	{
		void scale_image(mtl::command_context& cmd, mtl::image* src, mtl::image* dst, areai src_area, areai dst_area, bool interpolate, const rsx::typeless_xfer& xfer_info);
	};
}
