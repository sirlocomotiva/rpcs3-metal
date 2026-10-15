#pragma once

#include "Emu/RSX/Metal/mtlutils/mtl_common.h"
#include "Emu/RSX/Metal/mtlutils/sampler.h" // mtl::get_border_color(u32) (vk::get_border_color equivalent, defined in mtlutils/sampler.mm)

#include <array>
#include <utility>

namespace mtl
{
	class image;
	class render_device;

	/**
	 * Mirrors vk::gpu_formats_support.
	 * Apple silicon has no D24S8 support; z24s8 surfaces always use MTLPixelFormatDepth32Float_Stencil8.
	 * The remaining flags are always set on Apple GPUs but are kept so that the VK texture cache logic ports unchanged.
	 */
	struct gpu_formats_support
	{
		bool d24_unorm_s8 = false;
		bool d32_sfloat_s8 = true;
		bool bgra8_linear = true;
		bool argb8_linear = true;
	};

	// Mirrors vk::get_optimal_tiling_supported_formats(physical_device)
	gpu_formats_support get_optimal_tiling_supported_formats(const render_device& dev);

	// Mirrors vk::render_device::get_formats_support() for the current device (mtl::get_current_device())
	gpu_formats_support get_formats_support();

	struct minification_filter
	{
		MTLSamplerMinMagFilter filter;
		MTLSamplerMipFilter mipmap_mode;  // Nearest or Linear. Callers use NotMipmapped (or clamp the LOD range to 0) when !sample_mipmaps
		bool sample_mipmaps;
	};

	// NOTE: vk::get_border_color(u32) equivalent is mtl::get_border_color(u32) declared in mtlutils/sampler.h (included above)

	MTLPixelFormat get_compatible_depth_surface_format(const gpu_formats_support& support, rsx::surface_depth_format2 format);
	MTLPixelFormat get_compatible_sampler_format(const gpu_formats_support& support, u32 format);
	MTLPixelFormat get_compatible_srgb_format(MTLPixelFormat rgb_format);   // MTLPixelFormatInvalid if there is no sRGB variant
	MTLPixelFormat get_compatible_snorm_format(MTLPixelFormat rgb_format);  // MTLPixelFormatInvalid if there is no SNORM variant
	u8 get_format_texel_width(MTLPixelFormat format);
	std::pair<u8, u8> get_format_element_size(MTLPixelFormat format);       // {ELEMENT_SIZE, NUM_ELEMENTS_PER_TEXEL}
	std::pair<bool, u32> get_format_convert_flags(MTLPixelFormat format);   // {requires byteswap, swap block size}
	bool formats_are_bitcast_compatible(MTLPixelFormat format1, MTLPixelFormat format2);
	bool formats_are_bitcast_compatible(image* image1, image* image2);

	// Surface color format -> {host format, sampling swizzle}. The swizzle only applies to sampled views, never to attachments.
	// (vk::get_compatible_surface_format lives in VKHelpers.h; it is part of the formats module here)
	std::pair<MTLPixelFormat, MTLTextureSwizzleChannels> get_compatible_surface_format(rsx::surface_color_format color_format);

	// Replacement for the VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT device query used by the draw path
	bool is_linear_filtering_supported(MTLPixelFormat format);

	// Sampler state
	minification_filter get_min_filter(rsx::texture_minify_filter min_filter);
	MTLSamplerMinMagFilter get_mag_filter(rsx::texture_magnify_filter mag_filter);
	MTLSamplerAddressMode mtl_wrap_mode(rsx::texture_wrap_mode gcm_wrap);
	u8 max_aniso(rsx::texture_max_anisotropy gcm_aniso);

	// Base component mapping of a texture format, in ARGB order (feed into mtl::apply_swizzle_remap)
	std::array<MTLTextureSwizzle, 4> get_component_mapping(u32 format);

	// {native primitive type, emulated}. Emulated primitives are expanded to indexed lists by the RSX BufferUtils helpers.
	// Consistent with mtl::is_primitive_native() / mtl::get_primitive_type() (mtlutils/mtl_common.h).
	std::pair<MTLPrimitiveType, bool> get_appropriate_topology(rsx::primitive_type mode);

	// Fixed-function state (VK keeps these in VKGSRender.cpp / VKDraw.cpp)
	MTLCompareFunction get_compare_func(rsx::comparison_function op, bool reverse_direction = false);
	MTLStencilOperation get_stencil_op(rsx::stencil_op op);
	MTLBlendFactor get_blend_factor(rsx::blend_factor factor);
	MTLBlendOperation get_blend_op(rsx::blend_equation op);
	MTLWinding get_front_face(rsx::front_face ffv);

	// Metal has no FRONT_AND_BACK cull mode: front_and_back returns MTLCullModeNone and cull_face_rejects_all_polygons()
	// returns true. The renderer must then skip triangle-class draws (lines/points are never face-culled), which is
	// exactly what VK_CULL_MODE_FRONT_AND_BACK does.
	MTLCullMode get_cull_face(rsx::cull_face cfv);
	bool cull_face_rejects_all_polygons(rsx::cull_face cfv);

	// Metal only supports fill and line rasterization of polygons. Point mode falls back to fill.
	// (The VK backend ignores the RSX polygon mode entirely; this is provided for parity with GL)
	MTLTriangleFillMode get_polygon_mode(rsx::polygon_mode mode);
}
