#include "stdafx.h"
#include "MTLFormats.h"
#include "MTLHelpers.h"
#include "mtlutils/device.h"
#include "mtlutils/image.h"

#include <atomic>

namespace mtl
{
	gpu_formats_support get_optimal_tiling_supported_formats(const render_device& dev)
	{
		gpu_formats_support result = {};

		// Apple silicon GPUs cannot render to or sample D24S8. VK already supports running without it, so the
		// z24s8 surfaces always take the D32S8 path (equivalent to VK with force_high_precision_z_buffer).
		// Depth32Float_Stencil8 is a required format on every Apple and Mac2 family GPU.
		result.d24_unorm_s8 = false;
		result.d32_sfloat_s8 = true;

		if (dev.caps().supports_depth24_stencil8)
		{
			rsx_log.notice("Metal: D24S8 is reported as supported by '%s' but will not be used", dev.caps().name);
		}

		// Linear (shared storage) BGRA8/RGBA8 textures can always be sampled and copied on Apple GPUs
		result.bgra8_linear = true;
		result.argb8_linear = true;

		return result;
	}

	gpu_formats_support get_formats_support()
	{
		return get_optimal_tiling_supported_formats(*ensure(get_current_device()));
	}

	MTLPixelFormat get_compatible_depth_surface_format(const gpu_formats_support& support, rsx::surface_depth_format2 format)
	{
		switch (format)
		{
		case rsx::surface_depth_format2::z16_uint:
			return MTLPixelFormatDepth16Unorm;
		case rsx::surface_depth_format2::z16_float:
			return MTLPixelFormatDepth32Float;
		case rsx::surface_depth_format2::z24s8_uint:
		{
			// No D24S8 on Apple silicon (support.d24_unorm_s8 is always false)
			if (support.d32_sfloat_s8) return MTLPixelFormatDepth32Float_Stencil8;
			fmt::throw_exception("No hardware support for z24s8");
		}
		case rsx::surface_depth_format2::z24s8_float:
		{
			if (support.d32_sfloat_s8) return MTLPixelFormatDepth32Float_Stencil8;
			fmt::throw_exception("No hardware support for z24s8_float");
		}
		default:
			break;
		}
		fmt::throw_exception("Invalid format (0x%x)", static_cast<u32>(format));
	}

	minification_filter get_min_filter(rsx::texture_minify_filter min_filter)
	{
		switch (min_filter)
		{
		case rsx::texture_minify_filter::nearest: return { MTLSamplerMinMagFilterNearest, MTLSamplerMipFilterNearest, false };
		case rsx::texture_minify_filter::linear: return { MTLSamplerMinMagFilterLinear, MTLSamplerMipFilterNearest, false };
		case rsx::texture_minify_filter::nearest_nearest: return { MTLSamplerMinMagFilterNearest, MTLSamplerMipFilterNearest, true };
		case rsx::texture_minify_filter::linear_nearest: return { MTLSamplerMinMagFilterLinear, MTLSamplerMipFilterNearest, true };
		case rsx::texture_minify_filter::nearest_linear: return { MTLSamplerMinMagFilterNearest, MTLSamplerMipFilterLinear, true };
		case rsx::texture_minify_filter::linear_linear: return { MTLSamplerMinMagFilterLinear, MTLSamplerMipFilterLinear, true };
		case rsx::texture_minify_filter::convolution_min: return { MTLSamplerMinMagFilterLinear, MTLSamplerMipFilterNearest, false };
		default:
			fmt::throw_exception("Invalid min filter");
		}
	}

	MTLSamplerMinMagFilter get_mag_filter(rsx::texture_magnify_filter mag_filter)
	{
		switch (mag_filter)
		{
		case rsx::texture_magnify_filter::nearest: return MTLSamplerMinMagFilterNearest;
		case rsx::texture_magnify_filter::linear: return MTLSamplerMinMagFilterLinear;
		case rsx::texture_magnify_filter::convolution_mag: return MTLSamplerMinMagFilterLinear;
		default:
			break;
		}

		fmt::throw_exception("Invalid mag filter (0x%x)", static_cast<u32>(mag_filter));
	}

	MTLSamplerAddressMode mtl_wrap_mode(rsx::texture_wrap_mode gcm_wrap)
	{
		switch (gcm_wrap)
		{
		case rsx::texture_wrap_mode::wrap: return MTLSamplerAddressModeRepeat;
		case rsx::texture_wrap_mode::mirror: return MTLSamplerAddressModeMirrorRepeat;
		case rsx::texture_wrap_mode::clamp_to_edge: return MTLSamplerAddressModeClampToEdge;
		case rsx::texture_wrap_mode::border: return MTLSamplerAddressModeClampToBorderColor;
		case rsx::texture_wrap_mode::clamp: return MTLSamplerAddressModeClampToEdge;
		case rsx::texture_wrap_mode::mirror_once_clamp_to_edge: return MTLSamplerAddressModeMirrorClampToEdge;
		case rsx::texture_wrap_mode::mirror_once_border: return MTLSamplerAddressModeMirrorClampToEdge;
		case rsx::texture_wrap_mode::mirror_once_clamp: return MTLSamplerAddressModeMirrorClampToEdge;
		default:
			fmt::throw_exception("Unhandled texture clamp mode");
		}
	}

	u8 max_aniso(rsx::texture_max_anisotropy gcm_aniso)
	{
		switch (gcm_aniso)
		{
		case rsx::texture_max_anisotropy::x1: return 1;
		case rsx::texture_max_anisotropy::x2: return 2;
		case rsx::texture_max_anisotropy::x4: return 4;
		case rsx::texture_max_anisotropy::x6: return 6;
		case rsx::texture_max_anisotropy::x8: return 8;
		case rsx::texture_max_anisotropy::x10: return 10;
		case rsx::texture_max_anisotropy::x12: return 12;
		case rsx::texture_max_anisotropy::x16: return 16;
		default:
			break;
		}

		fmt::throw_exception("Texture anisotropy error: bad max aniso (%d)", static_cast<u32>(gcm_aniso));
	}

	std::array<MTLTextureSwizzle, 4> get_component_mapping(u32 format)
	{
		// Component map in ARGB format
		std::array<MTLTextureSwizzle, 4> mapping = {};

		// NOTE: The packed 16-bit host formats are bit-identical to the VK formats chosen by the VK backend
		// (Metal names list components from LSB to MSB, e.g. MTLPixelFormatBGR5A1Unorm == VK_FORMAT_A1R5G5B5_UNORM_PACK16),
		// so the VK channel mappings apply unchanged.
		switch (format)
		{
		case CELL_GCM_TEXTURE_A1R5G5B5:
		case CELL_GCM_TEXTURE_R5G5B5A1:
		case CELL_GCM_TEXTURE_R6G5B5:
		case CELL_GCM_TEXTURE_R5G6B5:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT1:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT23:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT45:
			mapping = { MTLTextureSwizzleAlpha, MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue }; break;

		case CELL_GCM_TEXTURE_DEPTH24_D8:
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
		case CELL_GCM_TEXTURE_DEPTH16:
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
			mapping = { MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed }; break;

		case CELL_GCM_TEXTURE_A4R4G4B4:
			// Host format is ABGR4Unorm (== VK_FORMAT_R4G4B4A4_UNORM_PACK16): host R holds A, G holds R, B holds G, A holds B
			mapping = { MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha }; break;

		case CELL_GCM_TEXTURE_G8B8:
			mapping = { MTLTextureSwizzleGreen, MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleRed }; break;

		case CELL_GCM_TEXTURE_B8:
			mapping = { MTLTextureSwizzleOne, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed }; break;

		case CELL_GCM_TEXTURE_X16:
			mapping = { MTLTextureSwizzleRed, MTLTextureSwizzleOne, MTLTextureSwizzleRed, MTLTextureSwizzleOne }; break;

		case CELL_GCM_TEXTURE_X32_FLOAT:
			mapping = { MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed }; break;

		case CELL_GCM_TEXTURE_Y16_X16:
			mapping = { MTLTextureSwizzleGreen, MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleRed }; break;

		case CELL_GCM_TEXTURE_Y16_X16_FLOAT:
			mapping = { MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleRed, MTLTextureSwizzleGreen }; break;

		case CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT:
		case CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT:
			mapping = { MTLTextureSwizzleAlpha, MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue }; break;

		case CELL_GCM_TEXTURE_D8R8G8B8:
			mapping = { MTLTextureSwizzleOne, MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue }; break;

		case CELL_GCM_TEXTURE_D1R5G5B5:
			mapping = { MTLTextureSwizzleOne, MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue }; break;

		case CELL_GCM_TEXTURE_COMPRESSED_HILO8:
		case CELL_GCM_TEXTURE_COMPRESSED_HILO_S8:
			mapping = { MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleRed, MTLTextureSwizzleGreen }; break;

		case CELL_GCM_TEXTURE_COMPRESSED_B8R8_G8R8:
		case CELL_GCM_TEXTURE_COMPRESSED_R8B8_R8G8:
			mapping = { MTLTextureSwizzleAlpha, MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue }; break;

		case CELL_GCM_TEXTURE_A8R8G8B8:
			mapping = { MTLTextureSwizzleAlpha, MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue }; break;

		default:
			fmt::throw_exception("Invalid or unsupported component mapping for texture format (0x%x)", format);
		}

		return mapping;
	}

	MTLPixelFormat get_compatible_sampler_format([[maybe_unused]] const gpu_formats_support& support, u32 format)
	{
		const bool supports_dxt = get_caps().supports_bc_texture_compression;
		switch (format)
		{
		// Packed 16-bit formats are natively supported by Apple GPUs. The upload path (rsx::upload_texture_subresource)
		// performs the same 16-bit word copies as for VK (R6G5B5 is expanded to R5G6B5 on upload).
		// Metal equivalents of the VK formats (component names are listed from LSB to MSB in Metal):
		//   VK_FORMAT_R5G6B5_UNORM_PACK16   -> MTLPixelFormatB5G6R5Unorm
		//   VK_FORMAT_A1R5G5B5_UNORM_PACK16 -> MTLPixelFormatBGR5A1Unorm
		//   VK_FORMAT_R5G5B5A1_UNORM_PACK16 -> MTLPixelFormatA1BGR5Unorm
		//   VK_FORMAT_R4G4B4A4_UNORM_PACK16 -> MTLPixelFormatABGR4Unorm
		case CELL_GCM_TEXTURE_R5G6B5: return MTLPixelFormatB5G6R5Unorm;
		case CELL_GCM_TEXTURE_R6G5B5: return MTLPixelFormatB5G6R5Unorm; // Expand, discard high bit?
		case CELL_GCM_TEXTURE_R5G5B5A1: return MTLPixelFormatA1BGR5Unorm;
		case CELL_GCM_TEXTURE_D1R5G5B5: return MTLPixelFormatBGR5A1Unorm;
		case CELL_GCM_TEXTURE_A1R5G5B5: return MTLPixelFormatBGR5A1Unorm;
		case CELL_GCM_TEXTURE_A4R4G4B4: return MTLPixelFormatABGR4Unorm;
		case CELL_GCM_TEXTURE_B8: return MTLPixelFormatR8Unorm;
		case CELL_GCM_TEXTURE_A8R8G8B8: return MTLPixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT1: return supports_dxt ? MTLPixelFormatBC1_RGBA : MTLPixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT23: return supports_dxt ? MTLPixelFormatBC2_RGBA : MTLPixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT45: return supports_dxt ? MTLPixelFormatBC3_RGBA : MTLPixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_G8B8: return MTLPixelFormatRG8Unorm;
		case CELL_GCM_TEXTURE_DEPTH24_D8: return MTLPixelFormatDepth32Float_Stencil8; // No D24S8 on Apple silicon (support.d24_unorm_s8 is always false)
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT: return MTLPixelFormatDepth32Float_Stencil8;
		case CELL_GCM_TEXTURE_DEPTH16: return MTLPixelFormatDepth16Unorm;
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT: return MTLPixelFormatDepth32Float;
		case CELL_GCM_TEXTURE_X16: return MTLPixelFormatR16Unorm;
		case CELL_GCM_TEXTURE_Y16_X16: return MTLPixelFormatRG16Unorm;
		case CELL_GCM_TEXTURE_Y16_X16_FLOAT: return MTLPixelFormatRG16Float;
		case CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT: return MTLPixelFormatRGBA16Float;
		case CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT: return MTLPixelFormatRGBA32Float;
		case CELL_GCM_TEXTURE_X32_FLOAT: return MTLPixelFormatR32Float;
		case CELL_GCM_TEXTURE_D8R8G8B8: return MTLPixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_HILO8: return MTLPixelFormatRG8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_HILO_S8: return MTLPixelFormatRG8Snorm;
		case CELL_GCM_TEXTURE_COMPRESSED_B8R8_G8R8: return MTLPixelFormatBGRA8Unorm;
		case CELL_GCM_TEXTURE_COMPRESSED_R8B8_R8G8: return MTLPixelFormatBGRA8Unorm;
		default:
			break;
		}
		fmt::throw_exception("Invalid or unsupported sampler format for texture format (0x%x)", format);
	}

	MTLPixelFormat get_compatible_srgb_format(MTLPixelFormat rgb_format)
	{
		switch (rgb_format)
		{
		// 8-bit
		case MTLPixelFormatR8Unorm:
			return MTLPixelFormatR8Unorm_sRGB;
		case MTLPixelFormatRG8Unorm:
			return MTLPixelFormatRG8Unorm_sRGB;
		case MTLPixelFormatRGBA8Unorm:
			return MTLPixelFormatRGBA8Unorm_sRGB;
		case MTLPixelFormatBGRA8Unorm:
			return MTLPixelFormatBGRA8Unorm_sRGB;
		// DXT
		case MTLPixelFormatBC1_RGBA:
			return MTLPixelFormatBC1_RGBA_sRGB;
		case MTLPixelFormatBC2_RGBA:
			return MTLPixelFormatBC2_RGBA_sRGB;
		case MTLPixelFormatBC3_RGBA:
			return MTLPixelFormatBC3_RGBA_sRGB;
		default:
			return MTLPixelFormatInvalid;
		}
	}

	MTLPixelFormat get_compatible_snorm_format(MTLPixelFormat rgb_format)
	{
		switch (rgb_format)
		{
		// 8-bit
		case MTLPixelFormatR8Unorm:
			return MTLPixelFormatR8Snorm;
		case MTLPixelFormatRG8Unorm:
			return MTLPixelFormatRG8Snorm;
		case MTLPixelFormatRGBA8Unorm:
			return MTLPixelFormatRGBA8Snorm;
		// NOTE: Metal has no BGRA8 SNORM format (VK maps B8G8R8A8_UNORM -> B8G8R8A8_SNORM).
		// Returning Invalid keeps the SEXT bits in the texel remap control so the shader performs the conversion instead.
		// 16-bit
		case MTLPixelFormatR16Unorm:
			return MTLPixelFormatR16Snorm;
		case MTLPixelFormatRG16Unorm:
			return MTLPixelFormatRG16Snorm;
		default:
			return MTLPixelFormatInvalid;
		}
	}

	std::pair<MTLPixelFormat, MTLTextureSwizzleChannels> get_compatible_surface_format(rsx::surface_color_format color_format)
	{
		// NOTE: Swizzles only apply to sampled views. All formats below render correctly without any attachment-side swizzle.
		const MTLTextureSwizzleChannels o_rgb = { MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleOne };
		const MTLTextureSwizzleChannels z_rgb = { MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleZero };

		switch (color_format)
		{
		// Packed 16-bit formats are color renderable on Apple GPUs.
		// MTLPixelFormatB5G6R5Unorm == VK_FORMAT_R5G6B5_UNORM_PACK16, MTLPixelFormatBGR5A1Unorm == VK_FORMAT_A1R5G5B5_UNORM_PACK16
		case rsx::surface_color_format::r5g6b5:
			return std::make_pair(MTLPixelFormatB5G6R5Unorm, mtl::default_component_map);

		case rsx::surface_color_format::x1r5g5b5_o1r5g5b5:
			return std::make_pair(MTLPixelFormatBGR5A1Unorm, o_rgb);

		case rsx::surface_color_format::x1r5g5b5_z1r5g5b5:
			return std::make_pair(MTLPixelFormatBGR5A1Unorm, z_rgb);

		case rsx::surface_color_format::a8r8g8b8:
			return std::make_pair(MTLPixelFormatBGRA8Unorm, mtl::default_component_map);

		case rsx::surface_color_format::a8b8g8r8:
			return std::make_pair(MTLPixelFormatRGBA8Unorm, mtl::default_component_map);

		case rsx::surface_color_format::x8b8g8r8_o8b8g8r8:
			return std::make_pair(MTLPixelFormatRGBA8Unorm, o_rgb);

		case rsx::surface_color_format::x8b8g8r8_z8b8g8r8:
			return std::make_pair(MTLPixelFormatRGBA8Unorm, z_rgb);

		case rsx::surface_color_format::x8r8g8b8_z8r8g8b8:
			return std::make_pair(MTLPixelFormatBGRA8Unorm, z_rgb);

		case rsx::surface_color_format::x8r8g8b8_o8r8g8b8:
			return std::make_pair(MTLPixelFormatBGRA8Unorm, o_rgb);

		case rsx::surface_color_format::w16z16y16x16:
			return std::make_pair(MTLPixelFormatRGBA16Float, mtl::default_component_map);

		case rsx::surface_color_format::w32z32y32x32:
			return std::make_pair(MTLPixelFormatRGBA32Float, mtl::default_component_map);

		case rsx::surface_color_format::b8:
		{
			const MTLTextureSwizzleChannels no_alpha = { MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleOne };
			return std::make_pair(MTLPixelFormatR8Unorm, no_alpha);
		}

		case rsx::surface_color_format::g8b8:
		{
			const MTLTextureSwizzleChannels gb_rg = { MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleRed, MTLTextureSwizzleGreen };
			return std::make_pair(MTLPixelFormatRG8Unorm, gb_rg);
		}

		case rsx::surface_color_format::x32:
		{
			const MTLTextureSwizzleChannels rrrr = { MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed, MTLTextureSwizzleRed };
			return std::make_pair(MTLPixelFormatR32Float, rrrr);
		}

		default:
			rsx_log.error("Surface color buffer: Unsupported surface color format (0x%x)", static_cast<u32>(color_format));
			return std::make_pair(MTLPixelFormatBGRA8Unorm, mtl::default_component_map);
		}
	}

	u8 get_format_texel_width(MTLPixelFormat format)
	{
		switch (format)
		{
		case MTLPixelFormatR8Unorm:
		case MTLPixelFormatR8Unorm_sRGB:
		case MTLPixelFormatR8Snorm:
		case MTLPixelFormatR8Uint:
			return 1;
		case MTLPixelFormatR16Uint:
		case MTLPixelFormatR16Float:
		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatR16Snorm:
		case MTLPixelFormatRG8Unorm:
		case MTLPixelFormatRG8Unorm_sRGB:
		case MTLPixelFormatRG8Snorm:
		case MTLPixelFormatBGR5A1Unorm:
		case MTLPixelFormatABGR4Unorm:
		case MTLPixelFormatB5G6R5Unorm:
		case MTLPixelFormatA1BGR5Unorm:
			return 2;
		case MTLPixelFormatR32Uint:
		case MTLPixelFormatR32Float:
		case MTLPixelFormatRG16Unorm:
		case MTLPixelFormatRG16Snorm:
		case MTLPixelFormatRG16Float:
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB:
		case MTLPixelFormatRGBA8Snorm:
		case MTLPixelFormatBGRA8Unorm:
		case MTLPixelFormatBGRA8Unorm_sRGB:
		case MTLPixelFormatBC1_RGBA:
		case MTLPixelFormatBC2_RGBA:
		case MTLPixelFormatBC3_RGBA:
		case MTLPixelFormatBC1_RGBA_sRGB:
		case MTLPixelFormatBC2_RGBA_sRGB:
		case MTLPixelFormatBC3_RGBA_sRGB:
			return 4;
		case MTLPixelFormatRGBA16Float:
			return 8;
		case MTLPixelFormatRGBA32Float:
			return 16;
		case MTLPixelFormatDepth16Unorm:
		case MTLPixelFormatDepth32Float: // Holds z16_float data, RSX texel is 2 bytes wide
			return 2;
		case MTLPixelFormatDepth32Float_Stencil8: // Holds z24s8 data, RSX texel is 4 bytes wide
		case MTLPixelFormatDepth24Unorm_Stencil8:
			return 4;
		default:
			break;
		}

		fmt::throw_exception("Unexpected MTLPixelFormat 0x%X", static_cast<u32>(format));
	}

	std::pair<u8, u8> get_format_element_size(MTLPixelFormat format)
	{
		// Return value is {ELEMENT_SIZE, NUM_ELEMENTS_PER_TEXEL}
		// NOTE: Due to endianness issues, coalesced larger types are preferred
		// e.g UINT1 to hold 4x1 bytes instead of UBYTE4 to hold 4x1

		switch (format)
		{
		//8-bit
		case MTLPixelFormatR8Unorm:
		case MTLPixelFormatR8Unorm_sRGB:
		case MTLPixelFormatR8Snorm:
		case MTLPixelFormatR8Uint:
			return{ 1, 1 };
		case MTLPixelFormatRG8Unorm:
		case MTLPixelFormatRG8Unorm_sRGB:
		case MTLPixelFormatRG8Snorm:
			return{ 2, 1 }; //UNSIGNED_SHORT_8_8
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB:
		case MTLPixelFormatRGBA8Snorm:
		case MTLPixelFormatBGRA8Unorm:
		case MTLPixelFormatBGRA8Unorm_sRGB:
			return{ 4, 1 }; //UNSIGNED_INT_8_8_8_8
		//16-bit
		case MTLPixelFormatR16Uint:
		case MTLPixelFormatR16Float:
		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatR16Snorm:
			return{ 2, 1 }; //UNSIGNED_SHORT and HALF_FLOAT
		case MTLPixelFormatRG16Unorm:
		case MTLPixelFormatRG16Snorm:
		case MTLPixelFormatRG16Float:
			return{ 2, 2 }; //HALF_FLOAT
		case MTLPixelFormatRGBA16Float:
			return{ 2, 4 }; //HALF_FLOAT
		case MTLPixelFormatBGR5A1Unorm:
		case MTLPixelFormatABGR4Unorm:
		case MTLPixelFormatB5G6R5Unorm:
		case MTLPixelFormatA1BGR5Unorm:
			return{ 2, 1 }; //UNSIGNED_SHORT_X_Y_Z_W
		//32-bit
		case MTLPixelFormatR32Uint:
		case MTLPixelFormatR32Float:
			return{ 4, 1 }; //FLOAT
		case MTLPixelFormatRGBA32Float:
			return{ 4, 4 }; //FLOAT
		//DXT
		case MTLPixelFormatBC1_RGBA:
		case MTLPixelFormatBC2_RGBA:
		case MTLPixelFormatBC3_RGBA:
		case MTLPixelFormatBC1_RGBA_sRGB:
		case MTLPixelFormatBC2_RGBA_sRGB:
		case MTLPixelFormatBC3_RGBA_sRGB:
			return{ 4, 1 };
		//Depth
		case MTLPixelFormatDepth16Unorm:
		case MTLPixelFormatDepth32Float:
			return{ 2, 1 };
		case MTLPixelFormatDepth32Float_Stencil8:
		case MTLPixelFormatDepth24Unorm_Stencil8:
			return{ 4, 1 };
		default:
			break;
		}

		fmt::throw_exception("Unexpected MTLPixelFormat 0x%X", static_cast<u32>(format));
	}

	std::pair<bool, u32> get_format_convert_flags(MTLPixelFormat format)
	{
		switch (format)
		{
			//8-bit
		case MTLPixelFormatR8Unorm:
		case MTLPixelFormatR8Unorm_sRGB:
		case MTLPixelFormatR8Snorm:
		case MTLPixelFormatR8Uint:
			return{ false, 1 };
		case MTLPixelFormatBGRA8Unorm:
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatBGRA8Unorm_sRGB:
		case MTLPixelFormatRGBA8Unorm_sRGB:
		case MTLPixelFormatRGBA8Snorm:
			return{ true, 4 };
			//16-bit
		case MTLPixelFormatR16Uint:
		case MTLPixelFormatR16Float:
		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatR16Snorm:
		case MTLPixelFormatRG8Unorm:
		case MTLPixelFormatRG8Unorm_sRGB:
		case MTLPixelFormatRG8Snorm:
		case MTLPixelFormatRG16Unorm:
		case MTLPixelFormatRG16Snorm:
		case MTLPixelFormatRG16Float:
		case MTLPixelFormatRGBA16Float:
		case MTLPixelFormatBGR5A1Unorm:
		case MTLPixelFormatABGR4Unorm:
		case MTLPixelFormatB5G6R5Unorm:
		case MTLPixelFormatA1BGR5Unorm:
			return{ true, 2 };
			//32-bit
		case MTLPixelFormatR32Uint:
		case MTLPixelFormatR32Float:
		case MTLPixelFormatRGBA32Float:
			return{ true, 4 };
			//DXT
		case MTLPixelFormatBC1_RGBA:
		case MTLPixelFormatBC2_RGBA:
		case MTLPixelFormatBC3_RGBA:
		case MTLPixelFormatBC1_RGBA_sRGB:
		case MTLPixelFormatBC2_RGBA_sRGB:
		case MTLPixelFormatBC3_RGBA_sRGB:
			return{ false, 1 };
			//Depth
		case MTLPixelFormatDepth16Unorm:
		case MTLPixelFormatDepth32Float:
			return{ true, 2 };
		case MTLPixelFormatDepth32Float_Stencil8:
		case MTLPixelFormatDepth24Unorm_Stencil8:
			return{ true, 4 };
		default:
			break;
		}

		fmt::throw_exception("Unknown MTLPixelFormat 0x%x", static_cast<u32>(format));
	}

	bool formats_are_bitcast_compatible(MTLPixelFormat format1, MTLPixelFormat format2)
	{
		if (format1 == format2) [[likely]]
		{
			return true;
		}

		// Formats are compatible if the following conditions are met:
		// 1. Texel sizes must match
		// 2. Both formats require no transforms (basic memcpy) or...
		// 3. Both formats have the same transform (e.g RG16_UNORM to RG16_SFLOAT, both are down and uploaded with a 2-byte byteswap)

		if (get_format_texel_width(format1) != get_format_texel_width(format2))
		{
			return false;
		}

		const auto transform_a = get_format_convert_flags(format1);
		const auto transform_b = get_format_convert_flags(format2);

		if (transform_a.first == transform_b.first)
		{
			return !transform_a.first || (transform_a.second == transform_b.second);
		}

		return false;
	}

	bool formats_are_bitcast_compatible(image* image1, image* image2)
	{
		if (const u32 transfer_class = image1->format_class() | image2->format_class();
			transfer_class & RSX_FORMAT_CLASS_DEPTH_FLOAT_MASK)
		{
			// If any one of the two images is a depth float, the other must match exactly or bust
			return (image1->format_class() == image2->format_class());
		}

		return formats_are_bitcast_compatible(image1->format(), image2->format());
	}

	bool is_linear_filtering_supported(MTLPixelFormat format)
	{
		// Metal has no format feature query; capabilities follow the Metal feature set tables for Apple7+/Mac2 GPUs.
		switch (format)
		{
		// Integer and stencil formats can never be filtered
		case MTLPixelFormatR8Uint:
		case MTLPixelFormatR8Sint:
		case MTLPixelFormatR16Uint:
		case MTLPixelFormatR16Sint:
		case MTLPixelFormatRG8Uint:
		case MTLPixelFormatRG16Uint:
		case MTLPixelFormatR32Uint:
		case MTLPixelFormatR32Sint:
		case MTLPixelFormatRG32Uint:
		case MTLPixelFormatRGBA8Uint:
		case MTLPixelFormatRGBA16Uint:
		case MTLPixelFormatRGBA32Uint:
		case MTLPixelFormatStencil8:
		case MTLPixelFormatX32_Stencil8:
		case MTLPixelFormatX24_Stencil8:
		case MTLPixelFormatInvalid:
			return false;

		// 32-bit float formats (including 32-bit float depth) are filterable only when the device reports it
		case MTLPixelFormatR32Float:
		case MTLPixelFormatRG32Float:
		case MTLPixelFormatRGBA32Float:
		case MTLPixelFormatDepth32Float:
		case MTLPixelFormatDepth32Float_Stencil8:
			return get_caps().supports_32bit_float_filtering;

		default:
			// Normalized, 16-bit float, packed and BCn formats
			return true;
		}
	}

	std::pair<MTLPrimitiveType, bool> get_appropriate_topology(rsx::primitive_type mode)
	{
		// Mirrors the __APPLE__ path of vk::get_appropriate_topology (no triangle fans in Metal).
		// The primitive tables live in mtlutils/mtl_common.mm so that both helpers always agree.
		return { mtl::get_primitive_type(mode), !mtl::is_primitive_native(mode) };
	}

	MTLCompareFunction get_compare_func(rsx::comparison_function op, bool reverse_direction)
	{
		switch (op)
		{
		case rsx::comparison_function::never: return MTLCompareFunctionNever;
		case rsx::comparison_function::greater: return reverse_direction ? MTLCompareFunctionLess : MTLCompareFunctionGreater;
		case rsx::comparison_function::less: return reverse_direction ? MTLCompareFunctionGreater : MTLCompareFunctionLess;
		case rsx::comparison_function::less_or_equal: return reverse_direction ? MTLCompareFunctionGreaterEqual : MTLCompareFunctionLessEqual;
		case rsx::comparison_function::greater_or_equal: return reverse_direction ? MTLCompareFunctionLessEqual : MTLCompareFunctionGreaterEqual;
		case rsx::comparison_function::equal: return MTLCompareFunctionEqual;
		case rsx::comparison_function::not_equal: return MTLCompareFunctionNotEqual;
		case rsx::comparison_function::always: return MTLCompareFunctionAlways;
		default:
			fmt::throw_exception("Unknown compare op: 0x%x", static_cast<u32>(op));
		}
	}

	MTLBlendFactor get_blend_factor(rsx::blend_factor factor)
	{
		switch (factor)
		{
		case rsx::blend_factor::one: return MTLBlendFactorOne;
		case rsx::blend_factor::zero: return MTLBlendFactorZero;
		case rsx::blend_factor::src_alpha: return MTLBlendFactorSourceAlpha;
		case rsx::blend_factor::dst_alpha: return MTLBlendFactorDestinationAlpha;
		case rsx::blend_factor::src_color: return MTLBlendFactorSourceColor;
		case rsx::blend_factor::dst_color: return MTLBlendFactorDestinationColor;
		case rsx::blend_factor::constant_color: return MTLBlendFactorBlendColor;
		case rsx::blend_factor::constant_alpha: return MTLBlendFactorBlendAlpha;
		case rsx::blend_factor::one_minus_src_color: return MTLBlendFactorOneMinusSourceColor;
		case rsx::blend_factor::one_minus_dst_color: return MTLBlendFactorOneMinusDestinationColor;
		case rsx::blend_factor::one_minus_src_alpha: return MTLBlendFactorOneMinusSourceAlpha;
		case rsx::blend_factor::one_minus_dst_alpha: return MTLBlendFactorOneMinusDestinationAlpha;
		case rsx::blend_factor::one_minus_constant_alpha: return MTLBlendFactorOneMinusBlendAlpha;
		case rsx::blend_factor::one_minus_constant_color: return MTLBlendFactorOneMinusBlendColor;
		case rsx::blend_factor::src_alpha_saturate: return MTLBlendFactorSourceAlphaSaturated;
		default:
			fmt::throw_exception("Unknown blend factor 0x%x", static_cast<u32>(factor));
		}
	}

	MTLBlendOperation get_blend_op(rsx::blend_equation op)
	{
		switch (op)
		{
		case rsx::blend_equation::add_signed:
			rsx_log.error("blend equation add_signed used. Emulating using FUNC_ADD");
			[[fallthrough]];
		case rsx::blend_equation::add: return MTLBlendOperationAdd;
		case rsx::blend_equation::reverse_add_signed:
			rsx_log.error("blend equation reverse_add_signed used. Emulating using FUNC_ADD");
			return MTLBlendOperationAdd;
		case rsx::blend_equation::subtract: return MTLBlendOperationSubtract;
		case rsx::blend_equation::reverse_subtract_signed:
			rsx_log.error("blend equation reverse_subtract_signed used. Emulating using FUNC_REVERSE_SUBTRACT");
			[[fallthrough]];
		case rsx::blend_equation::reverse_subtract: return MTLBlendOperationReverseSubtract;
		case rsx::blend_equation::min: return MTLBlendOperationMin;
		case rsx::blend_equation::max: return MTLBlendOperationMax;
		default:
			fmt::throw_exception("Unknown blend op: 0x%x", static_cast<u32>(op));
		}
	}

	MTLStencilOperation get_stencil_op(rsx::stencil_op op)
	{
		switch (op)
		{
		case rsx::stencil_op::keep: return MTLStencilOperationKeep;
		case rsx::stencil_op::zero: return MTLStencilOperationZero;
		case rsx::stencil_op::replace: return MTLStencilOperationReplace;
		case rsx::stencil_op::incr: return MTLStencilOperationIncrementClamp;
		case rsx::stencil_op::decr: return MTLStencilOperationDecrementClamp;
		case rsx::stencil_op::invert: return MTLStencilOperationInvert;
		case rsx::stencil_op::incr_wrap: return MTLStencilOperationIncrementWrap;
		case rsx::stencil_op::decr_wrap: return MTLStencilOperationDecrementWrap;
		default:
			fmt::throw_exception("Unknown stencil op: 0x%x", static_cast<u32>(op));
		}
	}

	MTLWinding get_front_face(rsx::front_face ffv)
	{
		// RSX face winding is always based off of the upper-left corner, like Vulkan (GL inverts it because its origin is
		// bottom-left; VK maps it 1:1 with no inversion: cw -> VK_FRONT_FACE_CLOCKWISE).
		// mtl::compile_glsl() flips Y in every vertex shader (SPIRV-Cross flip_vert_y, like MoltenVK), so primitives land
		// at the same top-left based window coordinates as on Vulkan and Metal evaluates the winding there. Hence the VK
		// mapping carries over unchanged (MoltenVK maps VK_FRONT_FACE_CLOCKWISE -> MTLWindingClockwise as well).
		switch (ffv)
		{
		case rsx::front_face::cw: return MTLWindingClockwise;
		case rsx::front_face::ccw: return MTLWindingCounterClockwise;
		default:
			fmt::throw_exception("Unknown front face value: 0x%x", static_cast<u32>(ffv));
		}
	}

	MTLCullMode get_cull_face(rsx::cull_face cfv)
	{
		switch (cfv)
		{
		case rsx::cull_face::back: return MTLCullModeBack;
		case rsx::cull_face::front: return MTLCullModeFront;
		case rsx::cull_face::front_and_back:
			// No Metal equivalent of VK_CULL_MODE_FRONT_AND_BACK. See cull_face_rejects_all_polygons().
			return MTLCullModeNone;
		default:
			fmt::throw_exception("Unknown cull face value: 0x%x", static_cast<u32>(cfv));
		}
	}

	bool cull_face_rejects_all_polygons(rsx::cull_face cfv)
	{
		return cfv == rsx::cull_face::front_and_back;
	}

	MTLTriangleFillMode get_polygon_mode(rsx::polygon_mode mode)
	{
		switch (mode)
		{
		case rsx::polygon_mode::fill: return MTLTriangleFillModeFill;
		case rsx::polygon_mode::line: return MTLTriangleFillModeLines;
		case rsx::polygon_mode::point:
		{
			static std::atomic<bool> s_warned = false;
			if (!s_warned.exchange(true))
			{
				rsx_log.warning("Polygon point mode is not supported by Metal. Falling back to fill mode.");
			}
			return MTLTriangleFillModeFill;
		}
		default:
			fmt::throw_exception("Unknown polygon mode: 0x%x", static_cast<u32>(mode));
		}
	}
}
