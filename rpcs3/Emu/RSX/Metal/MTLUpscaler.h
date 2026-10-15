#pragma once

// Metal port of VK/upscalers (upscaling.h, nearest_pass.hpp, bilinear_pass.hpp, fsr_pass.h).
// Implementations live in MTLUpscaler.mm (H3).
//
// Scaling request semantics (replaces VkImageBlit):
// - src_area / dst_area hold the two corner offsets of the source and destination regions exactly like
//   VkImageBlit::srcOffsets[0..1] / dstOffsets[0..1]: (x1, y1) = offsets[0], (x2, y2) = offsets[1].
// - A reversed axis (x1 > x2 or y1 > y2) on either side mirrors the image along that axis, like vkCmdBlitImage.
//   areai::width()/height() return absolute extents.
// - The destination region is in present surface coordinates when committing, otherwise it only describes the output
//   size (offsets are relative to the returned image).
//
// Metal has no scaled blits: committing passes render into the present surface with mtl::copy_scaled_image()
// (MTLTexture.h, backed by mtl::blit_pass), so present_surface needs MTLTextureUsageRenderTarget (drawables have it).
// copy_scaled_image falls back to an unscaled blit-encoder copy (mtl::copy_image) when no scaling/flip is needed and the
// formats are compatible, which is illegal on framebuffer-only drawables: the CAMetalLayer must be configured with
// framebufferOnly = NO (MTLGSRender::init_metal_layer).

// Foundation pulls in CoreServices/CarbonCore MachineExceptions.h, which declares a global 'Vector128' type that collides with
// the Vector128 concept from util/v128.hpp (included by most RSX headers). Rename the Apple type while the frameworks are
// imported (same workaround as MTLCommonDecompiler.h). No-op if the frameworks were already imported by this translation unit.
#pragma push_macro("Vector128")
#undef Vector128
#define Vector128 CarbonCore_Vector128
#include "Emu/RSX/Metal/mtlutils/mtl_common.h"
#undef Vector128
#pragma pop_macro("Vector128")

#include "Emu/RSX/Metal/mtlutils/command_context.h"
#include "Emu/RSX/Metal/mtlutils/image.h"

#include "Emu/RSX/Utils/rsx_utils.h"
#include "Utilities/geometry.h"

#include <memory>

// MetalFX is only imported by MTLUpscaler.mm
@protocol MTLFXSpatialScaler;

namespace mtl
{
	namespace upscaling_flags_
	{
		enum upscaling_flags
		{
			UPSCALE_DEFAULT_VIEW = (1 << 0),
			UPSCALE_LEFT_VIEW    = (1 << 0),
			UPSCALE_RIGHT_VIEW   = (1 << 1),
			UPSCALE_AND_COMMIT   = (1 << 2)
		};
	}

	using namespace upscaling_flags_;

	struct upscaler
	{
		virtual ~upscaler() {}

		virtual mtl::viewable_image* scale_output(
			mtl::command_context& cmd,              // Command context (RSX thread)
			mtl::viewable_image* src,               // Source input
			id<MTLTexture> present_surface,         // Present target. May be nil for some passes (required with UPSCALE_AND_COMMIT)
			const areai& src_area,                  // Source region (VkImageBlit::srcOffsets semantics)
			const areai& dst_area,                  // Destination region (VkImageBlit::dstOffsets semantics)
			rsx::flags32_t mode                     // Mode (upscaling_flags)
		) = 0;
	};

	// Nearest filtering. Only meaningful with UPSCALE_AND_COMMIT (scaled copy into present_surface, returns nullptr);
	// otherwise the source is returned unchanged (same as VK).
	struct nearest_upscale_pass : public upscaler
	{
		mtl::viewable_image* scale_output(
			mtl::command_context& cmd,
			mtl::viewable_image* src,
			id<MTLTexture> present_surface,
			const areai& src_area,
			const areai& dst_area,
			rsx::flags32_t mode
		) override;
	};

	// Bilinear filtering. Same behavior as nearest_upscale_pass with linear filtering.
	struct bilinear_upscale_pass : public upscaler
	{
		mtl::viewable_image* scale_output(
			mtl::command_context& cmd,
			mtl::viewable_image* src,
			id<MTLTexture> present_surface,
			const areai& src_area,
			const areai& dst_area,
			rsx::flags32_t mode
		) override;
	};

	/**
	 * MetalFX spatial upscaler, selected when g_cfg.video.output_scaling == output_scaling_mode::fsr
	 * (replaces vk::fsr_upscale_pass: FSR1 EASU + RCAS).
	 * Behavior mirrors vk::fsr_upscale_pass::scale_output:
	 * - Upscales only when the input is smaller than the output on both axes (and the scaler is available,
	 *   mtl::get_caps().supports_metalfx_spatial); otherwise behaves like bilinear_upscale_pass.
	 * - mode selects the output slot (UPSCALE_LEFT_VIEW / UPSCALE_RIGHT_VIEW, never both).
	 * - Without UPSCALE_AND_COMMIT: returns the upscaled image (m_output_left / m_output_right).
	 * - With UPSCALE_AND_COMMIT: the upscaled image is copied (linear, full output rect, preserving any mirroring of the
	 *   request) into present_surface and nullptr is returned.
	 * The MTLFXSpatialScaler encodes into cmd.get() directly, so the active encoder is ended first (cmd.end_encoding()).
	 */
	class metalfx_upscale_pass : public upscaler
	{
		id<MTLFXSpatialScaler> m_scaler = nil;

		std::unique_ptr<mtl::viewable_image> m_output_left;
		std::unique_ptr<mtl::viewable_image> m_output_right;

		// Configuration the scaler was created for. The scaler is recreated whenever any of these change.
		u32 m_input_width = 0;
		u32 m_input_height = 0;
		MTLPixelFormat m_input_format = MTLPixelFormatInvalid;
		u32 m_output_width = 0;
		u32 m_output_height = 0;
		MTLPixelFormat m_output_format = MTLPixelFormatInvalid;

		void dispose_images();

		// Output storage: private 2D texture with RenderTarget | ShaderRead | ShaderWrite usage.
		// The format is BGRA8Unorm for BGRA inputs, RGBA8Unorm otherwise (the MetalFX output format must be writable).
		void initialize_image(u32 output_w, u32 output_h, MTLPixelFormat format, rsx::flags32_t mode);

		// (Re)creates m_scaler for the given configuration (colorProcessingMode = perceptual). Returns false if
		// MetalFX is unavailable or the configuration is unsupported.
		bool initialize_scaler(u32 input_w, u32 input_h, MTLPixelFormat input_format, u32 output_w, u32 output_h, MTLPixelFormat output_format);

	public:
		metalfx_upscale_pass() = default;
		~metalfx_upscale_pass() override;

		mtl::viewable_image* scale_output(
			mtl::command_context& cmd,
			mtl::viewable_image* src,
			id<MTLTexture> present_surface,
			const areai& src_area,
			const areai& dst_area,
			rsx::flags32_t mode
		) override;
	};
}
