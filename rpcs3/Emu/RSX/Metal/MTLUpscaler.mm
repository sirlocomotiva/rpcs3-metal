#include "stdafx.h"
#include "MTLUpscaler.h"
#include "MTLTexture.h"
#include "mtlutils/device.h"

#import <MetalFX/MetalFX.h>

// Metal port of VK/upscalers (nearest_pass.hpp, bilinear_pass.hpp, fsr_pass.h / fsr1/fsr_pass.cpp).
// FSR1 (EASU + RCAS compute passes) is replaced by the MetalFX spatial scaler.

namespace mtl
{
	namespace
	{
		// vkCmdBlitImage replacement for committing passes: scaled (and possibly mirrored) copy into the present surface.
		void commit_scaled_copy(mtl::command_context& cmd, mtl::image* src, id<MTLTexture> present_surface,
			const areai& src_area, const areai& dst_area, bool linear_filter)
		{
			ensure(src && present_surface);

			// Wrap the present surface (usually a drawable texture). The wrapper only retains the texture; the command buffer
			// keeps it alive for as long as the GPU needs it.
			mtl::image target(present_surface, RSX_FORMAT_CLASS_COLOR);

			// coord3i(areai) keeps the signed extents (x2 - x1, y2 - y1), i.e. the VkImageBlit offsets semantics.
			mtl::copy_scaled_image(cmd, src, &target, coord3i(src_area), coord3i(dst_area), {}, false, linear_filter);
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Nearest / bilinear (VK: upscalers/nearest_pass.hpp, upscalers/bilinear_pass.hpp)
	// ---------------------------------------------------------------------------------------------------------------

	mtl::viewable_image* nearest_upscale_pass::scale_output(
		mtl::command_context& cmd,
		mtl::viewable_image* src,
		id<MTLTexture> present_surface,
		const areai& src_area,
		const areai& dst_area,
		rsx::flags32_t mode)
	{
		if (mode & UPSCALE_AND_COMMIT)
		{
			commit_scaled_copy(cmd, src, present_surface, src_area, dst_area, false);
			return nullptr;
		}

		// Upscaling source only is unsupported
		return src;
	}

	mtl::viewable_image* bilinear_upscale_pass::scale_output(
		mtl::command_context& cmd,
		mtl::viewable_image* src,
		id<MTLTexture> present_surface,
		const areai& src_area,
		const areai& dst_area,
		rsx::flags32_t mode)
	{
		if (mode & UPSCALE_AND_COMMIT)
		{
			commit_scaled_copy(cmd, src, present_surface, src_area, dst_area, true);
			return nullptr;
		}

		// Upscaling source only is unsupported
		return src;
	}

	// ---------------------------------------------------------------------------------------------------------------
	// MetalFX spatial (VK: upscalers/fsr_pass.h, fsr1/fsr_pass.cpp)
	// ---------------------------------------------------------------------------------------------------------------

	metalfx_upscale_pass::~metalfx_upscale_pass()
	{
		dispose_images();
		m_scaler = nil;
	}

	void metalfx_upscale_pass::dispose_images()
	{
		// Command buffers retain the textures they reference, so the images can be released right away
		m_output_left.reset();
		m_output_right.reset();
	}

	void metalfx_upscale_pass::initialize_image(u32 output_w, u32 output_h, MTLPixelFormat format, rsx::flags32_t mode)
	{
		// Unlike VK, only the requested view is (re)created. VK disposes both views through the deferred resource manager,
		// which keeps a pointer returned earlier in the frame (left view) valid; here the image object would be freed at once.
		const bool left_view = !!(mode & UPSCALE_LEFT_VIEW);
		auto& output = left_view ? m_output_left : m_output_right;

		MTLTextureUsage usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
		if (m_scaler)
		{
			usage |= [m_scaler outputTextureUsage];
		}

		// Private storage is a MetalFX requirement for the output texture
		output = std::make_unique<mtl::viewable_image>(
			MTLTextureType2D,
			format,
			output_w, output_h, 1, 1, 1,
			u8{1},
			usage,
			RSX_FORMAT_CLASS_COLOR,
			MTLStorageModePrivate);

		output->set_debug_name(left_view ? "MetalFX output (left)" : "MetalFX output (right)");
	}

	bool metalfx_upscale_pass::initialize_scaler(u32 input_w, u32 input_h, MTLPixelFormat input_format, u32 output_w, u32 output_h, MTLPixelFormat output_format)
	{
		if (m_input_width == input_w && m_input_height == input_h && m_input_format == input_format &&
			m_output_width == output_w && m_output_height == output_h && m_output_format == output_format)
		{
			// Same configuration. If the scaler could not be created for it, do not retry every frame.
			return m_scaler != nil;
		}

		// The output images are kept: images handed out earlier in the frame must stay valid. Their usage is validated
		// against the new scaler's requirements in scale_output().
		m_scaler = nil;

		m_input_width = input_w;
		m_input_height = input_h;
		m_input_format = input_format;
		m_output_width = output_w;
		m_output_height = output_h;
		m_output_format = output_format;

		if (!mtl::get_caps().supports_metalfx_spatial)
		{
			return false;
		}

		@autoreleasepool
		{
			id<MTLDevice> device = mtl::get_device();
			if (![MTLFXSpatialScalerDescriptor supportsDevice:device])
			{
				return false;
			}

			MTLFXSpatialScalerDescriptor* desc = [MTLFXSpatialScalerDescriptor new];
			desc.colorTextureFormat = input_format;
			desc.outputTextureFormat = output_format;
			desc.inputWidth = input_w;
			desc.inputHeight = input_h;
			desc.outputWidth = output_w;
			desc.outputHeight = output_h;
			// RSX output is display referred (sRGB encoded values in UNORM storage)
			desc.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;

			m_scaler = [desc newSpatialScalerWithDevice:device];
		}

		if (!m_scaler)
		{
			rsx_log.error("MetalFX: Failed to create a spatial scaler for %ux%u (format %u) -> %ux%u (format %u). Falling back to bilinear scaling.",
				input_w, input_h, static_cast<u32>(input_format), output_w, output_h, static_cast<u32>(output_format));
			return false;
		}

		rsx_log.notice("MetalFX: Spatial scaler initialized for %ux%u -> %ux%u", input_w, input_h, output_w, output_h);
		return true;
	}

	mtl::viewable_image* metalfx_upscale_pass::scale_output(
		mtl::command_context& cmd,
		mtl::viewable_image* src,
		id<MTLTexture> present_surface,
		const areai& src_area,
		const areai& dst_area,
		rsx::flags32_t mode)
	{
		size2u input_size, output_size;
		input_size.width = std::abs(src_area.x2 - src_area.x1);
		input_size.height = std::abs(src_area.y2 - src_area.y1);
		output_size.width = std::abs(dst_area.x2 - dst_area.x1);
		output_size.height = std::abs(dst_area.y2 - dst_area.y1);

		mtl::viewable_image* src_image = src;
		areai output_src_area = src_area;

		// The MetalFX input region always starts at the texture origin (inputContentWidth/Height); requests that sample
		// from an offset are not upscaled (VK's EASU pass makes the same assumption implicitly).
		const bool src_at_origin = (std::min(src_area.x1, src_area.x2) == 0 && std::min(src_area.y1, src_area.y2) == 0);

		if (src &&
			input_size.width < output_size.width && input_size.height < output_size.height &&
			src_at_origin && input_size.width <= src->width() && input_size.height <= src->height())
		{
			// Cannot upscale both LEFT and RIGHT images at the same time.
			// Default maps to LEFT for simplicity
			ensure((mode & (UPSCALE_LEFT_VIEW | UPSCALE_RIGHT_VIEW)) != (UPSCALE_LEFT_VIEW | UPSCALE_RIGHT_VIEW));

			// The output format must be writable by the scaler. Keep the channel order of 8-bit sources.
			const MTLPixelFormat output_format = (src->format() == MTLPixelFormatBGRA8Unorm)
				? MTLPixelFormatBGRA8Unorm
				: MTLPixelFormatRGBA8Unorm;

			// MetalFX requires the input texture to match the dimensions the scaler was created for
			const bool scaler_ok = initialize_scaler(src->width(), src->height(), src->format(), output_size.width, output_size.height, output_format) &&
				([src->value usage] & [m_scaler colorTextureUsage]) == [m_scaler colorTextureUsage];

			if (scaler_ok)
			{
				const MTLTextureUsage required_output_usage = [m_scaler outputTextureUsage];

				auto& m_output_data = (mode & UPSCALE_LEFT_VIEW) ? m_output_left : m_output_right;
				if (!m_output_data ||
					m_output_data->width() != output_size.width ||
					m_output_data->height() != output_size.height ||
					m_output_data->format() != output_format ||
					([m_output_data->value usage] & required_output_usage) != required_output_usage)
				{
					initialize_image(output_size.width, output_size.height, output_format, mode);
				}

				if (m_output_data)
				{
					// The scaler encodes its own passes into the command buffer
					cmd.end_encoding();

					m_scaler.colorTexture = src->value;
					m_scaler.inputContentWidth = input_size.width;
					m_scaler.inputContentHeight = input_size.height;
					m_scaler.outputTexture = m_output_data->value;

					[m_scaler encodeToCommandBuffer:cmd.get()];

					// Do not keep the textures alive through the scaler; the command buffer retains them
					m_scaler.colorTexture = nil;
					m_scaler.outputTexture = nil;

					// Swap input for the upscaled target
					src_image = m_output_data.get();

					// Update output parameters to match expected output
					if (mode & UPSCALE_AND_COMMIT)
					{
						output_src_area.x1 = 0;
						output_src_area.x2 = static_cast<s32>(output_size.width);
						output_src_area.y1 = 0;
						output_src_area.y2 = static_cast<s32>(output_size.height);

						// Preserve mirroring/flipping
						if (src_area.x1 > src_area.x2)
						{
							std::swap(output_src_area.x1, output_src_area.x2);
						}

						if (src_area.y1 > src_area.y2)
						{
							std::swap(output_src_area.y1, output_src_area.y2);
						}
					}
				}
			}
		}

		if (mode & UPSCALE_AND_COMMIT)
		{
			commit_scaled_copy(cmd, src_image, present_surface, output_src_area, dst_area, true);
			return nullptr;
		}

		return src_image;
	}
}
