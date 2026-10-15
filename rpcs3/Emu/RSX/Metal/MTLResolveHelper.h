#pragma once

#include "MTLOverlays.h"

#include "Emu/RSX/Metal/mtlutils/image.h"

#include <memory>
#include <vector>

/**
 * MSAA resolve helpers. Port of VKResolveHelper.h.
 *
 * RPCS3 "resolve" is not an average: the planar MSAA surface (w x h, N samples) is expanded into a linear single-sample
 * surface of (w * samples_x) x (h * samples_y) texels where every sample becomes one texel, and "unresolve" is the
 * reverse operation. Metal's MultisampleResolve store action cannot be used for this.
 *
 * Differences with the VK backend:
 * - Color (un)resolve is a graphics pass instead of the cs_resolve_task / cs_unresolve_task compute kernels: MSL can not
 *   write multisampled textures from compute, and rendering also handles the packed 16-bit formats natively. A single
 *   pass object serves every format (the pipeline cache is keyed by the target format).
 * - Metal always supports stencil export ([[stencil]]), so depth/stencil surfaces always go through the
 *   depthstencil_*_EXT passes. The stencilonly_* fallbacks (bit-by-bit stencil write loops) are not needed.
 * - Unresolve passes use gl_SampleID, which makes Metal run the fragment shader per sample.
 */

namespace mtl
{
	struct depth_resolve_base : public overlay_pass
	{
		u8 samples_x = 1;
		u8 samples_y = 1;
		s32 static_parameters[4] = {};
		s32 static_parameters_width = 2;

		static constexpr usz fragment_push_constants_size = sizeof(static_parameters);

		depth_resolve_base();

		// resolve_depth == resolve_stencil == false selects the color variant
		void build(bool resolve_depth, bool resolve_stencil, bool unresolve);

		void update_uniforms(mtl::command_context& cmd, mtl::program* program) override;

		void update_sample_configuration(mtl::image* msaa_image);

		void destroy() override;

	protected:
		bool m_writes_color = false;
		bool m_writes_depth = false;
		bool m_writes_stencil = false;

		// Views created by get_stencil_view when the image view cache can not provide a stencil-only view
		std::vector<std::unique_ptr<mtl::image_view>> m_temp_views;

		// DontCare when the pass overwrites every aspect of the target, Load otherwise
		MTLLoadAction get_load_action(const mtl::image* target) const;

		mtl::image_view* get_color_view(mtl::viewable_image* img, u32 encoding);
		mtl::image_view* get_depth_view(mtl::viewable_image* img, u32 encoding);
		mtl::image_view* get_stencil_view(mtl::viewable_image* img, u32 encoding);

		// Opens a render pass on target, draws and drops the temporary views
		void run_resolve_pass(mtl::command_context& cmd, mtl::viewable_image* target, const std::vector<mtl::image_view*>& src);
	};

	// Replaces vk::cs_resolve_task: planar MSAA color -> linear color
	struct coloronly_resolve : depth_resolve_base
	{
		coloronly_resolve();

		void run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image);
	};

	// Replaces vk::cs_unresolve_task: linear color -> planar MSAA color
	struct coloronly_unresolve : depth_resolve_base
	{
		coloronly_unresolve();

		void run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image);
	};

	struct depthonly_resolve : depth_resolve_base
	{
		depthonly_resolve();

		void run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image);
	};

	struct depthonly_unresolve : depth_resolve_base
	{
		depthonly_unresolve();

		void run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image);
	};

	struct depthstencil_resolve_EXT : depth_resolve_base
	{
		depthstencil_resolve_EXT();

		void run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image);
	};

	struct depthstencil_unresolve_EXT : depth_resolve_base
	{
		depthstencil_unresolve_EXT();

		void run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image);
	};

	// resolve_image: planar MSAA src -> linear single-sample dst (dst is width*samples_x by height*samples_y).
	// unresolve_image: linear single-sample src -> planar MSAA dst.
	void resolve_image(mtl::command_context& cmd, mtl::viewable_image* dst, mtl::viewable_image* src);
	void unresolve_image(mtl::command_context& cmd, mtl::viewable_image* dst, mtl::viewable_image* src);

	// Per-frame cleanup
	void reset_resolve_resources();

	// Shutdown
	void clear_resolve_helpers();
}
