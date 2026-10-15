#include "stdafx.h"
#include "MTLResolveHelper.h"

#include "Utilities/StrUtil.h"

#include <initializer_list>

namespace
{
	// Shared MSAA snippets are written for VK descriptor set 0 + push constants. Move the textures to set=1 and the
	// push constant block to the std140 parameter block at set=0, binding=0 (guide section 5).
	std::string patch_msaa_snippet(std::string src)
	{
		const std::initializer_list<std::pair<std::string_view, std::string_view>> replacements =
		{
			{ "layout(set=0, binding=", "layout(set=1, binding=" },
			{ "layout(push_constant) uniform static_data", "layout(std140, set=0, binding=0) uniform static_data" }
		};

		for (const auto& [from, to] : replacements)
		{
			if (src.find(from) == std::string::npos)
			{
				fmt::throw_exception("Metal resolve: shader patch target '%s' not found", std::string(from));
			}

			src = fmt::replace_all(src, from, to);
		}

		return src;
	}
}

namespace mtl
{
	namespace
	{
		std::unique_ptr<mtl::coloronly_resolve> g_color_resolver;
		std::unique_ptr<mtl::coloronly_unresolve> g_color_unresolver;
		std::unique_ptr<mtl::depthonly_resolve> g_depth_resolver;
		std::unique_ptr<mtl::depthonly_unresolve> g_depth_unresolver;
		std::unique_ptr<mtl::depthstencil_resolve_EXT> g_depthstencil_resolver;
		std::unique_ptr<mtl::depthstencil_unresolve_EXT> g_depthstencil_unresolver;

		template <typename T, typename ...Args>
		void initialize_pass(std::unique_ptr<T>& ptr, const mtl::render_device& dev, Args&&... extras)
		{
			if (!ptr)
			{
				ptr = std::make_unique<T>(std::forward<Args>(extras)...);
				ptr->create(dev);
			}
		}
	}

	// --------------------------------------------------------------------------------------------
	// depth_resolve_base
	// --------------------------------------------------------------------------------------------

	depth_resolve_base::depth_resolve_base()
	{
		renderpass_config.set_depth_mask(true);
		renderpass_config.enable_depth_test(MTLCompareFunctionAlways);
		renderpass_config.set_attachment_count(0);

		// Depth-stencil buffers are almost never filterable, and we do not need it here (1:1 mapping)
		m_sampler_filter = MTLSamplerMinMagFilterNearest;
	}

	void depth_resolve_base::build(bool resolve_depth, bool resolve_stencil, bool is_unresolve)
	{
		vs_src =
			#include "Emu/RSX/Program/GLSLSnippets/GenericVSPassthrough.glsl"
			;

		static const char* depth_resolver =
			#include "Emu/RSX/Program/MSAA/DepthResolvePass.glsl"
			;

		static const char* depth_unresolver =
			#include "Emu/RSX/Program/MSAA/DepthUnresolvePass.glsl"
			;

		static const char* depth_stencil_resolver =
			#include "Emu/RSX/Program/MSAA/DepthStencilResolvePass.glsl"
			;

		static const char* depth_stencil_unresolver =
			#include "Emu/RSX/Program/MSAA/DepthStencilUnresolvePass.glsl"
			;

		// Fragment equivalents of ColorResolvePass.glsl / ColorUnresolvePass.glsl (same sample index mapping)
		static const char* color_resolver =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"\n"
			"layout(set=1, binding=0) uniform sampler2DMS fs0;\n"
			"layout(std140, set=0, binding=0) uniform static_data { ivec2 sample_count; };\n"
			"layout(location=0) out vec4 ocol;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	ivec2 out_coord = ivec2(gl_FragCoord.xy);\n"
			"	ivec2 in_coord = (out_coord / sample_count.xy);\n"
			"	ivec2 sample_loc = out_coord % sample_count.xy;\n"
			"	int sample_index = sample_loc.x + (sample_loc.y * sample_count.y);\n"
			"	ocol = texelFetch(fs0, in_coord, sample_index);\n"
			"}\n";

		static const char* color_unresolver =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"\n"
			"layout(set=1, binding=0) uniform sampler2D fs0;\n"
			"layout(std140, set=0, binding=0) uniform static_data { ivec2 sample_count; };\n"
			"layout(location=0) out vec4 ocol;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	ivec2 pixel_coord = ivec2(gl_FragCoord.xy);\n"
			"	pixel_coord *= sample_count.xy;\n"
			"	pixel_coord.x += (gl_SampleID % sample_count.x);\n"
			"	pixel_coord.y += (gl_SampleID / sample_count.x);\n"
			"	ocol = texelFetch(fs0, pixel_coord, 0);\n"
			"}\n";

		if (resolve_depth && resolve_stencil)
		{
			fs_src = patch_msaa_snippet(is_unresolve ? depth_stencil_unresolver : depth_stencil_resolver);
		}
		else if (resolve_depth)
		{
			fs_src = patch_msaa_snippet(is_unresolve ? depth_unresolver : depth_resolver);
		}
		else if (resolve_stencil)
		{
			// Stencil-only passes only exist as a fallback for hosts without stencil export
			fmt::throw_exception("Metal resolve: stencil-only resolve passes are not supported (stencil export is always available)");
		}
		else
		{
			fs_src = is_unresolve ? color_unresolver : color_resolver;
		}

		m_writes_color = !resolve_depth && !resolve_stencil;
		m_writes_depth = resolve_depth;
		m_writes_stencil = resolve_stencil;

		rsx_log.notice("Resolve shader:\n%s", fs_src);
	}

	void depth_resolve_base::update_uniforms(mtl::command_context& cmd, mtl::program* /*program*/)
	{
		const u32 size_to_push = static_parameters_width * sizeof(decltype(static_parameters[0]));
		ensure(size_to_push <= fragment_push_constants_size);

		// The whole block is bound so that the buffer is never smaller than the MSL struct
		[cmd.render_encoder() setFragmentBytes:static_parameters length:fragment_push_constants_size atIndex:mtl::binding::helper_params_buffer];
	}

	void depth_resolve_base::update_sample_configuration(mtl::image* msaa_image)
	{
		switch (msaa_image->samples())
		{
		case 1:
			fmt::throw_exception("MSAA input not multisampled!");
		case 2:
			samples_x = 2;
			samples_y = 1;
			break;
		case 4:
			samples_x = samples_y = 2;
			break;
		default:
			fmt::throw_exception("Unsupported sample count %d", msaa_image->samples());
		}

		static_parameters[0] = samples_x;
		static_parameters[1] = samples_y;
	}

	void depth_resolve_base::destroy()
	{
		m_temp_views.clear();
		overlay_pass::destroy();
	}

	MTLLoadAction depth_resolve_base::get_load_action(const mtl::image* target) const
	{
		const u32 target_aspect = mtl::get_aspect_flags(target->format());

		if (target_aspect & mtl::image_aspect::color)
		{
			return m_writes_color ? MTLLoadActionDontCare : MTLLoadActionLoad;
		}

		if ((target_aspect & mtl::image_aspect::depth) && !m_writes_depth)
		{
			return MTLLoadActionLoad;
		}

		if ((target_aspect & mtl::image_aspect::stencil) && !m_writes_stencil)
		{
			return MTLLoadActionLoad;
		}

		// Every texel of every aspect is overwritten
		return MTLLoadActionDontCare;
	}

	mtl::image_view* depth_resolve_base::get_color_view(mtl::viewable_image* img, u32 encoding)
	{
		auto view = img->get_view(rsx::default_remap_vector.with_encoding(encoding), mtl::image_aspect::color);
		ensure(view && view->value);
		return view;
	}

	mtl::image_view* depth_resolve_base::get_depth_view(mtl::viewable_image* img, u32 encoding)
	{
		auto view = img->get_view(rsx::default_remap_vector.with_encoding(encoding), mtl::image_aspect::depth);
		ensure(view && view->value);
		return view;
	}

	mtl::image_view* depth_resolve_base::get_stencil_view(mtl::viewable_image* img, u32 encoding)
	{
		const MTLPixelFormat stencil_format = mtl::get_stencil_view_format(img->format());
		ensure(stencil_format != MTLPixelFormatInvalid);

		auto view = img->get_view(rsx::default_remap_vector.with_encoding(encoding), mtl::image_aspect::stencil);
		if (view && view->value && view->format() == stencil_format)
		{
			return view;
		}

		// Sampling the stencil aspect requires a stencil view format (X32_Stencil8). Build one locally.
		// The image is either multisampled (msaa input) or 2D (resolve input), the view keeps the image's texture type.
		m_temp_views.push_back(std::make_unique<mtl::image_view>(img, stencil_format, img->type(),
			mtl::default_component_map, mtl::image_aspect::stencil, 0, 1, 0, 1));

		return m_temp_views.back().get();
	}

	void depth_resolve_base::run_resolve_pass(mtl::command_context& cmd, mtl::viewable_image* target, const std::vector<mtl::image_view*>& src)
	{
		overlay_pass::run(
			cmd,
			{ 0, 0, target->width(), target->height() },
			target, src,
			get_load_action(target));

		// The command buffer retains the textures, the view objects themselves are no longer needed
		m_temp_views.clear();
	}

	// --------------------------------------------------------------------------------------------
	// Color
	// --------------------------------------------------------------------------------------------

	coloronly_resolve::coloronly_resolve()
	{
		renderpass_config.disable_depth_test();
		renderpass_config.set_depth_mask(false);
		renderpass_config.set_attachment_count(1);
		renderpass_config.set_color_mask(0, true, true, true, true);

		build(false, false, false);
	}

	void coloronly_resolve::run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
	{
		ensure(msaa_image->samples() > 1);
		ensure(resolve_image->samples() == 1);

		update_sample_configuration(msaa_image);
		auto src_view = get_color_view(msaa_image, MTL_REMAP_VIEW_MULTISAMPLED);

		run_resolve_pass(cmd, resolve_image, { src_view });
	}

	coloronly_unresolve::coloronly_unresolve()
	{
		renderpass_config.disable_depth_test();
		renderpass_config.set_depth_mask(false);
		renderpass_config.set_attachment_count(1);
		renderpass_config.set_color_mask(0, true, true, true, true);

		build(false, false, true);
	}

	void coloronly_unresolve::run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
	{
		ensure(msaa_image->samples() > 1);
		ensure(resolve_image->samples() == 1);

		// Per-sample shading is implied by gl_SampleID, the sample count comes from the target
		update_sample_configuration(msaa_image);
		auto src_view = get_color_view(resolve_image, MTL_REMAP_IDENTITY);

		run_resolve_pass(cmd, msaa_image, { src_view });
	}

	// --------------------------------------------------------------------------------------------
	// Depth
	// --------------------------------------------------------------------------------------------

	depthonly_resolve::depthonly_resolve()
	{
		build(true, false, false);
	}

	void depthonly_resolve::run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
	{
		update_sample_configuration(msaa_image);
		auto src_view = get_depth_view(msaa_image, MTL_REMAP_VIEW_MULTISAMPLED);

		run_resolve_pass(cmd, resolve_image, { src_view });
	}

	depthonly_unresolve::depthonly_unresolve()
	{
		build(true, false, true);
	}

	void depthonly_unresolve::run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
	{
		update_sample_configuration(msaa_image);
		auto src_view = get_depth_view(resolve_image, MTL_REMAP_IDENTITY);

		run_resolve_pass(cmd, msaa_image, { src_view });
	}

	// --------------------------------------------------------------------------------------------
	// Depth + stencil (stencil export)
	// --------------------------------------------------------------------------------------------

	depthstencil_resolve_EXT::depthstencil_resolve_EXT()
	{
		renderpass_config.enable_stencil_test(
			MTLStencilOperationReplace, MTLStencilOperationReplace, MTLStencilOperationReplace, // Always replace
			MTLCompareFunctionAlways,                                                            // Always pass
			0xFF,                                                                                // Full write-through
			0);                                                                                  // Unused

		renderpass_config.set_stencil_mask(0xFF);
		m_num_usable_samplers = 2;

		build(true, true, false);
	}

	void depthstencil_resolve_EXT::run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
	{
		update_sample_configuration(msaa_image);
		auto depth_view = get_depth_view(msaa_image, MTL_REMAP_VIEW_MULTISAMPLED);
		auto stencil_view = get_stencil_view(msaa_image, MTL_REMAP_VIEW_MULTISAMPLED);

		run_resolve_pass(cmd, resolve_image, { depth_view, stencil_view });
	}

	depthstencil_unresolve_EXT::depthstencil_unresolve_EXT()
	{
		renderpass_config.enable_stencil_test(
			MTLStencilOperationReplace, MTLStencilOperationReplace, MTLStencilOperationReplace, // Always replace
			MTLCompareFunctionAlways,                                                            // Always pass
			0xFF,                                                                                // Full write-through
			0);                                                                                  // Unused

		renderpass_config.set_stencil_mask(0xFF);
		m_num_usable_samplers = 2;

		build(true, true, true);
	}

	void depthstencil_unresolve_EXT::run(mtl::command_context& cmd, mtl::viewable_image* msaa_image, mtl::viewable_image* resolve_image)
	{
		update_sample_configuration(msaa_image);
		auto depth_view = get_depth_view(resolve_image, MTL_REMAP_IDENTITY);
		auto stencil_view = get_stencil_view(resolve_image, MTL_REMAP_IDENTITY);

		run_resolve_pass(cmd, msaa_image, { depth_view, stencil_view });
	}

	// --------------------------------------------------------------------------------------------
	// Entry points
	// --------------------------------------------------------------------------------------------

	void resolve_image(mtl::command_context& cmd, mtl::viewable_image* dst, mtl::viewable_image* src)
	{
		ensure(dst && src);
		const auto& dev = *ensure(mtl::get_current_device());

		if (src->aspect() == mtl::image_aspect::color)
		{
			initialize_pass(g_color_resolver, dev);
			g_color_resolver->run(cmd, src, dst);
		}
		else if (src->aspect() & mtl::image_aspect::stencil)
		{
			// Metal always supports stencil export
			initialize_pass(g_depthstencil_resolver, dev);
			g_depthstencil_resolver->run(cmd, src, dst);
		}
		else
		{
			initialize_pass(g_depth_resolver, dev);
			g_depth_resolver->run(cmd, src, dst);
		}
	}

	void unresolve_image(mtl::command_context& cmd, mtl::viewable_image* dst, mtl::viewable_image* src)
	{
		ensure(dst && src);
		const auto& dev = *ensure(mtl::get_current_device());

		if (src->aspect() == mtl::image_aspect::color)
		{
			initialize_pass(g_color_unresolver, dev);
			g_color_unresolver->run(cmd, dst, src);
		}
		else if (src->aspect() & mtl::image_aspect::stencil)
		{
			initialize_pass(g_depthstencil_unresolver, dev);
			g_depthstencil_unresolver->run(cmd, dst, src);
		}
		else
		{
			initialize_pass(g_depth_unresolver, dev);
			g_depth_unresolver->run(cmd, dst, src);
		}
	}

	void clear_resolve_helpers()
	{
		auto destroy_pass = [](auto& pass)
		{
			if (pass)
			{
				pass->destroy();
				pass.reset();
			}
		};

		destroy_pass(g_color_resolver);
		destroy_pass(g_color_unresolver);
		destroy_pass(g_depth_resolver);
		destroy_pass(g_depth_unresolver);
		destroy_pass(g_depthstencil_resolver);
		destroy_pass(g_depthstencil_unresolver);
	}

	void reset_resolve_resources()
	{
		if (g_color_resolver) g_color_resolver->free_resources();
		if (g_color_unresolver) g_color_unresolver->free_resources();
		if (g_depth_resolver) g_depth_resolver->free_resources();
		if (g_depth_unresolver) g_depth_unresolver->free_resources();
		if (g_depthstencil_resolver) g_depthstencil_resolver->free_resources();
		if (g_depthstencil_unresolver) g_depthstencil_unresolver->free_resources();
	}
}
