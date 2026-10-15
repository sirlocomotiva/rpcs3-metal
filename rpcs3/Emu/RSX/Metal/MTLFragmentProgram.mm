#include "stdafx.h"
#include "MTLFragmentProgram.h"

#include "MTLCommonDecompiler.h"
#include "MTLHelpers.h"
#include "Emu/RSX/Metal/mtlutils/device.h"
#include "Emu/RSX/Program/GLSLCommon.h"
#include "Emu/system_config.h"
#include "Utilities/File.h"

std::string MTLFragmentDecompilerThread::getFloatTypeName(usz elementCount)
{
	return glsl::getFloatTypeNameImpl(elementCount);
}

std::string MTLFragmentDecompilerThread::getHalfTypeName(usz elementCount)
{
	return glsl::getHalfTypeNameImpl(elementCount);
}

std::string MTLFragmentDecompilerThread::getFunction(FUNCTION f)
{
	return glsl::getFunctionImpl(f);
}

std::string MTLFragmentDecompilerThread::compareFunction(COMPARE f, std::string_view Op0, std::string_view Op1)
{
	return glsl::compareFunctionImpl(f, Op0, Op1);
}

void MTLFragmentDecompilerThread::insertHeader(std::stringstream & OS)
{
	std::vector<const char*> required_extensions;

	if (device_props.has_native_half_support)
	{
		required_extensions.emplace_back("GL_EXT_shader_explicit_arithmetic_types_float16");
	}

	if (properties.multisampled_sampler_mask)
	{
		required_extensions.emplace_back("GL_ARB_shader_texture_image_samples");
	}

	if (m_prog.ctrl & RSX_SHADER_CONTROL_ATTRIBUTE_INTERPOLATION)
	{
		required_extensions.emplace_back("GL_EXT_fragment_shader_barycentric");
	}

	OS << "#version 450\n";
	for (const auto ext : required_extensions)
	{
		OS << "#extension " << ext << ": require\n";
	}

	OS << "#extension GL_ARB_separate_shader_objects: enable\n\n";

	mtl::insert_rsx_program_preamble(OS, glsl::glsl_fragment_program);

	glsl::insert_subheader_block(OS);
}

void MTLFragmentDecompilerThread::insertInputs(std::stringstream & OS)
{
	glsl::insert_fragment_shader_inputs_block(
		OS,
		glsl::extension_flavour::EXT,
		m_prog,
		m_parr.params[PF_PARAM_IN],
		{
			.two_sided_color = !!(properties.in_register_mask & in_diff_color),
			.two_sided_specular = !!(properties.in_register_mask & in_spec_color)
		},
		mtl::get_varying_register_location
	);

	if (m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_RANGE)
	{
		OS << "layout(location=" << mtl::get_varying_register_location("depth_range") << ") in flat vec2 depth_range;\n\n";
	}
}

void MTLFragmentDecompilerThread::insertOutputs(std::stringstream & OS)
{
	const std::pair<std::string, std::string> table[] =
	{
		{ "ocol0", m_prog.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS ? "r0" : "h0" },
		{ "ocol1", m_prog.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS ? "r2" : "h4" },
		{ "ocol2", m_prog.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS ? "r3" : "h6" },
		{ "ocol3", m_prog.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS ? "r4" : "h8" },
	};

	// NOTE: We do not skip outputs, the only possible combinations are a(0), b(0), ab(0,1), abc(0,1,2), abcd(0,1,2,3)
	u8 output_index = 0;
	const bool float_type = (m_prog.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS) || !device_props.has_native_half_support;
	const auto reg_type = float_type ? "vec4" : getHalfTypeName(4);
	for (uint i = 0; i < std::size(table); ++i)
	{
		if (!m_parr.HasParam(PF_PARAM_NONE, reg_type, table[i].second))
		{
			continue;
		}

		if (i >= m_prog.mrt_buffers_count)
		{
			// Dead writes. Declare as temp variables for DCE to clean up.
			OS << "vec4 " << table[i].first << "; // Unused\n";
			mtl_prog->output_color_masks[i] = 0;
			continue;
		}

		OS << "layout(location=" << std::to_string(output_index++) << ") " << "out vec4 " << table[i].first << ";\n";
		mtl_prog->output_color_masks[i] = -1;
	}
}

void MTLFragmentDecompilerThread::insertConstants(std::stringstream & OS)
{
	// Fixed inputs from shader decompilation process
	for (const ParamType& PT : m_parr.params[PF_PARAM_UNIFORM])
	{
		if (!PT.type.starts_with("sampler"))
		{
			continue;
		}

		for (const ParamItem& PI : PT.items)
		{
			std::string samplerType = PT.type;

			const int index = mtl::get_texture_index(PI.name);
			const auto mask = (1 << index);

			ensure(index >= 0 && index < 16, "Unexpected fragment texture index");

			if (properties.multisampled_sampler_mask & mask)
			{
				if (samplerType != "sampler1D" && samplerType != "sampler2D")
				{
					rsx_log.error("Unexpected multisampled image type '%s'", samplerType);
				}

				samplerType = "sampler2DMS";
			}
			else if (properties.shadow_sampler_mask & mask)
			{
				if (properties.common_access_sampler_mask & mask)
				{
					rsx_log.error("Texture unit %d is sampled as both a shadow texture and a depth texture", index);
				}
				else
				{
					samplerType += "Shadow";
				}
			}

			OS << "layout(set=" << mtl::descriptor_set::textures << ", binding=" << (mtl::binding::fs_texture0 + index) << ") uniform " << samplerType << " " << PI.name << ";\n";
			mtl_prog->referenced_textures_mask |= mask;

			if (properties.redirected_sampler_mask & mask)
			{
				// Insert stencil mirror declaration
				OS << "layout(set=" << mtl::descriptor_set::textures << ", binding=" << (mtl::binding::fs_stencil_mirror0 + index) << ") uniform u" << samplerType << " " << PI.name << "_stencil;\n";
				mtl_prog->stencil_mirror_mask |= mask;
			}
		}
	}

	if (m_prog.ctrl & (RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE | RSX_SHADER_CONTROL_EMULATE_DEPTH_BOUNDS))
	{
		const auto frag_depth_type = (m_prog.ctrl & RSX_SHADER_CONTROL_ROP_MULTISAMPLED)
			? "sampler2DMS"
			: "sampler2D";

		OS << "layout(set=" << mtl::descriptor_set::textures << ", binding=" << mtl::binding::fs_depth_input << ") uniform " << frag_depth_type << " frag_depth;\n";
		mtl_prog->uses_depth_input = true;
	}

	if ((m_prog.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING) && m_prog.mrt_buffers_count)
	{
		// Color attachments are read with framebuffer fetch ([[color(n)]]).
		// Metal framebuffer fetch always returns the sample being shaded, so multisampled targets do not need subpassInputMS.
		for (u32 i = 0; i < m_prog.mrt_buffers_count; ++i)
		{
			OS << "layout(input_attachment_index=" << i << ", set=" << mtl::descriptor_set::input_attachments << ", binding=" << i << ") uniform subpassInput frag_src_" << i << ";\n";
		}

		if (m_prog.ctrl & RSX_SHADER_CONTROL_ROP_MULTISAMPLED)
		{
			// The ROP prologue loads the attachment at gl_SampleID when the output is multisampled.
			// This is implicit with framebuffer fetch, drop the sample index.
			OS << "#define subpassLoad(input, sample_index) subpassLoad(input)\n";
		}

		mtl_prog->uses_framebuffer_fetch = true;
	}

	OS << "\n";

	if (!properties.constant_offsets.empty())
	{
		OS <<
			"layout(std140, set=" << mtl::descriptor_set::buffers << ", binding=" << mtl::binding::fs_constants_buffer << ") uniform FragmentConstantsBuffer\n"
			"{\n"
			"	vec4 fc[" << properties.constant_offsets.size() << "];\n"
			"};\n"
			"#define _fetch_constant(x) fc[x]\n\n";
	}

	// Same layouts as the OpenGL backend (filled by the renderer exactly like GLGSRender::load_program_env)
	OS <<
		"layout(std140, set=" << mtl::descriptor_set::buffers << ", binding=" << mtl::binding::fs_state_buffer << ") uniform FragmentStateBuffer\n"
		"{\n"
		"	float fog_param0;\n"
		"	float fog_param1;\n"
		"	uint rop_control;\n"
		"	float alpha_ref;\n"
		"	uint fog_mode;\n"
		"	float wpos_scale;\n"
		"	vec2 wpos_bias;\n"
		"	vec2 depth_bounds;\n"   // Metal only: emulated depth bounds test (RSX_SHADER_CONTROL_EMULATE_DEPTH_BOUNDS)
		"};\n\n"

		"layout(std140, set=" << mtl::descriptor_set::buffers << ", binding=" << mtl::binding::fs_texture_params_buffer << ") uniform TextureParametersBuffer\n"
		"{\n"
		"	sampler_info texture_parameters[16];\n"
		"};\n\n"

		"layout(std140, set=" << mtl::descriptor_set::buffers << ", binding=" << mtl::binding::fs_stipple_buffer << ") uniform RasterizerHeap\n"
		"{\n"
		"	uvec4 stipple_pattern[8];\n"
		"};\n\n";

	if (m_prog.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING)
	{
		OS <<
			"layout(std140, set=" << mtl::descriptor_set::buffers << ", binding=" << mtl::binding::fs_blend_params_buffer << ") uniform BlendParameters\n"
			"{\n"
			"	uint blend_eqn;\n"
			"	uint blend_sfactors;\n"
			"	uint blend_dfactors;\n"
			"	vec4 blend_constants;\n"
			"};\n\n";
	}
}

void MTLFragmentDecompilerThread::insertGlobalFunctions(std::stringstream &OS)
{
	m_shader_props.domain = glsl::glsl_fragment_program;
	m_shader_props.require_lit_emulation = properties.has_lit_op;
	m_shader_props.fp32_outputs = !!(m_prog.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS);
	m_shader_props.require_depth_conversion = properties.redirected_sampler_mask != 0;
	m_shader_props.require_wpos = !!(properties.in_register_mask & in_wpos);
	m_shader_props.require_texture_ops = properties.has_tex_op;
	m_shader_props.require_tex_shadow_ops = properties.shadow_sampler_mask != 0;
	m_shader_props.require_msaa_ops = m_prog.texture_state.multisampled_textures != 0;
	m_shader_props.require_texture_expand = properties.has_exp_tex_op;
	m_shader_props.require_srgb_to_linear = properties.has_upg;
	m_shader_props.require_linear_to_srgb = properties.has_pkg;
	m_shader_props.require_fog_read = properties.in_register_mask & in_fogc;
	m_shader_props.emulate_shadow_compare = device_props.emulate_depth_compare || device_props.emulated_depth_storage;

	m_shader_props.low_precision_tests = device_props.has_low_precision_rounding && !(m_prog.ctrl & RSX_SHADER_CONTROL_ATTRIBUTE_INTERPOLATION);
	m_shader_props.disable_early_discard = true;
	m_shader_props.supports_native_fp16 = device_props.has_native_half_support;

	m_shader_props.ROP_output_rounding = (g_cfg.video.shader_precision != gpu_preset_level::low) && !!(m_prog.ctrl & RSX_SHADER_CONTROL_8BIT_FRAMEBUFFER);
	m_shader_props.ROP_sRGB_packing = !!(m_prog.ctrl & RSX_SHADER_CONTROL_SRGB_FRAMEBUFFER);
	m_shader_props.ROP_alpha_test = !!(m_prog.ctrl & RSX_SHADER_CONTROL_ALPHA_TEST);
	m_shader_props.ROP_alpha_to_coverage_test = !!(m_prog.ctrl & RSX_SHADER_CONTROL_ALPHA_TO_COVERAGE);
	m_shader_props.ROP_polygon_stipple_test = !!(m_prog.ctrl & RSX_SHADER_CONTROL_POLYGON_STIPPLE);
	m_shader_props.ROP_discard = !!(m_prog.ctrl & RSX_SHADER_CONTROL_USES_KIL);
	m_shader_props.ROP_channel_remap = !!(m_prog.ctrl & RSX_SHADER_CONTROL_ROP_OUTPUT_REMAP);
	m_shader_props.ROP_programmable_blend = !!(m_prog.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING);

	m_shader_props.require_tex1D_ops = properties.has_tex1D;
	m_shader_props.require_tex2D_ops = properties.has_tex2D;
	m_shader_props.require_tex3D_ops = properties.has_tex3D;
	m_shader_props.require_shadowProj_ops = properties.shadow_sampler_mask != 0 && properties.has_texShadowProj;
	m_shader_props.require_alpha_kill = !!(m_prog.ctrl & RSX_SHADER_CONTROL_TEXTURE_ALPHA_KILL);
	m_shader_props.require_color_format_convert = !!(m_prog.ctrl & RSX_SHADER_CONTROL_TEXTURE_FORMAT_CONVERT);
	m_shader_props.emulate_depth_compare = !!(m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE);
	m_shader_props.ROP_output_multisampled = !!(m_prog.ctrl & RSX_SHADER_CONTROL_ROP_MULTISAMPLED);
	m_shader_props.ROP_emulate_depth_range = !!(m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_RANGE);
	m_shader_props.ROP_depth_export = !!(m_prog.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT) && m_shader_props.ROP_emulate_depth_range;
	m_shader_props.emulated_depth_storage = device_props.emulated_depth_storage;

	// Texture parameters are indexed directly (no per-draw base offset)
	OS <<
		"#define texture_base_index 0\n"
		"#define TEX_PARAM(index) texture_parameters[index]\n\n";

	glsl::insert_glsl_legacy_function(OS, m_shader_props);
}

void MTLFragmentDecompilerThread::insertMainStart(std::stringstream & OS)
{
	std::set<std::string> output_registers;
	if (m_prog.ctrl & CELL_GCM_SHADER_CONTROL_32_BITS_EXPORTS)
	{
		output_registers = { "r0", "r2", "r3", "r4" };
	}
	else
	{
		output_registers = { "h0", "h4", "h6", "h8" };
	}

	if (m_prog.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT)
	{
		output_registers.insert("r1");
	}

	std::string registers;
	std::string reg_type;
	const auto half4 = getHalfTypeName(4);
	for (auto &reg_name : output_registers)
	{
		const auto type = (reg_name[0] == 'r' || !device_props.has_native_half_support)? "vec4" : half4;
		if (reg_type == type) [[likely]]
		{
			registers += ", " + reg_name + " = " + type + "(0.)";
		}
		else
		{
			if (!registers.empty())
				registers += ";\n";

			registers += type + " " + reg_name + " = " + type + "(0.)";
		}

		reg_type = type;
	}

	if (!registers.empty())
	{
		OS << registers << ";\n";
	}

	OS << "void fs_main()\n";
	OS << "{\n";

	for (const ParamType& PT : m_parr.params[PF_PARAM_NONE])
	{
		for (const ParamItem& PI : PT.items)
		{
			if (output_registers.find(PI.name) != output_registers.end())
				continue;

			OS << "	" << PT.type << " " << PI.name;
			if (!PI.value.empty())
				OS << " = " << PI.value;

			OS << ";\n";
		}
	}

	if (properties.has_w_access)
		OS << "	float in_w = (1. / gl_FragCoord.w);\n";

	if (properties.in_register_mask & in_ssa)
		OS << "	vec4 ssa = gl_FrontFacing ? vec4(1.) : vec4(-1.);\n";

	if (properties.in_register_mask & in_wpos)
		OS << "	vec4 wpos = get_wpos();\n";

	if (properties.in_register_mask & in_fogc)
		OS << "	vec4 fogc = fetch_fog_value(fog_mode);\n";

	if (m_prog.two_sided_lighting)
	{
		if (properties.in_register_mask & in_diff_color)
			OS << "	vec4 diff_color = gl_FrontFacing ? diff_color0 : diff_color1;\n";

		if (properties.in_register_mask & in_spec_color)
			OS << "	vec4 spec_color = gl_FrontFacing ? spec_color0 : spec_color1;\n";
	}
}

void MTLFragmentDecompilerThread::insertMainEnd(std::stringstream & OS)
{
	OS << "}\n\n";

	OS << "void main()\n";
	OS << "{\n";

	if (m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_BOUNDS)
	{
		// Metal has no depth bounds test. The stored depth uses the same encoding as the bounds (see load_program_env).
		const auto sample_index = (m_prog.ctrl & RSX_SHADER_CONTROL_ROP_MULTISAMPLED) ? "gl_SampleID" : "0";
		OS <<
			"	// Depth bounds test\n"
			"	{\n"
			"		const float stored_depth = texelFetch(frag_depth, ivec2(gl_FragCoord.xy), " << sample_index << ").r;\n"
			"		if (stored_depth < depth_bounds.x || stored_depth > depth_bounds.y)\n"
			"		{\n"
			"			discard;\n"
			"		}\n"
			"	}\n\n";
	}

	// NOTE: rop_control and alpha_ref are read directly from FragmentStateBuffer

	::glsl::insert_rop_init(OS, m_prog.mrt_buffers_count);

	OS << "\n" << "	fs_main();\n\n";

	if (m_prog.ctrl & RSX_SHADER_CONTROL_DISABLE_EARLY_Z)
	{
		// This is effectively unreachable code, but good enough to trick the GPU to skip early Z
		// For vulkan, depth export has stronger semantics than discard.
		// Metal rejects pipelines whose fragment function writes depth without a depth attachment. Depth writes are
		// compiled out of a second variant of the program (MTL_NO_DEPTH_EXPORT) that is linked in that case.
		OS <<
			"	// Insert pseudo-barrier sequence to disable early-Z\n"
			"#ifndef MTL_NO_DEPTH_EXPORT\n"
			"	gl_FragDepth = gl_FragCoord.z;\n"
			"#endif\n\n";

		mtl_prog->writes_depth = true;
	}

	glsl::insert_rop(OS, m_shader_props);

	if (m_prog.ctrl & CELL_GCM_SHADER_CONTROL_DEPTH_EXPORT)
	{
		if (m_parr.HasParam(PF_PARAM_NONE, "vec4", "r1"))
		{
			// NOTE: Depth writes are always from a fp32 register. See issues section on nvidia's NV_fragment_program spec
			// https://www.khronos.org/registry/OpenGL/extensions/NV/NV_fragment_program.txt

			// NOTE: Depth writes are unclamped. Targeted hardware tests disprove the theory that depth must be clamped.
			OS <<
				"#ifndef MTL_NO_DEPTH_EXPORT\n"
				"	gl_FragDepth = r1.z;\n"
				"#endif\n";

			mtl_prog->writes_depth = true;
		}
		else
		{
			//Input not declared. Leave commented to assist in debugging the shader
			OS << "	//gl_FragDepth = r1.z;\n";
		}
	}

	glsl::insert_fragment_epilogue(OS, m_shader_props);

	OS << "}\n";
}

void MTLFragmentDecompilerThread::Task()
{
	m_shader = Decompile();
}

MTLFragmentProgram::MTLFragmentProgram() = default;

MTLFragmentProgram::~MTLFragmentProgram()
{
	Delete();
}

void MTLFragmentProgram::Decompile(const RSXFragmentProgram& prog)
{
	output_color_masks = {};
	referenced_textures_mask = 0;
	stencil_mirror_mask = 0;
	uses_depth_input = false;
	uses_framebuffer_fetch = false;
	writes_depth = false;

	u32 size;
	std::string source;
	MTLFragmentDecompilerThread decompiler(source, parr, prog, size, *this);

	if (g_cfg.video.shader_precision == gpu_preset_level::low)
	{
		// All Apple GPUs have native fp16 ALUs
		decompiler.device_props.has_native_half_support = true;
	}

	// Apple GPUs have no D24S8 support; depth surfaces are always stored as floating point
	decompiler.device_props.emulate_depth_compare = !mtl::get_caps().supports_depth24_stencil8;
	decompiler.device_props.emulated_depth_storage = mtl::emulate_extended_depth_range();
	decompiler.device_props.has_low_precision_rounding = false;
	decompiler.Task();

	constant_offsets = std::move(decompiler.properties.constant_offsets);
	shader.create(::glsl::program_domain::glsl_fragment_program, source);
}

void MTLFragmentProgram::Compile()
{
	if (g_cfg.video.log_programs)
		fs::write_file(fs::get_cache_dir() + "shaderlog/FragmentProgram" + std::to_string(id) + ".glsl", fs::rewrite, shader.get_source());

	// Compiled to a Metal function when first linked (pipeline compiler worker in asynchronous mode)
	handle = shader.compile(fmt::format("FragmentProgram%u", id),
		g_cfg.video.log_programs ? fs::get_cache_dir() + "shaderlog/FragmentProgram" + std::to_string(id) + ".metal" : std::string{});

	if (writes_depth)
	{
		// Variant without depth export, linked when the render pass has no depth attachment
		std::string source = shader.get_source();
		const usz version_end = source.find('\n');
		ensure(version_end != umax);
		source.insert(version_end + 1, "#define MTL_NO_DEPTH_EXPORT\n");

		shader_no_depth.create(::glsl::program_domain::glsl_fragment_program, source);
		handle_no_depth = shader_no_depth.compile(fmt::format("FragmentProgram%u_nodepth", id),
			g_cfg.video.log_programs ? fs::get_cache_dir() + "shaderlog/FragmentProgram" + std::to_string(id) + "_nodepth.metal" : std::string{});
	}
}

void MTLFragmentProgram::Delete()
{
	shader.destroy();
	shader_no_depth.destroy();
	handle.reset();
	handle_no_depth.reset();
}
