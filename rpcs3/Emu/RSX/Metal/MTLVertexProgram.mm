#include "stdafx.h"
#include "MTLVertexProgram.h"

#include "MTLCommonDecompiler.h"
#include "Emu/RSX/Program/GLSLCommon.h"
#include "Emu/system_config.h"
#include "Utilities/File.h"

std::string MTLVertexDecompilerThread::getFloatTypeName(usz elementCount)
{
	return glsl::getFloatTypeNameImpl(elementCount);
}

std::string MTLVertexDecompilerThread::getIntTypeName(usz /*elementCount*/)
{
	return "ivec4";
}

std::string MTLVertexDecompilerThread::getFunction(FUNCTION f)
{
	return glsl::getFunctionImpl(f);
}

std::string MTLVertexDecompilerThread::compareFunction(COMPARE f, std::string_view Op0, std::string_view Op1, bool scalar)
{
	return glsl::compareFunctionImpl(f, Op0, Op1, scalar);
}

void MTLVertexDecompilerThread::insertHeader(std::stringstream& OS)
{
	OS <<
		"#version 450\n\n"
		"#extension GL_ARB_separate_shader_objects : enable\n\n";

	mtl::insert_rsx_program_preamble(OS, glsl::glsl_vertex_program);

	glsl::insert_subheader_block(OS);

	// Same layout as the OpenGL backend (filled by the renderer exactly like GLGSRender::load_program_env)
	OS <<
		"layout(std140, set=" << mtl::descriptor_set::buffers << ", binding=" << mtl::binding::vs_context_buffer << ") uniform VertexContextBuffer\n"
		"{\n"
		"	mat4 scale_offset_mat;\n"
		"	uint user_clip_configuration_bits;\n"
		"	uint transform_branch_bits;\n"
		"	float point_size;\n"
		"	float z_near;\n"
		"	float z_far;\n"
		"};\n"
		"#define get_user_clip_config() user_clip_configuration_bits\n\n";

	// Same layout as the OpenGL backend (filled by the renderer exactly like GLGSRender::update_vertex_env)
	OS <<
		"layout(std140, set=" << mtl::descriptor_set::buffers << ", binding=" << mtl::binding::vs_layout_buffer << ") uniform VertexLayoutBuffer\n"
		"{\n"
		"	uint  vertex_base_index;\n"
		"	uint  vertex_index_offset;\n"
		"	uvec4 input_attributes_blob[16 / 2];\n"
		"};\n\n";
}

void MTLVertexDecompilerThread::insertInputs(std::stringstream& OS, const std::vector<ParamType>& /*inputs*/)
{
	static const std::pair<const char*, u32> input_streams[] =
	{
		{ "persistent_input_stream", mtl::binding::vs_persistent_stream },    // Data stream with persistent vertex data (cacheable)
		{ "volatile_input_stream", mtl::binding::vs_volatile_stream }         // Data stream with per-draw data (registers and immediate draw data)
	};

	for (const auto& [stream, location] : input_streams)
	{
		OS << "layout(set=" << mtl::descriptor_set::textures << ", binding=" << location << ") uniform usamplerBuffer " << stream << ";\n";
	}

	OS << "\n";
}

void MTLVertexDecompilerThread::insertConstants(std::stringstream& OS, const std::vector<ParamType>& constants)
{
	for (const ParamType &PT : constants)
	{
		for (const ParamItem &PI : PT.items)
		{
			if (PI.name.starts_with("vc["))
			{
				if (!(m_prog.ctrl & RSX_SHADER_CONTROL_INSTANCED_CONSTANTS))
				{
					// Either the full 468 register file or the compacted list of referenced constants (see VertexProgramBase::constant_ids)
					OS <<
						"layout(std140, set=" << mtl::descriptor_set::buffers << ", binding=" << mtl::binding::vs_constants_buffer << ") uniform VertexConstantsBuffer\n"
						"{\n"
						"	vec4 " << PI.name << ";\n"
						"};\n\n";
				}
				else
				{
					// 1. Indirection lookup table, 2. Actual constants storage
					OS <<
						"layout(std430, set=" << mtl::descriptor_set::buffers << ", binding=" << mtl::binding::vs_instancing_lut_buffer << ") readonly buffer InstancingIndirectionLUT\n"
						"{\n"
						"	int constants_addressing_lookup[];\n"
						"};\n\n"

						"layout(std430, set=" << mtl::descriptor_set::buffers << ", binding=" << mtl::binding::vs_instancing_data_buffer << ") readonly buffer InstancingVertexConstantsBlock\n"
						"{\n"
						"	vec4 instanced_constants_array[];\n"
						"};\n\n"

						"#define CONSTANTS_ARRAY_LENGTH " << (properties.has_indexed_constants ? 468 : ::size32(m_constant_ids)) << "\n\n";
				}

				continue;
			}

			if (PT.type.starts_with("sampler"))
			{
				const int id = mtl::get_texture_index(PI.name);
				ensure(id >= 0 && id < 4, "Unexpected vertex texture index");

				auto samplerType = PT.type;

				if (m_prog.texture_state.multisampled_textures) [[ unlikely ]]
				{
					ensure(PI.name.length() > 3);
					int index = atoi(&PI.name[3]);

					if (m_prog.texture_state.multisampled_textures & (1 << index))
					{
						if (samplerType != "sampler1D" && samplerType != "sampler2D")
						{
							rsx_log.error("Unexpected multisampled sampler type '%s'", samplerType);
						}

						samplerType = "sampler2DMS";
					}
				}

				OS << "layout(set=" << mtl::descriptor_set::textures << ", binding=" << (mtl::binding::vs_texture0 + id) << ") uniform " << samplerType << " " << PI.name << ";\n";

				mtl_prog->referenced_textures_mask |= (1u << id);
			}
		}
	}

	OS << "\n";
}

static const vertex_reg_info reg_table[] =
{
	{ "gl_Position", false, "dst_reg0", "", false },
	//Technically these two are for both back and front
	{ "diff_color", true, "dst_reg1", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_FRONTDIFFUSE | CELL_GCM_ATTRIB_OUTPUT_MASK_BACKDIFFUSE },
	{ "spec_color", true, "dst_reg2", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_FRONTSPECULAR | CELL_GCM_ATTRIB_OUTPUT_MASK_BACKSPECULAR },
	{ "diff_color1", true, "dst_reg3", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_FRONTDIFFUSE | CELL_GCM_ATTRIB_OUTPUT_MASK_BACKDIFFUSE },
	{ "spec_color1", true, "dst_reg4", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_FRONTSPECULAR | CELL_GCM_ATTRIB_OUTPUT_MASK_BACKSPECULAR },
	{ "fog_c", true, "dst_reg5", ".xxxx", true, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_FOG },
	//Warning: With spir-v if you declare clip distance var, you must assign a value even when its disabled! Runtime does not assign a default value
	{ "gl_ClipDistance[0]", false, "dst_reg5", ".y * user_clip_factor(0)", false, "is_user_clip_enabled(0)", "0.5", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_UC0 },
	{ "gl_ClipDistance[1]", false, "dst_reg5", ".z * user_clip_factor(1)", false, "is_user_clip_enabled(1)", "0.5", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_UC1 },
	{ "gl_ClipDistance[2]", false, "dst_reg5", ".w * user_clip_factor(2)", false, "is_user_clip_enabled(2)", "0.5", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_UC2 },
	{ "gl_PointSize", false, "dst_reg6", ".x", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_POINTSIZE },
	{ "gl_ClipDistance[3]", false, "dst_reg6", ".y * user_clip_factor(3)", false, "is_user_clip_enabled(3)", "0.5", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_UC3 },
	{ "gl_ClipDistance[4]", false, "dst_reg6", ".z * user_clip_factor(4)", false, "is_user_clip_enabled(4)", "0.5", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_UC4 },
	{ "gl_ClipDistance[5]", false, "dst_reg6", ".w * user_clip_factor(5)", false, "is_user_clip_enabled(5)", "0.5", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_UC5 },
	{ "tc0", true, "dst_reg7", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_TEX0 },
	{ "tc1", true, "dst_reg8", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_TEX1 },
	{ "tc2", true, "dst_reg9", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_TEX2 },
	{ "tc3", true, "dst_reg10", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_TEX3 },
	{ "tc4", true, "dst_reg11", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_TEX4 },
	{ "tc5", true, "dst_reg12", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_TEX5 },
	{ "tc6", true, "dst_reg13", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_TEX6 },
	{ "tc7", true, "dst_reg14", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_TEX7 },
	{ "tc8", true, "dst_reg15", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_TEX8 },
	{ "tc9", true, "dst_reg6", "", false, "", "", "", true, CELL_GCM_ATTRIB_OUTPUT_MASK_TEX9 }  // In this line, dst_reg6 is correct since dst_reg goes from 0 to 15.
};

void MTLVertexDecompilerThread::insertOutputs(std::stringstream& OS, const std::vector<ParamType>& /*outputs*/)
{
	for (auto &i : reg_table)
	{
		if (i.need_declare)
		{
			// All outputs must be declared always to allow setting default values.
			// NV4097_SET_SHADE_MODE applies to the front/back diffuse and specular colors.
			const bool flat_color = (m_prog.ctrl & RSX_SHADER_CONTROL_FLAT_SHADING) &&
				(i.name.starts_with("diff_color"sv) || i.name.starts_with("spec_color"sv));
			OS << "layout(location=" << mtl::get_varying_register_location(i.name) << ") out "
				<< (flat_color ? "flat " : "") << "vec4 " << i.name << ";\n";
		}
	}

	if (m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_RANGE)
	{
		OS << "layout(location=" << mtl::get_varying_register_location("depth_range") << ") out flat vec2 depth_range;\n";
	}

	OS << "\n";
}

void MTLVertexDecompilerThread::insertMainStart(std::stringstream& OS)
{
	glsl::shader_properties properties2{};
	properties2.domain = glsl::glsl_vertex_program;
	properties2.require_lit_emulation = properties.has_lit_op;
	properties2.require_clip_plane_functions = true;
	properties2.emulate_zclip_transform = true;
	properties2.emulate_depth_clip_only = false; // Metal has no double precision support, use the float fallback
	properties2.low_precision_tests = false;
	properties2.require_explicit_invariance = false;
	properties2.require_instanced_render = !!(m_prog.ctrl & RSX_SHADER_CONTROL_INSTANCED_CONSTANTS);
	properties2.emulate_depth_range = !!(m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_RANGE);

	glsl::insert_glsl_legacy_function(OS, properties2);
	glsl::insert_vertex_input_fetch(OS, glsl::glsl_rules_vulkan);

	// Declare global registers with optional initialization
	std::string registers;
	if (ParamType *vec4Types = m_parr.SearchParam(PF_PARAM_OUT, "vec4"))
	{
		for (auto &PI : vec4Types->items)
		{
			if (registers.length())
				registers += ", ";
			else
				registers = "vec4 ";

			registers += PI.name;

			if (!PI.value.empty())
			{
				// Simplify default initialization
				if (PI.value == "vec4(0.0, 0.0, 0.0, 0.0)")
					registers += " = vec4(0.)";
				else
					registers += " = " + PI.value;
			}
		}
	}

	if (!registers.empty())
	{
		OS << registers << ";\n";
	}

	OS << "void vs_main()\n";
	OS << "{\n";

	//Declare temporary registers, ignoring those mapped to outputs
	for (const ParamType& PT : m_parr.params[PF_PARAM_NONE])
	{
		for (const ParamItem& PI : PT.items)
		{
			if (PI.name.starts_with("dst_reg"))
				continue;

			OS << "	" << PT.type << " " << PI.name;
			if (!PI.value.empty())
				OS << " = " << PI.value;

			OS << ";\n";
		}
	}

	for (const ParamType& PT : m_parr.params[PF_PARAM_IN])
	{
		for (const ParamItem& PI : PT.items)
		{
			OS << "	vec4 " << PI.name << "= read_location(" << std::to_string(PI.location) << ");\n";
		}
	}
}

void MTLVertexDecompilerThread::insertMainEnd(std::stringstream& OS)
{
	OS << "}\n\n";

	OS << "void main ()\n";
	OS << "{\n\n";

	OS << "	vs_main();\n\n";

	for (auto &i : reg_table)
	{
		if (!i.check_mask || i.test(rsx_vertex_program.output_mask))
		{
			if (m_parr.HasParam(PF_PARAM_OUT, "vec4", i.src_reg))
			{
				std::string condition = (!i.cond.empty()) ? "(" + i.cond + ") " : "";

				if (condition.empty() || i.default_val.empty())
				{
					if (!condition.empty()) condition = "if " + condition;
					OS << "	" << condition << i.name << " = " << i.src_reg << i.src_reg_mask << ";\n";
				}
				else
				{
					//Insert if-else condition
					OS << "	" << i.name << " = " << condition << "? " << i.src_reg << i.src_reg_mask << ": " << i.default_val << ";\n";
				}

				// Register was marked for output and a properly initialized source register exists
				// Nothing more to do
				continue;
			}
		}

		if (i.need_declare)
		{
			OS << "	" << i.name << " = vec4(0., 0., 0., 1.);\n";
		}
		else if (i.check_mask_value == CELL_GCM_ATTRIB_OUTPUT_MASK_POINTSIZE)
		{
			// Default point size if none was generated by the program
			OS << "	gl_PointSize = point_size;\n";
		}
	}

	OS << "	gl_Position = gl_Position * scale_offset_mat;\n";

	if (!(m_prog.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_RANGE)) [[ likely ]]
	{
		OS << "	gl_Position = apply_zclip_xform(gl_Position, z_near, z_far);\n";
	}
	else
	{
		OS <<
			"	depth_range = vec2(min(z_near, z_far), max(z_near, z_far));\n"
			"	gl_Position = apply_depth_range_xform(gl_Position, depth_range);\n";
	}

	OS << "}\n";
}

void MTLVertexDecompilerThread::Task()
{
	m_shader = Decompile();
}

MTLVertexProgram::MTLVertexProgram() = default;

MTLVertexProgram::~MTLVertexProgram()
{
	Delete();
}

void MTLVertexProgram::Decompile(const RSXVertexProgram& prog)
{
	m_ctrl = prog.ctrl;
	referenced_textures_mask = 0;

	std::string source;
	MTLVertexDecompilerThread decompiler(prog, source, parr, *this);
	decompiler.Task();

	has_indexed_constants = decompiler.properties.has_indexed_constants;
	constant_ids = std::vector<u16>(decompiler.m_constant_ids.begin(), decompiler.m_constant_ids.end());

	shader.create(::glsl::program_domain::glsl_vertex_program, source);
}

void MTLVertexProgram::Compile()
{
	if (g_cfg.video.log_programs)
		fs::write_file(fs::get_cache_dir() + "shaderlog/VertexProgram" + std::to_string(id) + ".glsl", fs::rewrite, shader.get_source());

	// Compiled to a Metal function when first linked (pipeline compiler worker in asynchronous mode)
	handle = shader.compile(fmt::format("VertexProgram%u", id),
		g_cfg.video.log_programs ? fs::get_cache_dir() + "shaderlog/VertexProgram" + std::to_string(id) + ".metal" : std::string{});
}

void MTLVertexProgram::Delete()
{
	shader.destroy();
	handle.reset();
}
