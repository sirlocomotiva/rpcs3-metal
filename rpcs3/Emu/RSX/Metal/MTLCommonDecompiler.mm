#include "stdafx.h"
#include "MTLCommonDecompiler.h"
#include "util/cctype.hpp"

#include <array>

namespace mtl
{
	static constexpr std::array<std::pair<std::string_view, int>, 18> varying_registers =
	{ {
		{ "tc0", 0 },
		{ "tc1", 1 },
		{ "tc2", 2 },
		{ "tc3", 3 },
		{ "tc4", 4 },
		{ "tc5", 5 },
		{ "tc6", 6 },
		{ "tc7", 7 },
		{ "tc8", 8 },
		{ "tc9", 9 },
		{ "diff_color", 10 },
		{ "diff_color1", 11 },
		{ "spec_color", 12 },
		{ "spec_color1", 13 },
		{ "fog_c", 14 },
		{ "fogc", 14 },
		{ "usr", 15 }, // custom injected stuff
		{ "depth_range", 17 }, // clip range for depth range emulation; 16 is reserved (shader interpreter on VK)
	} };

	int get_varying_register_location(std::string_view varying_register_name)
	{
		for (const auto& varying_register : varying_registers)
		{
			if (varying_register.first == varying_register_name)
			{
				return varying_register.second;
			}
		}

		fmt::throw_exception("Unknown register name: %s", varying_register_name);
	}

	int get_texture_index(std::string_view name)
	{
		if (name.length() < 2)
		{
			fmt::throw_exception("Invalid texture name: '%s'", name);
		}

		constexpr int max_index_length = 2;
		const int name_length = static_cast<int>(name.length());
		std::string index;

		for (int char_idx = name_length - max_index_length; char_idx < name_length; ++char_idx)
		{
			if (utils::isdigit(name[char_idx]))
			{
				index += name[char_idx];
			}
		}

		if (index.empty())
		{
			fmt::throw_exception("Invalid texture name: '%s'", name);
		}

		return std::atoi(index.c_str());
	}

	void insert_rsx_program_preamble(std::ostream& OS, ::glsl::program_domain domain)
	{
		// The shared RSX snippets select the per-draw parameter indirection used by the Vulkan backend when VULKAN is defined
		// (draw_parameters[] lookups, offset constant/stipple arrays). The Metal backend binds every block directly, like OpenGL.
		OS <<
			"// Metal: RSX resources are bound directly (OpenGL resource model), disable the Vulkan draw parameter indirection\n"
			"#undef VULKAN\n";

		if (domain == ::glsl::program_domain::glsl_vertex_program)
		{
			// Without VULKAN the snippets use the OpenGL builtin names. Redirect them to the Vulkan GLSL builtins.
			// gl_VertexIndex/gl_InstanceIndex include the base vertex/instance like OpenGL's gl_VertexID; the renderer never uses a base instance.
			OS <<
				"#define gl_VertexID gl_VertexIndex\n"
				"#define gl_InstanceID gl_InstanceIndex\n";
		}

		OS << "\n";
	}

	void decompiled_shader::create(::glsl::program_domain domain, const std::string& source)
	{
		type = domain;
		m_source = source;
	}

	deferred_function_ptr decompiled_shader::compile(const std::string& label, const std::string& msl_log_path) const
	{
		shader_stage stage;
		switch (type)
		{
		case ::glsl::program_domain::glsl_vertex_program:
			stage = shader_stage::vertex;
			break;
		case ::glsl::program_domain::glsl_fragment_program:
			stage = shader_stage::fragment;
			break;
		case ::glsl::program_domain::glsl_compute_program:
			stage = shader_stage::compute;
			break;
		default:
			fmt::throw_exception("Unexpected program domain %d", static_cast<int>(type));
		}

		shader_compile_options options{};
		options.label = label;

		return std::make_shared<deferred_function>(m_source, stage, std::move(options), msl_log_path);
	}

	void decompiled_shader::destroy()
	{
		m_source.clear();
	}
}
