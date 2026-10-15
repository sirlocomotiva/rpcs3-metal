#pragma once
#include "Emu/RSX/Program/GLSLTypes.h"

// Foundation pulls in CoreServices/CarbonCore MachineExceptions.h, which declares a global 'Vector128' type that collides with
// the Vector128 concept from util/v128.hpp (included by ProgramStateCache.h and most RSX headers). Rename the Apple type.
// No-op if the frameworks were already imported by this translation unit.
#pragma push_macro("Vector128")
#undef Vector128
#define Vector128 CarbonCore_Vector128
#include "Emu/RSX/Metal/mtlutils/shader.h"
#undef Vector128
#pragma pop_macro("Vector128")

#include <ostream>
#include <string>
#include <string_view>

namespace mtl
{
	using namespace ::glsl;

	int get_varying_register_location(std::string_view varying_register_name);

	int get_texture_index(std::string_view name);

	/**
	 * Emitted right after the #version/#extension block of every RSX program.
	 * RSX programs on Metal use the OpenGL resource model (std140 blocks indexed directly, see the binding
	 * convention in mtl::binding) while being compiled with Vulkan GLSL rules. This disables the Vulkan-only
	 * draw-parameter indirection paths of the shared GLSL snippets and maps the legacy builtins they use
	 * to their Vulkan GLSL names.
	 */
	void insert_rsx_program_preamble(std::ostream& OS, ::glsl::program_domain domain);

	/**
	 * Holds the generated GLSL of an RSX program. Mirrors vk::glsl::shader.
	 * The Metal function is compiled on first use by whoever links the program (see mtl::deferred_function).
	 */
	class decompiled_shader
	{
		::glsl::program_domain type = ::glsl::program_domain::glsl_vertex_program;
		std::string m_source;

	public:
		decompiled_shader() = default;
		~decompiled_shader() = default;

		void create(::glsl::program_domain domain, const std::string& source);

		// Returns the function compiled from the GLSL source on first use. msl_log_path: optional MSL dump (log_programs).
		deferred_function_ptr compile(const std::string& label = {}, const std::string& msl_log_path = {}) const;

		void destroy();

		const std::string& get_source() const { return m_source; }
	};
}
