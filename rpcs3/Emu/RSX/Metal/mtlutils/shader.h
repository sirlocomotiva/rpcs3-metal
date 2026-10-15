#pragma once

#include "mtl_common.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace mtl
{
	enum class shader_stage : u8
	{
		vertex,
		fragment,
		compute
	};

	/**
	 * GLSL resource declaration convention used by every shader fed into the Metal backend.
	 * Shaders are written in Vulkan-flavoured GLSL (glsl_rules_vulkan) and the descriptor set selects the Metal argument table:
	 *
	 *   layout(set = 0, binding = N) uniform/buffer ...     -> [[buffer(N)]]
	 *   layout(set = 1, binding = N) uniform samplerXX ...  -> [[texture(N)]] + [[sampler(N)]] (N < 16)
	 *                                                        -> [[texture(N)]] + constexpr nearest/clamp sampler (N >= 16)
	 *   layout(set = 1, binding = N) uniform usamplerBuffer -> [[texture(N)]] (native texture_buffer<uint>)
	 *   layout(set = 1, binding = N) uniform imageXX ...    -> [[texture(N)]]
	 *   layout(input_attachment_index = I, set = 2, binding = N) uniform subpassInput ...
	 *                                                        -> [[color(I)]] (framebuffer fetch)
	 *   layout(location = N) in (vertex stage)              -> [[attribute(N)]]
	 *
	 * Push constants are not supported; use a small UBO bound with setVertexBytes/setFragmentBytes instead.
	 */
	namespace descriptor_set
	{
		constexpr u32 buffers = 0;
		constexpr u32 textures = 1;
		constexpr u32 input_attachments = 2;
	}

	// Texture slots at or above this index get a constexpr sampler instead of a sampler binding
	constexpr u32 first_constexpr_sampler_slot = 16;

	struct shader_compile_options
	{
		// Entry point name for the generated MSL function
		std::string entry_point = "main0";

		// Allow use of framebuffer fetch for subpass inputs
		bool framebuffer_fetch = true;

		// Flip Y in vertex shaders so that Vulkan NDC conventions (+Y down) carry over to Metal (+Y up)
		bool flip_vert_y = true;

		// Emit [[invariant]] for gl_Position
		bool invariant_position = false;

		// Fast math is disabled by default to match the precision expectations of RSX programs
		bool fast_math = false;

		// Emit the point size builtin when written by the shader
		bool enable_point_size = true;

		// Debug label for the library
		std::string label;
	};

	struct shader_module
	{
		id<MTLLibrary> library = nil;
		id<MTLFunction> function = nil;
		std::string msl_source;

		// Compute only: threadgroup size declared in the shader
		u32 local_size_x = 1;
		u32 local_size_y = 1;
		u32 local_size_z = 1;

		explicit operator bool() const { return function != nil; }
	};

	// GLSL (Vulkan rules) -> SPIR-V. Thread-safe.
	bool compile_glsl_to_spirv(std::vector<u32>& spv, const std::string& glsl, shader_stage stage);

	// SPIR-V -> MSL source. Thread-safe. Returns an empty string on failure.
	std::string compile_spirv_to_msl(const std::vector<u32>& spv, shader_stage stage, const shader_compile_options& options, u32* local_size = nullptr);

	// MSL -> MTLLibrary + entry point. Thread-safe.
	shader_module compile_msl(const std::string& msl, const shader_compile_options& options);

	// Full pipeline: GLSL -> SPIR-V -> MSL -> MTLFunction. Errors are logged; check the returned module.
	shader_module compile_glsl(const std::string& glsl, shader_stage stage, const shader_compile_options& options = {});

	/**
	 * A shader stage compiled on first use. RSX programs are decompiled on the RSX thread, but the expensive part
	 * (GLSL -> SPIR-V -> MSL -> MTLLibrary, tens to hundreds of milliseconds) runs on the thread that links the stage first:
	 * a pipeline compiler worker when pipelines are compiled asynchronously. Thread-safe.
	 */
	class deferred_function
	{
		std::string m_glsl;
		shader_stage m_stage;
		shader_compile_options m_options;
		std::string m_msl_log_path;

		std::once_flag m_compiled;
		shader_module m_module;

	public:
		// msl_log_path: if not empty, the generated MSL is written there once compiled
		deferred_function(std::string glsl, shader_stage stage, shader_compile_options options, std::string msl_log_path = {});

		// Compiles the function on the first call; concurrent callers wait for it. Returns nil if compilation failed (logged once).
		id<MTLFunction> get();
	};

	using deferred_function_ptr = std::shared_ptr<deferred_function>;

	// Must be called once on startup/shutdown (wraps glslang process initialization)
	void initialize_shader_compiler();
	void finalize_shader_compiler();
}
