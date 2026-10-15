#include "stdafx.h"
#include "shader.h"
#include "device.h"

#include "Emu/RSX/Program/GLSLTypes.h"
#include "Utilities/File.h"
#include "Emu/RSX/Program/SPIRVCommon.h"

// SPIRV-Cross reports translation errors by throwing spirv_cross::CompilerError. RPCS3 builds with -fno-exceptions,
// so this translation unit alone is compiled with -fexceptions (see rpcs3/Emu/CMakeLists.txt) to catch them.
#if !defined(__cpp_exceptions)
#error "shader.mm must be compiled with -fexceptions: SPIRV-Cross reports translation errors with C++ exceptions"
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wshadow"
#ifdef __clang__
#pragma clang diagnostic ignored "-Winconsistent-missing-override"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#if __has_include(<spirv_cross/spirv_msl.hpp>)
#include <spirv_cross/spirv_msl.hpp>
#else
#include <spirv_msl.hpp>
#endif
#pragma GCC diagnostic pop

#include <exception>

namespace mtl
{
	namespace
	{
		const char* get_stage_name(shader_stage stage)
		{
			switch (stage)
			{
			case shader_stage::vertex: return "vertex";
			case shader_stage::fragment: return "fragment";
			case shader_stage::compute: return "compute";
			}

			return "unknown";
		}

		spv::ExecutionModel get_execution_model(shader_stage stage)
		{
			switch (stage)
			{
			case shader_stage::vertex: return spv::ExecutionModelVertex;
			case shader_stage::fragment: return spv::ExecutionModelFragment;
			case shader_stage::compute: return spv::ExecutionModelGLCompute;
			}

			fmt::throw_exception("Unknown shader stage %u", static_cast<u32>(stage));
		}

		// Constexpr sampler used for texture slots without a sampler binding (stencil mirrors, frag_depth)
		spirv_cross::MSLConstexprSampler get_default_constexpr_sampler()
		{
			spirv_cross::MSLConstexprSampler sampler{};
			sampler.coord = spirv_cross::MSL_SAMPLER_COORD_NORMALIZED;
			sampler.min_filter = spirv_cross::MSL_SAMPLER_FILTER_NEAREST;
			sampler.mag_filter = spirv_cross::MSL_SAMPLER_FILTER_NEAREST;
			sampler.mip_filter = spirv_cross::MSL_SAMPLER_MIP_FILTER_NONE;
			sampler.s_address = spirv_cross::MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE;
			sampler.t_address = spirv_cross::MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE;
			sampler.r_address = spirv_cross::MSL_SAMPLER_ADDRESS_CLAMP_TO_EDGE;
			return sampler;
		}

		// Applies the resource binding convention described in shader.h. Returns false (with error set) on unsupported resources.
		bool remap_resources(spirv_cross::CompilerMSL& compiler, spv::ExecutionModel model, std::string& error)
		{
			const spirv_cross::ShaderResources resources = compiler.get_shader_resources();

			auto get_set = [&](const spirv_cross::Resource& res)
			{
				return compiler.get_decoration(res.id, spv::DecorationDescriptorSet);
			};

			auto get_binding = [&](const spirv_cross::Resource& res)
			{
				return compiler.get_decoration(res.id, spv::DecorationBinding);
			};

			auto get_array_size = [&](const spirv_cross::Resource& res) -> u32
			{
				const auto& type = compiler.get_type(res.type_id);
				u32 count = 1;
				for (const u32 dim : type.array)
				{
					count *= std::max(dim, 1u);
				}
				return count;
			};

			auto make_binding = [&](const spirv_cross::Resource& res)
			{
				spirv_cross::MSLResourceBinding binding{};
				binding.stage = model;
				binding.desc_set = get_set(res);
				binding.binding = get_binding(res);
				binding.count = get_array_size(res);
				return binding;
			};

			auto check_set = [&](const spirv_cross::Resource& res, u32 expected_set, const char* kind)
			{
				const u32 set = get_set(res);
				if (set == expected_set)
				{
					return true;
				}

				error = fmt::format("%s '%s' (binding %u) is declared in descriptor set %u, expected set %u",
					kind, compiler.get_name(res.id), get_binding(res), set, expected_set);
				return false;
			};

			if (!resources.atomic_counters.empty())
			{
				error = "Atomic counters are not supported";
				return false;
			}

			if (!resources.acceleration_structures.empty() || !resources.shader_record_buffers.empty() || !resources.tensors.empty())
			{
				error = "Ray tracing / tensor resources are not supported";
				return false;
			}

			// set 0: buffers
			bool buffer_slot0_used = false;
			for (const auto* list : { &resources.uniform_buffers, &resources.storage_buffers })
			{
				for (const auto& res : *list)
				{
					if (!check_set(res, descriptor_set::buffers, "Buffer"))
					{
						return false;
					}

					auto binding = make_binding(res);
					binding.msl_buffer = binding.binding;
					buffer_slot0_used |= (binding.binding == 0);

					if (binding.binding + binding.count > binding::max_buffer_slots)
					{
						error = fmt::format("Buffer '%s' uses slot %u which exceeds the Metal buffer table", compiler.get_name(res.id), binding.binding);
						return false;
					}

					compiler.add_msl_resource_binding(binding);
				}
			}

			// Push constants are not part of the convention, but map them to buffer 0 like a parameter block for robustness
			if (!resources.push_constant_buffers.empty())
			{
				if (buffer_slot0_used)
				{
					error = "Push constant block conflicts with the buffer bound at set 0, binding 0";
					return false;
				}

				spirv_cross::MSLResourceBinding binding{};
				binding.stage = model;
				binding.desc_set = spirv_cross::kPushConstDescSet;
				binding.binding = spirv_cross::kPushConstBinding;
				binding.count = 1;
				binding.msl_buffer = 0;
				compiler.add_msl_resource_binding(binding);
			}

			// set 1: textures (+ samplers)
			for (const auto& res : resources.sampled_images)
			{
				if (!check_set(res, descriptor_set::textures, "Sampled image"))
				{
					return false;
				}

				auto binding = make_binding(res);
				binding.msl_texture = binding.binding;

				const auto& type = compiler.get_type(res.base_type_id);
				const bool is_texel_buffer = type.image.dim == spv::DimBuffer;

				if (is_texel_buffer)
				{
					// Native texture_buffer<T>, read with .read(); no sampler needed
				}
				else if (binding.binding < first_constexpr_sampler_slot)
				{
					binding.msl_sampler = binding.binding;
				}
				else
				{
					compiler.remap_constexpr_sampler_by_binding(binding.desc_set, binding.binding, get_default_constexpr_sampler());
				}

				compiler.add_msl_resource_binding(binding);
			}

			for (const auto* list : { &resources.separate_images, &resources.storage_images })
			{
				for (const auto& res : *list)
				{
					if (!check_set(res, descriptor_set::textures, "Image"))
					{
						return false;
					}

					auto binding = make_binding(res);
					binding.msl_texture = binding.binding;
					compiler.add_msl_resource_binding(binding);
				}
			}

			for (const auto& res : resources.separate_samplers)
			{
				if (!check_set(res, descriptor_set::textures, "Sampler"))
				{
					return false;
				}

				auto binding = make_binding(res);
				if (binding.binding >= first_constexpr_sampler_slot)
				{
					compiler.remap_constexpr_sampler_by_binding(binding.desc_set, binding.binding, get_default_constexpr_sampler());
					continue;
				}

				binding.msl_sampler = binding.binding;
				compiler.add_msl_resource_binding(binding);
			}

			// set 2: subpass inputs. With framebuffer fetch they become [[color(input_attachment_index)]] automatically.
			// Without it SPIRV-Cross reads them as textures, which then use texture slot = binding.
			for (const auto& res : resources.subpass_inputs)
			{
				if (!check_set(res, descriptor_set::input_attachments, "Subpass input"))
				{
					return false;
				}

				auto binding = make_binding(res);
				binding.msl_texture = binding.binding;
				compiler.add_msl_resource_binding(binding);
			}

			return true;
		}

		void decorate_invariant_position(spirv_cross::CompilerMSL& compiler)
		{
			const spirv_cross::ShaderResources resources = compiler.get_shader_resources();

			for (const auto& builtin : resources.builtin_outputs)
			{
				if (builtin.builtin != spv::BuiltInPosition)
				{
					continue;
				}

				const auto& type = compiler.get_type(builtin.resource.base_type_id);
				if (type.basetype == spirv_cross::SPIRType::Struct)
				{
					// gl_PerVertex block
					for (u32 index = 0; index < static_cast<u32>(type.member_types.size()); ++index)
					{
						if (compiler.has_member_decoration(type.self, index, spv::DecorationBuiltIn) &&
							compiler.get_member_decoration(type.self, index, spv::DecorationBuiltIn) == spv::BuiltInPosition)
						{
							compiler.set_member_decoration(type.self, index, spv::DecorationInvariant);
						}
					}
				}
				else
				{
					compiler.set_decoration(builtin.resource.id, spv::DecorationInvariant);
				}
			}
		}

		// Everything that can throw lives in here; the caller catches the exceptions
		bool translate_spirv_to_msl(const std::vector<u32>& spv, shader_stage stage, const shader_compile_options& options,
			u32* local_size, std::string& msl, std::string& error)
		{
			spirv_cross::CompilerMSL compiler(spv);

			const spv::ExecutionModel model = get_execution_model(stage);

			// Entry point
			if (!options.entry_point.empty() && options.entry_point != "main")
			{
				compiler.rename_entry_point("main", options.entry_point, model);
			}

			// MSL options
			spirv_cross::CompilerMSL::Options msl_options = compiler.get_msl_options();
			msl_options.platform = spirv_cross::CompilerMSL::Options::macOS;
			msl_options.set_msl_version(3, 2);
			msl_options.texture_buffer_native = true;
			msl_options.use_framebuffer_fetch_subpasses = options.framebuffer_fetch;
			msl_options.enable_point_size_builtin = options.enable_point_size;
			msl_options.pad_fragment_output_components = true;
			msl_options.argument_buffers = false;
			msl_options.enable_decoration_binding = false;
			msl_options.invariant_float_math = options.invariant_position;
			msl_options.agx_manual_cube_grad_fixup = get_current_device() ? get_caps().is_apple_gpu : true;
			compiler.set_msl_options(msl_options);

			// Common options. Y is flipped in vertex shaders so that the Vulkan conventions of the RSX code carry over.
			spirv_cross::CompilerGLSL::Options common_options = compiler.get_common_options();
			common_options.vertex.flip_vert_y = (stage == shader_stage::vertex) && options.flip_vert_y;
			common_options.vertex.fixup_clipspace = false;
			compiler.set_common_options(common_options);

			if (stage == shader_stage::vertex && options.invariant_position)
			{
				decorate_invariant_position(compiler);
			}

			if (!remap_resources(compiler, model, error))
			{
				return false;
			}

			if (stage == shader_stage::compute && local_size)
			{
				spirv_cross::SpecializationConstant x{}, y{}, z{};
				compiler.get_work_group_size_specialization_constants(x, y, z);

				const spirv_cross::SpecializationConstant* spec[3] = { &x, &y, &z };
				for (u32 i = 0; i < 3; ++i)
				{
					if (spec[i]->id)
					{
						// Work group dimension is a specialization constant, use its default value
						local_size[i] = compiler.get_constant(spec[i]->id).scalar();
					}
					else
					{
						local_size[i] = compiler.get_execution_mode_argument(spv::ExecutionModeLocalSize, i);
					}

					local_size[i] = std::max(local_size[i], 1u);
				}
			}

			msl = compiler.compile();
			return true;
		}
	}

	bool compile_glsl_to_spirv(std::vector<u32>& spv, const std::string& glsl, shader_stage stage)
	{
		::glsl::program_domain domain;
		switch (stage)
		{
		case shader_stage::vertex: domain = ::glsl::glsl_vertex_program; break;
		case shader_stage::fragment: domain = ::glsl::glsl_fragment_program; break;
		case shader_stage::compute: domain = ::glsl::glsl_compute_program; break;
		default: fmt::throw_exception("Unknown shader stage %u", static_cast<u32>(stage));
		}

		std::string source = glsl;
		return spirv::compile_glsl_to_spv(spv, source, domain, ::glsl::glsl_rules_vulkan);
	}

	std::string compile_spirv_to_msl(const std::vector<u32>& spv, shader_stage stage, const shader_compile_options& options, u32* local_size)
	{
		// A fresh compiler instance is used for every call, which makes this function thread-safe.
		std::string msl;
		std::string error;
		bool success = false;

		if (local_size)
		{
			local_size[0] = local_size[1] = local_size[2] = 1;
		}

		try
		{
			success = translate_spirv_to_msl(spv, stage, options, local_size, msl, error);
		}
		catch (const std::exception& e)
		{
			// spirv_cross::CompilerError derives from std::runtime_error
			success = false;
			error = e.what();
		}
		catch (...)
		{
			success = false;
			error = "<unknown exception>";
		}

		if (!success || msl.empty())
		{
			if (local_size)
			{
				local_size[0] = local_size[1] = local_size[2] = 1;
			}

			rsx_log.error("Metal: SPIR-V -> MSL translation of %s shader failed: %s", get_stage_name(stage), error.empty() ? "<no output>" : error);
			return {};
		}

		return msl;
	}

	shader_module compile_msl(const std::string& msl, const shader_compile_options& options)
	{
		shader_module result{};
		result.msl_source = msl;

		@autoreleasepool
		{
			MTLCompileOptions* compile_options = [MTLCompileOptions new];
			compile_options.languageVersion = MTLLanguageVersion3_2;
			compile_options.mathMode = options.fast_math ? MTLMathModeFast : MTLMathModeSafe;
			compile_options.mathFloatingPointFunctions = options.fast_math ? MTLMathFloatingPointFunctionsFast : MTLMathFloatingPointFunctionsPrecise;
			compile_options.preserveInvariance = options.invariant_position ? YES : NO;

			stall_trace_scope trace("MSL compile");

			NSError* error = nil;
			id<MTLLibrary> library = [get_device() newLibraryWithSource:to_ns_string(msl) options:compile_options error:&error];

			if (!library)
			{
				rsx_log.error("Metal: MSL compilation failed (%s): %s\nMSL source:\n%s",
					options.label.empty() ? "unnamed" : options.label, describe_error(error), msl);
				return result;
			}

			if (error)
			{
				// Compilation succeeded with warnings
				rsx_log.trace("Metal: MSL compiler warnings (%s): %s", options.label.empty() ? "unnamed" : options.label, describe_error(error));
			}

			if (!options.label.empty())
			{
				library.label = to_ns_string(options.label);
			}

			id<MTLFunction> function = [library newFunctionWithName:to_ns_string(options.entry_point)];

			if (!function && library.functionNames.count == 1)
			{
				// The entry point name was not applied (e.g. module translated with a different name), use the only function
				function = [library newFunctionWithName:library.functionNames.firstObject];
			}

			if (!function)
			{
				rsx_log.error("Metal: Entry point '%s' not found in the compiled library (%s)\nMSL source:\n%s",
					options.entry_point, options.label.empty() ? "unnamed" : options.label, msl);
				return result;
			}

			if (!options.label.empty())
			{
				function.label = to_ns_string(options.label);
			}

			result.library = library;
			result.function = function;
		}

		return result;
	}

	shader_module compile_glsl(const std::string& glsl, shader_stage stage, const shader_compile_options& options)
	{
		std::vector<u32> spv;
		if (!compile_glsl_to_spirv(spv, glsl, stage))
		{
			rsx_log.error("Metal: GLSL -> SPIR-V compilation of %s shader failed (%s)\nGLSL source:\n%s",
				get_stage_name(stage), options.label.empty() ? "unnamed" : options.label, glsl);
			return {};
		}

		u32 local_size[3] = { 1, 1, 1 };
		const std::string msl = compile_spirv_to_msl(spv, stage, options, local_size);
		if (msl.empty())
		{
			rsx_log.error("Metal: Failed to translate %s shader (%s)\nGLSL source:\n%s",
				get_stage_name(stage), options.label.empty() ? "unnamed" : options.label, glsl);
			return {};
		}

		shader_module result = compile_msl(msl, options);
		if (!result)
		{
			rsx_log.error("Metal: Failed to compile %s shader (%s)\nGLSL source:\n%s",
				get_stage_name(stage), options.label.empty() ? "unnamed" : options.label, glsl);
			return result;
		}

		result.local_size_x = local_size[0];
		result.local_size_y = local_size[1];
		result.local_size_z = local_size[2];
		return result;
	}

	deferred_function::deferred_function(std::string glsl, shader_stage stage, shader_compile_options options, std::string msl_log_path)
		: m_glsl(std::move(glsl))
		, m_stage(stage)
		, m_options(std::move(options))
		, m_msl_log_path(std::move(msl_log_path))
	{
	}

	id<MTLFunction> deferred_function::get()
	{
		std::call_once(m_compiled, [this]()
		{
			// compile_glsl logs the reason and the source on failure
			m_module = compile_glsl(m_glsl, m_stage, m_options);

			if (m_module && !m_msl_log_path.empty())
			{
				fs::write_file(m_msl_log_path, fs::rewrite, m_module.msl_source);
			}

			// Only needed to compile
			m_glsl = {};
			m_module.msl_source = {};
		});

		return m_module.function;
	}

	void initialize_shader_compiler()
	{
		spirv::initialize_compiler_context();
	}

	void finalize_shader_compiler()
	{
		spirv::finalize_compiler_context();
	}
}
