#pragma once

// Objective-C++ defines __EXCEPTIONS for Objective-C exceptions even though rpcs3 builds with C++ exceptions disabled.
// unordered_dense keys its error handling on that macro and would emit C++ 'throw' statements; make it take the same
// no-exceptions path as the C++ translation units (this also keeps its inline error handlers ODR-identical across TUs).
#if defined(__OBJC__) && defined(__EXCEPTIONS) && !defined(__cpp_exceptions)
#pragma push_macro("__EXCEPTIONS")
#undef __EXCEPTIONS
#include "Emu/RSX/Common/unordered_map.hpp"
#pragma pop_macro("__EXCEPTIONS")
#endif

#include "MTLVertexProgram.h"
#include "MTLFragmentProgram.h"
#include "Emu/RSX/Metal/mtlutils/pipeline.h"
#include "Emu/RSX/Program/ProgramStateCache.h"
#include "Emu/RSX/rsx_cache.h"

#include <functional>
#include <memory>

namespace mtl
{
	struct MTLTraits
	{
		using vertex_program_type = MTLVertexProgram;
		using fragment_program_type = MTLFragmentProgram;
		using pipeline_type = mtl::program;
		using pipeline_storage_type = std::unique_ptr<mtl::program>;
		using pipeline_properties = mtl::pipeline_props;

		static
			void recompile_fragment_program(const RSXFragmentProgram& RSXFP, fragment_program_type& fragmentProgramData, usz ID);

		static
			void recompile_vertex_program(const RSXVertexProgram& RSXVP, vertex_program_type& vertexProgramData, usz ID);

		static
			void validate_pipeline_properties(const MTLVertexProgram& vp, const MTLFragmentProgram& fp, mtl::pipeline_props& properties);

		// Links the two stages with pipe_compiler (inline or deferred) and copies the decompilers' reflection data into the program.
		// Returns the linked program for inline compilation, nullptr for deferred compilation or if a stage failed to compile.
		static
			pipeline_type* build_pipeline(
				const vertex_program_type& vertexProgramData,
				const fragment_program_type& fragmentProgramData,
				const mtl::pipeline_props& pipelineProperties,
				bool compile_async,
				std::function<pipeline_type*(pipeline_storage_type&)> callback);
	};

	struct program_cache : public program_state_cache<MTLTraits>
	{
		program_cache(decompiler_callback_t callback)
		{
			notify_pipeline_compiled = callback;
		}

		u64 get_hash(const mtl::pipeline_props& props)
		{
			return rpcs3::hash_struct<mtl::pipeline_props>(props);
		}

		u64 get_hash(const RSXVertexProgram& prog)
		{
			return program_hash_util::vertex_program_utils::get_vertex_program_ucode_hash(prog);
		}

		u64 get_hash(const RSXFragmentProgram& prog)
		{
			return program_hash_util::fragment_program_utils::get_fragment_program_ucode_hash(prog);
		}

		template <typename... Args>
		void add_pipeline_entry(RSXVertexProgram& vp, RSXFragmentProgram& fp, mtl::pipeline_props& props, Args&& ...args)
		{
			get_graphics_pipeline(nullptr, vp, fp, props, false, false, std::forward<Args>(args)...);
		}

		void preload_programs(rsx::program_cache_hint_t* cache_hint, const RSXVertexProgram& vp, const RSXFragmentProgram& fp)
		{
			search_vertex_program(cache_hint, vp);
			search_fragment_program(cache_hint, fp);
		}

		bool check_cache_missed() const
		{
			return m_cache_miss_flag;
		}
	};

	using shader_cache = rsx::shaders_cache<mtl::pipeline_props, mtl::program_cache>;
}
