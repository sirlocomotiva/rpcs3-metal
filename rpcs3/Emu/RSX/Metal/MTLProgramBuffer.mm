#include "stdafx.h"
#include "MTLProgramBuffer.h"

namespace mtl
{
	void MTLTraits::recompile_fragment_program(const RSXFragmentProgram& RSXFP, fragment_program_type& fragmentProgramData, usz ID)
	{
		fragmentProgramData.Decompile(RSXFP);
		fragmentProgramData.id = static_cast<u32>(ID);
		fragmentProgramData.Compile();
	}

	void MTLTraits::recompile_vertex_program(const RSXVertexProgram& RSXVP, vertex_program_type& vertexProgramData, usz ID)
	{
		vertexProgramData.Decompile(RSXVP);
		vertexProgramData.id = static_cast<u32>(ID);
		vertexProgramData.Compile();
	}

	void MTLTraits::validate_pipeline_properties(const MTLVertexProgram&, const MTLFragmentProgram& fp, mtl::pipeline_props& properties)
	{
		//Explicitly disable writing to undefined registers
		properties.blend[0].write_mask &= static_cast<u8>(fp.output_color_masks[0]);
		properties.blend[1].write_mask &= static_cast<u8>(fp.output_color_masks[1]);
		properties.blend[2].write_mask &= static_cast<u8>(fp.output_color_masks[2]);
		properties.blend[3].write_mask &= static_cast<u8>(fp.output_color_masks[3]);
	}

	MTLTraits::pipeline_type* MTLTraits::build_pipeline(
		const vertex_program_type& vertexProgramData,
		const fragment_program_type& fragmentProgramData,
		const mtl::pipeline_props& pipelineProperties,
		bool compile_async,
		std::function<pipeline_type*(pipeline_storage_type&)> callback)
	{
		if (!vertexProgramData.handle || !fragmentProgramData.handle) [[ unlikely ]]
		{
			// Should not happen, the stages are created by recompile_*_program. Compilation failures are reported when linking.
			// Nothing can be linked; the renderer skips draws without a program.
			rsx_log.error("Cannot link program (vp id = %u, fp id = %u): missing shader stage", vertexProgramData.id, fragmentProgramData.id);

			pipeline_storage_type null_pipeline;
			return callback(null_pipeline);
		}

		const mtl::pipe_compiler::op_flags compiler_flags = compile_async ? mtl::pipe_compiler::COMPILE_DEFERRED : mtl::pipe_compiler::COMPILE_INLINE;

		// Reflection data is captured by value; deferred jobs may complete after the program cache was cleared.
		const auto apply_reflection = [
			vs_texture_mask = vertexProgramData.referenced_textures_mask,
			fs_texture_mask = fragmentProgramData.referenced_textures_mask,
			fs_stencil_mirror_mask = fragmentProgramData.stencil_mirror_mask,
			fs_uses_depth_input = fragmentProgramData.uses_depth_input,
			fs_uses_framebuffer_fetch = fragmentProgramData.uses_framebuffer_fetch](pipeline_storage_type& pipeline)
		{
			if (!pipeline)
			{
				return;
			}

			pipeline->vs_texture_mask = vs_texture_mask;
			pipeline->fs_texture_mask = fs_texture_mask;
			pipeline->fs_stencil_mirror_mask = fs_stencil_mirror_mask;
			pipeline->fs_uses_depth_input = fs_uses_depth_input;
			pipeline->fs_uses_framebuffer_fetch = fs_uses_framebuffer_fetch;
		};

		// Invoked by a compiler worker thread when the pipeline was compiled with COMPILE_DEFERRED
		mtl::pipe_compiler::callback_t compiler_callback = [apply_reflection, callback](pipeline_storage_type& pipeline)
		{
			apply_reflection(pipeline);
			callback(pipeline);
		};

		// Fragment functions that write depth cannot be linked without a depth attachment
		const mtl::deferred_function_ptr& fragment_function =
			(pipelineProperties.depth_format == MTLPixelFormatInvalid && fragmentProgramData.handle_no_depth)
				? fragmentProgramData.handle_no_depth
				: fragmentProgramData.handle;

		auto compiler = mtl::get_pipe_compiler();
		auto result = compiler->compile(
			pipelineProperties,
			vertexProgramData.handle,
			fragment_function,
			compiler_flags, compiler_callback);

		apply_reflection(result);
		return callback(result);
	}
}
