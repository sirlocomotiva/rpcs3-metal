#include "stdafx.h"
#include "MTLCompute.h"

#include "Emu/RSX/Metal/mtlutils/device.h"

#include <array>
#include <cmath>
#include <cstring>

namespace mtl
{
	std::unordered_map<u32, std::unique_ptr<mtl::compute_task>> g_compute_tasks;

	// Extracts the workgroup size declared in a GLSL compute shader (layout(local_size_x = N, ...) in;)
	static void parse_local_size(const std::string& src, u32 (&size)[3])
	{
		static constexpr std::string_view qualifiers[3] = { "local_size_x", "local_size_y", "local_size_z" };

		for (u32 i = 0; i < 3; ++i)
		{
			size[i] = 1;

			const auto pos = src.find(qualifiers[i]);
			if (pos == std::string::npos)
			{
				continue;
			}

			usz cursor = pos + qualifiers[i].size();
			while (cursor < src.size() && (src[cursor] == ' ' || src[cursor] == '\t' || src[cursor] == '='))
			{
				cursor++;
			}

			u32 value = 0;
			bool found = false;
			while (cursor < src.size() && src[cursor] >= '0' && src[cursor] <= '9')
			{
				value = (value * 10) + static_cast<u32>(src[cursor] - '0');
				found = true;
				cursor++;
			}

			if (found && value)
			{
				size[i] = value;
			}
		}
	}

	void compute_task::create()
	{
		if (!initialized)
		{
			// Apple GPUs execute SIMD-groups of 32 threads (threadExecutionWidth = 32).
			// Use the same configuration the VK backend picks for Apple hardware (MVK/Honeykrisp): unrolled loops,
			// one element per thread and 256-thread groups, which keeps every core fully occupied for these memory-bound kernels.
			unroll_loops = true;
			optimal_kernel_size = 1;
			optimal_group_size = 256;

			if (const auto dev = mtl::get_current_device())
			{
				const u32 max_threads = std::max<u32>(dev->caps().max_threads_per_threadgroup, 32u);
				optimal_group_size = std::min<u32>(optimal_group_size, max_threads);
			}

			// Metal does not have the small per-dimension threadgroup count limits found on some VK drivers.
			// Keep the conservative VK default anyway; large jobs are split into 2D grids which every kernel supports.
			max_invocations_x = 65535;

			initialized = true;
		}
	}

	void compute_task::destroy()
	{
		if (initialized)
		{
			m_shader = {};
			m_program.reset();

			initialized = false;
		}
	}

	void compute_task::load_program(mtl::command_context& cmd)
	{
		if (!m_program)
		{
			mtl::shader_compile_options options{};
			options.label = "compute_task";

			m_shader = mtl::compile_glsl(m_src, mtl::shader_stage::compute, options);
			if (!m_shader)
			{
				rsx_log.error("Failed to compile compute task shader. Source:\n%s", m_src);
				fmt::throw_exception("Failed to compile compute task shader");
			}

			auto compiler = mtl::get_pipe_compiler();
			m_program = compiler->compile(m_shader.function, mtl::pipe_compiler::COMPILE_INLINE);

			if (!m_program || !m_program->compute_pipeline)
			{
				fmt::throw_exception("Failed to create compute task pipeline");
			}

			// Workgroup size declared in the shader (Metal takes it at dispatch time)
			u32 local_size[3] = { m_shader.local_size_x, m_shader.local_size_y, m_shader.local_size_z };
			if (local_size[0] <= 1 && local_size[1] <= 1 && local_size[2] <= 1)
			{
				// Fall back to the declaration in the source if the compiler did not report it
				parse_local_size(m_src, local_size);
			}

			m_program->local_size[0] = std::max<u32>(local_size[0], 1);
			m_program->local_size[1] = std::max<u32>(local_size[1], 1);
			m_program->local_size[2] = std::max<u32>(local_size[2], 1);

			const u64 threads_per_group = u64{ m_program->local_size[0] } * m_program->local_size[1] * m_program->local_size[2];
			const u64 max_threads_per_group = [m_program->compute_pipeline maxTotalThreadsPerThreadgroup];

			if (threads_per_group > max_threads_per_group)
			{
				fmt::throw_exception("Compute task workgroup size (%llu) exceeds the pipeline limit (%llu)", threads_per_group, max_threads_per_group);
			}
		}

		bind_resources(cmd);
		[cmd.compute() setComputePipelineState:m_program->compute_pipeline];
	}

	void compute_task::run(mtl::command_context& cmd, u32 invocations_x, u32 invocations_y, u32 invocations_z)
	{
		if (!invocations_x || !invocations_y || !invocations_z)
		{
			// Empty dispatch, nothing to do
			return;
		}

		// Dispatch is outside render pass scope only. Requesting the compute encoder ends any open render pass.
		id<MTLComputeCommandEncoder> encoder = cmd.compute();

		load_program(cmd);

		[encoder dispatchThreadgroups:MTLSizeMake(invocations_x, invocations_y, invocations_z)
			threadsPerThreadgroup:MTLSizeMake(m_program->local_size[0], m_program->local_size[1], m_program->local_size[2])];
	}

	void compute_task::run(mtl::command_context& cmd, u32 num_invocations)
	{
		u32 invocations_x, invocations_y;
		if (num_invocations > max_invocations_x)
		{
			// Split the 1D job into 2 dimensions to stay within the threadgroup count limits
			invocations_x = static_cast<u32>(floor(std::sqrt(num_invocations)));
			invocations_y = invocations_x;

			if (num_invocations % invocations_x) invocations_y++;
		}
		else
		{
			invocations_x = num_invocations;
			invocations_y = 1;
		}

		run(cmd, invocations_x, invocations_y, 1);
	}

	void compute_task::push_params(mtl::command_context& cmd, const void* data, u32 size, u32 slot)
	{
		ensure(data && size);

		// std140 blocks are sized in multiples of 16 bytes
		const u32 padded_size = utils::align<u32>(size, 16);
		ensure(padded_size <= 4096, "Parameter block is too large for setBytes");

		id<MTLComputeCommandEncoder> encoder = cmd.compute();

		if (padded_size == size)
		{
			[encoder setBytes:data length:size atIndex:slot];
			return;
		}

		alignas(16) std::array<u8, 4096> padded;
		std::memcpy(padded.data(), data, size);
		std::memset(padded.data() + size, 0, padded_size - size);
		[encoder setBytes:padded.data() length:padded_size atIndex:slot];
	}

	void compute_task::bind_buffer(mtl::command_context& cmd, id<MTLBuffer> buffer, u32 offset, u32 slot)
	{
		ensure(buffer != nil, "Null storage buffer binding");

		// Storage buffers are declared as uint arrays; device-space bindings must be aligned to the element type
		ensure((offset & 3u) == 0, "Storage buffer offsets must be 4-byte aligned");

		[cmd.compute() setBuffer:buffer offset:offset atIndex:slot];
	}

	void compute_task::bind_texture(mtl::command_context& cmd, id<MTLTexture> texture, u32 slot)
	{
		[cmd.compute() setTexture:texture atIndex:slot];
	}

	void compute_task::bind_sampler(mtl::command_context& cmd, id<MTLSamplerState> sampler, u32 slot)
	{
		[cmd.compute() setSamplerState:sampler atIndex:slot];
	}

	cs_shuffle_base::cs_shuffle_base()
	{
		// NOTE: Metal has no robust buffer access. VK relies on the bound descriptor range to discard the invocations
		// past the end of the data block (the dispatch is rounded up to the workgroup size), here it is done explicitly.
		work_kernel =
			"		if (index >= data_range.x)\n"
			"			return;\n"
			"\n"
			"		value = data[index];\n"
			"		data[index] = %f(value);\n";

		loop_advance =
			"		index++;\n";

		suffix =
			"}\n";
	}

	void cs_shuffle_base::build(const char* function_name, u32 _kernel_size)
	{
		// Initialize to allow detecting optimal settings
		create();

		kernel_size = _kernel_size? _kernel_size : optimal_kernel_size;

		m_src =
		#include "Emu/RSX/Program/GLSLSnippets/ShuffleBytes.glsl"
		;

		// Parameter block replacing the VK push constants. Always present since it also carries the bound data range.
		// params[] mirrors the VK push constant layout, data_range.x is the length of the bound data range in words.
		const auto parameters_size = use_push_constants ? utils::align(push_constants_size, 16) / 16 : 0;
		std::string parameter_block = "layout(std140, set = 0, binding = " + std::to_string(params_slot) + ") uniform ubo\n{\n";
		if (parameters_size)
		{
			parameter_block += "	uvec4 params[" + std::to_string(parameters_size) + "];\n";
		}
		parameter_block += "	uvec4 data_range;\n};\n";

		const std::pair<std::string_view, std::string> syntax_replace[] =
		{
			{ "%loc", std::to_string(ssbo_slot(0)) },
			{ "%set", "set = 0"},
			{ "%ws", std::to_string(optimal_group_size) },
			{ "%ks", std::to_string(kernel_size) },
			{ "%vars", variables },
			{ "%f", function_name },
			{ "%md", method_declarations },
			{ "%ub", parameter_block },
		};

		m_src = fmt::replace_all(m_src, syntax_replace);
		work_kernel = fmt::replace_all(work_kernel, syntax_replace);

		if (kernel_size <= 1)
		{
			m_src += "	{\n" + work_kernel + "	}\n";
		}
		else if (unroll_loops)
		{
			work_kernel += loop_advance + "\n";

			m_src += std::string
			(
				"	//Unrolled loop\n"
				"	{\n"
			);

			// Assemble body with manual loop unroll to try loweing GPR usage
			for (u32 n = 0; n < kernel_size; ++n)
			{
				m_src += work_kernel;
			}

			m_src += "	}\n";
		}
		else
		{
			m_src += "	for (int loop = 0; loop < KERNEL_SIZE; ++loop)\n";
			m_src += "	{\n";
			m_src += work_kernel;
			m_src += loop_advance;
			m_src += "	}\n";
		}

		m_src += suffix;
	}

	void cs_shuffle_base::bind_resources(mtl::command_context& cmd)
	{
		set_parameters(cmd);
		bind_buffer(cmd, m_data, m_data_offset, ssbo_slot(0));
	}

	void cs_shuffle_base::free_resources()
	{
		m_data = nil;
	}

	void cs_shuffle_base::set_parameters(mtl::command_context& cmd)
	{
		const u32 parameters_size = use_push_constants ? utils::align<u32>(push_constants_size, 16) : 0;

		// params[] (VK push constants) followed by data_range
		std::array<u32, 1024> block{};
		ensure((parameters_size + 16) <= sizeof(block));

		if (!m_params.empty())
		{
			ensure(use_push_constants);
			ensure(m_params.size_bytes32() <= parameters_size);
			std::memcpy(block.data(), m_params.data(), m_params.size_bytes32());
		}

		block[parameters_size / 4] = m_data_length / 4;
		push_params(cmd, block.data(), parameters_size + 16);
	}

	void cs_shuffle_base::run(mtl::command_context& cmd, id<MTLBuffer> data, u32 data_length, u32 data_offset)
	{
		ensure(data != nil);

		m_data = data;
		m_data_offset = data_offset;
		m_data_length = data_length;

		const auto num_bytes_per_invocation = optimal_group_size * kernel_size * 4;
		const auto num_bytes_to_process = rsx::align2(data_length, num_bytes_per_invocation);
		const auto num_invocations = num_bytes_to_process / num_bytes_per_invocation;

		if ((u64{ data_offset } + data_length) > [data length])
		{
			// There is no robust buffer access in Metal, this would read/write out of bounds
			rsx_log.error("Inadequate buffer length submitted for a compute operation."
				"Required=%d bytes, Available=%d bytes", u64{ data_offset } + data_length, [data length]);
		}

		compute_task::run(cmd, num_invocations);

		free_resources();
	}

	cs_interleave_task::cs_interleave_task()
	{
		use_push_constants = true;
		push_constants_size = 16;

		variables =
			"	uint block_length = params[0].x >> 2;\n"
			"	uint z_offset = params[0].y >> 2;\n"
			"	uint s_offset = params[0].z >> 2;\n"
			"	uint depth;\n"
			"	uint stencil;\n"
			"	uint stencil_shift;\n"
			"	uint stencil_offset;\n";
	}

	void cs_interleave_task::bind_resources(mtl::command_context& cmd)
	{
		set_parameters(cmd);
		bind_buffer(cmd, m_data, m_data_offset, ssbo_slot(0));
	}

	void cs_interleave_task::run(mtl::command_context& cmd, id<MTLBuffer> data, u32 data_offset, u32 data_length, u32 zeta_offset, u32 stencil_offset)
	{
		m_params = { data_length, zeta_offset - data_offset, stencil_offset - data_offset, 0 };

		ensure(stencil_offset > data_offset);
		m_ssbo_length = stencil_offset + (data_length / 4) - data_offset;

		if ((u64{ data_offset } + m_ssbo_length) > [data length])
		{
			rsx_log.error("Inadequate buffer length submitted for an interleave operation."
				"Required=%d bytes, Available=%d bytes", u64{ data_offset } + m_ssbo_length, [data length]);
		}

		cs_shuffle_base::run(cmd, data, data_length, data_offset);
	}

	cs_scatter_d24x8::cs_scatter_d24x8()
	{
		work_kernel =
			"		if (index >= block_length)\n"
			"			return;\n"
			"\n"
			"		value = data[index];\n"
			"		data[index + z_offset] = (value >> 8);\n"
			"		stencil_offset = (index / 4);\n"
			"		stencil_shift = (index % 4) * 8;\n"
			"		stencil = (value & 0xFF) << stencil_shift;\n"
			"		atomicOr(data[stencil_offset + s_offset], stencil);\n";

		cs_shuffle_base::build("");
	}

	cs_aggregator::cs_aggregator()
	{
		ssbo_count = 2;
		use_push_constants = true;
		push_constants_size = 16;

		create();

		// NOTE: VK bounds the loop with src.length(). Runtime array lengths need an extra buffer-size binding in MSL,
		// so the word count is passed through the parameter block instead. The invocation index is linearized
		// since large jobs are split into 2D grids.
		m_src =
			"#version 450\n"
			"layout(local_size_x = %ws, local_size_y = 1, local_size_z = 1) in;\n\n"

			"layout(set=0, binding=%params, std140) uniform parameters{ uvec4 params; };\n"
			"layout(set=0, binding=%src, std430) readonly buffer ssbo0{ uint src[]; };\n"
			"layout(set=0, binding=%dst, std430) writeonly buffer ssbo1{ uint result; };\n\n"

			"void main()\n"
			"{\n"
			"	const uint index = (gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x) + gl_GlobalInvocationID.x;\n"
			"	if (index < params.x)\n"
			"	{\n"
			"		atomicAdd(result, src[index]);\n"
			"	}\n"
			"}\n";

		const std::pair<std::string_view, std::string> syntax_replace[] =
		{
			{ "%ws", std::to_string(optimal_group_size) },
			{ "%params", std::to_string(params_slot) },
			{ "%src", std::to_string(ssbo_slot(0)) },
			{ "%dst", std::to_string(ssbo_slot(1)) },
		};

		m_src = fmt::replace_all(m_src, syntax_replace);
	}

	void cs_aggregator::bind_resources(mtl::command_context& cmd)
	{
		const u32 parameters[4] = { word_count, 0, 0, 0 };
		push_params(cmd, parameters, sizeof(parameters));

		bind_buffer(cmd, src, src_offset, ssbo_slot(0));
		bind_buffer(cmd, dst, dst_offset, ssbo_slot(1));
	}

	void cs_aggregator::free_resources()
	{
		src = nil;
		dst = nil;
	}

	void cs_aggregator::run(mtl::command_context& cmd, id<MTLBuffer> dst, id<MTLBuffer> src, u32 num_words, u32 dst_offset, u32 src_offset)
	{
		this->dst = dst;
		this->src = src;
		this->dst_offset = dst_offset;
		this->src_offset = src_offset;
		word_count = num_words;
		block_length = num_words * 4;

		if ((u64{ src_offset } + block_length) > [src length] || (u64{ dst_offset } + 4) > [dst length])
		{
			rsx_log.error("Inadequate buffer length submitted for an aggregation operation. "
				"Required=(src: %u, dst: %u) bytes, Available=(src: %u, dst: %u) bytes",
				u64{ src_offset } + block_length, u64{ dst_offset } + 4, [src length], [dst length]);
		}

		const u32 linear_invocations = utils::aligned_div(word_count, optimal_group_size);
		compute_task::run(cmd, linear_invocations);

		free_resources();
	}

	void reset_compute_tasks()
	{
		for (const auto& [key, task] : g_compute_tasks)
		{
			task->free_resources();
		}
	}

	void destroy_compute_tasks()
	{
		g_compute_tasks.clear();
	}
}
