#pragma once

#include "Emu/RSX/Metal/mtlutils/mtl_common.h"
#include "Emu/RSX/Metal/mtlutils/command_context.h"
#include "Emu/RSX/Metal/mtlutils/pipeline.h"
#include "Emu/RSX/Metal/mtlutils/shader.h"
#include "Emu/RSX/Metal/MTLHelpers.h" // emulate_extended_depth_range

#include "Emu/IdManager.h"
#include "Emu/RSX/Common/simple_array.hpp"
#include "Emu/RSX/Utils/algorithm.hpp"

#include "Utilities/StrUtil.h"
#include "util/asm.hpp"
#include "util/logs.hpp"

#include <memory>
#include <string>
#include <unordered_map>

/**
 * Compute helper tasks. Faithful port of VKCompute.h.
 *
 * Differences with the VK backend:
 * - Push constants are replaced by a small std140 parameter block declared as `layout(std140, set=0, binding=0) uniform`
 *   and bound with setBytes:length:atIndex:0 (see compute_task::push_params).
 * - Storage buffers are declared at set=0, binding=1.. and bound with setBuffer:offset:atIndex:(binding).
 *   Textures/images (set=1) map to texture slots, see compute_task::bind_texture.
 * - Buffers are passed as id<MTLBuffer> + byte offset. Bound offsets must be 4-byte aligned (uint arrays).
 * - Metal has no robust buffer access, so kernels that relied on the bound descriptor range to clip out-of-range
 *   invocations get explicit bounds checks through the parameter block.
 * - Metal tracks hazards on tracked resources and compute encoders dispatch serially, so VK's memory barriers are dropped.
 */
namespace mtl
{
	struct compute_task
	{
		// Metal argument table slots used by compute helper tasks
		static constexpr u32 params_slot = binding::helper_params_buffer; // set=0, binding=0: std140 parameter block (replaces push constants)
		static constexpr u32 first_ssbo_slot = 1;                          // set=0, binding=1..: storage buffers

		static constexpr u32 ssbo_slot(u32 index)
		{
			return first_ssbo_slot + index;
		}

		std::string m_src;
		mtl::shader_module m_shader;
		std::unique_ptr<mtl::program> m_program;

		bool initialized = false;
		bool unroll_loops = true;
		bool use_push_constants = false;   // The task declares a parameter block bound at slot 0
		u32 ssbo_count = 1;
		u32 push_constants_size = 0;       // Size of the parameter block in bytes
		u32 optimal_group_size = 1;
		u32 optimal_kernel_size = 1;
		u32 max_invocations_x = 65535;

		compute_task() = default;
		virtual ~compute_task() { destroy(); }

		compute_task(const compute_task&) = delete;
		compute_task& operator=(const compute_task&) = delete;

		void create();
		void destroy();

		virtual void bind_resources(mtl::command_context& /*cmd*/) {}

		// Release transient references held between runs (per-frame reset). Compiled pipelines are kept.
		virtual void free_resources() {}

		void load_program(mtl::command_context& cmd);

		void run(mtl::command_context& cmd, u32 invocations_x, u32 invocations_y, u32 invocations_z);
		void run(mtl::command_context& cmd, u32 num_invocations);

		// Binding helpers (operate on the active compute encoder of cmd)
		// Equivalent of vkCmdPushConstants: uploads the parameter block to slot 0 (padded to 16 bytes, max 4 KiB)
		static void push_params(mtl::command_context& cmd, const void* data, u32 size, u32 slot = params_slot);
		// Storage buffer binding. The offset must be 4-byte aligned.
		static void bind_buffer(mtl::command_context& cmd, id<MTLBuffer> buffer, u32 offset, u32 slot);
		// set=1, binding=N resources
		static void bind_texture(mtl::command_context& cmd, id<MTLTexture> texture, u32 slot);
		static void bind_sampler(mtl::command_context& cmd, id<MTLSamplerState> sampler, u32 slot);
	};

	struct cs_shuffle_base : compute_task
	{
		id<MTLBuffer> m_data = nil;
		u32 m_data_offset = 0;
		u32 m_data_length = 0;
		u32 kernel_size = 1;

		rsx::simple_array<u32> m_params;

		std::string variables, work_kernel, loop_advance, suffix;
		std::string method_declarations;

		cs_shuffle_base();

		void build(const char* function_name, u32 _kernel_size = 0);

		void bind_resources(mtl::command_context& cmd) override;

		void free_resources() override;

		void set_parameters(mtl::command_context& cmd);

		void run(mtl::command_context& cmd, id<MTLBuffer> data, u32 data_length, u32 data_offset = 0);
	};

	struct cs_shuffle_16 : cs_shuffle_base
	{
		// byteswap ushort
		cs_shuffle_16()
		{
			cs_shuffle_base::build("bswap_u16");
		}
	};

	struct cs_shuffle_32 : cs_shuffle_base
	{
		// byteswap_ulong
		cs_shuffle_32()
		{
			cs_shuffle_base::build("bswap_u32");
		}
	};

	struct cs_shuffle_32_16 : cs_shuffle_base
	{
		// byteswap_ulong + byteswap_ushort
		cs_shuffle_32_16()
		{
			cs_shuffle_base::build("bswap_u16_u32");
		}
	};

	struct cs_shuffle_d24x8_f32 : cs_shuffle_base
	{
		// convert d24x8 to f32
		cs_shuffle_d24x8_f32()
		{
			cs_shuffle_base::build("d24x8_to_f32");
		}
	};

	struct cs_shuffle_se_f32_d24x8 : cs_shuffle_base
	{
		// convert f32 to d24x8 and swap endianness
		cs_shuffle_se_f32_d24x8()
		{
			cs_shuffle_base::build("f32_to_d24x8_swapped");
		}
	};

	struct cs_shuffle_se_d24x8 : cs_shuffle_base
	{
		// swap endianness of d24x8
		cs_shuffle_se_d24x8()
		{
			cs_shuffle_base::build("d24x8_to_d24x8_swapped");
		}
	};

	// NOTE: D24S8 layout has the stencil in the MSB! Its actually S8|D24|S8|D24 starting at offset 0
	struct cs_interleave_task : cs_shuffle_base
	{
		u32 m_ssbo_length = 0;

		cs_interleave_task();

		void bind_resources(mtl::command_context& cmd) override;

		void run(mtl::command_context& cmd, id<MTLBuffer> data, u32 data_offset, u32 data_length, u32 zeta_offset, u32 stencil_offset);
	};

	template<bool _SwapBytes = false>
	struct cs_gather_d24x8 : cs_interleave_task
	{
		cs_gather_d24x8()
		{
			work_kernel =
				"		if (index >= block_length)\n"
				"			return;\n"
				"\n"
				"		depth = data[index + z_offset] & 0x00FFFFFF;\n"
				"		stencil_offset = (index / 4);\n"
				"		stencil_shift = (index % 4) * 8;\n"
				"		stencil = data[stencil_offset + s_offset];\n"
				"		stencil = (stencil >> stencil_shift) & 0xFF;\n"
				"		value = (depth << 8) | stencil;\n";

			if constexpr (!_SwapBytes)
			{
				work_kernel +=
				"		data[index] = value;\n";
			}
			else
			{
				work_kernel +=
				"		data[index] = bswap_u32(value);\n";
			}

			cs_shuffle_base::build("");
		}
	};

	template<bool _SwapBytes = false, bool _DepthFloat = false>
	struct cs_gather_d32x8 : cs_interleave_task
	{
		cs_gather_d32x8()
		{
			work_kernel =
				"		if (index >= block_length)\n"
				"			return;\n"
				"\n";

			if constexpr (!_DepthFloat)
			{
				work_kernel +=
				"		depth = f32_to_d24(data[index + z_offset]);\n";
			}
			else
			{
				// With depth range emulation, float depth surfaces hold the emulated encoding (half the bit pattern)
				work_kernel += !mtl::emulate_extended_depth_range()
					? "		depth = f32_to_d24f(data[index + z_offset]);\n"
					: "		depth = f32_to_d24f(data[index + z_offset] << 1);\n";
			}

			work_kernel +=
				"		stencil_offset = (index / 4);\n"
				"		stencil_shift = (index % 4) * 8;\n"
				"		stencil = data[stencil_offset + s_offset];\n"
				"		stencil = (stencil >> stencil_shift) & 0xFF;\n"
				"		value = (depth << 8) | stencil;\n";

			if constexpr (!_SwapBytes)
			{
				work_kernel +=
				"		data[index] = value;\n";
			}
			else
			{
				work_kernel +=
				"		data[index] = bswap_u32(value);\n";
			}

			cs_shuffle_base::build("");
		}
	};

	struct cs_scatter_d24x8 : cs_interleave_task
	{
		cs_scatter_d24x8();
	};

	template<bool _DepthFloat = false>
	struct cs_scatter_d32x8 : cs_interleave_task
	{
		cs_scatter_d32x8()
		{
			work_kernel =
				"		if (index >= block_length)\n"
				"			return;\n"
				"\n"
				"		value = data[index];\n";

			if constexpr (!_DepthFloat)
			{
				work_kernel +=
				"		data[index + z_offset] = d24_to_f32(value >> 8);\n";
			}
			else
			{
				work_kernel += !mtl::emulate_extended_depth_range()
					? "		data[index + z_offset] = d24f_to_f32(value >> 8);\n"
					: "		data[index + z_offset] = d24f_to_f32(value >> 8) >> 1;\n";
			}

			work_kernel +=
				"		stencil_offset = (index / 4);\n"
				"		stencil_shift = (index % 4) * 8;\n"
				"		stencil = (value & 0xFF) << stencil_shift;\n"
				"		atomicOr(data[stencil_offset + s_offset], stencil);\n";

			cs_shuffle_base::build("");
		}
	};

	template<typename From, typename To, bool _SwapSrc = false, bool _SwapDst = false>
	struct cs_fconvert_task : cs_shuffle_base
	{
		u32 m_ssbo_length = 0;

		cs_fconvert_task()
		{
			use_push_constants = true;
			push_constants_size = 16;

			variables =
				"	uint block_length = params[0].x >> 2;\n"
				"	uint in_offset = params[0].y >> 2;\n"
				"	uint out_offset = params[0].z >> 2;\n"
				"	uvec4 tmp;\n";

			if constexpr (sizeof(From) == 4)
			{
				// NOTE: We're halving the data on output, so our index is 2x as large
				work_kernel =
					"		if ((index * 2) >= block_length)\n"
					"			return;\n";

				method_declarations += "#define _CONVERT_F32_TO_E4M12 1\n";

				static_assert(sizeof(To) == 2);
				work_kernel +=
					"		const uint src_offset = (index * 2) + in_offset;\n"
					"		const uint dst_offset = index + out_offset;\n"
					"		tmp.x = data[src_offset];\n"
					"		tmp.y = data[src_offset + 1];\n";

				if constexpr (_SwapSrc)
				{
					work_kernel +=
						"		tmp = bswap_u32(tmp);\n";
				}

				if (mtl::emulate_extended_depth_range())
				{
					// Float depth surfaces hold the emulated encoding (half the bit pattern)
					work_kernel += "		tmp.xy <<= 1;\n";
				}

				// Convert
				work_kernel += "		tmp.z = pack_e4m12_pack16(tmp.xy);\n";

				if constexpr (_SwapDst)
				{
					work_kernel += "		tmp.z = bswap_u16(tmp.z);\n";
				}

				work_kernel += "		data[dst_offset] = tmp.z;\n";
			}
			else
			{
				work_kernel =
					"		if (index >= block_length)\n"
					"			return;\n";

				static_assert(sizeof(To) == 4);
				work_kernel +=
					"		const uint src_offset = index + in_offset;\n"
					"		const uint dst_offset = (index * 2) + out_offset;\n"
					"		tmp.x = data[src_offset];\n";

				if constexpr (_SwapSrc)
				{
					work_kernel +=
						"		tmp.x = bswap_u16(tmp.x);\n";
				}

				// Convert
				work_kernel += "		tmp.yz = unpack_e4m12_pack16(tmp.x);\n";

				if (mtl::emulate_extended_depth_range())
				{
					work_kernel += "		tmp.yz >>= 1;\n";
				}

				if constexpr (_SwapDst)
				{
					work_kernel += "		tmp.yz = bswap_u32(tmp.yz);\n";
				}

				work_kernel +=
					"		data[dst_offset] = tmp.y;\n"
					"		data[dst_offset + 1] = tmp.z;\n";
			}

			method_declarations +=
				#include "Emu/RSX/Program/GLSLSnippets/E4M12Conversion.glsl"
				;

			cs_shuffle_base::build("");
		}

		void bind_resources(mtl::command_context& cmd) override
		{
			set_parameters(cmd);
			bind_buffer(cmd, m_data, m_data_offset, ssbo_slot(0));
		}

		void run(mtl::command_context& cmd, id<MTLBuffer> data, u32 src_offset, u32 src_length, u32 dst_offset)
		{
			u32 data_offset;
			if (src_offset > dst_offset)
			{
				m_ssbo_length = (src_offset + src_length) - dst_offset;
				data_offset = dst_offset;
			}
			else
			{
				m_ssbo_length = (dst_offset - src_offset) + (src_length / sizeof(From)) * sizeof(To);
				data_offset = src_offset;
			}

			if ((data_offset + m_ssbo_length) > [data length])
			{
				rsx_log.error("Inadequate buffer length submitted for a compute operation."
					"Required=%d bytes, Available=%d bytes", data_offset + m_ssbo_length, [data length]);
			}

			m_params = { src_length, src_offset - data_offset, dst_offset - data_offset, 0 };
			cs_shuffle_base::run(cmd, data, src_length, data_offset);
		}
	};

	// Reverse morton-order block arrangement
	struct cs_deswizzle_base : compute_task
	{
		virtual void run(mtl::command_context& cmd, id<MTLBuffer> dst, u32 out_offset, id<MTLBuffer> src, u32 in_offset, u32 data_length, u32 width, u32 height, u32 depth, u32 mipmaps) = 0;
	};

	template <typename _BlockType, typename _BaseType, bool _SwapBytes>
	struct cs_deswizzle_3d : cs_deswizzle_base
	{
		union params_t
		{
			u32 data[7];

			struct
			{
				u32 width;
				u32 height;
				u32 depth;
				u32 logw;
				u32 logh;
				u32 logd;
				u32 mipmaps;
			};
		}
		params;

		id<MTLBuffer> src_buffer = nil;
		id<MTLBuffer> dst_buffer = nil;
		u32 in_offset = 0;
		u32 out_offset = 0;
		u32 block_length = 0;

		cs_deswizzle_3d()
		{
			ssbo_count = 2;
			use_push_constants = true;
			push_constants_size = 28;

			create();

			m_src =
			#include "Emu/RSX/Program/GLSLSnippets/GPUDeswizzle.glsl"
			;

			std::string transform;
			if constexpr (_SwapBytes)
			{
				if constexpr (sizeof(_BaseType) == 4)
				{
					transform = "bswap_u32";
				}
				else if constexpr (sizeof(_BaseType) == 2)
				{
					transform = "bswap_u16";
				}
				else
				{
					fmt::throw_exception("Unreachable");
				}
			}

			const std::pair<std::string_view, std::string> syntax_replace[] =
			{
				{ "%loc", std::to_string(first_ssbo_slot) },
				{ "%set", "set = 0" },
				{ "%push_block", "std140, set = 0, binding = " + std::to_string(params_slot) },
				{ "%ws", std::to_string(optimal_group_size) },
				{ "%_wordcount", std::to_string(std::max<u32>(sizeof(_BlockType) / 4u, 1u)) },
				{ "%f", transform },
				{ "%_8bit", sizeof(_BlockType) == 1 ? "1" : "0" },
				{ "%_16bit", sizeof(_BlockType) == 2 ? "1" : "0" },
			};

			m_src = fmt::replace_all(m_src, syntax_replace);
		}

		void bind_resources(mtl::command_context& cmd) override
		{
			set_parameters(cmd);

			bind_buffer(cmd, src_buffer, in_offset, ssbo_slot(0));
			bind_buffer(cmd, dst_buffer, out_offset, ssbo_slot(1));
		}

		void free_resources() override
		{
			src_buffer = nil;
			dst_buffer = nil;
		}

		void set_parameters(mtl::command_context& cmd)
		{
			push_params(cmd, params.data, push_constants_size);
		}

		void run(mtl::command_context& cmd, id<MTLBuffer> dst, u32 out_offset, id<MTLBuffer> src, u32 in_offset, u32 data_length, u32 width, u32 height, u32 depth, u32 mipmaps) override
		{
			dst_buffer = dst;
			src_buffer = src;

			this->in_offset = in_offset;
			this->out_offset = out_offset;
			this->block_length = data_length;

			params.width = width;
			params.height = height;
			params.depth = depth;
			params.mipmaps = mipmaps;
			params.logw = rsx::ceil_log2(width);
			params.logh = rsx::ceil_log2(height);
			params.logd = rsx::ceil_log2(depth);

			if ((in_offset + data_length) > [src length] || (out_offset + data_length) > [dst length])
			{
				rsx_log.error("Inadequate buffer length submitted for a deswizzle operation. "
					"Required=(src: %u, dst: %u) bytes, Available=(src: %u, dst: %u) bytes",
					in_offset + data_length, out_offset + data_length, [src length], [dst length]);
			}

			const u32 word_count_per_invocation = std::max<u32>(sizeof(_BlockType) / 4u, 1u);
			const u32 num_bytes_per_invocation = (word_count_per_invocation * 4u * optimal_group_size);
			const u32 workgroup_invocations = utils::aligned_div(data_length, num_bytes_per_invocation);
			compute_task::run(cmd, workgroup_invocations);

			free_resources();
		}
	};

	struct cs_aggregator : compute_task
	{
		id<MTLBuffer> src = nil;
		id<MTLBuffer> dst = nil;
		u32 src_offset = 0;
		u32 dst_offset = 0;
		u32 block_length = 0;
		u32 word_count = 0;

		cs_aggregator();

		void bind_resources(mtl::command_context& cmd) override;

		void free_resources() override;

		// Sums num_words 32-bit words from src (starting at src_offset) into the 32-bit word at dst + dst_offset
		void run(mtl::command_context& cmd, id<MTLBuffer> dst, id<MTLBuffer> src, u32 num_words, u32 dst_offset = 0, u32 src_offset = 0);
	};

	enum RSX_detiler_op
	{
		decode = 0,
		encode = 1
	};

	struct RSX_detiler_config
	{
		u32 tile_base_address;
		u32 tile_base_offset;
		u32 tile_rw_offset;
		u32 tile_size;
		u32 tile_pitch;
		u32 bank;

		id<MTLBuffer> dst;
		u32 dst_offset;
		id<MTLBuffer> src;
		u32 src_offset;

		u16 image_width;
		u16 image_height;
		u32 image_pitch;
		u8  image_bpp;
	};

	template <RSX_detiler_op Op>
	struct cs_tile_memcpy : compute_task
	{
#pragma pack (push, 1)
		struct
		{
			u32 prime;
			u32 factor;
			u32 num_tiles_per_row;
			u32 tile_base_address;
			u32 tile_size;
			u32 tile_address_offset;
			u32 tile_rw_offset;
			u32 tile_pitch;
			u32 tile_bank;
			u32 image_width;
			u32 image_height;
			u32 image_pitch;
			u32 image_bpp;
		} params;
#pragma pack (pop)

		// Metal-only: byte distance between the (4-byte aligned) bound buffer offset and the actual data start.
		// RSX surfaces with 8/16-bit texels can produce data offsets that are only 1/2-byte aligned.
		struct
		{
			u32 tiled_data_bias;
			u32 linear_data_bias;
			u32 reserved[2];
		} bias_params;

		static constexpr u32 bias_params_slot = 3; // set=0, binding=3 (after the two storage buffers)

		id<MTLBuffer> src_buffer = nil;
		id<MTLBuffer> dst_buffer = nil;
		u32 in_offset = 0;
		u32 out_offset = 0;
		u32 in_block_length = 0;
		u32 out_block_length = 0;

		cs_tile_memcpy()
		{
			ssbo_count = 2;
			use_push_constants = true;
			push_constants_size = sizeof(params);

			create();

			m_src =
			#include "Emu/RSX/Program/GLSLSnippets/RSXMemoryTiling.glsl"
				;

			const std::pair<std::string_view, std::string> syntax_replace[] =
			{
				{ "%loc", std::to_string(first_ssbo_slot) },
				{ "%set", "set = 0" },
				{ "%push_block", "std140, set = 0, binding = " + std::to_string(params_slot) },
				{ "%ws", std::to_string(optimal_group_size) },
				{ "%op", std::to_string(Op) }
			};

			m_src = fmt::replace_all(m_src, syntax_replace);

			// Data bias support (Metal requires 4-byte aligned storage buffer offsets)
			const auto version_pos = m_src.find("#version");
			const auto version_end = (version_pos == std::string::npos) ? std::string::npos : m_src.find('\n', version_pos);
			ensure(version_end != std::string::npos, "Unexpected memory tiling shader layout");

			m_src.insert(version_end + 1,
				"layout(std140, set = 0, binding = " + std::to_string(bias_params_slot) + ") uniform DataBias\n"
				"{\n"
				"	uint tiled_data_bias;\n"
				"	uint linear_data_bias;\n"
				"};\n");

			const std::pair<std::string_view, std::string> bias_replace[] =
			{
				{ "linear_image_offset /= image_bpp;", "linear_image_offset = (linear_image_offset + linear_data_bias) / image_bpp;" },
				{ "tile_data_offset /= image_bpp;", "tile_data_offset = (tile_data_offset + tiled_data_bias) / image_bpp;" },
			};

			m_src = fmt::replace_all(m_src, bias_replace);
			ensure(m_src.find("linear_data_bias) / image_bpp") != std::string::npos && m_src.find("tiled_data_bias) / image_bpp") != std::string::npos,
				"Unexpected memory tiling shader layout");
		}

		void bind_resources(mtl::command_context& cmd) override
		{
			set_parameters(cmd);

			const auto op = static_cast<u32>(Op);
			bind_buffer(cmd, src_buffer, in_offset, ssbo_slot(0u ^ op));
			bind_buffer(cmd, dst_buffer, out_offset, ssbo_slot(1u ^ op));
		}

		void free_resources() override
		{
			src_buffer = nil;
			dst_buffer = nil;
		}

		void set_parameters(mtl::command_context& cmd)
		{
			push_params(cmd, &params, push_constants_size);
			push_params(cmd, &bias_params, sizeof(bias_params), bias_params_slot);
		}

		void run(mtl::command_context& cmd, const RSX_detiler_config& config)
		{
			dst_buffer = config.dst;
			src_buffer = config.src;

			// Bind the 4-byte aligned base and pass the remainder to the shader
			ensure(config.image_bpp, "Invalid texel size");
			const u32 in_bias = config.src_offset & 3u;
			const u32 out_bias = config.dst_offset & 3u;
			ensure((in_bias % std::min<u32>(config.image_bpp, 4u)) == 0 && (out_bias % std::min<u32>(config.image_bpp, 4u)) == 0,
				"Misaligned tiled memory transfer");

			this->in_offset = config.src_offset - in_bias;
			this->out_offset = config.dst_offset - out_bias;

			const auto tile_aligned_height = std::min(
				utils::align<u32>(config.image_height, 64),
				utils::aligned_div(config.tile_size - config.tile_base_offset, config.tile_pitch)
			);

			if constexpr (Op == RSX_detiler_op::decode)
			{
				this->in_block_length = tile_aligned_height * config.tile_pitch;
				this->out_block_length = config.image_height * config.image_pitch;

				bias_params.tiled_data_bias = in_bias;
				bias_params.linear_data_bias = out_bias;
			}
			else
			{
				this->in_block_length = config.image_height * config.image_pitch;
				this->out_block_length = tile_aligned_height * config.tile_pitch;

				bias_params.tiled_data_bias = out_bias;
				bias_params.linear_data_bias = in_bias;
			}

			bias_params.reserved[0] = bias_params.reserved[1] = 0;

			if ((config.src_offset + in_block_length) > [src_buffer length] || (config.dst_offset + out_block_length) > [dst_buffer length])
			{
				rsx_log.error("Inadequate buffer length submitted for a tiled memory operation. "
					"Required=(src: %u, dst: %u) bytes, Available=(src: %u, dst: %u) bytes",
					config.src_offset + in_block_length, config.dst_offset + out_block_length, [src_buffer length], [dst_buffer length]);
			}

			auto get_prime_factor = [](u32 pitch) -> std::pair<u32, u32>
			{
				const u32 base = (pitch >> 8);
				if ((pitch & (pitch - 1)) == 0)
				{
					return { 1u, base };
				}

				for (const auto prime : { 3, 5, 7, 11, 13 })
				{
					if ((base % prime) == 0)
					{
						return { prime, base / prime };
					}
				}

				rsx_log.error("Unexpected pitch value 0x%x", pitch);
				return {};
			};

			const auto [prime, factor] = get_prime_factor(config.tile_pitch);
			const u32 tiles_per_row = prime * factor;

			params.prime = prime;
			params.factor = factor;
			params.num_tiles_per_row = tiles_per_row;
			params.tile_base_address = config.tile_base_address;
			params.tile_rw_offset = config.tile_rw_offset;
			params.tile_size = config.tile_size;
			params.tile_address_offset = config.tile_base_offset;
			params.tile_pitch = config.tile_pitch;
			params.tile_bank = config.bank;
			params.image_width = config.image_width;
			params.image_height = (Op == RSX_detiler_op::decode) ? tile_aligned_height : config.image_height;
			params.image_pitch = config.image_pitch;
			params.image_bpp = config.image_bpp;

			const u32 subtexels_per_invocation = (config.image_bpp < 4) ? (4 / config.image_bpp) : 1;
			const u32 virtual_width = config.image_width / subtexels_per_invocation;
			const u32 invocations_x = utils::aligned_div(virtual_width, optimal_group_size);
			compute_task::run(cmd, invocations_x, config.image_height, 1);

			free_resources();
		}
	};

	// TODO: Replace with a proper manager
	extern std::unordered_map<u32, std::unique_ptr<mtl::compute_task>> g_compute_tasks;

	template<class T>
	T* get_compute_task()
	{
		u32 index = stx::typeindex<id_manager::typeinfo, T>();
		auto &e = g_compute_tasks[index];

		if (!e)
		{
			e = std::make_unique<T>();
			e->create();
		}

		return static_cast<T*>(e.get());
	}

	// Per-frame reset: drops transient references held by the tasks. Compiled pipelines are kept.
	void reset_compute_tasks();

	// Destroys every compute task (renderer shutdown)
	void destroy_compute_tasks();
}
