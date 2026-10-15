#pragma once

#include "mtl_common.h"
#include "shader.h"
#include "Utilities/lockless.h"
#include "util/fnv_hash.hpp"
#include "Emu/RSX/Utils/rsx_utils.h"

#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace mtl
{
	struct blend_attachment_state
	{
		u8 enable;
		u8 write_mask;   // MTLColorWriteMask
		u8 src_rgb;      // MTLBlendFactor
		u8 dst_rgb;
		u8 src_alpha;
		u8 dst_alpha;
		u8 op_rgb;       // MTLBlendOperation
		u8 op_alpha;
	};

	/**
	 * Fixed-function state baked into an MTLRenderPipelineState.
	 * MUST stay POD with no implicit padding holes: it is hashed as raw bytes and stored verbatim in the shader cache.
	 * Always value-initialize (pipeline_props props{};) before filling.
	 */
	struct pipeline_props
	{
		u32 color_formats[4];         // MTLPixelFormat per attachment (MTLPixelFormatInvalid if unused)
		u32 depth_format;             // MTLPixelFormat of the depth attachment (Invalid if none)
		u32 stencil_format;           // MTLPixelFormat of the stencil attachment (Invalid if none)
		blend_attachment_state blend[4];
		u8 color_count;               // Number of bound color attachments
		u8 sample_count;              // Raster sample count
		u8 alpha_to_coverage;
		u8 alpha_to_one;
		u8 topology_class;            // MTLPrimitiveTopologyClass
		u8 reserved[3];

		bool operator == (const pipeline_props& other) const
		{
			return std::memcmp(this, &other, sizeof(pipeline_props)) == 0;
		}
	};

	static_assert(sizeof(pipeline_props) == 4 * 4 + 4 + 4 + 4 * 8 + 8, "pipeline_props must not contain padding");

	/**
	 * Attachment formats of a render pass. Helper passes that draw into an already open render pass
	 * (e.g. clears) need this to build a compatible pipeline.
	 */
	struct render_pass_formats
	{
		u32 color_formats[4];         // MTLPixelFormat per attachment (MTLPixelFormatInvalid if unused)
		u32 depth_format;             // MTLPixelFormatInvalid if none
		u32 stencil_format;           // MTLPixelFormatInvalid if none
		u8 color_count;
		u8 sample_count;
		u8 reserved[2];

		bool operator == (const render_pass_formats& other) const
		{
			return std::memcmp(this, &other, sizeof(render_pass_formats)) == 0;
		}

		static render_pass_formats from(const pipeline_props& props)
		{
			render_pass_formats r{};
			std::memcpy(r.color_formats, props.color_formats, sizeof(r.color_formats));
			r.depth_format = props.depth_format;
			r.stencil_format = props.stencil_format;
			r.color_count = props.color_count;
			r.sample_count = props.sample_count;
			return r;
		}

		// Fill the attachment formats of a pipeline descriptor
		void apply(MTLRenderPipelineDescriptor* desc) const;
	};

	/**
	 * A linked graphics or compute pipeline.
	 */
	class program
	{
	public:
		id<MTLRenderPipelineState> render_pipeline = nil;
		id<MTLComputePipelineState> compute_pipeline = nil;

		// Compute only
		u32 local_size[3] = { 1, 1, 1 };

		// Reflection data filled by the decompilers, used to skip unnecessary bindings
		u32 vs_texture_mask = 0;      // Bit N set if vertex texture N is referenced
		u32 fs_texture_mask = 0;      // Bit N set if fragment texture N is referenced
		u32 fs_stencil_mirror_mask = 0;
		bool fs_uses_depth_input = false;
		bool fs_uses_framebuffer_fetch = false;

		program() = default;
		program(id<MTLRenderPipelineState> pipeline) : render_pipeline(pipeline) {}
		program(id<MTLComputePipelineState> pipeline) : compute_pipeline(pipeline) {}

		bool is_compute() const { return compute_pipeline != nil; }
		explicit operator bool() const { return render_pipeline != nil || compute_pipeline != nil; }
	};

	/**
	 * Asynchronous pipeline compiler. Mirrors vk::pipe_compiler.
	 */
	class pipe_compiler
	{
	public:
		enum op_flag_bits
		{
			COMPILE_DEFAULT = 0,
			COMPILE_INLINE = 1,
			COMPILE_DEFERRED = 2,
		};

		using op_flags = rsx::flags32_t;
		using callback_t = std::function<void(std::unique_ptr<program>&)>;
		using descriptor_callback_t = std::function<MTLRenderPipelineDescriptor*()>;

		pipe_compiler() = default;

		// Graphics pipeline from RSX state. The stages are compiled by the thread that builds the pipeline if needed.
		std::unique_ptr<program> compile(
			const pipeline_props& props,
			const deferred_function_ptr& vs,
			const deferred_function_ptr& fs,
			op_flags flags,
			callback_t callback = {});

		// Graphics pipeline from an arbitrary descriptor (helper passes)
		std::unique_ptr<program> compile(
			MTLRenderPipelineDescriptor* desc,
			op_flags flags,
			callback_t callback = {});

		// Compute pipeline
		std::unique_ptr<program> compile(
			id<MTLFunction> cs,
			op_flags flags,
			callback_t callback = {});

		void operator()();

	private:
		struct pipe_compiler_job
		{
			pipeline_props props{};
			deferred_function_ptr vs;
			deferred_function_ptr fs;
			id<MTLFunction> cs = nil;
			MTLRenderPipelineDescriptor* desc = nil;
			op_flags flags = 0;
			callback_t callback;
		};

		lf_queue<pipe_compiler_job> m_work_queue;

		std::unique_ptr<program> int_compile(const pipe_compiler_job& job);
	};

	// Builds the render pipeline descriptor for an RSX draw
	MTLRenderPipelineDescriptor* make_pipeline_descriptor(const pipeline_props& props, id<MTLFunction> vs, id<MTLFunction> fs);

	// num_worker_threads <= 0 selects the worker count automatically (same heuristic as VK)
	void initialize_pipe_compiler(int num_worker_threads = 0);
	void resize_pipe_compiler(int num_worker_threads = 0);
	void destroy_pipe_compiler();
	pipe_compiler* get_pipe_compiler();

	// Pipeline binary archives: persist compiled pipeline binaries across runs. dir holds the archive layers of one game.
	// Pipelines compiled during the session are written to a new layer in the background and when the archive is closed.
	void open_pipeline_binary_archive(const std::string& dir);
	void close_pipeline_binary_archive();
	bool is_pipeline_binary_archive_open();

	/**
	 * Depth-stencil state cache. MTLDepthStencilState objects are immutable; the key is the full state.
	 */
	struct depth_stencil_desc
	{
		u8 depth_test;      // bool
		u8 depth_write;     // bool
		u8 depth_func;      // MTLCompareFunction
		u8 stencil_test;    // bool

		struct stencil_face
		{
			u8 func;        // MTLCompareFunction
			u8 fail_op;     // MTLStencilOperation
			u8 zfail_op;
			u8 pass_op;
			u8 read_mask;
			u8 write_mask;
			u8 reserved[2];
		} front, back;

		bool operator == (const depth_stencil_desc& other) const
		{
			return std::memcmp(this, &other, sizeof(depth_stencil_desc)) == 0;
		}
	};

	id<MTLDepthStencilState> get_depth_stencil_state(const depth_stencil_desc& desc);
	void clear_depth_stencil_states();
}
