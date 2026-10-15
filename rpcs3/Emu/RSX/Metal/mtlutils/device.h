#pragma once

#include "mtl_common.h"

namespace rsx
{
	enum class problem_severity : u8;
}

namespace mtl
{
	struct device_caps
	{
		std::string name;

		// Apple GPU family (e.g. 7 for M1, 8 for M2, 9 for M3/M4). 0 if not an Apple GPU.
		u32 apple_family = 0;
		bool is_apple_gpu = false;
		bool has_unified_memory = false;

		// Feature support
		bool supports_bc_texture_compression = false;
		bool supports_depth24_stencil8 = false;          // Never available on Apple silicon
		bool supports_framebuffer_fetch = false;         // [[color(n)]] inputs (programmable blending)
		bool supports_raster_order_groups = false;
		bool supports_barycentrics = false;              // [[barycentric_coord]]
		bool supports_packed_16bit_formats = false;      // B5G6R5, A1BGR5, ABGR4 etc
		bool supports_32bit_float_filtering = false;
		bool supports_query_texture_lod = false;
		bool supports_binary_archives = false;
		bool supports_metalfx_spatial = false;
		bool supports_metalfx_temporal = false;
		bool supports_visibility_accumulate = false;     // MTLVisibilityResultTypeAccumulate (macOS 15+)
		bool supports_shader_validation = false;

		// Limits
		u64 max_buffer_length = 0;
		u64 recommended_working_set_size = 0;
		u32 max_texture_size_2d = 16384;
		u32 max_texture_buffer_width = 1u << 26;
		u32 texture_buffer_alignment = 16;               // minimumTextureBufferAlignmentForPixelFormat(R8Uint)
		u32 linear_texture_alignment = 16;
		u32 max_threads_per_threadgroup = 1024;
		u32 max_anisotropy = 16;
	};

	class render_device
	{
		id<MTLDevice> m_device = nil;
		id<MTLCommandQueue> m_queue = nil;
		device_caps m_caps{};

	public:
		render_device() = default;
		~render_device();

		render_device(const render_device&) = delete;
		render_device& operator=(const render_device&) = delete;

		// Create the default system device and the primary command queue
		bool create();
		void destroy();

		id<MTLDevice> get() const { return m_device; }
		id<MTLCommandQueue> queue() const { return m_queue; }
		const device_caps& caps() const { return m_caps; }

		operator id<MTLDevice>() const { return m_device; }
	};

	// Global device accessors. Valid between MTLGSRender::on_init_thread and on_exit.
	void set_current_device(render_device* dev);
	render_device* get_current_device();

	inline id<MTLDevice> get_device()
	{
		return get_current_device()->get();
	}

	inline const device_caps& get_caps()
	{
		return get_current_device()->caps();
	}

	// Human readable name of the system default Metal device. Safe to call at any time.
	std::string get_default_device_name();

	// Memory pressure estimate (mirrors vk::vmm_determine_memory_load_severity).
	// Compares [MTLDevice currentAllocatedSize] against the recommended working set size.
	rsx::problem_severity vmm_determine_memory_load_severity();
}
