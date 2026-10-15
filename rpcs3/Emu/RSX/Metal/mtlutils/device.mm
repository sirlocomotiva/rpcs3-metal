#include "stdafx.h"
#include "device.h"
#include "Emu/RSX/Utils/rsx_utils.h"

#import <MetalFX/MetalFX.h>

namespace mtl
{
	namespace
	{
		render_device* g_current_device = nullptr;

		// Maximum number of command buffers that can be in flight on the primary queue
		constexpr NSUInteger c_max_command_buffer_count = 64;

		u32 query_apple_family(id<MTLDevice> device)
		{
			// Highest supported family first
			for (u32 family = 11; family >= 1; --family)
			{
				const auto gpu_family = static_cast<MTLGPUFamily>(static_cast<NSInteger>(MTLGPUFamilyApple1) + family - 1);
				if ([device supportsFamily:gpu_family])
				{
					return family;
				}
			}

			return 0;
		}

		bool query_env_flag(const char* name)
		{
			const char* value = std::getenv(name);
			return value && value[0] != '\0' && value[0] != '0';
		}
	}

	render_device::~render_device()
	{
		destroy();
	}

	bool render_device::create()
	{
		ensure(!m_device, "Metal device already created");

		@autoreleasepool
		{
			m_device = MTLCreateSystemDefaultDevice();
			if (!m_device)
			{
				rsx_log.fatal("Metal: No Metal capable device could be found. The Metal renderer cannot be used on this system.");
				return false;
			}

			m_queue = [m_device newCommandQueueWithMaxCommandBufferCount:c_max_command_buffer_count];
			if (!m_queue)
			{
				rsx_log.fatal("Metal: Failed to create the primary command queue on device '%s'", from_ns_string(m_device.name));
				m_device = nil;
				return false;
			}

			m_queue.label = @"RSX primary queue";

			// Let the driver use as many threads as possible when compiling shaders and pipelines (we compile on many threads)
			if (@available(macOS 13.3, *))
			{
				m_device.shouldMaximizeConcurrentCompilation = YES;
			}

			auto& caps = m_caps;
			caps = {};
			caps.name = from_ns_string(m_device.name);

			caps.apple_family = query_apple_family(m_device);
			caps.is_apple_gpu = caps.apple_family != 0;
			caps.has_unified_memory = m_device.hasUnifiedMemory;

			// Features
			caps.supports_bc_texture_compression = m_device.supportsBCTextureCompression;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
			caps.supports_depth24_stencil8 = m_device.depth24Stencil8PixelFormatSupported;
#pragma clang diagnostic pop

			// Programmable blending ([[color(n)]] inputs) and the packed 16-bit formats are available on all Apple GPU families
			caps.supports_framebuffer_fetch = caps.is_apple_gpu;
			caps.supports_packed_16bit_formats = caps.is_apple_gpu;
			caps.supports_raster_order_groups = m_device.rasterOrderGroupsSupported;
			caps.supports_barycentrics = m_device.supportsShaderBarycentricCoordinates;
			caps.supports_32bit_float_filtering = m_device.supports32BitFloatFiltering;
			caps.supports_query_texture_lod = m_device.supportsQueryTextureLOD;
			caps.supports_binary_archives = caps.is_apple_gpu || [m_device supportsFamily:MTLGPUFamilyMetal3];
			caps.supports_metalfx_spatial = [MTLFXSpatialScalerDescriptor supportsDevice:m_device];
			caps.supports_metalfx_temporal = [MTLFXTemporalScalerDescriptor supportsDevice:m_device];

			if (@available(macOS 26.0, *))
			{
				// MTLRenderPassDescriptor.visibilityResultType
				caps.supports_visibility_accumulate = true;
			}

			// Shader validation is a debug layer toggled through the environment
			caps.supports_shader_validation = query_env_flag("MTL_SHADER_VALIDATION");

			// Limits
			caps.max_buffer_length = static_cast<u64>(m_device.maxBufferLength);
			caps.recommended_working_set_size = static_cast<u64>(m_device.recommendedMaxWorkingSetSize);
			caps.max_texture_size_2d = 16384; // All Apple silicon GPUs (Apple3+) support 16K textures
			caps.texture_buffer_alignment = static_cast<u32>([m_device minimumTextureBufferAlignmentForPixelFormat:MTLPixelFormatR8Uint]);

			u32 linear_alignment = 1;
			for (const MTLPixelFormat format : { MTLPixelFormatR8Uint, MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA16Float, MTLPixelFormatRGBA32Float })
			{
				linear_alignment = std::max(linear_alignment, static_cast<u32>([m_device minimumLinearTextureAlignmentForPixelFormat:format]));
			}
			caps.linear_texture_alignment = linear_alignment;

			const MTLSize max_threads = m_device.maxThreadsPerThreadgroup;
			caps.max_threads_per_threadgroup = static_cast<u32>(std::max<NSUInteger>({ max_threads.width, max_threads.height, max_threads.depth }));
			caps.max_anisotropy = 16;

			if (!caps.texture_buffer_alignment)
			{
				caps.texture_buffer_alignment = 16;
			}

			if (!caps.linear_texture_alignment)
			{
				caps.linear_texture_alignment = 16;
			}

			rsx_log.notice("Metal: Using device '%s' (Apple GPU family %u, unified memory: %s, recommended working set: %u MiB, max buffer length: %u MiB)",
				caps.name, caps.apple_family, caps.has_unified_memory ? "yes" : "no",
				caps.recommended_working_set_size / 0x100000, caps.max_buffer_length / 0x100000);

			rsx_log.notice("Metal: Features: BC=%s, D24S8=%s, framebuffer_fetch=%s, ROG=%s, barycentrics=%s, float32_filtering=%s, query_lod=%s, binary_archives=%s, MetalFX spatial=%s, MetalFX temporal=%s, visibility_accumulate=%s, shader_validation=%s",
				caps.supports_bc_texture_compression, caps.supports_depth24_stencil8, caps.supports_framebuffer_fetch,
				caps.supports_raster_order_groups, caps.supports_barycentrics, caps.supports_32bit_float_filtering,
				caps.supports_query_texture_lod, caps.supports_binary_archives, caps.supports_metalfx_spatial,
				caps.supports_metalfx_temporal, caps.supports_visibility_accumulate, caps.supports_shader_validation);

			rsx_log.notice("Metal: Limits: texture buffer alignment=%u, linear texture alignment=%u, max threads per threadgroup=%u",
				caps.texture_buffer_alignment, caps.linear_texture_alignment, caps.max_threads_per_threadgroup);

			if (!caps.is_apple_gpu)
			{
				rsx_log.error("Metal: Device '%s' is not an Apple GPU. The Metal renderer only supports Apple silicon; expect missing features.", caps.name);
			}
			else if (caps.apple_family < 7)
			{
				rsx_log.error("Metal: Apple GPU family %u is older than the minimum supported family (Apple7).", caps.apple_family);
			}
		}

		return true;
	}

	void render_device::destroy()
	{
		if (g_current_device == this)
		{
			g_current_device = nullptr;
		}

		m_queue = nil;
		m_device = nil;
		m_caps = {};
	}

	void set_current_device(render_device* dev)
	{
		g_current_device = dev;
	}

	render_device* get_current_device()
	{
		return g_current_device;
	}

	std::string get_default_device_name()
	{
		@autoreleasepool
		{
			if (g_current_device && g_current_device->get())
			{
				return g_current_device->caps().name;
			}

			id<MTLDevice> device = MTLCreateSystemDefaultDevice();
			return device ? from_ns_string(device.name) : std::string{};
		}
	}

	rsx::problem_severity vmm_determine_memory_load_severity()
	{
		const auto dev = get_current_device();
		if (!dev || !dev->get())
		{
			return rsx::problem_severity::low;
		}

		const u64 budget = dev->caps().recommended_working_set_size;
		if (!budget)
		{
			return rsx::problem_severity::low;
		}

		const u64 usage = [dev->get() currentAllocatedSize];
		const double load = (static_cast<double>(usage) * 100.) / static_cast<double>(budget);

		if (load > 95.)
		{
			return rsx::problem_severity::fatal;
		}

		if (load > 90.)
		{
			return rsx::problem_severity::severe;
		}

		if (load > 75.)
		{
			return rsx::problem_severity::moderate;
		}

		return rsx::problem_severity::low;
	}
}
