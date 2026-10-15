#include "stdafx.h"
#include "MTLHelpers.h"

#include "Emu/system_config.h"
#include "Emu/RSX/Common/bitfield.hpp"

#include "MTLCompute.h"
#include "MTLOverlays.h"
#include "MTLResolveHelper.h"
#include "MTLTexture.h"

#include "mtlutils/device.h"
#include "mtlutils/buffer.h"
#include "mtlutils/pipeline.h"
#include "mtlutils/sampler.h"

namespace mtl
{
	namespace
	{
		rsx::atomic_bitmask_t<runtime_state, u64> g_runtime_state;

		// Compatibility workarounds
		bool g_drv_no_primitive_restart = false;
		bool g_drv_sanitize_fp_values = false;
		bool g_drv_emulate_cond_render = true;
		bool g_drv_strict_query_scopes = false;
		bool g_drv_force_reuse_query_pools = false;

		u64 g_num_processed_frames = 0;
		u64 g_num_total_frames = 0;
	}

	void reset_global_resources()
	{
		// FIXME: These two shouldn't exist
		mtl::reset_resolve_resources();
		mtl::reset_overlay_passes();

		// Drops transient references held by the compute tasks (pipelines are kept)
		mtl::reset_compute_tasks();
	}

	void destroy_global_resources()
	{
		// NOTE: There are no renderpass/framebuffer caches, DMA resources, resource manager or memory allocators to tear down.
		mtl::clear_resolve_helpers();

		// vk::clear_scratch_resources also releases the typeless helpers
		mtl::clear_scratch_resources();
		mtl::clear_typeless_helpers();
		mtl::destroy_upload_heap();

		mtl::destroy_compute_tasks();
		mtl::destroy_overlay_passes();

		// Immutable state objects are tied to the device
		mtl::get_sampler_cache().clear();
		mtl::clear_depth_stencil_states();
	}

	const mtl::render_device* get_current_renderer()
	{
		return mtl::get_current_device();
	}

	void set_current_renderer(mtl::render_device& device)
	{
		mtl::set_current_device(&device);

		g_runtime_state.clear();
		g_num_processed_frames = 0;
		g_num_total_frames = 0;

		const auto& caps = device.caps();

		// Primitive restart is always available (and always enabled) for indexed draws
		g_drv_no_primitive_restart = false;

		// Apple GPUs have not shown NaN/Inf poisoning issues in RSX shaders (vk enables this for NVIDIA only)
		g_drv_sanitize_fp_values = false;

		// There is no hardware conditional rendering in Metal
		g_drv_emulate_cond_render = true;

		g_drv_strict_query_scopes = !!g_cfg.video.strict_rendering_mode;
		g_drv_force_reuse_query_pools = false;

		if (!caps.is_apple_gpu)
		{
			rsx_log.warning("Metal: Unsupported device '%s'. Only Apple silicon GPUs are supported by the Metal renderer.", caps.name);
		}

		rsx_log.notice("Metal: Renderer initialized on device '%s' (Apple GPU family %u)", caps.name, caps.apple_family);
	}

	bool emulate_primitive_restart(rsx::primitive_type type)
	{
		if (g_drv_no_primitive_restart)
		{
			switch (type)
			{
			case rsx::primitive_type::triangle_strip:
			case rsx::primitive_type::quad_strip:
				return true;
			default:
				break;
			}
		}

		return false;
	}

	bool sanitize_fp_values()
	{
		return g_drv_sanitize_fp_values;
	}

	bool emulate_conditional_rendering()
	{
		return g_drv_emulate_cond_render;
	}

	bool emulate_extended_depth_range()
	{
		// Metal clamps depth to [0, 1] (no VK_EXT_depth_range_unrestricted equivalent)
		return !!g_cfg.video.emulate_extended_depth_range;
	}

	bool use_strict_query_scopes()
	{
		return g_drv_strict_query_scopes;
	}

	bool force_reuse_query_pools()
	{
		return g_drv_force_reuse_query_pools;
	}

	void raise_status_interrupt(runtime_state status)
	{
		g_runtime_state |= status;
	}

	void clear_status_interrupt(runtime_state status)
	{
		g_runtime_state.clear(status);
	}

	bool test_status_interrupt(runtime_state status)
	{
		return g_runtime_state & status;
	}

	void enter_uninterruptible()
	{
		raise_status_interrupt(runtime_state::uninterruptible);
	}

	void leave_uninterruptible()
	{
		clear_status_interrupt(runtime_state::uninterruptible);
	}

	bool is_uninterruptible()
	{
		return test_status_interrupt(runtime_state::uninterruptible);
	}

	void advance_completed_frame_counter()
	{
		g_num_processed_frames++;
	}

	void advance_frame_counter()
	{
		ensure(g_num_processed_frames <= g_num_total_frames);
		g_num_total_frames++;
	}

	u64 get_current_frame_id()
	{
		return g_num_total_frames;
	}

	u64 get_last_completed_frame_id()
	{
		return (g_num_processed_frames > 0)? g_num_processed_frames - 1: 0;
	}
}
