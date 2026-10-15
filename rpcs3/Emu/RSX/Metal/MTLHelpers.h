#pragma once

#include "util/types.hpp"

#include "Emu/RSX/Metal/mtlutils/mtl_common.h"
#include "Emu/RSX/Metal/MTLFormats.h" // get_compatible_surface_format (declared in VKHelpers.h in the VK backend)

// NOTE: The texture section of VKHelpers.h (upload_image, copy_image*, buffer_image_copy, blitter...) lives in MTLTexture.h

namespace mtl
{
	// Forward declarations
	class render_device;

	enum runtime_state
	{
		uninterruptible = 1,
		heap_dirty      = 2,
		heap_changed    = 4,
	};

	// Mirrors vk::get_current_renderer / vk::set_current_renderer.
	// set_current_renderer() calls mtl::set_current_device() and resets all runtime state (interrupts, frame counters,
	// compatibility flags). Call it once the device is created (MTLGSRender::on_init_thread).
	const mtl::render_device* get_current_renderer();
	void set_current_renderer(mtl::render_device& device);

	// Compatibility workarounds
	// Metal always enables primitive restart for indexed draws with a fixed restart value (0xFFFF / 0xFFFFFFFF) that
	// write_index_array_data_to_buffer already substitutes for the RSX restart index, so restart never needs CPU
	// emulation. Caveat: restart cannot be disabled; when RSX restart is OFF, a u16 index buffer whose max index is 0xFFFF
	// must be promoted to u32 by the caller to avoid a spurious restart.
	bool emulate_primitive_restart(rsx::primitive_type type);
	bool sanitize_fp_values();
	bool emulate_conditional_rendering();  // Always true: Metal has no hardware conditional rendering
	bool emulate_extended_depth_range();   // Metal has no unrestricted depth range; follows g_cfg.video.emulate_extended_depth_range
	bool use_strict_query_scopes();
	bool force_reuse_query_pools();

	void destroy_global_resources();
	void reset_global_resources();

	// Runtime stuff
	void raise_status_interrupt(runtime_state status);
	void clear_status_interrupt(runtime_state status);
	bool test_status_interrupt(runtime_state status);
	void enter_uninterruptible();
	void leave_uninterruptible();
	bool is_uninterruptible();

	void advance_completed_frame_counter();
	void advance_frame_counter();
	u64 get_current_frame_id();
	u64 get_last_completed_frame_id();
}
