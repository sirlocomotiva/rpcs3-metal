#include "stdafx.h"
#include "MTLGSRender.h"
#include "../Overlays/overlay_compile_notification.h"
#include "../Overlays/Shaders/shader_loading_dialog_native.h"

#include "mtlutils/shader.h"

#include "Emu/RSX/rsx_methods.h"
#include "Emu/RSX/RSXOffload.h"
#include "Emu/RSX/Host/MM.h"
#include "Emu/RSX/NV47/HW/context_accessors.define.h"
#include "Emu/Memory/vm_locking.h"
#include "Emu/Cell/timers.hpp"
#include "Emu/cache_utils.hpp"
#include "Emu/system_config.h"
#include "Utilities/File.h"
#include "Utilities/Thread.h"

#include "util/asm.hpp"
#include "util/sysinfo.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>
#include <span>

// Metal port of VKGSRender.cpp (module H1).
//
// Differences to VK (see the porting guide):
// - There is a single command stream (mtl::command_context m_cmd). Command buffers are created on demand and identified by
//   sync ids; waits and frame pacing use those ids instead of fences and semaphores. Per command buffer flags are tracked by
//   get_cb_flags(), which also notices command buffers committed by other modules.
// - The main render pass (m_draw_fbo) is opened lazily by begin_render_pass() and stays open until another encoder is needed.
//   Full-surface clears are folded into the load actions of the pass when possible (VK: vkCmdClearAttachments), partial
//   clears are drawn into the open pass (attachment_clear_pass).
// - Occlusion queries use a visibility result buffer (query_pool_manager below). Visibility counting is render encoder
//   state, see begin_render_pass() for how queries survive pass interruptions.
// - Small uniform blocks are uploaded with set*Bytes from CPU shadow copies (guide 9.11). This module fills the shadow
//   copies and ring offsets, MTLDraw.mm (H2) binds them.
// - Memory faults on threads other than the RSX thread hand GPU readbacks to the RSX thread (GL work queue), since only the
//   RSX thread records into m_cmd.
// - There is no shader interpreter (interpreter shader modes use the asynchronous recompiler), no hardware conditional
//   rendering, no host GPU labels, no passthrough DMA and no asynchronous compute queue.
//
// GPU capture: RPCS3_METAL_CAPTURE=N captures the N-th presented frame (1-based) into a .gputrace document in
// <cache dir>/metal_captures. Metal only allows programmatic captures when the process runs with MTL_CAPTURE_ENABLED=1.

using cbf = mtl::command_buffer_flags;

namespace
{
	// On-disk shader cache identification (rsx::shaders_cache class and version) and the pipeline binary archive kept next to it
	constexpr const char* shader_cache_class_name = "metal";
	constexpr const char* shader_cache_version = "v1.00";

	// ---------------------------------------------------------------------------------------------------------------
	// Programmatic GPU capture (RPCS3_METAL_CAPTURE)
	// ---------------------------------------------------------------------------------------------------------------
	struct gpu_capture_state_t
	{
		u64 target_frame = 0;       // 1-based index of the presented frame to capture, 0 = disabled
		u64 presented_frames = 0;   // Number of frames presented so far
		bool active = false;
	};

	gpu_capture_state_t g_gpu_capture{};

	void gpu_capture_init()
	{
		g_gpu_capture = {};

		const char* value = std::getenv("RPCS3_METAL_CAPTURE");
		if (!value || !*value)
		{
			return;
		}

		char* end = nullptr;
		const unsigned long long frame = std::strtoull(value, &end, 10);

		if (end == value || *end != '\0' || frame == 0)
		{
			rsx_log.error("Metal: Invalid RPCS3_METAL_CAPTURE value '%s'. Expected the index (starting at 1) of the presented frame to capture.", value);
			return;
		}

		g_gpu_capture.target_frame = frame;

		if (const char* capture_enabled = std::getenv("MTL_CAPTURE_ENABLED"); !capture_enabled || std::strcmp(capture_enabled, "1") != 0)
		{
			rsx_log.warning("Metal: GPU capture of frame %u requested, but MTL_CAPTURE_ENABLED=1 is not set. "
				"Metal only allows programmatic captures when the process is started with MTL_CAPTURE_ENABLED=1; the capture will most likely fail.", frame);
		}
		else
		{
			rsx_log.notice("Metal: GPU capture of presented frame %u requested (RPCS3_METAL_CAPTURE)", frame);
		}
	}

	void gpu_capture_start(id<MTLDevice> device)
	{
		if (g_gpu_capture.active || !g_gpu_capture.target_frame || !device)
		{
			return;
		}

		const u64 frame = g_gpu_capture.target_frame;

		@autoreleasepool
		{
			MTLCaptureManager* manager = [MTLCaptureManager sharedCaptureManager];

			if (manager.isCapturing)
			{
				rsx_log.error("Metal: Cannot capture frame %u, another GPU capture is already in progress", frame);
				g_gpu_capture.target_frame = 0;
				return;
			}

			if (![manager supportsDestination:MTLCaptureDestinationGPUTraceDocument])
			{
				rsx_log.error("Metal: GPU trace documents are not supported. Start RPCS3 with MTL_CAPTURE_ENABLED=1 to capture frame %u.", frame);
				g_gpu_capture.target_frame = 0;
				return;
			}

			std::string directory = fs::get_cache_dir();
			if (!directory.empty() && directory.back() != '/')
			{
				directory += '/';
			}

			directory += "metal_captures";

			if (!fs::is_dir(directory) && !fs::create_path(directory))
			{
				rsx_log.error("Metal: Failed to create the GPU capture directory '%s' (%s)", directory, fs::g_tls_error);
				g_gpu_capture.target_frame = 0;
				return;
			}

			// The capture fails if the document already exists
			const std::string path = fmt::format("%s/frame_%u_%u.gputrace", directory, frame, get_system_time());

			MTLCaptureDescriptor* desc = [MTLCaptureDescriptor new];
			desc.captureObject = device;
			desc.destination = MTLCaptureDestinationGPUTraceDocument;
			desc.outputURL = [NSURL fileURLWithPath:mtl::to_ns_string(path)];

			NSError* error = nil;
			if (![manager startCaptureWithDescriptor:desc error:&error])
			{
				rsx_log.error("Metal: Failed to start the GPU capture of frame %u: %s (MTL_CAPTURE_ENABLED=1 is required)", frame, mtl::describe_error(error));
				g_gpu_capture.target_frame = 0;
				return;
			}

			g_gpu_capture.active = true;
			rsx_log.notice("Metal: Capturing presented frame %u to '%s'", frame, path);
		}
	}

	void gpu_capture_stop()
	{
		if (!g_gpu_capture.active)
		{
			return;
		}

		@autoreleasepool
		{
			[[MTLCaptureManager sharedCaptureManager] stopCapture];
		}

		rsx_log.notice("Metal: GPU capture of frame %u finished", g_gpu_capture.target_frame);

		// Capture only once
		g_gpu_capture.active = false;
		g_gpu_capture.target_frame = 0;
	}

	// Called after every command buffer that presents a drawable was committed
	void gpu_capture_on_frame_presented(id<MTLDevice> device)
	{
		if (!g_gpu_capture.target_frame)
		{
			return;
		}

		const u64 presented = ++g_gpu_capture.presented_frames;

		if (g_gpu_capture.active)
		{
			if (presented >= g_gpu_capture.target_frame)
			{
				gpu_capture_stop();
			}
		}
		else if (presented + 1 == g_gpu_capture.target_frame)
		{
			// The next frame is the one to capture
			gpu_capture_start(device);
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Pipeline state (VK: vk::decode_rsx_state)
	// Depth/stencil, rasterizer and stencil MSAA bookkeeping are dynamic state on Metal (MTLDraw.mm, update_draw_state).
	// ---------------------------------------------------------------------------------------------------------------
	mtl::pipeline_props decode_rsx_state(
		const rsx::context* ctx,
		const mtl::draw_framebuffer& fbo,
		const rsx::backend_configuration& backend_config,
		u8 topology_class,
		bool force_disable_blending)
	{
		mtl::pipeline_props properties{};

		// Attachments
		std::memcpy(properties.color_formats, fbo.formats.color_formats, sizeof(properties.color_formats));
		properties.depth_format = fbo.formats.depth_format;
		properties.stencil_format = fbo.formats.stencil_format;
		properties.color_count = fbo.formats.color_count;
		properties.sample_count = fbo.formats.sample_count;

		// Input assembly. Primitive restart is always enabled for indexed draws on Metal (see MTLHelpers.h).
		properties.topology_class = topology_class;

		const u8 num_draw_buffers = fbo.formats.color_count;
		const auto host_write_mask = rsx::get_write_output_mask(REGS(ctx)->surface_color());

		for (u8 index = 0; index < num_draw_buffers; ++index)
		{
			bool color_mask_b = REGS(ctx)->color_mask_b(index);
			bool color_mask_g = REGS(ctx)->color_mask_g(index);
			bool color_mask_r = REGS(ctx)->color_mask_r(index);
			bool color_mask_a = REGS(ctx)->color_mask_a(index);

			switch (REGS(ctx)->surface_color())
			{
			case rsx::surface_color_format::b8:
				rsx::get_b8_colormask(color_mask_r, color_mask_g, color_mask_b, color_mask_a);
				break;
			case rsx::surface_color_format::g8b8:
				rsx::get_g8b8_r8g8_colormask(color_mask_r, color_mask_g, color_mask_b, color_mask_a);
				break;
			default:
				break;
			}

			u8 write_mask = 0;
			if (color_mask_r && host_write_mask[0]) write_mask |= MTLColorWriteMaskRed;
			if (color_mask_g && host_write_mask[1]) write_mask |= MTLColorWriteMaskGreen;
			if (color_mask_b && host_write_mask[2]) write_mask |= MTLColorWriteMaskBlue;
			if (color_mask_a && host_write_mask[3]) write_mask |= MTLColorWriteMaskAlpha;

			properties.blend[index].write_mask = write_mask;
		}

		// LogicOp and Blend are mutually exclusive. If both are enabled, LogicOp takes precedence.
		if (REGS(ctx)->logic_op_enabled())
		{
			// Metal has no logic operations. Blending stays disabled, which matches the common logic_copy case.
			static bool s_logic_op_reported = false;
			if (!s_logic_op_reported)
			{
				s_logic_op_reported = true;
				rsx_log.warning("Metal: Logic operations are not supported (logic op 0x%x). Further occurrences are not reported.",
					static_cast<u32>(REGS(ctx)->logic_operation()));
			}
		}
		else if (!force_disable_blending)
		{
			if (const auto blend_enabled = REGS(ctx)->blend_enabled_mask())
			{
				const u8 sfactor_rgb = static_cast<u8>(mtl::get_blend_factor(REGS(ctx)->blend_func_sfactor_rgb()));
				const u8 sfactor_a = static_cast<u8>(mtl::get_blend_factor(REGS(ctx)->blend_func_sfactor_a()));
				const u8 dfactor_rgb = static_cast<u8>(mtl::get_blend_factor(REGS(ctx)->blend_func_dfactor_rgb()));
				const u8 dfactor_a = static_cast<u8>(mtl::get_blend_factor(REGS(ctx)->blend_func_dfactor_a()));
				const u8 equation_rgb = static_cast<u8>(mtl::get_blend_op(REGS(ctx)->blend_equation_rgb()));
				const u8 equation_a = static_cast<u8>(mtl::get_blend_op(REGS(ctx)->blend_equation_a()));

				for (u8 idx = 0; idx < num_draw_buffers; ++idx)
				{
					if (blend_enabled & (1u << idx))
					{
						auto& blend = properties.blend[idx];
						blend.enable = 1;
						blend.src_rgb = sfactor_rgb;
						blend.dst_rgb = dfactor_rgb;
						blend.src_alpha = sfactor_a;
						blend.dst_alpha = dfactor_a;
						blend.op_rgb = equation_rgb;
						blend.op_alpha = equation_a;
					}
				}
			}
		}

		if (backend_config.supports_hw_a2c || fbo.formats.sample_count > 1)
		{
			const bool alpha_to_one_enable = REGS(ctx)->msaa_alpha_to_one_enabled() && backend_config.supports_hw_a2one;

			properties.alpha_to_coverage = REGS(ctx)->msaa_alpha_to_coverage_enabled() ? 1 : 0;
			properties.alpha_to_one = alpha_to_one_enable ? 1 : 0;
		}

		return properties;
	}
}

// -------------------------------------------------------------------------------------------------------------------
// mtl::query_pool_manager (VK: VKQueryPool.cpp)
// -------------------------------------------------------------------------------------------------------------------
namespace mtl
{
	query_pool_manager::query_pool_manager(mtl::command_context& cmd, u32 num_entries)
		: m_cmd(&cmd)
	{
		ensure(num_entries > 0);

		// Metal caps visibility result buffer offsets at 256 KiB
		ensure(static_cast<u64>(num_entries) * slot_size <= 0x40000);

		m_visibility_buffer = std::make_unique<mtl::buffer>(static_cast<usz>(num_entries) * slot_size,
			MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked, "Visibility results");
		std::memset(m_visibility_buffer->map(), 0, static_cast<usz>(num_entries) * slot_size);

		query_slot_status.resize(num_entries);
		m_release_sync_id.resize(num_entries, 0);

		for (u32 i = 0; i < num_entries; ++i)
		{
			m_available_slots.push_back(i);
		}
	}

	query_pool_manager::~query_pool_manager() = default;

	void query_pool_manager::set_control_flags(bool precise)
	{
		m_result_mode = precise ? MTLVisibilityResultModeCounting : MTLVisibilityResultModeBoolean;
	}

	bool query_pool_manager::poke_query(query_slot_info& query, u32 index)
	{
		if (query.ready)
		{
			return true;
		}

		if (!query.end_sync_id || !m_cmd->is_complete(query.end_sync_id))
		{
			return false;
		}

		// The command buffer that recorded the last counting pass has completed; the shared storage slot holds the result
		u64 value = 0;
		std::memcpy(&value, m_visibility_buffer->map(get_slot_offset(index)), sizeof(u64));

		query.data = static_cast<u32>(std::min<u64>(value, u32{umax}));
		query.any_passed = (value != 0);
		query.ready = true;
		return true;
	}

	void query_pool_manager::begin_query(mtl::command_context& /*cmd*/, u32 index)
	{
		ensure(index < query_slot_status.size());

		auto& query = query_slot_status[index];
		ensure(!query.active);

		// The slot was zeroed when it was allocated. Counting is enabled by the renderer (encoder state).
		query.any_passed = false;
		query.ready = false;
		query.data = 0;
		query.end_sync_id = 0;
		query.active = true;
	}

	void query_pool_manager::end_query(mtl::command_context& cmd, u32 index)
	{
		ensure(index < query_slot_status.size());

		if (cmd.has_pending_work())
		{
			// Counted (at most) by the command buffer being recorded
			end_query(index, cmd.current_sync_id());
			return;
		}

		// Nothing was recorded since the slot was started: no GPU work counted into it, the CPU-zeroed value is final.
		// (A slot counted by a command buffer that was committed elsewhere is ended through end_query(index, sync_id).)
		auto& query = query_slot_status[index];
		query.end_sync_id = cmd.last_submitted_sync_id();
		query.data = 0;
		query.any_passed = false;
		query.ready = true;
	}

	void query_pool_manager::end_query(u32 index, u64 sync_id)
	{
		ensure(index < query_slot_status.size());

		auto& query = query_slot_status[index];
		ensure(query.active);

		query.end_sync_id = sync_id;
		query.ready = false;
	}

	bool query_pool_manager::check_query_status(u32 index)
	{
		ensure(index < query_slot_status.size());
		return poke_query(query_slot_status[index], index);
	}

	u32 query_pool_manager::get_query_result(u32 index)
	{
		ensure(index < query_slot_status.size());

		auto& query = query_slot_status[index];
		if (poke_query(query, index))
		{
			return query.data;
		}

		if (!query.end_sync_id)
		{
			rsx_log.error("Metal: Occlusion query slot %u was read before it was ended", index);
			return 0;
		}

		// Submits the command buffer first if it is still being recorded
		m_cmd->wait(query.end_sync_id);

		if (!poke_query(query, index))
		{
			// The command buffer never completed (e.g. GPU error). Return whatever was written to the slot so far.
			rsx_log.error("Metal: Occlusion query slot %u did not complete (sync id %u)", index, query.end_sync_id);

			u64 value = 0;
			std::memcpy(&value, m_visibility_buffer->map(get_slot_offset(index)), sizeof(u64));

			query.data = static_cast<u32>(std::min<u64>(value, u32{umax}));
			query.any_passed = (value != 0);
			query.ready = true;
		}

		return query.data;
	}

	u32 query_pool_manager::allocate_query(mtl::command_context& cmd)
	{
		if (m_available_slots.empty())
		{
			return ~0u;
		}

		auto is_reusable = [&](u32 slot)
		{
			const u64 release_id = m_release_sync_id[slot];
			return !release_id || cmd.is_complete(release_id);
		};

		// Slots are recycled in the order they were freed. Fall back to a scan if the oldest one is still in flight.
		auto found = m_available_slots.begin();
		if (!is_reusable(*found))
		{
			found = std::find_if(m_available_slots.begin(), m_available_slots.end(), is_reusable);
			if (found == m_available_slots.end())
			{
				return ~0u;
			}
		}

		const u32 index = *found;
		m_available_slots.erase(found);

		// No GPU work references the slot anymore; reset it
		query_slot_status[index] = {};
		m_release_sync_id[index] = 0;
		std::memset(m_visibility_buffer->map(get_slot_offset(index)), 0, slot_size);

		return index;
	}

	void query_pool_manager::free_query(mtl::command_context& cmd, u32 index)
	{
		ensure(index < query_slot_status.size());

		auto& query = query_slot_status[index];
		ensure(query.active);

		// The slot may only be reused once the last command buffer that could count into it has completed
		m_release_sync_id[index] = query.end_sync_id ? query.end_sync_id : cmd.current_sync_id();

		query.active = false;
		query.ready = false;
		query.any_passed = false;
		query.data = 0;
		query.end_sync_id = 0;

		m_available_slots.push_back(index);
	}
}

// -------------------------------------------------------------------------------------------------------------------
// MTLGSRender
// -------------------------------------------------------------------------------------------------------------------

u64 MTLGSRender::get_cycles()
{
	return thread_ctrl::get_cycles(static_cast<named_thread<MTLGSRender>&>(*this));
}

MTLGSRender::MTLGSRender(utils::serial* ar) noexcept : GSRender(ar)
{
	@autoreleasepool
	{
		// Initialize dependencies
		g_fxo->need<rsx::dma_manager>();

		m_device = std::make_unique<mtl::render_device>();
		if (!m_device->create())
		{
			rsx_log.fatal("Could not find a Metal compatible GPU. The Metal renderer requires an Apple silicon GPU running macOS 15 or later.");
			m_device.reset();
			return;
		}

		const auto& caps = m_device->caps();
		mtl::set_current_renderer(*m_device);

		// Command streams
		m_cmd.create(m_device->queue());
		m_secondary_cmd.create(m_device->queue());

		// Auxiliary upload heap (mtl::get_upload_heap) records into the primary stream
		mtl::init_upload_heap(m_cmd);

		// Occlusion
		m_occlusion_query_manager = std::make_unique<mtl::query_pool_manager>(m_cmd, mtl::occlusion_max_pool_size);
		m_occlusion_map.resize(rsx::reports::occlusion_query_count);

		for (u32 n = 0; n < rsx::reports::occlusion_query_count; ++n)
			m_occlusion_query_data[n].driver_handle = n;

		m_occlusion_query_manager->set_control_flags(!!g_cfg.video.precise_zpass_count);

		// Ring buffers. They are growable; handles are fetched after allocation by the users.
		m_attrib_ring_info.create(m_cmd, mtl::attrib_ring_buffer_size_m * 0x100000, "attrib buffer", 0x400000);
		m_fragment_constants_ring_info.create(m_cmd, mtl::fragment_constants_buffer_size_m * 0x100000, "fragment constants buffer");
		m_transform_constants_ring_info.create(m_cmd, mtl::transform_constants_buffer_size_m * 0x100000, "transform constants buffer");
		m_instancing_buffer_ring_info.create(m_cmd, mtl::transform_constants_buffer_size_m * 0x100000, "instancing data buffer");
		m_index_buffer_ring_info.create(m_cmd, mtl::index_ring_buffer_size_m * 0x100000, "index buffer");
		m_texture_upload_buffer_ring_info.create(m_cmd, mtl::texture_upload_ring_buffer_size_m * 0x100000, "texture upload buffer", 32 * 0x100000);

		m_texbuffer_view_size = std::min<u32>(caps.max_texture_buffer_width, mtl::attrib_ring_buffer_size_m * 0x100000u);

		if (m_texbuffer_view_size < 0x800000)
		{
			// Warn, only possibly expected on old devices
			rsx_log.warning("Current driver may crash due to memory limitations (%uk)", m_texbuffer_view_size / 1024);
		}

		// Fallback bindables
		null_buffer = std::make_unique<mtl::buffer>(32, MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked, "null buffer");
		std::memset(null_buffer->map(), 0, 32);
		null_buffer_view = std::make_unique<mtl::texel_buffer_view>(null_buffer->handle(), MTLPixelFormatR8Uint, 0, 32);

		// Initialize the shader compiler stack
		const int preferred_compiler_threads = g_cfg.video.shader_compiler_threads_count == 0
			? utils::get_thread_count()                   // We spawn the initial pipe compiler with all threads during the boot sequence by default.
			: g_cfg.video.shader_compiler_threads_count;  // Respect user override if set

		mtl::initialize_shader_compiler();
		mtl::initialize_pipe_compiler(preferred_compiler_threads);

		m_prog_buffer = std::make_unique<mtl::program_cache>
		(
			[this](const mtl::pipeline_props& props, const RSXVertexProgram& vp, const RSXFragmentProgram& fp)
			{
				// Program was linked or queued for linking
				m_shaders_cache->store(props, vp, fp);
			}
		);

		if (g_cfg.video.disable_vertex_cache)
			m_vertex_cache = std::make_unique<mtl::null_vertex_cache>();
		else
			m_vertex_cache = std::make_unique<mtl::weak_vertex_cache>();

		m_shaders_cache = std::make_unique<mtl::shader_cache>(*m_prog_buffer, shader_cache_class_name, shader_cache_version);

		// Compiled pipeline binaries are kept next to the shader cache (outside of the version directory enumerated by
		// shaders_cache::load). Not used when the on-disk shader cache is disabled.
		if (!g_cfg.video.disable_on_disk_shader_cache)
		{
			if (const std::string cache_path = rpcs3::cache::get_ppu_cache(); !cache_path.empty())
			{
				const std::string archive_root = fmt::format("%sshaders_cache/pipelines/%s/", cache_path, shader_cache_class_name);

				// Single archive file written by earlier builds, superseded by the layer directory
				if (const std::string legacy_archive = fmt::format("%sbinary_archive_%s.bin", archive_root, shader_cache_version); fs::is_file(legacy_archive))
				{
					fs::remove_file(legacy_archive);
				}

				mtl::open_pipeline_binary_archive(fmt::format("%sarchive_%s/", archive_root, shader_cache_version));
			}
		}

		// Presentation surface (VK: swapchain)
		m_swapchain_dims.width = m_frame->client_width();
		m_swapchain_dims.height = m_frame->client_height();

		if (!init_metal_layer())
		{
			swapchain_unavailable = true;
		}

		m_max_async_frames = m_metal_layer ? static_cast<u32>(m_metal_layer.maximumDrawableCount) : 3u;
		m_max_async_frames = std::max(m_max_async_frames, 2u);
		m_frame_context_storage.resize(m_max_async_frames);
		m_current_frame = &m_frame_context_storage[0];

		m_texture_cache.initialize(*m_device, mtl::get_formats_support(), m_texture_upload_buffer_ring_info);

		mtl::get_overlay_pass<mtl::ui_overlay_renderer>()->init(m_cmd, m_texture_upload_buffer_ring_info);

		const auto shadermode = g_cfg.video.shadermode.get();
		if (shadermode == shader_mode::async_with_interpreter || shadermode == shader_mode::interpreter_only)
		{
			rsx_log.warning("Metal: The shader interpreter is not available. Shaders are compiled asynchronously instead (shader mode: %s).", shadermode);
		}

		backend_config.supports_multidraw = true;
		backend_config.supports_hw_instanced_rendering = true;

		// Programmable blending reads the attachments with framebuffer fetch
		backend_config.supports_programmable_blending = caps.supports_framebuffer_fetch;

		// Metal clamps viewport depth to [0, 1]; out of range depth is emulated in the shaders when enabled
		backend_config.supports_extended_depth_range = false;

		// Flat shading always uses the first vertex of the primitive on Metal
		backend_config.supports_last_provoking_vertex = false;
		rsx_log.warning("Metal has no last vertex provoking convention; RSX flat shading will fall back to smooth interpolation.");

		backend_config.supports_normalized_barycentrics = true;

		// Metal has no depth bounds test. The fragment shader reads the bound Z buffer and discards fragments outside the bounds.
		backend_config.emulate_depth_bounds_test = true;

		// NOTE: We do not actually need multiple sample support for A2C to work
		// This is here for visual consistency - will be removed when AA problems due to mipmaps are fixed
		if (g_cfg.video.antialiasing_level != msaa_level::none)
		{
			backend_config.supports_hw_msaa = true;
			backend_config.supports_hw_a2c = true;
			backend_config.supports_hw_a2c_1spp = true;
			backend_config.supports_hw_a2one = true;
		}

		// Framebufferless rendering. Attachment-less passes can only rasterize at the sample counts supported by the device.
		{
			const auto device = m_device->get();
			const bool needs_msaa_counts = (g_cfg.video.antialiasing_level == msaa_level::_auto);

			bool supported = [device supportsTextureSampleCount:1];
			if (needs_msaa_counts)
			{
				supported = supported && [device supportsTextureSampleCount:2] && [device supportsTextureSampleCount:4];
			}

			if (supported)
			{
				backend_config.supports_framebufferless_rendering = true;
			}
			else
			{
				rsx_log.warning("Framebufferless rendering is not supported with the current MSAA configuration. Some occlusion queries may return incorrect results.");
			}
		}

		// NVIDIA only on VK
		backend_config.supports_hw_renormalization = false;

		// No hardware conditional rendering (rsx::thread CPU fallback), no host labels, no passthrough DMA, no async queues
		backend_config.supports_hw_conditional_render = false;
		backend_config.supports_passthrough_dma = false;
		backend_config.supports_host_gpu_labels = false;
		backend_config.supports_asynchronous_compute = false;

		if (g_cfg.video.host_label_synchronization)
		{
			rsx_log.warning("Metal: Host GPU labels are not supported by the Metal renderer and will be disabled.");
		}

		gpu_capture_init();
	}
}

MTLGSRender::~MTLGSRender()
{
	// Force-release ZCULL-ctrl since it is a pointer to self.
	zcull_ctrl.release();

	if (!m_device)
	{
		// Initialization failed
		return;
	}

	@autoreleasepool
	{
		// Flush DMA queue
		while (!g_fxo->get<rsx::dma_manager>().sync())
		{
			do_local_task(rsx::FIFO::state::lock_wait);
		}

		// Wait for the GPU to finish up with resources
		close_render_pass();
		m_cmd.wait_idle();
		m_secondary_cmd.wait_idle();

		// Clear flush requests
		m_flush_requests.clear_pending_flag();

		gpu_capture_stop();

		// Shaders
		mtl::destroy_pipe_compiler();         // Ensure no pending shaders being compiled
		mtl::finalize_shader_compiler();      // Shut down the glslang compiler
		m_prog_buffer->clear();               // Delete shader objects
		mtl::close_pipeline_binary_archive(); // Save compiled pipeline binaries

		m_persistent_attribute_storage.reset();
		m_volatile_attribute_storage.reset();

		// Upscaler (references some global resources)
		m_upscaler.reset();

		// Heaps
		m_attrib_ring_info.destroy();
		m_fragment_constants_ring_info.destroy();
		m_transform_constants_ring_info.destroy();
		m_instancing_buffer_ring_info.destroy();
		m_index_buffer_ring_info.destroy();
		m_texture_upload_buffer_ring_info.destroy();

		// Fallback bindables
		null_buffer_view.reset();
		null_buffer.reset();

		if (m_current_frame == &m_aux_frame_context && m_current_queue_index < m_frame_context_storage.size())
		{
			// Return resources back to the owner
			m_current_frame = &m_frame_context_storage[m_current_queue_index];
			m_current_frame->grab_resources(m_aux_frame_context);
		}

		// CPU frame contexts
		for (auto& ctx : m_frame_context_storage)
		{
			ctx.present_drawable = nil;
		}

		m_aux_frame_context.present_drawable = nil;
		m_queued_frames.clear();
		m_current_frame = nullptr;
		m_frame_context_storage.clear();

		// Textures
		m_rtts.destroy();
		m_texture_cache.destroy();

		m_overlay_recording_img.reset();
		m_null_textures.clear();

		// Queries
		m_occlusion_query_manager.reset();

		// Main pass
		m_draw_fbo.reset();
		m_encoder_state.reset();
		m_depth_stencil_state = nil;
		fs_sampler_handles = {};
		vs_sampler_handles = {};

		// Global resources (also releases the auxiliary upload heap)
		mtl::destroy_global_resources();

		// Command streams
		m_cmd.destroy();
		m_secondary_cmd.destroy();

		// Device
		m_device->destroy();
	}
}

bool MTLGSRender::on_access_violation(u32 address, bool is_writing)
{
	rsx::mm_flush(address);

	const bool is_rsx_thread = is_current_thread();

	mtl::texture_cache::thrashed_set result;
	{
		// Deferred causes record nothing; the flush happens below. Only the RSX thread may use the primary stream.
		const rsx::invalidation_cause cause = is_writing ? rsx::invalidation_cause::deferred_write : rsx::invalidation_cause::deferred_read;
		result = m_texture_cache.invalidate_address(is_rsx_thread ? m_cmd : m_secondary_cmd, address, cause);
	}

	if (result.invalidate_samplers)
	{
		std::lock_guard lock(m_sampler_mutex);
		m_samplers_dirty.store(true);
	}

	if (!result.violation_handled)
	{
		return zcull_ctrl->on_access_violation(address);
	}

	if (result.num_flushable > 0)
	{
		if (g_fxo->get<rsx::dma_manager>().is_current_thread())
		{
			// The offloader thread cannot handle flush requests
			ensure(!(m_queue_status & flush_queue_state::deadlock));

			m_offloader_fault_range = g_fxo->get<rsx::dma_manager>().get_fault_range(is_writing);
			m_offloader_fault_cause = (is_writing) ? rsx::invalidation_cause::write : rsx::invalidation_cause::read;

			g_fxo->get<rsx::dma_manager>().set_mem_fault_flag();
			m_queue_status |= flush_queue_state::deadlock;
			m_eng_interrupt_mask |= rsx::backend_interrupt;

			// Wait for deadlock to clear
			while (m_queue_status & flush_queue_state::deadlock)
			{
				utils::pause();
			}

			g_fxo->get<rsx::dma_manager>().clear_mem_fault_flag();
			return true;
		}

		if (!is_rsx_thread) [[likely]]
		{
			// Only the RSX thread records into the primary command stream. Post the readback and wait for it.
			auto& task = post_flush_request(address, result);

			m_eng_interrupt_mask |= rsx::backend_interrupt;
			vm::temporary_unlock();
			task.producer_wait();
		}
		else
		{
			if (mtl::is_uninterruptible())
			{
				rsx_log.error("Fault in uninterruptible code!");
			}

			// Flush primary cb queue to sync pending changes
			flush_command_queue();

			m_texture_cache.flush_all(m_cmd, result);
		}
	}

	return true;
}

void MTLGSRender::on_invalidate_memory_range(const utils::address_range32 &range, rsx::invalidation_cause cause)
{
	std::lock_guard lock(m_secondary_cb_guard);

	// Unmap invalidations from other threads record nothing; flushing causes are only raised on the RSX thread
	auto& cmd = is_current_thread() ? m_cmd : m_secondary_cmd;

	auto data = m_texture_cache.invalidate_range(cmd, range, cause);
	AUDIT(data.empty());

	if (cause == rsx::invalidation_cause::unmap && data.violation_handled)
	{
		m_texture_cache.purge_unreleased_sections();
		{
			std::lock_guard lock(m_sampler_mutex);
			m_samplers_dirty.store(true);
		}
	}
}

void MTLGSRender::on_semaphore_acquire_wait()
{
	bool flush_work_pending = false;
	{
		std::lock_guard lock(m_work_queue_guard);
		flush_work_pending = std::any_of(m_work_queue.begin(), m_work_queue.end(), [](const mtl::work_item& item)
		{
			return !item.processed.load();
		});
	}

	if (flush_work_pending ||
		m_flush_requests.pending() ||
		(async_flip_requested & flip_request::emu_requested) ||
		(m_queue_status & flush_queue_state::deadlock))
	{
		do_local_task(rsx::FIFO::state::lock_wait);
	}
}

bool MTLGSRender::on_vram_exhausted(rsx::problem_severity severity)
{
	ensure(!mtl::is_uninterruptible() && rsx::get_current_renderer()->is_current_thread());

	bool texture_cache_relieved = false;
	if (severity >= rsx::problem_severity::fatal)
	{
		// Hard sync before trying to evict anything. This guarantees no UAF crashes in the driver.
		flush_command_queue(true, true);

		if (m_texture_cache.is_overallocated())
		{
			// Evict some unused textures. Do not evict any active references
			std::set<u32> exclusion_list;
			auto scan_array = [&](const auto& texture_array)
			{
				for (auto i = 0ull; i < texture_array.size(); ++i)
				{
					const auto& tex = texture_array[i];
					const auto addr = rsx::get_address(tex.offset(), tex.location());
					exclusion_list.insert(addr);
				}
			};

			scan_array(rsx::method_registers.fragment_textures);
			scan_array(rsx::method_registers.vertex_textures);

			// Hold the secondary lock guard to prevent threads from trying to touch access violation handler stuff
			std::lock_guard lock(m_secondary_cb_guard);

			rsx_log.warning("Texture cache is overallocated. Will evict unnecessary textures.");
			texture_cache_relieved = m_texture_cache.evict_unused(exclusion_list);
		}
	}

	texture_cache_relieved |= m_texture_cache.handle_memory_pressure(severity);
	if (severity == rsx::problem_severity::low)
	{
		// Low severity only handles invalidating unused textures
		return texture_cache_relieved;
	}

	bool surface_cache_relieved = false;

	// Check if we need to spill
	if (severity >= rsx::problem_severity::fatal &&      // Only spill for fatal errors
		!m_device->caps().has_unified_memory &&          // Do not spill on unified memory, there is nowhere to spill to
		m_rtts.is_overallocated())                       // Surface cache must be over-allocated by the design quota
	{
		// Queue a VRAM spill operation.
		m_rtts.spill_unused_memory();
	}

	// Moderate severity and higher also starts removing stale render target objects
	if (m_rtts.handle_memory_pressure(m_cmd, severity))
	{
		surface_cache_relieved = true;
		m_rtts.trim(m_cmd, severity);
	}

	const bool any_cache_relieved = (texture_cache_relieved || surface_cache_relieved);
	if (severity < rsx::problem_severity::fatal)
	{
		return any_cache_relieved;
	}

	if (surface_cache_relieved && !m_samplers_dirty)
	{
		// If surface cache was modified destructively, then we must reload samplers touching the surface cache.
		bool invalidate_samplers = false;
		auto scan_array = [&](const auto& texture_array, const auto& sampler_states)
		{
			if (invalidate_samplers)
			{
				return;
			}

			for (auto i = 0ull; i < texture_array.size(); ++i)
			{
				if (texture_array[i].enabled() &&
					sampler_states[i] &&
					sampler_states[i]->upload_context == rsx::texture_upload_context::framebuffer_storage)
				{
					invalidate_samplers = true;
					break;
				}
			}
		};

		scan_array(rsx::method_registers.fragment_textures, fs_sampler_state);
		scan_array(rsx::method_registers.vertex_textures, vs_sampler_state);

		if (invalidate_samplers)
		{
			m_samplers_dirty.store(true);
		}
	}

	// Imminent crash, full GPU sync is the least of our problems
	flush_command_queue(true, true);

	return any_cache_relieved;
}

void MTLGSRender::notify_tile_unbound(u32 /*tile*/)
{
	//TODO: Handle texture writeback (same as VK)
	{
		std::lock_guard lock(m_sampler_mutex);
		m_samplers_dirty.store(true);
	}
}

void MTLGSRender::on_init_thread()
{
	if (!m_device)
	{
		fmt::throw_exception("No Metal device was created");
	}

	GSRender::on_init_thread();
	zcull_ctrl.reset(static_cast<::rsx::reports::ZCULL_control*>(this));

	// There is no shader interpreter; the shader cache is loaded in every shader mode
	@autoreleasepool
	{
		if (!m_overlay_manager)
		{
			m_frame->hide();
			m_shaders_cache->load(nullptr);
			m_frame->show();
		}
		else
		{
			rsx::shader_loading_dialog_native dlg(this);

			m_shaders_cache->load(&dlg);
		}
	}

	// Now we properly initialize the pipe compiler as per the user's configuration
	if (!g_cfg.video.shader_compiler_threads_count)
	{
		mtl::resize_pipe_compiler(g_cfg.video.shader_compiler_threads_count);
	}

	if (g_gpu_capture.target_frame == 1)
	{
		// Capture from the very first command buffer on
		gpu_capture_start(m_device->get());
	}
}

void MTLGSRender::on_exit()
{
	GSRender::on_exit();
	mtl::destroy_pipe_compiler();         // Ensure no pending shaders being compiled
	mtl::close_pipeline_binary_archive(); // Persist the pipelines compiled during this session
	gpu_capture_stop();
	zcull_ctrl.release();
}

rsx::flags32_t& MTLGSRender::get_cb_flags()
{
	if (m_cb_flags.is_stale(m_cmd)) [[unlikely]]
	{
		// m_cmd moved on to a new command buffer (committed by close_and_submit_command_buffer, by a wait, or by another
		// module such as a texture cache readback).
		if ((m_cb_flags.flags & cbf::cb_has_open_query) && m_active_query_info)
		{
			// The counting draws were recorded into the committed command buffer: the slot ends there
			const auto& indices = m_occlusion_map[m_active_query_info->driver_handle].indices;
			if (!indices.empty())
			{
				m_occlusion_query_manager->end_query(indices.back(), m_cb_flags.sync_id);
			}
		}

		// An active query continues in a new slot of the new command buffer (VK: flush_command_queue sets the load task)
		m_cb_flags.reset(m_cmd, (m_occlusion_query_active ? cbf::cb_load_occluson_task : 0) | cbf::cb_reload_dynamic_state);

		m_visibility_slot = umax;
		m_visibility_slot_pass_epoch = 0;
		m_encoder_state.set_visibility_result_mode(MTLVisibilityResultModeDisabled, 0);
	}

	return m_cb_flags.flags;
}

void MTLGSRender::clear_surface(u32 mask)
{
	if (skip_current_frame) return;

	// If stencil write mask is disabled, remove clear_stencil bit
	if (!REGS(m_ctx)->stencil_mask()) mask &= ~RSX_GCM_CLEAR_STENCIL_BIT;

	// Ignore invalid clear flags
	if (!(mask & RSX_GCM_CLEAR_ANY_MASK)) return;

	u8 ctx = rsx::framebuffer_creation_context::context_draw;
	if (mask & RSX_GCM_CLEAR_COLOR_RGBA_MASK) ctx |= rsx::framebuffer_creation_context::context_clear_color;
	if (mask & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK) ctx |= rsx::framebuffer_creation_context::context_clear_depth;
	init_buffers(rsx::framebuffer_creation_context{ctx});

	if (!m_graphics_state.test(rsx::rtt_config_valid) || !m_draw_fbo.is_valid())
	{
		return;
	}

	f32 depth_clear = 1.f;
	u32 stencil_clear = 0;
	u32 depth_stencil_mask = 0;     // mtl::image_aspect bits

	u16 scissor_x = static_cast<u16>(m_scissor.x);
	u16 scissor_w = static_cast<u16>(m_scissor.width);
	u16 scissor_y = static_cast<u16>(m_scissor.y);
	u16 scissor_h = static_cast<u16>(m_scissor.height);

	const u16 fb_width = static_cast<u16>(m_draw_fbo.width);
	const u16 fb_height = static_cast<u16>(m_draw_fbo.height);

	//clip region
	std::tie(scissor_x, scissor_y, scissor_w, scissor_h) = rsx::clip_region<u16>(fb_width, fb_height, scissor_x, scissor_y, scissor_w, scissor_h, true);
	const MTLScissorRect region = { scissor_x, scissor_y, scissor_w, scissor_h };

	const bool full_frame = (scissor_w == fb_width && scissor_h == fb_height);
	bool update_color = false, update_z = false;
	auto surface_depth_format = REGS(m_ctx)->surface_depth_fmt();

	// Color clear parameters
	bool color_fast_clear = false;
	u8 color_write_mask = 0;        // MTLColorWriteMask
	color4f color_clear_value = { 0.f, 0.f, 0.f, 0.f };

	if (auto ds = std::get<1>(m_rtts.m_bound_depth_stencil); ds && (mask & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK))
	{
		if (mask & RSX_GCM_CLEAR_DEPTH_BIT)
		{
			const u32 clear_depth_bits = REGS(m_ctx)->z_clear_value(is_depth_stencil_format(surface_depth_format));
			depth_clear = rsx::get_depth_clear_value(surface_depth_format, clear_depth_bits);
			if (mtl::emulate_extended_depth_range() && rsx::is_float_depth_format(surface_depth_format)) [[ unlikely ]]
			{
				depth_clear = rsx::encode_emulated_depth(depth_clear);
			}

			depth_stencil_mask |= mtl::image_aspect::depth;
		}

		if (is_depth_stencil_format(surface_depth_format))
		{
			if (mask & RSX_GCM_CLEAR_STENCIL_BIT)
			{
				u8 clear_stencil = REGS(m_ctx)->stencil_clear_value();
				stencil_clear = clear_stencil;

				depth_stencil_mask |= mtl::image_aspect::stencil;

				if (ds->samples() > 1)
				{
					if (full_frame) ds->stencil_init_flags &= 0xFF;
					ds->stencil_init_flags |= clear_stencil;
				}
			}
		}

		if ((depth_stencil_mask && depth_stencil_mask != ds->aspect()) || !full_frame)
		{
			// At least one aspect is not being cleared or the clear does not cover the full frame
			// Steps to initialize memory are required

			if (ds->state_flags & rsx::surface_state_flags::erase_bkgnd &&  // Needs initialization
				ds->old_contents.empty() && !g_cfg.video.read_depth_buffer) // No way to load data from memory, so no initialization given
			{
				// Only one aspect was cleared. Make sure to memory initialize the other before removing dirty flag
				const auto ds_mask = (mask & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK);
				if (ds_mask == RSX_GCM_CLEAR_DEPTH_BIT && (ds->aspect() & mtl::image_aspect::stencil))
				{
					// Depth was cleared, initialize stencil
					stencil_clear = 0xFF;
					depth_stencil_mask |= mtl::image_aspect::stencil;
				}
				else if (ds_mask == RSX_GCM_CLEAR_STENCIL_BIT)
				{
					// Stencil was cleared, initialize depth
					depth_clear = 1.f;
					depth_stencil_mask |= mtl::image_aspect::depth;
				}
			}
			else
			{
				// Barrier required before any writes
				ds->write_barrier(m_cmd);
			}
		}
	}

	if (auto colormask = (mask & RSX_GCM_CLEAR_COLOR_RGBA_MASK))
	{
		if (!m_draw_buffers.empty())
		{
			bool use_fast_clear = (colormask == RSX_GCM_CLEAR_COLOR_RGBA_MASK);
			u8 clear_a = REGS(m_ctx)->clear_color_a();
			u8 clear_r = REGS(m_ctx)->clear_color_r();
			u8 clear_g = REGS(m_ctx)->clear_color_g();
			u8 clear_b = REGS(m_ctx)->clear_color_b();

			switch (REGS(m_ctx)->surface_color())
			{
			case rsx::surface_color_format::x32:
			case rsx::surface_color_format::w16z16y16x16:
			case rsx::surface_color_format::w32z32y32x32:
			{
				//NOP
				colormask = 0;
				break;
			}
			case rsx::surface_color_format::b8:
			{
				rsx::get_b8_clear_color(clear_r, clear_g, clear_b, clear_a);
				colormask = rsx::get_b8_clearmask(colormask);
				use_fast_clear = (colormask & RSX_GCM_CLEAR_RED_BIT);
				break;
			}
			case rsx::surface_color_format::g8b8:
			{
				rsx::get_g8b8_clear_color(clear_r, clear_g, clear_b, clear_a);
				colormask = rsx::get_g8b8_r8g8_clearmask(colormask);
				use_fast_clear = ((colormask & RSX_GCM_CLEAR_COLOR_RG_MASK) == RSX_GCM_CLEAR_COLOR_RG_MASK);
				break;
			}
			case rsx::surface_color_format::r5g6b5:
			{
				rsx::get_rgb565_clear_color(clear_r, clear_g, clear_b, clear_a);
				use_fast_clear = ((colormask & RSX_GCM_CLEAR_COLOR_RGB_MASK) == RSX_GCM_CLEAR_COLOR_RGB_MASK);
				break;
			}
			case rsx::surface_color_format::x1r5g5b5_o1r5g5b5:
			{
				rsx::get_a1rgb555_clear_color(clear_r, clear_g, clear_b, clear_a, 255);
				break;
			}
			case rsx::surface_color_format::x1r5g5b5_z1r5g5b5:
			{
				rsx::get_a1rgb555_clear_color(clear_r, clear_g, clear_b, clear_a, 0);
				break;
			}
			case rsx::surface_color_format::a8b8g8r8:
			case rsx::surface_color_format::x8b8g8r8_o8b8g8r8:
			case rsx::surface_color_format::x8b8g8r8_z8b8g8r8:
			{
				rsx::get_abgr8_clear_color(clear_r, clear_g, clear_b, clear_a);
				colormask = rsx::get_abgr8_clearmask(colormask);
				break;
			}
			default:
			{
				break;
			}
			}

			if (colormask)
			{
				if (!use_fast_clear || !full_frame)
				{
					// If we're not clobber all the memory, a barrier is required
					for (const auto& index : m_rtts.m_bound_render_target_ids)
					{
						m_rtts.m_bound_render_targets[index].second->write_barrier(m_cmd);
					}
				}

				color_clear_value =
				{
					static_cast<float>(clear_r) / 255,
					static_cast<float>(clear_g) / 255,
					static_cast<float>(clear_b) / 255,
					static_cast<float>(clear_a) / 255
				};

				color_fast_clear = use_fast_clear;

				if (use_fast_clear)
				{
					color_write_mask = MTLColorWriteMaskAll;
				}
				else
				{
					if (colormask & RSX_GCM_CLEAR_RED_BIT) color_write_mask |= MTLColorWriteMaskRed;
					if (colormask & RSX_GCM_CLEAR_GREEN_BIT) color_write_mask |= MTLColorWriteMaskGreen;
					if (colormask & RSX_GCM_CLEAR_BLUE_BIT) color_write_mask |= MTLColorWriteMaskBlue;
					if (colormask & RSX_GCM_CLEAR_ALPHA_BIT) color_write_mask |= MTLColorWriteMaskAlpha;
				}

				update_color = true;
			}
		}
	}

	if (depth_stencil_mask)
	{
		update_z = true;
	}

	if (update_color || update_z)
	{
		m_rtts.on_write({ update_color, update_color, update_color, update_color }, update_z);
	}
	else
	{
		return;
	}

	// Clears that cover every attachment entirely are folded into the load actions of a new main pass (VK fast clears).
	// The pass is opened right away so that the clear is ordered before any later access to the surfaces.
	const bool attachments_match_render_area = [&]()
	{
		for (u32 index = 0; index < m_draw_fbo.color_count; ++index)
		{
			const auto surface = m_draw_fbo.color_targets[index];
			if (!surface || surface->width() != m_draw_fbo.width || surface->height() != m_draw_fbo.height)
			{
				return false;
			}
		}

		if (const auto ds = m_draw_fbo.depth_stencil;
			ds && (ds->width() != m_draw_fbo.width || ds->height() != m_draw_fbo.height))
		{
			return false;
		}

		return true;
	}();

	const bool can_use_load_actions = full_frame && attachments_match_render_area && !is_main_pass_open();

	mtl::attachment_clear_pass::clear_info clear_info{};
	bool needs_clear_pass = false;
	u32 load_clear_mask = 0;

	if (update_color)
	{
		if (color_fast_clear && can_use_load_actions)
		{
			for (u32 index = 0; index < m_draw_fbo.color_count; ++index)
			{
				load_clear_mask |= (mtl::draw_framebuffer::clear_bit_color0 << index);
				m_draw_fbo.clear_colors[index] = MTLClearColorMake(color_clear_value.r, color_clear_value.g, color_clear_value.b, color_clear_value.a);
			}
		}
		else
		{
			for (u32 index = 0; index < m_draw_fbo.color_count; ++index)
			{
				clear_info.color_write_mask[index] = color_write_mask;
			}

			clear_info.color = color_clear_value;
			needs_clear_pass = true;
		}
	}

	if (depth_stencil_mask & mtl::image_aspect::depth)
	{
		if (can_use_load_actions)
		{
			load_clear_mask |= mtl::draw_framebuffer::clear_bit_depth;
			m_draw_fbo.clear_depth = depth_clear;
		}
		else
		{
			clear_info.clear_depth = true;
			clear_info.depth = depth_clear;
			needs_clear_pass = true;
		}
	}

	if (depth_stencil_mask & mtl::image_aspect::stencil)
	{
		// Partial stencil clear. Disables fast stencil clear
		const u8 stencil_write_mask = static_cast<u8>(REGS(m_ctx)->stencil_mask());
		const bool partial_stencil_clear = (stencil_write_mask != 0xff);

		if (can_use_load_actions && !partial_stencil_clear)
		{
			load_clear_mask |= mtl::draw_framebuffer::clear_bit_stencil;
			m_draw_fbo.clear_stencil = stencil_clear;
		}
		else
		{
			clear_info.clear_stencil = true;
			clear_info.stencil = static_cast<u8>(stencil_clear);
			clear_info.stencil_write_mask = partial_stencil_clear ? stencil_write_mask : u8{0xff};
			needs_clear_pass = true;
		}
	}

	if (!load_clear_mask && !needs_clear_pass)
	{
		return;
	}

	m_draw_fbo.pending_clear_mask |= load_clear_mask;

	// Opens the main pass if needed, consuming the pending clears
	begin_render_pass();

	if (needs_clear_pass)
	{
		mtl::get_overlay_pass<mtl::attachment_clear_pass>()->run(m_cmd, get_render_pass_formats(), region, clear_info);

		// The helper replaced pipeline, depth-stencil, viewport, scissor and visibility state
		m_encoder_state.invalidate_helper_clobbered_state();
	}

	if (depth_stencil_mask)
	{
		// The clear was written in the open pass. Shader reads of the Z buffer (depth bounds/compare emulation, feedback
		// loops) only see it once the pass ends, so a read-only chain cannot continue in this pass.
		m_rtts.m_bound_depth_stencil.second->on_depth_stencil_clear();
	}
}

u64 MTLGSRender::close_and_submit_command_buffer(mtl::drawable_handle present_drawable)
{
	ensure(!m_queue_status.test_and_set(flush_queue_state::flushing));

	// Host MM sync before executing anything on the GPU
	rsx::mm_flush();

	// Workaround for deadlock occuring during RSX offloader fault (VK uses the result to force a flush)
	g_fxo->get<rsx::dma_manager>().sync();

	// End any active render pass; the caller should handle reopening
	close_render_pass();

	// End open queries. Flags will be automatically reset once the command buffer is committed (get_cb_flags)
	auto& cb_flags = get_cb_flags();
	if (cb_flags & cbf::cb_has_open_query)
	{
		if (m_active_query_info)
		{
			const auto& indices = m_occlusion_map[m_active_query_info->driver_handle].indices;
			if (!indices.empty())
			{
				m_occlusion_query_manager->end_query(m_cmd, indices.back());
			}
		}

		cb_flags &= ~cbf::cb_has_open_query;
	}

	m_visibility_slot = umax;
	m_visibility_slot_pass_epoch = 0;
	m_encoder_state.set_visibility_result_mode(MTLVisibilityResultModeDisabled, 0);

	if (present_drawable)
	{
		// The drawable is shown once this command buffer has completed
		m_cmd.present_drawable(present_drawable);
	}

	const u64 sync_id = m_cmd.submit();
	m_frame_stats.submit_count++;

	if (present_drawable)
	{
		gpu_capture_on_frame_presented(m_device->get());
	}

	m_queue_status.clear(flush_queue_state::flushing);
	return sync_id;
}

void MTLGSRender::flush_command_queue(bool hard_sync, bool do_not_switch)
{
	close_and_submit_command_buffer();

	if (hard_sync)
	{
		mtl::stall_trace_scope trace("Hard sync flush");

		// wait for the latest instruction to execute
		m_cmd.wait(m_cmd.last_submitted_sync_id());

		// Drain present queue
		while (!m_queued_frames.empty())
		{
			const auto queued = m_queued_frames.size();
			check_present_status();

			if (m_queued_frames.size() == queued)
			{
				// Everything submitted has completed; a frame left behind was never committed
				const auto frame = m_queued_frames.front();
				m_cmd.wait(frame->swap_sync_id);
				check_present_status();

				if (m_queued_frames.size() == queued)
				{
					rsx_log.error("Metal: Queued frame (sync id %u) did not complete during a hard sync", frame->swap_sync_id);
					break;
				}
			}
		}

		m_flush_requests.clear_pending_flag();
	}

	if (do_not_switch)
	{
		// Special hard-sync where VK must preserve the command buffer. Command buffers are transient on Metal.
		ensure(hard_sync);
	}

	// Just in case a queued frame holds a ref to this cb, drain the present queue
	check_present_status();
}

void MTLGSRender::write_barrier(u32 address, u32 range)
{
	ensure(is_current_thread());
	m_rtts.invalidate_range(utils::address_range32::start_length(address, range));
}

void MTLGSRender::sync_hint(rsx::FIFO::interrupt_hint hint, rsx::reports::sync_hint_payload_t payload)
{
	rsx::thread::sync_hint(hint, payload);

	if (!(get_cb_flags() & cbf::cb_has_occlusion_task))
	{
		// Occlusion queries not enabled, do nothing
		return;
	}

	// Occlusion test result evaluation is coming up, avoid a hard sync
	switch (hint)
	{
	case rsx::FIFO::interrupt_hint::conditional_render_eval:
	{
		// If a flush request is already enqueued, do nothing
		if (m_flush_requests.pending())
		{
			return;
		}

		// If the result is not going to be read by CELL, do nothing
		const auto ref_addr = static_cast<u32>(payload.address);
		if (!zcull_ctrl->is_query_result_urgent(ref_addr))
		{
			// No effect on CELL behaviour, it will be faster to handle this in RSX code
			return;
		}

		// OK, cell will be accessing the results, probably.
		// Try to avoid flush spam, it is more costly to flush the CB than it is to just upload the vertex data
		// This is supposed to be an optimization afterall.
		const auto now = get_system_time();
		if ((now - m_last_cond_render_eval_hint) > 50)
		{
			// Schedule a sync on the next loop iteration
			m_flush_requests.post(false);
			m_flush_requests.remove_one();
		}

		m_last_cond_render_eval_hint = now;
		break;
	}
	case rsx::FIFO::interrupt_hint::zcull_sync:
	{
		// Check if the required report is synced to this CB
		auto& data = m_occlusion_map[payload.query->driver_handle];

		// NOTE: Currently, a special condition exists where the indices can be empty even with active draw count.
		// This is caused by async compiler and should be removed when ubershaders are added in
		if (!data.is_current(m_cmd) || data.indices.empty())
		{
			return;
		}

		// Unavoidable hard sync coming up, flush immediately
		// This heavyweight hint should be used with caution
		std::lock_guard lock(m_flush_queue_mutex);
		flush_command_queue();

		if (m_flush_requests.pending())
		{
			// Clear without wait
			m_flush_requests.clear_pending_flag();
		}
		break;
	}
	}
}

void MTLGSRender::do_local_task(rsx::FIFO::state state)
{
	if (m_queue_status & flush_queue_state::deadlock)
	{
		// Clear offloader deadlock
		// NOTE: It is not possible to handle regular flush requests before this is cleared
		// NOTE: This may cause graphics corruption due to unsynchronized modification
		on_invalidate_memory_range(m_offloader_fault_range, m_offloader_fault_cause);
		m_queue_status.clear(flush_queue_state::deadlock);
	}

	if (m_queue_status & flush_queue_state::flushing)
	{
		// Abort recursive CB submit requests.
		// When flushing flag is already set, only deadlock events may be processed.
		return;
	}

	bool serviced_requests = false;

	if (!m_work_queue.empty())
	{
		// Readbacks requested by memory faults on other threads (GL work queue)
		std::lock_guard lock(m_work_queue_guard);

		m_work_queue.remove_if([](const mtl::work_item& q) { return q.received.load(); });

		for (auto& q : m_work_queue)
		{
			if (q.processed.load()) continue;

			q.result = m_texture_cache.flush_all(m_cmd, q.section_data);
			q.processed = true;
		}

		serviced_requests = true;
	}

	if (m_flush_requests.pending())
	{
		if (m_flush_queue_mutex.try_lock())
		{
			// TODO: Determine if a hard sync is necessary
			// Pipeline barriers later may do a better job synchronizing than wholly stalling the pipeline
			flush_command_queue();

			m_flush_requests.clear_pending_flag();
			m_flush_requests.consumer_wait();
			m_flush_queue_mutex.unlock();
		}

		serviced_requests = true;
	}

	if (!serviced_requests && !in_begin_end && state != rsx::FIFO::state::lock_wait)
	{
		if (m_graphics_state & rsx::pipeline_state::framebuffer_reads_dirty)
		{
			//This will re-engage locks and break the texture cache if another thread is waiting in access violation handler!
			//Only call when there are no waiters
			m_texture_cache.do_update();
			m_graphics_state.clear(rsx::pipeline_state::framebuffer_reads_dirty);
		}
	}

	rsx::thread::do_local_task(state);

	if (state == rsx::FIFO::state::lock_wait)
	{
		// Critical check finished
		return;
	}

	if (m_overlay_manager)
	{
		const auto should_ignore = in_begin_end && state != rsx::FIFO::state::empty;
		if ((async_flip_requested & flip_request::native_ui) && !should_ignore && !is_stopped())
		{
			flush_command_queue(true);
			rsx::display_flip_info_t info{};
			info.buffer = current_display_buffer;
			flip(info);
		}
	}
}

mtl::work_item& MTLGSRender::post_flush_request(u32 address, mtl::texture_cache::thrashed_set& flush_data)
{
	std::lock_guard lock(m_work_queue_guard);

	auto& result = m_work_queue.emplace_back();
	result.address_to_flush = address;
	result.section_data = std::move(flush_data);
	return result;
}

bool MTLGSRender::load_program()
{
	const auto shadermode = g_cfg.video.shadermode.get();
	// RSX vertex programs always write gl_PointSize. Metal rejects pipelines that write the point size with a line or triangle
	// topology class, so only point rendering declares its class (the class is only required for layered rendering).
	const u8 topology_class = static_cast<u8>(
		mtl::get_primitive_topology_class(REGS(m_ctx)->current_draw_clause.primitive) == MTLPrimitiveTopologyClassPoint
			? MTLPrimitiveTopologyClassPoint
			: MTLPrimitiveTopologyClassUnspecified);

	// Attachment formats are part of the pipeline state
	const bool formats_changed = !(mtl::render_pass_formats::from(m_pipeline_properties) == get_render_pass_formats());

	if (m_graphics_state & rsx::pipeline_state::invalidate_pipeline_bits)
	{
		get_current_fragment_program(fs_sampler_state);
		ensure(current_fragment_program.valid);

		get_current_vertex_program(vs_sampler_state);

		m_graphics_state.clear(rsx::pipeline_state::invalidate_pipeline_bits);
	}
	else if (!(m_graphics_state & rsx::pipeline_state::pipeline_config_dirty) &&
		!formats_changed &&
		m_program &&
		m_pipeline_properties.topology_class == topology_class)
	{
		return true;
	}

	auto &vertex_program = current_vertex_program;
	auto &fragment_program = current_fragment_program;

	if ((m_graphics_state & rsx::pipeline_state::pipeline_config_dirty) || formats_changed)
	{
		mtl::pipeline_props properties = decode_rsx_state(
			m_ctx,
			m_draw_fbo,
			backend_config,
			topology_class,
			!!(current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING));

		if (m_program && m_pipeline_properties == properties)
		{
			// Nothing changed
			m_graphics_state.clear(rsx::pipeline_state::pipeline_config_dirty);
			return true;
		}

		// Fallthrough
		m_pipeline_properties = properties;
		m_graphics_state.clear(rsx::pipeline_state::pipeline_config_dirty);
	}
	else
	{
		// Update primitive topology class
		m_pipeline_properties.topology_class = topology_class;
	}

	m_vertex_prog = nullptr;
	m_fragment_prog = nullptr;

	mtl::enter_uninterruptible();

	if (g_cfg.video.debug_overlay)
	{
		m_frame_stats.program_cache_lookups_total += 2;
		if (m_program_cache_hint.has_fragment_program())
		{
			m_frame_stats.program_cache_lookups_ellided++;
		}
		if (m_program_cache_hint.has_vertex_program())
		{
			m_frame_stats.program_cache_lookups_ellided++;
		}
	}

	// Load current program from cache
	// Programs that emulate the depth range are compiled synchronously (same as VK)
	const bool allow_async = shadermode != shader_mode::recompiler &&
		!(fragment_program.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_RANGE);

	std::tie(m_program, m_vertex_prog, m_fragment_prog) = m_prog_buffer->get_graphics_pipeline(
		&m_program_cache_hint,
		vertex_program,
		fragment_program,
		m_pipeline_properties,
		allow_async, true);

	mtl::leave_uninterruptible();

	if (m_prog_buffer->check_cache_missed())
	{
		// Notify the user with HUD notification
		if (g_cfg.misc.show_shader_compilation_hint)
		{
			if (m_overlay_manager)
			{
				rsx::overlays::show_shader_compile_notification();
			}
		}
	}

	return m_program != nullptr;
}

void MTLGSRender::load_program_env()
{
	if (!m_program)
	{
		fmt::throw_exception("Unreachable right now");
	}

	const auto& ctx = REGS(m_ctx);

	const u32 fragment_constants_size = current_fp_metadata.program_constants_buffer_length;

	const bool update_transform_constants = !!(m_graphics_state & rsx::pipeline_state::transform_constants_dirty);
	const bool update_fragment_constants = !!(m_graphics_state & rsx::pipeline_state::fragment_constants_dirty);
	const bool update_vertex_env = !!(m_graphics_state & rsx::pipeline_state::vertex_state_dirty);
	const bool update_fragment_env = !!(m_graphics_state & rsx::pipeline_state::fragment_state_dirty);
	const bool update_fragment_texture_env = !!(m_graphics_state & rsx::pipeline_state::fragment_texture_state_dirty);
	const bool update_raster_env = (ctx->polygon_stipple_enabled() && !!(m_graphics_state & rsx::pipeline_state::polygon_stipple_pattern_dirty));
	const bool update_instancing_data = ctx->current_draw_clause.is_trivial_instanced_draw;

	if (update_vertex_env)
	{
		// Vertex state (VertexContextBuffer, std140)
		auto buf = m_vertex_env_data.data();

		m_draw_processor.fill_scale_offset_data(buf, false);
		m_draw_processor.fill_user_clip_data(buf + 64);
		*(reinterpret_cast<u32*>(buf + 68)) = ctx->transform_branch_bits();
		*(reinterpret_cast<f32*>(buf + 72)) = ctx->point_size() * resolution_scaling_config.scale_factor();
		*(reinterpret_cast<f32*>(buf + 76)) = ctx->clip_min();
		*(reinterpret_cast<f32*>(buf + 80)) = ctx->clip_max();
	}

	if (update_instancing_data)
	{
		// Combines transform load + instancing lookup table
		rsx::io_buffer indirection_table_buf([&](usz size) -> std::pair<void*, usz>
		{
			const usz offset = m_instancing_buffer_ring_info.alloc<256>(size);
			m_instancing_indirection_offset = static_cast<u32>(offset);
			return std::make_pair(m_instancing_buffer_ring_info.map(offset), size);
		});

		rsx::io_buffer constants_array_buf([&](usz size) -> std::pair<void*, usz>
		{
			const usz offset = m_instancing_buffer_ring_info.alloc<256>(size);
			m_instancing_constants_offset = static_cast<u32>(offset);
			return std::make_pair(m_instancing_buffer_ring_info.map(offset), size);
		});

		m_draw_processor.fill_constants_instancing_buffer(indirection_table_buf, constants_array_buf, m_vertex_prog);
	}
	else if (update_transform_constants)
	{
		// Transform constants
		usz mem_offset = 0;
		auto alloc_storage = [&](usz size) -> std::pair<void*, usz>
		{
			mem_offset = m_transform_constants_ring_info.alloc<256>(size);
			return std::make_pair(m_transform_constants_ring_info.map(mem_offset), size);
		};

		auto io_buf = rsx::io_buffer(alloc_storage);
		upload_transform_constants(io_buf);

		if (!io_buf.empty())
		{
			m_xform_constants_dynamic_offset = static_cast<u32>(mem_offset);
		}
	}

	if (update_fragment_constants)
	{
		// Fragment constants
		if (fragment_constants_size)
		{
			if (fragment_constants_size <= mtl::max_inline_uniform_block_size)
			{
				// Small blocks are uploaded inline (setFragmentBytes)
				m_fragment_constants_data.resize(fragment_constants_size);

				m_prog_buffer->fill_fragment_constants_buffer({ reinterpret_cast<f32*>(m_fragment_constants_data.data()), fragment_constants_size / 4 },
					*ensure(m_fragment_prog), current_fragment_program, true);

				m_fragment_constants_inline = true;
			}
			else
			{
				const usz offset = m_fragment_constants_ring_info.alloc<256>(fragment_constants_size);
				auto buf = m_fragment_constants_ring_info.map(offset);

				m_prog_buffer->fill_fragment_constants_buffer({ reinterpret_cast<f32*>(buf), fragment_constants_size / 4 },
					*ensure(m_fragment_prog), current_fragment_program, true);

				m_fragment_constants_dynamic_offset = static_cast<u32>(offset);
				m_fragment_constants_inline = false;
			}
		}
		else
		{
			// The program does not declare a constants block
			m_fragment_constants_data.clear();
			m_fragment_constants_inline = true;
		}
	}

	if (update_fragment_env)
	{
		m_draw_processor.fill_fragment_state_buffer(m_fragment_env_data.data(), current_fragment_program);
	}

	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_BOUNDS)
	{
		// Bounds of the emulated depth bounds test, saturated into the RSX depth format's range and encoded like the stored
		// depth values (VK: vkCmdSetDepthBounds in update_draw_state). Not covered by fragment_state_dirty, refreshed every draw.
		const auto depth_format = ctx->surface_depth_fmt();
		f32 bounds_min = rsx::clamp_depth_bounds_value(depth_format, ctx->depth_bounds_min());
		f32 bounds_max = rsx::clamp_depth_bounds_value(depth_format, ctx->depth_bounds_max());

		if (requires_depth_range_emulation()) [[unlikely]]
		{
			bounds_min = rsx::encode_emulated_depth(bounds_min);
			bounds_max = rsx::encode_emulated_depth(bounds_max);
		}

		const f32 depth_bounds[2] = { std::clamp(bounds_min, 0.f, 1.f), std::clamp(bounds_max, 0.f, 1.f) };
		std::memcpy(m_fragment_env_data.data() + 32, depth_bounds, sizeof(depth_bounds));
	}

	if (update_fragment_texture_env)
	{
		current_fragment_program.texture_params.write_to(m_fragment_texture_params_data.data(), current_fp_metadata.referenced_textures_mask);
	}

	if (update_raster_env)
	{
		std::memcpy(m_raster_env_data.data(), ctx->polygon_stipple_pattern(), 128);
		m_graphics_state.clear(rsx::pipeline_state::polygon_stipple_pattern_dirty);
	}

	// Clear flags
	rsx::flags32_t handled_flags =
		rsx::pipeline_state::fragment_state_dirty |
		rsx::pipeline_state::vertex_state_dirty |
		rsx::pipeline_state::fragment_texture_state_dirty;

	if (!update_instancing_data)
	{
		handled_flags |= rsx::pipeline_state::transform_constants_dirty;
	}

	if (update_fragment_constants)
	{
		handled_flags |= rsx::pipeline_state::fragment_constants_dirty;
	}

	m_graphics_state.clear(handled_flags);
}

bool MTLGSRender::is_current_program_interpreted() const
{
	// There is no shader interpreter on Metal
	return false;
}

void MTLGSRender::upload_transform_constants(const rsx::io_buffer& buffer)
{
	ensure(m_vertex_prog);

	const usz transform_constants_size = m_vertex_prog->has_indexed_constants ? 8192 : m_vertex_prog->constant_ids.size() * 16;

	if (transform_constants_size)
	{
		buffer.reserve(transform_constants_size);
		auto buf = buffer.data();

		const auto constant_ids = (transform_constants_size == 8192)
			? std::span<const u16>{}
			: std::span<const u16>(m_vertex_prog->constant_ids);
		m_draw_processor.fill_vertex_program_constants_data(buf, constant_ids);
	}
}

void MTLGSRender::update_vertex_env(u32 /*id*/, const mtl::vertex_upload_info& vertex_info)
{
	// Vertex layout state (VertexLayoutBuffer, std140). Bound inline by emit_geometry.
	auto buf = reinterpret_cast<u32*>(m_vertex_layout_data.data());

	buf[0] = vertex_info.vertex_index_base;
	buf[1] = vertex_info.vertex_index_offset;
	buf[2] = 0;
	buf[3] = 0;

	m_draw_processor.fill_vertex_layout_state(
		m_vertex_layout,
		current_vp_metadata,
		vertex_info.first_vertex,
		vertex_info.allocated_vertex_count,
		reinterpret_cast<s32*>(buf + 4),
		vertex_info.persistent_window_offset,
		vertex_info.volatile_window_offset);

	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING)
	{
		// BlendParameters (std140): blend_eqn, blend_sfactors, blend_dfactors, vec4 blend_constants at offset 16
		auto blend_config = reinterpret_cast<u32*>(m_blend_params_data.data());
		blend_config[0] = REGS(m_ctx)->registers[NV4097_SET_BLEND_EQUATION];
		blend_config[1] = REGS(m_ctx)->registers[NV4097_SET_BLEND_FUNC_SFACTOR];
		blend_config[2] = REGS(m_ctx)->registers[NV4097_SET_BLEND_FUNC_DFACTOR];
		blend_config[3] = 0;

		const auto blend_colors = rsx::get_constant_blend_colors();
		std::memcpy(m_blend_params_data.data() + 16, blend_colors.data(), sizeof(f32) * 4);
	}
}

void MTLGSRender::patch_transform_constants(rsx::context* /*ctx*/, u32 index, u32 count)
{
	if (!m_program || !m_vertex_prog)
	{
		// Shouldn't be reachable, but handle it correctly anyway
		m_graphics_state |= rsx::pipeline_state::transform_constants_dirty;
		return;
	}

	if (!m_vertex_prog->overlaps_constants_range(index, count))
	{
		// Nothing meaningful to us
		return;
	}

	// Data already recorded keeps reading the previous block; the next draw binds the new one
	auto allocate_mem = [&](usz size) -> std::pair<void*, usz>
	{
		const usz offset = m_transform_constants_ring_info.alloc<256>(size);
		m_xform_constants_dynamic_offset = static_cast<u32>(offset);
		return std::make_pair(m_transform_constants_ring_info.map(offset), size);
	};

	rsx::io_buffer iobuf(allocate_mem);
	upload_transform_constants(iobuf);
}

void MTLGSRender::init_buffers(rsx::framebuffer_creation_context context, bool)
{
	prepare_rtts(context);
}

void MTLGSRender::prepare_rtts(rsx::framebuffer_creation_context context)
{
	const bool clipped_scissor = (context == rsx::framebuffer_creation_context::context_draw);
	if (m_current_framebuffer_context == context && !m_graphics_state.test(rsx::rtt_config_dirty) && m_draw_fbo.is_valid())
	{
		// Fast path
		// Framebuffer usage has not changed, framebuffer exists and config regs have not changed
		set_scissor(clipped_scissor);
		return;
	}

	m_graphics_state.clear(
		rsx::rtt_config_dirty |
		rsx::rtt_config_contested |
		rsx::rtt_config_valid |
		rsx::rtt_cache_state_dirty);

	get_framebuffer_layout(context, m_framebuffer_layout);
	if (!m_graphics_state.test(rsx::rtt_config_valid))
	{
		return;
	}

	if (m_draw_fbo.is_valid() && m_framebuffer_layout.ignore_change)
	{
		// Nothing has changed, we're still using the same framebuffer
		// Update flags to match current
		set_scissor(clipped_scissor);
		return;
	}

	m_rtts.prepare_render_target(m_cmd,
		m_framebuffer_layout.color_format, m_framebuffer_layout.depth_format,
		m_framebuffer_layout.width, m_framebuffer_layout.height,
		m_framebuffer_layout.target, m_framebuffer_layout.aa_mode, m_framebuffer_layout.raster_type,
		m_framebuffer_layout.color_addresses, m_framebuffer_layout.zeta_address,
		m_framebuffer_layout.actual_color_pitch, m_framebuffer_layout.actual_zeta_pitch,
		resolution_scaling_config);

	// Reset framebuffer information
	const auto color_bpp = get_format_block_size_in_bytes(m_framebuffer_layout.color_format);
	const auto samples = get_format_sample_count(m_framebuffer_layout.aa_mode);

	for (u8 i = 0; i < rsx::limits::color_buffers_count; ++i)
	{
		// Flush old address if we keep missing it
		if (m_surface_info[i].pitch && g_cfg.video.write_color_buffers)
		{
			const utils::address_range32 rsx_range = m_surface_info[i].get_memory_range();
			m_texture_cache.set_memory_read_flags(rsx_range, rsx::memory_read_flags::flush_once);
			m_texture_cache.flush_if_cache_miss_likely(m_cmd, rsx_range);
		}

		m_surface_info[i].address = m_surface_info[i].pitch = 0;
		m_surface_info[i].width = m_framebuffer_layout.width;
		m_surface_info[i].height = m_framebuffer_layout.height;
		m_surface_info[i].color_format = m_framebuffer_layout.color_format;
		m_surface_info[i].bpp = color_bpp;
		m_surface_info[i].samples = samples;
	}

	// Process depth surface as well
	{
		if (m_depth_surface_info.pitch && g_cfg.video.write_depth_buffer)
		{
			const utils::address_range32 surface_range = m_depth_surface_info.get_memory_range();
			m_texture_cache.set_memory_read_flags(surface_range, rsx::memory_read_flags::flush_once);
			m_texture_cache.flush_if_cache_miss_likely(m_cmd, surface_range);
		}

		m_depth_surface_info.address = m_depth_surface_info.pitch = 0;
		m_depth_surface_info.width = m_framebuffer_layout.width;
		m_depth_surface_info.height = m_framebuffer_layout.height;
		m_depth_surface_info.depth_format = m_framebuffer_layout.depth_format;
		m_depth_surface_info.bpp = get_format_block_size_in_bytes(m_framebuffer_layout.depth_format);
		m_depth_surface_info.samples = samples;
	}

	// Bind created rtts as current fbo...
	const auto draw_buffers = rsx::utility::get_rtt_indexes(m_framebuffer_layout.target);
	m_draw_buffers.clear();
	m_fbo_images.clear();

	for (u8 index : draw_buffers)
	{
		if (auto surface = std::get<1>(m_rtts.m_bound_render_targets[index]))
		{
			m_fbo_images.push_back(surface);

			m_surface_info[index].address = m_framebuffer_layout.color_addresses[index];
			m_surface_info[index].pitch = m_framebuffer_layout.actual_color_pitch[index];
			ensure(surface->rsx_pitch == m_framebuffer_layout.actual_color_pitch[index]);

			m_texture_cache.notify_surface_changed(m_surface_info[index].get_memory_range(m_framebuffer_layout.aa_factors));
			m_draw_buffers.push_back(index);
		}
	}

	if (std::get<0>(m_rtts.m_bound_depth_stencil) != 0)
	{
		auto ds = std::get<1>(m_rtts.m_bound_depth_stencil);
		m_fbo_images.push_back(ds);

		m_depth_surface_info.address = m_framebuffer_layout.zeta_address;
		m_depth_surface_info.pitch = m_framebuffer_layout.actual_zeta_pitch;
		ensure(ds->rsx_pitch == m_framebuffer_layout.actual_zeta_pitch);

		m_texture_cache.notify_surface_changed(m_depth_surface_info.get_memory_range(m_framebuffer_layout.aa_factors));
	}

	// Before messing with memory properties, flush command queue if there are dma transfers queued up
	if (get_cb_flags() & cbf::cb_has_dma_transfer)
	{
		flush_command_queue();
	}

	if (!m_rtts.superseded_surfaces.empty())
	{
		for (auto& surface : m_rtts.superseded_surfaces)
		{
			m_texture_cache.discard_framebuffer_memory_region(m_cmd, surface->get_memory_range());
		}

		m_rtts.superseded_surfaces.clear();
	}

	if (!m_rtts.orphaned_surfaces.empty())
	{
		for (auto& [base_addr, surface] : m_rtts.orphaned_surfaces)
		{
			bool lock = surface->is_depth_surface() ? !!g_cfg.video.write_depth_buffer :
				!!g_cfg.video.write_color_buffers;

			if (lock &&
#ifdef TEXTURE_CACHE_DEBUG
				!m_texture_cache.is_protected(
					base_addr,
					surface->get_memory_range(),
					rsx::texture_upload_context::framebuffer_storage)
#else
				!surface->is_locked()
#endif
				)
			{
				lock = false;
			}

			if (!lock) [[likely]]
			{
				m_texture_cache.commit_framebuffer_memory_region(m_cmd, surface->get_memory_range());
				continue;
			}

			m_texture_cache.lock_memory_region(
				m_cmd, surface, surface->get_memory_range(), false,
				surface->get_surface_width<rsx::surface_metrics::pixels>(), surface->get_surface_height<rsx::surface_metrics::pixels>(), surface->get_rsx_pitch(),
				surface);
		}

		m_rtts.orphaned_surfaces.clear();
	}

	for (u8 index : m_draw_buffers)
	{
		if (!m_surface_info[index].address || !m_surface_info[index].pitch) continue;

		const utils::address_range32 surface_range = m_surface_info[index].get_memory_range();
		if (g_cfg.video.write_color_buffers)
		{
			m_texture_cache.lock_memory_region(
				m_cmd, m_rtts.m_bound_render_targets[index].second, surface_range, true,
				m_surface_info[index].width, m_surface_info[index].height, m_framebuffer_layout.actual_color_pitch[index],
				m_rtts.m_bound_render_targets[index].second);
		}
		else
		{
			m_texture_cache.commit_framebuffer_memory_region(m_cmd, surface_range);
		}
	}

	if (m_depth_surface_info.address && m_depth_surface_info.pitch)
	{
		const utils::address_range32 surface_range = m_depth_surface_info.get_memory_range();
		if (g_cfg.video.write_depth_buffer)
		{
			m_texture_cache.lock_memory_region(
				m_cmd, m_rtts.m_bound_depth_stencil.second, surface_range, true,
				m_depth_surface_info.width, m_depth_surface_info.height, m_framebuffer_layout.actual_zeta_pitch,
				m_rtts.m_bound_depth_stencil.second);
		}
		else
		{
			m_texture_cache.commit_framebuffer_memory_region(m_cmd, surface_range);
		}
	}

	// Main pass description (VK: render pass key + framebuffer object)
	const auto [fbo_width, fbo_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, m_framebuffer_layout.width, m_framebuffer_layout.height);

	mtl::draw_framebuffer fbo{};
	fbo.width = fbo_width;
	fbo.height = fbo_height;
	fbo.layers = 1;

	for (u8 index : m_draw_buffers)
	{
		const auto surface = ensure(m_rtts.m_bound_render_targets[index].second);
		const u32 slot = fbo.color_count++;

		fbo.color_targets[slot] = surface;
		fbo.color_textures[slot] = surface->value;
		fbo.formats.color_formats[slot] = static_cast<u32>(surface->format());
		fbo.samples = surface->samples();

		fbo.width = std::min(fbo.width, surface->width());
		fbo.height = std::min(fbo.height, surface->height());
	}

	if (std::get<0>(m_rtts.m_bound_depth_stencil) != 0)
	{
		const auto ds = ensure(std::get<1>(m_rtts.m_bound_depth_stencil));

		fbo.depth_stencil = ds;
		fbo.depth_stencil_texture = ds->value;
		fbo.formats.depth_format = static_cast<u32>(ds->format());
		fbo.formats.stencil_format = mtl::is_stencil_format(ds->format()) ? static_cast<u32>(ds->format()) : static_cast<u32>(MTLPixelFormatInvalid);
		fbo.samples = ds->samples();

		fbo.width = std::min(fbo.width, ds->width());
		fbo.height = std::min(fbo.height, ds->height());
	}

	if (m_graphics_state.test(rsx::rtt_config_no_attachments) || (!fbo.color_count && !fbo.depth_stencil))
	{
		// Framebufferless rendering. The raster sample count must match what real surfaces would have used.
		ensure(m_fbo_images.empty());

		fbo.no_attachments = true;
		fbo.samples = (g_cfg.video.antialiasing_level == msaa_level::_auto) ? static_cast<u8>(samples) : u8{1};
	}

	fbo.formats.color_count = static_cast<u8>(fbo.color_count);
	fbo.formats.sample_count = fbo.samples;

	const bool fbo_changed =
		m_draw_fbo.color_count != fbo.color_count ||
		m_draw_fbo.color_targets != fbo.color_targets ||
		m_draw_fbo.color_textures != fbo.color_textures ||
		m_draw_fbo.depth_stencil != fbo.depth_stencil ||
		m_draw_fbo.depth_stencil_texture != fbo.depth_stencil_texture ||
		m_draw_fbo.width != fbo.width ||
		m_draw_fbo.height != fbo.height ||
		m_draw_fbo.layers != fbo.layers ||
		m_draw_fbo.samples != fbo.samples ||
		m_draw_fbo.no_attachments != fbo.no_attachments ||
		!(m_draw_fbo.formats == fbo.formats);

	if (fbo_changed)
	{
		// A new main pass is required for the next draw
		invalidate_render_pass();
		m_draw_fbo = fbo;
	}

	set_viewport();
	set_scissor(clipped_scissor);
	on_framebuffer_layout_updated();

	check_zcull_status(true);
}

MTLRenderPassDescriptor* MTLGSRender::create_main_pass_descriptor()
{
	MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor new];
	const u32 clear_mask = m_draw_fbo.pending_clear_mask;

	for (u32 index = 0; index < m_draw_fbo.color_count; ++index)
	{
		MTLRenderPassColorAttachmentDescriptor* attachment = desc.colorAttachments[index];
		attachment.texture = m_draw_fbo.color_textures[index];
		ensure(attachment.texture);

		if (clear_mask & (mtl::draw_framebuffer::clear_bit_color0 << index))
		{
			attachment.loadAction = MTLLoadActionClear;
			attachment.clearColor = m_draw_fbo.clear_colors[index];
		}
		else
		{
			attachment.loadAction = MTLLoadActionLoad;
		}

		attachment.storeAction = MTLStoreActionStore;
	}

	if (m_draw_fbo.depth_stencil_texture)
	{
		MTLRenderPassDepthAttachmentDescriptor* depth = desc.depthAttachment;
		depth.texture = m_draw_fbo.depth_stencil_texture;
		depth.storeAction = MTLStoreActionStore;

		if (clear_mask & mtl::draw_framebuffer::clear_bit_depth)
		{
			depth.loadAction = MTLLoadActionClear;
			depth.clearDepth = m_draw_fbo.clear_depth;
		}
		else
		{
			depth.loadAction = MTLLoadActionLoad;
		}

		if (m_draw_fbo.formats.stencil_format != static_cast<u32>(MTLPixelFormatInvalid))
		{
			MTLRenderPassStencilAttachmentDescriptor* stencil = desc.stencilAttachment;
			stencil.texture = m_draw_fbo.depth_stencil_texture;
			stencil.storeAction = MTLStoreActionStore;

			if (clear_mask & mtl::draw_framebuffer::clear_bit_stencil)
			{
				stencil.loadAction = MTLLoadActionClear;
				stencil.clearStencil = m_draw_fbo.clear_stencil;
			}
			else
			{
				stencil.loadAction = MTLLoadActionLoad;
			}
		}
	}

	desc.renderTargetWidth = m_draw_fbo.width;
	desc.renderTargetHeight = m_draw_fbo.height;

	if (m_draw_fbo.no_attachments)
	{
		desc.defaultRasterSampleCount = m_draw_fbo.samples;
	}

	if (m_draw_fbo.layers > 1)
	{
		desc.renderTargetArrayLength = m_draw_fbo.layers;
	}

	// Occlusion queries. Counting is enabled per draw (setVisibilityResultMode:offset:, MTLDraw.mm).
	if (m_occlusion_query_manager)
	{
		desc.visibilityResultBuffer = m_occlusion_query_manager->handle();

		if (@available(macOS 26.0, *))
		{
			if (m_device->caps().supports_visibility_accumulate)
			{
				desc.visibilityResultType = MTLVisibilityResultTypeAccumulate;
			}
		}
	}

	m_draw_fbo.pending_clear_mask = 0;
	return desc;
}

mtl::render_encoder_handle MTLGSRender::begin_render_pass()
{
	auto& cb_flags = get_cb_flags();

	if (!is_main_pass_open())
	{
		ensure(m_draw_fbo.is_valid());

		@autoreleasepool
		{
			m_cmd.begin_render_pass(create_main_pass_descriptor(), @"RSX main pass");
		}

		m_draw_fbo.pass_epoch = m_cmd.render_pass_epoch();

		// Encoder state does not survive the encoder
		m_encoder_state.invalidate();
	}

	// Without MTLVisibilityResultTypeAccumulate, every render encoder overwrites the counts of the visibility slots it uses.
	// A query that was already counted in an earlier pass segment continues in a new slot; the results of all its slots
	// are summed when the query is read (occlusion_data::indices).
	bool accumulate_supported = false;
	if (@available(macOS 26.0, *))
	{
		accumulate_supported = m_device->caps().supports_visibility_accumulate;
	}

	if (!accumulate_supported &&
		m_occlusion_query_active && m_active_query_info &&
		(cb_flags & cbf::cb_has_open_query) &&
		m_visibility_slot_pass_epoch && m_visibility_slot_pass_epoch != m_draw_fbo.pass_epoch)
	{
		auto& data = m_occlusion_map[m_active_query_info->driver_handle];
		const u32 current_slot = (m_visibility_slot != umax)
			? m_visibility_slot
			: (data.indices.empty() ? u32{umax} : data.indices.back());

		if (current_slot != umax)
		{
			// The counts of the previous segment are final once this command buffer completes
			m_occlusion_query_manager->end_query(m_cmd, current_slot);

			if (const u32 next_slot = m_occlusion_query_manager->allocate_query(m_cmd); next_slot != umax)
			{
				m_occlusion_query_manager->begin_query(m_cmd, next_slot);

				data.indices.push_back(next_slot);
				data.set_sync_command_buffer(m_cmd);

				// Counted from this pass segment on (the draw sets the encoder visibility mode)
				m_visibility_slot = next_slot;
				m_visibility_slot_pass_epoch = 0;
			}
			else
			{
				rsx_log.warning("[Performance Warning] Out of free occlusion slots while relocating an open query.");

				// Retry on the next draw (which may force a ZCULL sync to free slots)
				cb_flags &= ~cbf::cb_has_open_query;
				cb_flags |= cbf::cb_load_occluson_task;

				m_visibility_slot = umax;
				m_visibility_slot_pass_epoch = 0;
				m_encoder_state.set_visibility_result_mode(MTLVisibilityResultModeDisabled, 0);
			}
		}
	}

	return m_cmd.render_encoder();
}

void MTLGSRender::close_render_pass()
{
	if (m_cmd.is_render_pass_open())
	{
		m_cmd.end_encoding();
	}
}

void MTLGSRender::invalidate_render_pass()
{
	if (is_main_pass_open())
	{
		m_cmd.end_encoding();
	}

	m_draw_fbo.pass_epoch = 0;
}

void MTLGSRender::renderctl(u32 request_code, void* args)
{
	// There are no backend specific requests (VK: rctrl_queue_submit, offloaded queue submission)
	rsx::thread::renderctl(request_code, args);
}

bool MTLGSRender::scaled_image_from_memory(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate)
{
	if (m_texture_cache.blit(src, dst, interpolate, m_rtts, m_cmd))
	{
		m_samplers_dirty.store(true);

		auto& cb_flags = get_cb_flags();
		cb_flags |= cbf::cb_has_blit_transfer;

		if (cb_flags & cbf::cb_has_dma_transfer)
		{
			// A dma transfer has been queued onto this cb
			// This likely means that we're done with the tranfers to the target (writes_likely_completed=1)
			flush_command_queue();
		}
		return true;
	}

	return false;
}

void MTLGSRender::begin_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	ensure(!m_occlusion_query_active);

	query->result = 0;
	//query->sync_timestamp = get_system_time();
	m_active_query_info = query;
	m_occlusion_query_active = true;
	get_cb_flags() |= cbf::cb_load_occluson_task;
}

void MTLGSRender::end_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	ensure(query == m_active_query_info);

	// NOTE: flushing the queue is very expensive, do not flush just because query stopped
	auto& cb_flags = get_cb_flags();
	if (cb_flags & cbf::cb_has_open_query)
	{
		// Match the pass scope on drivers that require it (VK_1_0 rules in VK)
		if (mtl::use_strict_query_scopes() && is_main_pass_open())
		{
			close_render_pass();
		}

		// End query
		const auto& indices = m_occlusion_map[m_active_query_info->driver_handle].indices;
		if (!indices.empty())
		{
			m_occlusion_query_manager->end_query(m_cmd, indices.back());
		}

		cb_flags &= ~cbf::cb_has_open_query;
	}

	// Clear occlusion load flag
	cb_flags &= ~cbf::cb_load_occluson_task;

	// Stop counting from the next draw on
	m_visibility_slot = umax;
	m_visibility_slot_pass_epoch = 0;
	m_encoder_state.set_visibility_result_mode(MTLVisibilityResultModeDisabled, 0);

	m_occlusion_query_active = false;
	m_active_query_info = nullptr;
}

bool MTLGSRender::check_occlusion_query_status(rsx::reports::occlusion_query_info* query)
{
	if (!query->num_draws)
		return true;

	auto &data = m_occlusion_map[query->driver_handle];
	if (data.indices.empty())
		return true;

	if (data.is_current(m_cmd))
		return false;

	const u32 oldest = data.indices.front();
	return m_occlusion_query_manager->check_query_status(oldest);
}

void MTLGSRender::get_occlusion_query_result(rsx::reports::occlusion_query_info* query)
{
	auto &data = m_occlusion_map[query->driver_handle];
	if (data.indices.empty())
		return;

	if (query->num_draws)
	{
		if (data.is_current(m_cmd))
		{
			std::lock_guard lock(m_flush_queue_mutex);
			flush_command_queue();

			if (m_flush_requests.pending())
			{
				m_flush_requests.clear_pending_flag();
			}

			rsx_log.warning("[Performance warning] Unexpected ZCULL read caused a hard sync");
			utils::busy_wait();
		}

		data.sync();

		// Gather data
		for (const auto occlusion_id : data.indices)
		{
			query->result += m_occlusion_query_manager->get_query_result(occlusion_id);
			if (query->result && !g_cfg.video.precise_zpass_count)
			{
				// We only need one hit unless precise zcull is requested
				break;
			}
		}
	}

	m_occlusion_query_manager->free_queries(m_cmd, data.indices);
	data.indices.clear();
}

void MTLGSRender::discard_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	if (m_active_query_info == query)
	{
		end_occlusion_query(query);
	}

	auto &data = m_occlusion_map[query->driver_handle];
	if (data.indices.empty())
		return;

	m_occlusion_query_manager->free_queries(m_cmd, data.indices);
	data.indices.clear();
}

void MTLGSRender::emergency_query_cleanup(mtl::command_context* commands)
{
	ensure(commands == &m_cmd);

	auto& cb_flags = get_cb_flags();
	if (cb_flags & cbf::cb_has_open_query)
	{
		if (m_active_query_info)
		{
			const auto& indices = m_occlusion_map[m_active_query_info->driver_handle].indices;
			if (!indices.empty())
			{
				m_occlusion_query_manager->end_query(m_cmd, indices.back());
			}
		}

		cb_flags &= ~cbf::cb_has_open_query;
	}

	m_visibility_slot = umax;
	m_visibility_slot_pass_epoch = 0;
	m_encoder_state.set_visibility_result_mode(MTLVisibilityResultModeDisabled, 0);
}
