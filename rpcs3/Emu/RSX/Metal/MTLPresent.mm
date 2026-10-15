#include "stdafx.h"
#include "MTLGSRender.h"
#include "MTLTexture.h"
#include "Emu/RSX/Overlays/overlay_manager.h"
#include "Emu/RSX/Overlays/overlay_debug_overlay.h"
#include "Emu/Cell/Modules/cellVideoOut.h"
#include "Emu/System.h"
#include "Utilities/Thread.h"

#include "util/asm.hpp"
#include "util/video_provider.h"

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

#include <bit>
#include <cmath>
#include <cstdlib>

// Metal port of VKPresent.cpp (and VKGSRender::check_present_status).
//
// Presentation model:
// - VK's swapchain is replaced by the CAMetalLayer attached to the frame's native view. The layer owns a small pool of
//   drawables (maximumDrawableCount, VK: swap image count); m_max_async_frames follows it.
// - A drawable is acquired as late as possible in flip() and stored in the frame context. The command buffer that renders
//   into it also schedules the presentation (command_context::present_drawable) and is committed by queue_swap_request.
//   Its sync id (frame_context_t::swap_sync_id) replaces VK's swap_command_buffer.
// - Frame pacing is identical to VK: a frame context is only reused once the command buffer that presented it completed
//   (frame_context_cleanup), which bounds the CPU to m_max_async_frames frames ahead of the GPU.

extern atomic_t<bool> g_user_asked_for_screenshot;
extern atomic_t<recording_mode> g_recording_mode;

namespace
{
	MTLPixelFormat RSX_display_format_to_mtl_format(u8 format)
	{
		switch (format)
		{
		default:
			rsx_log.error("Unhandled video output format 0x%x", static_cast<s32>(format));
			[[fallthrough]];
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_X8R8G8B8:
			return MTLPixelFormatBGRA8Unorm;
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_X8B8G8R8:
			return MTLPixelFormatRGBA8Unorm;
		case CELL_VIDEO_OUT_BUFFER_COLOR_FORMAT_R16G16B16X16_FLOAT:
			return MTLPixelFormatRGBA16Float;
		}
	}

	// Number of drawables of the layer (VK: number of swapchain images).
	// VK negotiates a triple buffered swapchain. With full vsync and no frame limiter the display is the only pacer, and a
	// third drawable only adds a frame of input latency, so the layer is double buffered in that configuration.
	u32 get_desired_drawable_count()
	{
		const vsync_mode vsync = g_cfg.video.vsync;
		const frame_limit_type limiter = g_cfg.video.frame_limit;

		if (vsync == vsync_mode::full && (limiter == frame_limit_type::none || limiter == frame_limit_type::infinite))
		{
			return 2;
		}

		return 3;
	}

	f32 half_to_float(u16 value)
	{
		const u32 sign = static_cast<u32>(value & 0x8000u) << 16;
		const u32 exponent = (value >> 10) & 0x1fu;
		u32 mantissa = value & 0x3ffu;
		u32 bits;

		if (exponent == 0)
		{
			if (mantissa == 0)
			{
				bits = sign;
			}
			else
			{
				// Subnormal, normalize
				u32 shift = 0;
				while (!(mantissa & 0x400u))
				{
					mantissa <<= 1;
					shift++;
				}

				bits = sign | ((127u - 14u - shift) << 23) | ((mantissa & 0x3ffu) << 13);
			}
		}
		else if (exponent == 0x1f)
		{
			// Inf / NaN
			bits = sign | 0x7f800000u | (mantissa << 13);
		}
		else
		{
			bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
		}

		return std::bit_cast<f32>(bits);
	}

	u8 unorm_from_float(f32 value)
	{
		// NaN fails both comparisons and maps to 0
		if (!(value > 0.f))
		{
			return 0;
		}

		if (value >= 1.f)
		{
			return 255;
		}

		return static_cast<u8>(value * 255.f + 0.5f);
	}

	// Returns the CAMetalLayer backing the view, attaching a new one if the view is not backed by one.
	// The view is only touched on the main thread. When called from another thread, the request is posted to the main
	// queue and polled, giving up if the emulator is shutting down (the main thread may be waiting for this thread).
	CAMetalLayer* get_view_metal_layer(NSView* view)
	{
		CAMetalLayer* (^acquire)(void) = ^CAMetalLayer*(void)
		{
			CALayer* current = view.layer;
			if (current && [current isKindOfClass:[CAMetalLayer class]])
			{
				// Qt creates a CAMetalLayer for windows using QSurface::MetalSurface
				return static_cast<CAMetalLayer*>(current);
			}

			CAMetalLayer* layer = [CAMetalLayer layer];
			if (NSWindow* window = view.window)
			{
				layer.contentsScale = window.backingScaleFactor;
			}

			// Layer-hosting view: assign the layer first, then enable layer backing
			view.layer = layer;
			view.wantsLayer = YES;
			return layer;
		};

		if ([NSThread isMainThread])
		{
			return acquire();
		}

		__block CAMetalLayer* result = nil;
		dispatch_semaphore_t done = dispatch_semaphore_create(0);

		dispatch_async(dispatch_get_main_queue(), ^
		{
			result = acquire();
			dispatch_semaphore_signal(done);
		});

		const auto start = std::chrono::steady_clock::now();
		while (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 20 * static_cast<s64>(NSEC_PER_MSEC))) != 0)
		{
			const bool aborting = Emu.GetStatus(false) == system_state::stopping ||
				(thread_ctrl::get_current() && thread_ctrl::state() == thread_state::aborting);

			if (aborting)
			{
				rsx_log.warning("Metal: Gave up waiting for the main thread to set up the output layer (emulation is stopping).");
				return nil;
			}

			if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5))
			{
				rsx_log.error("Metal: Timed out waiting for the main thread to set up the output layer.");
				return nil;
			}
		}

		return result;
	}
}

bool MTLGSRender::init_metal_layer()
{
	@autoreleasepool
	{
		if (!m_metal_layer)
		{
			NSView* view = m_frame ? (__bridge NSView*)m_frame->handle() : nil;
			if (!view)
			{
				rsx_log.error("Metal: The output window has no native view.");
				return false;
			}

			m_metal_layer = get_view_metal_layer(view);
			if (!m_metal_layer)
			{
				rsx_log.error("Metal: Could not attach a CAMetalLayer to the output window.");
				return false;
			}
		}

		const vsync_mode vsync = g_cfg.video.vsync;
		const u32 drawable_count = get_desired_drawable_count();

		// Property changes must not be animated
		[CATransaction begin];
		[CATransaction setDisableActions:YES];

		m_metal_layer.device = m_device->get();
		m_metal_layer.pixelFormat = m_present_format;
		// Drawables are also targets of blit encoder copies (copy_scaled_image fallback) and clears
		m_metal_layer.framebufferOnly = NO;
		m_metal_layer.opaque = YES;
		m_metal_layer.colorspace = [[NSColorSpace sRGBColorSpace] CGColorSpace];
		m_metal_layer.wantsExtendedDynamicRangeContent = NO;
		// VK: FIFO for full/adaptive vsync, IMMEDIATE/MAILBOX when vsync is off
		m_metal_layer.displaySyncEnabled = (vsync != vsync_mode::off);
		m_metal_layer.maximumDrawableCount = drawable_count;
		// nextDrawable returns nil after one second instead of blocking forever (e.g. lost display)
		m_metal_layer.allowsNextDrawableTimeout = YES;

		if (m_swapchain_dims.width && m_swapchain_dims.height)
		{
			m_metal_layer.drawableSize = CGSizeMake(m_swapchain_dims.width, m_swapchain_dims.height);
		}

		[CATransaction commit];

		m_vsync_mode = vsync;

		rsx_log.notice("Metal: Output layer configured [%ux%u, %u drawables, vsync %s]",
			m_swapchain_dims.width, m_swapchain_dims.height, drawable_count, vsync != vsync_mode::off ? "on" : "off");
	}

	// Nothing can be presented to an empty surface (e.g. minimized window)
	return m_swapchain_dims.width && m_swapchain_dims.height;
}

bool MTLGSRender::reinitialize_swapchain()
{
	m_swapchain_dims.width = m_frame->client_width();
	m_swapchain_dims.height = m_frame->client_height();

	// Reject requests to acquire new drawables if the window is minimized
	if (m_swapchain_dims.width == 0 || m_swapchain_dims.height == 0)
	{
		swapchain_unavailable = true;
		return false;
	}

	// NOTE: This operation will create a hard sync point
	close_and_submit_command_buffer();

	for (auto& ctx : m_frame_context_storage)
	{
		// Drawables that were acquired but never scheduled for presentation return to the layer's pool when released
		ctx.present_drawable = nil;
	}

	// NOTE: frame_context_cleanup alters the queued_frames structure.
	while (!m_queued_frames.empty())
	{
		auto& frame = m_queued_frames.front();
		if (!frame->swap_sync_id)
		{
			// Drop it
			m_queued_frames.pop_front();
			continue;
		}

		frame_context_cleanup(frame);
	}
	ensure(m_queued_frames.empty());

	// Discard the current upscaling pipeline if any
	m_upscaler.reset();

	// Drain all the queues
	m_cmd.wait_idle();

	// Reset frame context storage
	m_current_frame = nullptr;
	m_max_async_frames = 0;
	m_current_queue_index = 0;
	m_frame_context_storage.clear();

	// Reconfigure the layer (drawable size, drawable count, vsync)
	if (!init_metal_layer())
	{
		rsx_log.warning("Swapchain initialization failed. Request ignored [%dx%d]", m_swapchain_dims.width, m_swapchain_dims.height);
		swapchain_unavailable = true;
		return false;
	}

	// Re-initialize CPU frame contexts
	m_max_async_frames = static_cast<u32>(m_metal_layer.maximumDrawableCount);
	m_frame_context_storage.resize(m_max_async_frames);
	m_current_queue_index = 0;
	m_current_frame = &m_frame_context_storage[0];

	swapchain_unavailable = false;
	should_reinitialize_swapchain = false;
	m_vsync_mode = g_cfg.video.vsync;
	return true;
}

void MTLGSRender::present(mtl::frame_context_t* ctx)
{
	// The presentation was scheduled by the command buffer that rendered into the drawable (queue_swap_request).
	// There is no present result to check: a lost or resized surface is detected when the next drawable is acquired.

	// Presentation image released; reset value
	ctx->present_drawable = nil;
}

void MTLGSRender::advance_queued_frames()
{
	// Check all other frames for completion and clear resources
	check_present_status();

	// Run video memory balancer (VK: vk::vmm_check_memory_usage)
	if (const auto load_severity = mtl::vmm_determine_memory_load_severity();
		load_severity >= rsx::problem_severity::moderate)
	{
		on_vram_exhausted(load_severity);
	}

	// m_rtts storage is double buffered and should be safe to tag on frame boundary
	m_rtts.trim(m_cmd, mtl::vmm_determine_memory_load_severity());

	// Texture cache is also double buffered to prevent use-after-free
	m_texture_cache.on_frame_end();
	m_samplers_dirty.store(true);

	if (m_vertex_cache)
	{
		m_vertex_cache->purge();
	}

	m_current_frame->tag_frame_end();

	m_queued_frames.push_back(m_current_frame);
	ensure(m_queued_frames.size() <= m_max_async_frames);

	m_current_queue_index = (m_current_queue_index + 1) % m_max_async_frames;
	m_current_frame = &m_frame_context_storage[m_current_queue_index];
	m_current_frame->flags |= frame_context_state::dirty;

	mtl::advance_frame_counter();
}

void MTLGSRender::queue_swap_request()
{
	ensure(!m_current_frame->swap_sync_id);

	// Make sure a command buffer exists so that the submission below always produces a new sync id,
	// even if nothing was recorded this frame (no drawable).
	m_cmd.get();
	const u64 expected_sync_id = m_cmd.current_sync_id();

	// Commit the frame. The drawable (if any) is presented once the command buffer completes.
	const u64 sync_id = close_and_submit_command_buffer(m_current_frame->present_drawable);
	m_current_frame->swap_sync_id = sync_id ? sync_id : expected_sync_id;

	// Set up a present request for this frame as well
	present(m_current_frame);

	// Set up new pointers for the next frame
	advance_queued_frames();
}

void MTLGSRender::frame_context_cleanup(mtl::frame_context_t* ctx)
{
	ensure(ctx->swap_sync_id);

	// Perform hard swap here
	{
		mtl::stall_trace_scope trace("Frame context wait", ctx->swap_sync_id);
		m_cmd.wait(ctx->swap_sync_id);
	}

	// Resource cleanup.
	{
		if (m_overlay_manager && m_overlay_manager->has_dirty())
		{
			auto ui_renderer = mtl::get_overlay_pass<mtl::ui_overlay_renderer>();
			m_overlay_manager->lock_shared();

			std::vector<u32> uids_to_dispose;
			uids_to_dispose.reserve(m_overlay_manager->get_dirty().size());

			for (const auto& view : m_overlay_manager->get_dirty())
			{
				ui_renderer->remove_temp_resources(view->uid);
				uids_to_dispose.push_back(view->uid);
			}

			m_overlay_manager->unlock_shared();
			m_overlay_manager->dispose(uids_to_dispose);
		}

		mtl::reset_global_resources();

		if (ctx->last_frame_sync_time > m_last_heap_sync_time)
		{
			m_last_heap_sync_time = ctx->last_frame_sync_time;

			// Heap cleanup; reclaims the ring buffer space consumed by command buffers that have completed
			// (VK: vk::data_heap_manager::restore_snapshot)
			m_attrib_ring_info.retire();
			m_fragment_constants_ring_info.retire();
			m_transform_constants_ring_info.retire();
			m_index_buffer_ring_info.retire();
			m_texture_upload_buffer_ring_info.retire();
			m_instancing_buffer_ring_info.retire();
		}
	}

	ctx->swap_sync_id = 0;

	// Remove from queued list
	while (!m_queued_frames.empty())
	{
		auto frame = m_queued_frames.front();
		m_queued_frames.pop_front();

		if (frame == ctx)
		{
			break;
		}

		// Older frames skipped over here. Command buffers complete in submission order, so these normally completed too;
		// retire them now instead of leaving a stale swap id behind (VK leaves them for a later cleanup).
		if (frame->swap_sync_id && m_cmd.is_complete(frame->swap_sync_id))
		{
			frame->swap_sync_id = 0;
			mtl::advance_completed_frame_counter();
		}
	}

	mtl::advance_completed_frame_counter();
}

void MTLGSRender::check_present_status()
{
	while (!m_queued_frames.empty())
	{
		auto ctx = m_queued_frames.front();
		if (!m_cmd.is_complete(ctx->swap_sync_id))
		{
			return;
		}

		frame_context_cleanup(ctx);
	}
}

mtl::viewable_image* MTLGSRender::get_present_source(/* inout */ mtl::present_surface_info* info, const rsx::avconf& avconfig)
{
	mtl::viewable_image* image_to_flip = nullptr;

	// @FIXME: This entire function needs to be rewritten to go through the texture cache's "upload_texture" routine.
	// That method is not a 1:1 replacement due to handling of insets that is done differently here.

	// Check the surface store first
	const auto format_bpp = rsx::get_format_block_size_in_bytes(info->format);
	const auto overlap_info = m_rtts.get_merged_texture_memory_region(m_cmd,
		info->address, info->width, info->height, info->pitch, format_bpp, rsx::surface_access::transfer_read);

	if (!overlap_info.empty())
	{
		const auto& section = overlap_info.back();
		auto surface = mtl::as_rtt(section.surface);
		bool viable = false;

		if (section.base_address >= info->address)
		{
			const auto surface_width = surface->get_surface_width<rsx::surface_metrics::samples>();
			const auto surface_height = surface->get_surface_height<rsx::surface_metrics::samples>();

			if (section.base_address == info->address)
			{
				// Check for fit or crop
				viable = (surface_width >= info->width && surface_height >= info->height);
			}
			else
			{
				// Check for borders and letterboxing
				const u32 inset_offset = section.base_address - info->address;
				const u32 inset_y = inset_offset / info->pitch;
				const u32 inset_x = (inset_offset % info->pitch) / format_bpp;

				const u32 full_width = surface_width + inset_x + inset_x;
				const u32 full_height = surface_height + inset_y + inset_y;

				viable = (full_width == info->width && full_height == info->height);
			}

			if (viable)
			{
				image_to_flip = section.surface->get_surface(rsx::surface_access::transfer_read);

				std::tie(info->width, info->height) = rsx::apply_resolution_scale<true>(
					resolution_scaling_config,
					std::min(surface_width, info->width),
					std::min(surface_height, info->height));
			}
		}
	}
	else if (auto surface = m_texture_cache.find_texture_from_dimensions<true>(info->address, info->format);
			 surface && surface->get_width() >= info->width && surface->get_height() >= info->height)
	{
		// Hack - this should be the first location to check for output
		// The render might have been done offscreen or in software and a blit used to display
		image_to_flip = surface->get_raw_texture();
	}

	// The correct output format is determined by the AV configuration set in CellVideoOutConfigure by the game.
	// 99.9% of the time, this will match the backbuffer fbo format used in rendering/compositing the output.
	// But in some cases, let's just say some devs are creative.
	const auto expected_format = RSX_display_format_to_mtl_format(avconfig.format);

	if (!image_to_flip) [[ unlikely ]]
	{
		// Read from cell
		const auto range = utils::address_range32::start_length(info->address, info->pitch * info->height);
		const u32  lookup_mask = rsx::texture_upload_context::blit_engine_dst | rsx::texture_upload_context::framebuffer_storage;
		const auto overlap = m_texture_cache.find_texture_from_range<true>(range, 0, lookup_mask);

		for (const auto & section : overlap)
		{
			if (!section->is_synchronized())
			{
				section->copy_texture(m_cmd, true);
			}
		}

		if (get_cb_flags() & mtl::command_buffer_flags::cb_has_dma_transfer)
		{
			// Submit for processing to lower hard fault penalty
			flush_command_queue();
		}

		m_texture_cache.invalidate_range(m_cmd, range, rsx::invalidation_cause::read);
		image_to_flip = m_texture_cache.upload_image_simple(m_cmd, expected_format, info->address, info->width, info->height, info->pitch);
	}
	else if (image_to_flip->format() != expected_format)
	{
		// Devs are being creative. Force-cast this to the proper pixel layout.
		auto dst_img = m_texture_cache.create_temporary_subresource_storage(
			RSX_FORMAT_CLASS_COLOR, expected_format, static_cast<u16>(info->width), static_cast<u16>(info->height), 1, 1, 1,
			MTLTextureType2D, MTLTextureUsageShaderRead);

		if (dst_img)
		{
			const areai src_rect = { 0, 0, static_cast<int>(info->width), static_cast<int>(info->height) };
			const areai dst_rect = src_rect;

			if (mtl::formats_are_bitcast_compatible(dst_img.get(), image_to_flip))
			{
				mtl::copy_image(m_cmd, image_to_flip, dst_img.get(), coord3i(src_rect), coord3i(dst_rect));
			}
			else
			{
				mtl::copy_image_typeless(m_cmd, image_to_flip, dst_img.get(), coord3i(src_rect), coord3i(dst_rect));
			}

			image_to_flip = dst_img.get();
			m_texture_cache.dispose_reusable_image(dst_img);
		}
	}

	return image_to_flip;
}

void MTLGSRender::flip(const rsx::display_flip_info_t& info)
{
	@autoreleasepool
	{
		// Check swapchain condition/status
		if (m_swapchain_dims.width != m_frame->client_width() + 0u ||
			m_swapchain_dims.height != m_frame->client_height() + 0u)
		{
			swapchain_unavailable = true;
		}

		if (m_vsync_mode != g_cfg.video.vsync)
		{
			swapchain_unavailable = true;
		}

		if (!m_metal_layer || m_frame_context_storage.empty() ||
			static_cast<u32>(m_metal_layer.maximumDrawableCount) != get_desired_drawable_count())
		{
			// Layer missing or drawable count (frame queue depth) changed
			swapchain_unavailable = true;
		}

		if (swapchain_unavailable || should_reinitialize_swapchain)
		{
			// Reinitializing the swapchain is a failable operation. However, not all failures are fatal (e.g minimized window).
			// In such scenarios, we have to retry a few times before giving up as we cannot proceed without a swapchain.
			for (int i = 0; i < 10; ++i)
			{
				if (reinitialize_swapchain() || m_current_frame)
				{
					// If m_current_frame exists, then the initialization failure is non-fatal. Proceed as usual.
					break;
				}

				if (Emu.IsStopped())
				{
					m_frame->flip(m_context);
					rsx::thread::flip(info);
					return;
				}

				std::this_thread::sleep_for(100ms);
			}
		}

		m_profiler.start();

		ensure(m_current_frame, "Invalid swapchain setup. Resizing the game window failed.");

		if (m_current_frame == &m_aux_frame_context)
		{
			m_current_frame = &m_frame_context_storage[m_current_queue_index];
			if (m_current_frame->swap_sync_id)
			{
				// Its possible this flip request is triggered by overlays and the flip queue is in undefined state
				frame_context_cleanup(m_current_frame);
			}

			// Swap aux storage and current frame; aux storage should always be ready for use at all times
			m_current_frame->grab_resources(m_aux_frame_context);
		}
		else if (m_current_frame->swap_sync_id)
		{
			if (info.stats.draw_calls > 0)
			{
				// This can be 'legal' if the window was being resized and no polling happened because of swapchain_unavailable flag
				rsx_log.error("Possible data corruption on frame context storage detected");
			}

			// There were no draws and back-to-back flips happened
			frame_context_cleanup(m_current_frame);
		}

		if (info.skip_frame || swapchain_unavailable)
		{
			if (!info.skip_frame)
			{
				ensure(swapchain_unavailable);

				// Perform a mini-flip here without invoking present code
				m_cmd.get();
				m_current_frame->swap_sync_id = m_cmd.current_sync_id();
				flush_command_queue(true);
				mtl::advance_frame_counter();
				frame_context_cleanup(m_current_frame);
			}

			m_frame->flip(m_context);
			rsx::thread::flip(info);
			return;
		}

		u32 buffer_width = display_buffers[info.buffer].width;
		u32 buffer_height = display_buffers[info.buffer].height;
		u32 buffer_pitch = display_buffers[info.buffer].pitch;

		u32 av_format;
		const auto& avconfig = g_fxo->get<rsx::avconf>();

		if (!buffer_width)
		{
			buffer_width = avconfig.resolution_x;
			buffer_height = avconfig.resolution_y;
		}

		if (avconfig.state)
		{
			av_format = avconfig.get_compatible_gcm_format();
			if (!buffer_pitch)
				buffer_pitch = buffer_width * avconfig.get_bpp();

			const size2u video_frame_size = avconfig.video_frame_size();
			buffer_width = std::min(buffer_width, video_frame_size.width);
			buffer_height = std::min(buffer_height, video_frame_size.height);
		}
		else
		{
			av_format = CELL_GCM_TEXTURE_A8R8G8B8;
			if (!buffer_pitch)
				buffer_pitch = buffer_width * 4;
		}

		// Scan memory for required data. This is done early to optimize waiting for the drawable acquire below.
		mtl::viewable_image* image_to_flip = nullptr;
		mtl::viewable_image* image_to_flip2 = nullptr;

		if (info.buffer < display_buffers_count && buffer_width && buffer_height)
		{
			mtl::present_surface_info present_info
			{
				.address = rsx::get_address(display_buffers[info.buffer].offset, CELL_GCM_LOCATION_LOCAL),
				.format = av_format,
				.width = buffer_width,
				.height = buffer_height,
				.pitch = buffer_pitch,
				.eye = 0
			};
			image_to_flip = get_present_source(&present_info, avconfig);

			if (avconfig.stereo_enabled && image_to_flip) [[unlikely]]
			{
				const auto [unused, min_expected_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, RSX_SURFACE_DIMENSION_IGNORED, buffer_height + 30);
				if (image_to_flip->height() < min_expected_height)
				{
					// Get image for second eye
					const u32 image_offset = (buffer_height + 30) * buffer_pitch + display_buffers[info.buffer].offset;
					present_info.width = buffer_width;
					present_info.height = buffer_height;
					present_info.address = rsx::get_address(image_offset, CELL_GCM_LOCATION_LOCAL);
					present_info.eye = 1;

					image_to_flip2 = get_present_source(&present_info, avconfig);
				}
				else
				{
					// Account for possible insets
					const auto [unused2, scaled_buffer_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, RSX_SURFACE_DIMENSION_IGNORED, buffer_height);
					buffer_height = std::min<u32>(image_to_flip->height() - min_expected_height, scaled_buffer_height);
				}
			}

			buffer_width = present_info.width;
			buffer_height = present_info.height;
		}

		if (info.emu_flip)
		{
			evaluate_cpu_usage_reduction_limits();
		}

		// Prepare surface for new frame
		ensure(!m_current_frame->present_drawable);
		ensure(!m_current_frame->swap_sync_id);

		// Calculate output dimensions (recomputed below if the acquired drawable does not have the expected size)
		const auto get_output_area = [&](const sizeu& surface_dims) -> areai
		{
			if (!g_cfg.video.stretch_to_display_area)
			{
				const auto converted = avconfig.aspect_convert_region({ buffer_width, buffer_height }, surface_dims);
				return static_cast<areai>(converted);
			}

			return { 0, 0, s32(surface_dims.width), s32(surface_dims.height) };
		};

		sizeu target_dims = m_swapchain_dims;
		areai aspect_ratio = get_output_area(target_dims);

		const bool has_overlay = (m_overlay_manager && m_overlay_manager->has_visible());

		// Debugging aids (environment):
		//   RPCS3_METAL_SCREENSHOT_EVERY=N  saves a screenshot of the flipped image every N flips
		//   RPCS3_METAL_TRACE_FLIP=N        logs the flip source every N flips
		//   RPCS3_METAL_DEBUG_CLEAR=1       clears the output to magenta instead of black
		//   RPCS3_METAL_DEBUG_CLEAR=2       presents only the magenta clear (checks presentation without the flip source)
		static const u32 s_debug_screenshot_every = [] { const char* v = ::getenv("RPCS3_METAL_SCREENSHOT_EVERY"); return v ? static_cast<u32>(std::atoi(v)) : 0u; }();
		static const u32 s_debug_trace_every = [] { const char* v = ::getenv("RPCS3_METAL_TRACE_FLIP"); return v ? static_cast<u32>(std::atoi(v)) : 0u; }();
		static const u32 s_debug_clear_mode = [] { const char* v = ::getenv("RPCS3_METAL_DEBUG_CLEAR"); return v ? static_cast<u32>(std::atoi(v)) : 0u; }();
		static const bool s_debug_clear = s_debug_clear_mode != 0;
		static u32 s_debug_flip_count = 0;
		s_debug_flip_count++;

		if (s_debug_trace_every && (s_debug_flip_count % s_debug_trace_every) == 0)
		{
			rsx_log.notice("Metal flip #%u: buffer=%u addr=0x%x %ux%u pitch=%u fmt=0x%x image=%p (%ux%u, mtlfmt=%u) draws=%u",
				s_debug_flip_count, info.buffer, info.buffer < display_buffers_count ? rsx::get_address(display_buffers[info.buffer].offset, CELL_GCM_LOCATION_LOCAL) : 0,
				buffer_width, buffer_height, buffer_pitch, av_format, static_cast<void*>(image_to_flip),
				image_to_flip ? image_to_flip->width() : 0, image_to_flip ? image_to_flip->height() : 0,
				image_to_flip ? static_cast<u32>(image_to_flip->format()) : 0, info.stats.draw_calls);
		}

		const bool user_asked_for_screenshot = g_user_asked_for_screenshot.exchange(false) ||
			(s_debug_screenshot_every && (s_debug_flip_count % s_debug_screenshot_every) == 0);
		const bool user_is_recording = (g_recording_mode != recording_mode::stopped && m_frame->can_consume_frame());
		const bool need_media_capture = user_asked_for_screenshot || user_is_recording;

		const auto render_overlays = [&](const mtl::overlay_render_target& target, const areau& area, const areau& display_area)
		{
			if (!has_overlay) return;

			// Lock to avoid modification during run-update chain
			auto ui_renderer = mtl::get_overlay_pass<mtl::ui_overlay_renderer>();
			std::lock_guard lock(*m_overlay_manager);

			for (const auto& view : m_overlay_manager->get_views())
			{
				const areau render_area = view->use_window_space ? display_area : area;
				ui_renderer->run(m_cmd, render_area, target, m_texture_upload_buffer_ring_info, *view.get());
			}
		};

		// NOTE: This is done before the drawable is acquired. The capture requires a hard sync, and the drawable must be
		// rendered to and presented by the same command buffer (VK has the same constraint on the acquired image).
		if (image_to_flip && need_media_capture)
		{
			mtl::image* image_to_copy = image_to_flip;

			const MTLPixelFormat source_format = image_to_flip->format();
			const bool can_render_overlays = (source_format == MTLPixelFormatBGRA8Unorm ||
				source_format == MTLPixelFormatRGBA8Unorm || source_format == MTLPixelFormatRGBA16Float);

			if (g_cfg.video.record_with_overlays && has_overlay && can_render_overlays)
			{
				if (m_overlay_recording_img)
				{
					// Validate
					if (m_overlay_recording_img->format() != image_to_flip->format() ||
						m_overlay_recording_img->width() != image_to_flip->width() ||
						m_overlay_recording_img->height() != image_to_flip->height())
					{
						// Command buffers retain the texture, it can be released right away
						m_overlay_recording_img.reset();
					}
				}

				if (!m_overlay_recording_img)
				{
					m_overlay_recording_img = std::make_unique<mtl::viewable_image>(
						MTLTextureType2D, image_to_flip->format(), image_to_flip->width(), image_to_flip->height(), 1, 1, 1, u8{1},
						MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead,
						RSX_FORMAT_CLASS_COLOR, MTLStorageModePrivate);

					m_overlay_recording_img->set_debug_name("Overlay recording image");
				}

				const areai rect = areai(0, 0, buffer_width, buffer_height);
				mtl::copy_image(m_cmd, image_to_flip, m_overlay_recording_img.get(), coord3i(rect), coord3i(rect));

				const areau display_area = { 0, 0, m_swapchain_dims.width, m_swapchain_dims.height };
				render_overlays(mtl::overlay_render_target::from(m_overlay_recording_img.get()), areau(rect), display_area);

				image_to_copy = m_overlay_recording_img.get();
			}

			// The capture is always handed over as 4 bytes per pixel. RGBA16F sources are read back as is and converted.
			const MTLPixelFormat capture_format = image_to_copy->format();
			const u32 texel_size = std::max<u32>(mtl::get_format_texel_size(capture_format), 1);
			const usz sshot_size = usz{buffer_height} * buffer_width * 4;
			const usz readback_size = usz{buffer_height} * buffer_width * texel_size;

			mtl::buffer sshot_buf(readback_size, MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked, "Screenshot readback");

			mtl::buffer_image_copy copy_info{};
			copy_info.buffer_offset = 0;
			copy_info.buffer_row_length = 0;
			copy_info.buffer_image_height = 0;
			copy_info.aspect = mtl::image_aspect::color;
			copy_info.mip_level = 0;
			copy_info.base_layer = 0;
			copy_info.layer_count = 1;
			copy_info.image_offset = { 0, 0, 0 };
			copy_info.image_extent = { buffer_width, buffer_height, 1 };

			mtl::copy_image_to_buffer(m_cmd, image_to_copy, sshot_buf.handle(), copy_info);

			flush_command_queue(true);

			const auto src = static_cast<const u8*>(sshot_buf.map(0));
			std::vector<u8> sshot_frame(sshot_size);
			bool is_bgra = capture_format == MTLPixelFormatBGRA8Unorm;

			if (capture_format == MTLPixelFormatRGBA16Float)
			{
				// Convert to RGBA8
				const usz texel_count = usz{buffer_height} * buffer_width;
				for (usz i = 0; i < texel_count; ++i)
				{
					for (usz c = 0; c < 4; ++c)
					{
						u16 half_value;
						std::memcpy(&half_value, src + (i * 4 + c) * sizeof(u16), sizeof(u16));
						sshot_frame[i * 4 + c] = unorm_from_float(half_to_float(half_value));
					}
				}

				is_bgra = false;
			}
			else
			{
				if (texel_size != 4)
				{
					rsx_log.error("Unexpected format of the captured image (%u), the capture will be corrupt", static_cast<u32>(capture_format));
				}

				std::memcpy(sshot_frame.data(), src, std::min(sshot_size, readback_size));
			}

			if (user_asked_for_screenshot)
			{
				m_frame->take_screenshot(std::move(sshot_frame), buffer_width, buffer_height, is_bgra);
			}
			else
			{
				m_frame->present_frame(std::move(sshot_frame), buffer_width * 4, buffer_width, buffer_height, is_bgra);
			}
		}

		const output_scaling_mode output_scaling = g_cfg.video.output_scaling.get();

		if (!m_upscaler || m_output_scaling != output_scaling)
		{
			m_output_scaling = output_scaling;

			if (m_output_scaling == output_scaling_mode::nearest)
			{
				m_upscaler = std::make_unique<mtl::nearest_upscale_pass>();
			}
			else if (m_output_scaling == output_scaling_mode::fsr && m_device->caps().supports_metalfx_spatial)
			{
				m_upscaler = std::make_unique<mtl::metalfx_upscale_pass>();
			}
			else
			{
				if (m_output_scaling == output_scaling_mode::fsr)
				{
					rsx_log.warning("MetalFX spatial scaling is not supported on this device. Falling back to bilinear output scaling.");
				}

				m_upscaler = std::make_unique<mtl::bilinear_upscale_pass>();
			}
		}

		rsx::simple_array<mtl::viewable_image*> calibration_src;
		const bool use_full_rgb_range_output = g_cfg.video.full_rgb_range_output.get();
		const bool use_calibration_pass = image_to_flip &&
			(!use_full_rgb_range_output || !rsx::fcmp(avconfig.gamma, 1.f) || avconfig.stereo_enabled);

		if (use_calibration_pass) [[unlikely]]
		{
			if (image_to_flip) calibration_src.push_back(image_to_flip);
			if (image_to_flip2) calibration_src.push_back(image_to_flip2);

			if (m_output_scaling == output_scaling_mode::fsr && !avconfig.stereo_enabled) // 3D will be implemented later
			{
				// Run upscaling pass before the rest of the output effects pipeline
				// This can be done with all upscalers but we already get bilinear upscaling for free if we just out the filters directly
				const areai src_area = { 0, 0, s32(buffer_width), s32(buffer_height) };
				const areai dst_area = { 0, 0, aspect_ratio.width(), aspect_ratio.height() };

				for (unsigned i = 0; i < calibration_src.size(); ++i)
				{
					const rsx::flags32_t mode = (i == 0) ? UPSCALE_LEFT_VIEW : UPSCALE_RIGHT_VIEW;
					calibration_src[i] = m_upscaler->scale_output(m_cmd, image_to_flip, nil, src_area, dst_area, mode);
				}
			}
		}

		// Acquire the drawable as late as possible: nextDrawable blocks until one of the layer's drawables is released.
		// The inner pool releases the autoreleased reference right away; the frame context keeps the drawable until present().
		mtl::drawable_handle drawable = nil;
		@autoreleasepool
		{
			drawable = [m_metal_layer nextDrawable];
		}

		mtl::texture_handle target_texture = nil;

		if (drawable)
		{
			m_current_frame->present_drawable = drawable;
			target_texture = drawable.texture;

			const u32 drawable_width = static_cast<u32>(target_texture.width);
			const u32 drawable_height = static_cast<u32>(target_texture.height);

			if (drawable_width != m_swapchain_dims.width || drawable_height != m_swapchain_dims.height)
			{
				// The layer was resized behind our back (e.g. by the window system). Draw to the real size and
				// reconfigure the layer on the next flip.
				should_reinitialize_swapchain = true;
				target_dims = { drawable_width, drawable_height };
				aspect_ratio = get_output_area(target_dims);
			}
		}
		else
		{
			// Timed out (allowsNextDrawableTimeout) or the layer cannot provide drawables. Skip presenting this frame.
			rsx_log.warning("Metal: Failed to acquire a drawable. Frame will not be displayed.");
			should_reinitialize_swapchain = true;
		}

		const areau display_area = { 0, 0, target_dims.width, target_dims.height };

		const auto clear_target = [&]()
		{
			MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor renderPassDescriptor];
			desc.colorAttachments[0].texture = target_texture;
			desc.colorAttachments[0].loadAction = MTLLoadActionClear;
			desc.colorAttachments[0].storeAction = MTLStoreActionStore;
			desc.colorAttachments[0].clearColor = s_debug_clear ? MTLClearColorMake(1., 0., 1., 1.) : MTLClearColorMake(0., 0., 0., 1.);

			m_cmd.begin_render_pass(desc, @"Present surface clear");
			m_cmd.end_encoding();
		};

		if (target_texture && s_debug_clear_mode == 2)
		{
			// Debug: present a solid magenta surface, ignoring the RSX image
			clear_target();
		}
		else if (target_texture)
		{
			// Drawable contents are undefined on acquire: everything outside the output area must be cleared
			const bool need_clear = !image_to_flip ||
				aspect_ratio.x1 > 0 || aspect_ratio.y1 > 0 ||
				aspect_ratio.x2 < s32(target_dims.width) || aspect_ratio.y2 < s32(target_dims.height);

			if (image_to_flip)
			{
				if (use_calibration_pass) [[unlikely]]
				{
					// The pass covers the output area; the load action takes care of the borders
					mtl::get_overlay_pass<mtl::video_out_calibration_pass>()->run(
						m_cmd, areau(aspect_ratio), mtl::overlay_render_target::from(target_texture), calibration_src,
						avconfig.gamma, !use_full_rgb_range_output, avconfig.stereo_enabled,
						need_clear ? MTLLoadActionClear : MTLLoadActionDontCare);
				}
				else
				{
					if (need_clear)
					{
						// Clear the window background to black
						clear_target();
					}

					const areai src_area = { 0, 0, s32(buffer_width), s32(buffer_height) };
					m_upscaler->scale_output(m_cmd, image_to_flip, target_texture, src_area, aspect_ratio, UPSCALE_AND_COMMIT | UPSCALE_DEFAULT_VIEW);
				}
			}
			else
			{
				// Clear the window background to black
				clear_target();
			}
		}

		if (g_cfg.video.debug_overlay || has_overlay)
		{
			if (target_texture)
			{
				render_overlays(mtl::overlay_render_target::from(target_texture), areau(aspect_ratio), display_area);
			}

			if (g_cfg.video.debug_overlay)
			{
				const auto num_dirty_textures = m_texture_cache.get_unreleased_textures_count();
				const auto texture_memory_size = m_texture_cache.get_texture_memory_in_use() / (1024 * 1024);
				const auto tmp_texture_memory_size = m_texture_cache.get_temporary_memory_in_use() / (1024 * 1024);
				const auto num_flushes = m_texture_cache.get_num_flush_requests();
				const auto num_mispredict = m_texture_cache.get_num_cache_mispredictions();
				const auto num_speculate = m_texture_cache.get_num_cache_speculative_writes();
				const auto num_misses = m_texture_cache.get_num_cache_misses();
				const auto num_unavoidable = m_texture_cache.get_num_unavoidable_hard_faults();
				const auto cache_miss_ratio = static_cast<u32>(std::ceil(m_texture_cache.get_cache_miss_ratio() * 100));
				const auto num_texture_upload = m_texture_cache.get_texture_upload_calls_this_frame();
				const auto num_texture_upload_miss = m_texture_cache.get_texture_upload_misses_this_frame();
				const auto texture_upload_miss_ratio = m_texture_cache.get_texture_upload_miss_percentage();
				const auto texture_copies_ellided = m_texture_cache.get_texture_copies_ellided_this_frame();
				const auto vertex_cache_hit_count = (info.stats.vertex_cache_request_count - info.stats.vertex_cache_miss_count);
				const auto vertex_cache_hit_ratio = info.stats.vertex_cache_request_count
					? (vertex_cache_hit_count * 100) / info.stats.vertex_cache_request_count
					: 0;
				const auto program_cache_lookups = info.stats.program_cache_lookups_total;
				const auto program_cache_ellided = info.stats.program_cache_lookups_ellided;
				const auto program_cache_ellision_rate = program_cache_lookups
					? (program_cache_ellided * 100) / program_cache_lookups
					: 0;

				rsx::overlays::set_debug_overlay_text(fmt::format(
					"Internal Resolution:      %s\n"
					"RSX Load:                 %3d%%\n"
					"draw calls: %17d\n"
					"submits: %20d\n"
					"draw call setup: %12dus\n"
					"vertex upload time: %9dus\n"
					"texture upload time: %8dus\n"
					"draw call execution: %8dus\n"
					"submit and flip: %12dus\n"
					"Unreleased textures: %8d\n"
					"Texture cache memory: %7dM\n"
					"Temporary texture memory: %3dM\n"
					"Flush requests: %13d  = %2d (%3d%%) hard faults, %2d unavoidable, %2d misprediction(s), %2d speculation(s)\n"
					"Texture uploads: %12u (%u from CPU - %02u%%, %u copies avoided)\n"
					"Vertex cache hits: %10u/%u (%u%%)\n"
					"Program cache lookup ellision: %u/%u (%u%%)",
					info.stats.framebuffer_stats.to_string(resolution_scaling_config, !backend_config.supports_hw_msaa),
					get_load(), info.stats.draw_calls, info.stats.submit_count, info.stats.setup_time, info.stats.vertex_upload_time,
					info.stats.textures_upload_time, info.stats.draw_exec_time, info.stats.flip_time,
					num_dirty_textures, texture_memory_size, tmp_texture_memory_size,
					num_flushes, num_misses, cache_miss_ratio, num_unavoidable, num_mispredict, num_speculate,
					num_texture_upload, num_texture_upload_miss, texture_upload_miss_ratio, texture_copies_ellided,
					vertex_cache_hit_count, info.stats.vertex_cache_request_count, vertex_cache_hit_ratio,
					program_cache_ellided, program_cache_lookups, program_cache_ellision_rate)
				);
			}
		}

		if (s_debug_trace_every && (s_debug_flip_count % s_debug_trace_every) == 0)
		{
			rsx_log.notice("Metal flip #%u: target=%p %ux%u output=(%d,%d)-(%d,%d) calibration=%d overlay=%d upscaler=%d",
				s_debug_flip_count, (__bridge void*)target_texture, target_dims.width, target_dims.height,
				aspect_ratio.x1, aspect_ratio.y1, aspect_ratio.x2, aspect_ratio.y2, use_calibration_pass ? 1 : 0, has_overlay ? 1 : 0,
				static_cast<int>(m_output_scaling));
		}

		queue_swap_request();

		m_frame_stats.flip_time = m_profiler.duration();

		m_frame->flip(m_context);
		rsx::thread::flip(info);

		// Data sync
		const rsx::surface_scaling_config_t active_res_scaling_config =
		{
			.scale_percent = static_cast<u16>(g_cfg.video.resolution_scale_percent),
			.min_scalable_dimension = static_cast<u16>(g_cfg.video.min_scalable_dimension),
		};

		if (active_res_scaling_config != this->resolution_scaling_config)
		{
			// First, try to reclaim any memory since the res scale upgrade is so memory intensive
			if (const auto severity = mtl::vmm_determine_memory_load_severity();
				severity > rsx::problem_severity::low && m_rtts.handle_memory_pressure(m_cmd, severity))
			{
				flush_command_queue(true);
			}

			// Then apply the change
			m_rtts.sync_scaling_config(m_cmd, active_res_scaling_config);
			this->resolution_scaling_config = active_res_scaling_config;

			// Finally reclaim any unused resources
			if (const auto severity = mtl::vmm_determine_memory_load_severity();
				severity > rsx::problem_severity::low && m_rtts.handle_memory_pressure(m_cmd, severity))
			{
				flush_command_queue(true);
			}
		}
	}
}
