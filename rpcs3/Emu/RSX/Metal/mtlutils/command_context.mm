#include "stdafx.h"
#include "command_context.h"

#include <memory>

namespace mtl
{
	namespace
	{
		const char* describe_encoder_error_state(MTLCommandEncoderErrorState state)
		{
			switch (state)
			{
			case MTLCommandEncoderErrorStateUnknown: return "unknown";
			case MTLCommandEncoderErrorStateCompleted: return "completed";
			case MTLCommandEncoderErrorStateAffected: return "affected";
			case MTLCommandEncoderErrorStatePending: return "pending";
			case MTLCommandEncoderErrorStateFaulted: return "faulted";
			}

			return "invalid";
		}

		void report_command_buffer_error(id<MTLCommandBuffer> cb, u64 sync_id, u32 error_count)
		{
			NSError* error = cb.error;
			rsx_log.error("Metal: Command buffer #%u ('%s') failed with error %d: %s (total GPU errors: %u)",
				sync_id, from_ns_string(cb.label), error ? static_cast<s32>(error.code) : 0, describe_error(error), error_count);

			if (error)
			{
				NSArray* encoder_infos = error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
				for (id<MTLCommandBufferEncoderInfo> info in encoder_infos)
				{
					std::string signposts;
					for (NSString* signpost in info.debugSignposts)
					{
						if (!signposts.empty())
						{
							signposts += " > ";
						}

						signposts += from_ns_string(signpost);
					}

					rsx_log.error("Metal:   Encoder '%s': %s%s%s", from_ns_string(info.label), describe_encoder_error_state(info.errorState),
						signposts.empty() ? "" : ", debug groups: ", signposts);
				}
			}

			for (id<MTLFunctionLog> log in cb.logs)
			{
				rsx_log.error("Metal:   Shader log: %s", from_ns_string([log description]));
			}
		}
	}

	command_context::~command_context()
	{
		destroy();
	}

	void command_context::create(id<MTLCommandQueue> queue)
	{
		ensure(queue);
		ensure(!m_queue, "Command context already created");

		m_queue = queue;

		m_cmd_desc = [MTLCommandBufferDescriptor new];
		m_cmd_desc.retainedReferences = YES;
		m_cmd_desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
	}

	void command_context::destroy()
	{
		if (!m_queue)
		{
			return;
		}

		// Flush outstanding work and wait for all completion handlers to run since they reference this object
		wait_idle();

		{
			std::unique_lock lock(m_completion_lock);
			m_completion_cv.wait(lock, [this]()
			{
				return m_inflight_command_buffers == 0;
			});
		}

		m_submit_hooks.clear();
		m_pending_completion_handlers.clear();
		m_debug_groups.clear();

		m_cmd = nil;
		m_cmd_desc = nil;
		m_queue = nil;
	}

	void command_context::begin_command_buffer()
	{
		ensure(m_queue, "Command context was not created");
		ensure(!m_cmd);

		@autoreleasepool
		{
			// The returned object is autoreleased. Take a strong reference inside a pool so that the pool's reference
			// does not keep the command buffer (and everything it retains) alive on threads that never drain.
			m_cmd = [m_queue commandBufferWithDescriptor:m_cmd_desc];

			if (!m_cmd)
			{
				fmt::throw_exception("Metal: Failed to allocate a command buffer");
			}

			m_cmd.label = [NSString stringWithFormat:@"RSX #%llu", static_cast<unsigned long long>(m_current_sync_id)];
		}
	}

	id<MTLCommandBuffer> command_context::get()
	{
		if (!m_cmd)
		{
			begin_command_buffer();
		}

		return m_cmd;
	}

	id<MTLRenderCommandEncoder> command_context::begin_render_pass(MTLRenderPassDescriptor* desc, NSString* label)
	{
		ensure(desc);
		end_encoding();

		id<MTLCommandBuffer> cmd = get();

		@autoreleasepool
		{
			m_render_encoder = [cmd renderCommandEncoderWithDescriptor:desc];
		}

		if (!m_render_encoder)
		{
			fmt::throw_exception("Metal: Failed to create a render command encoder");
		}

		if (label)
		{
			m_render_encoder.label = label;
		}

		m_encoder_type = encoder_type::render;
		m_encoder_id++;
		m_render_pass_epoch++;
		return m_render_encoder;
	}

	id<MTLBlitCommandEncoder> command_context::blit()
	{
		if (m_encoder_type == encoder_type::blit)
		{
			return m_blit_encoder;
		}

		end_encoding();

		id<MTLCommandBuffer> cmd = get();

		@autoreleasepool
		{
			m_blit_encoder = [cmd blitCommandEncoder];
		}

		if (!m_blit_encoder)
		{
			fmt::throw_exception("Metal: Failed to create a blit command encoder");
		}

		m_encoder_type = encoder_type::blit;
		m_encoder_id++;
		return m_blit_encoder;
	}

	id<MTLComputeCommandEncoder> command_context::compute()
	{
		if (m_encoder_type == encoder_type::compute)
		{
			return m_compute_encoder;
		}

		end_encoding();

		id<MTLCommandBuffer> cmd = get();

		@autoreleasepool
		{
			m_compute_encoder = [cmd computeCommandEncoder];
		}

		if (!m_compute_encoder)
		{
			fmt::throw_exception("Metal: Failed to create a compute command encoder");
		}

		m_encoder_type = encoder_type::compute;
		m_encoder_id++;
		return m_compute_encoder;
	}

	void command_context::end_encoding()
	{
		if (m_encoder_type == encoder_type::none)
		{
			return;
		}

		close_encoder_debug_groups();

		switch (m_encoder_type)
		{
		case encoder_type::render:
			[m_render_encoder endEncoding];
			m_render_encoder = nil;
			break;
		case encoder_type::blit:
			[m_blit_encoder endEncoding];
			m_blit_encoder = nil;
			break;
		case encoder_type::compute:
			[m_compute_encoder endEncoding];
			m_compute_encoder = nil;
			break;
		case encoder_type::none:
			break;
		}

		m_encoder_type = encoder_type::none;
	}

	void command_context::close_encoder_debug_groups()
	{
		id<MTLCommandEncoder> encoder = nil;
		switch (m_encoder_type)
		{
		case encoder_type::render: encoder = m_render_encoder; break;
		case encoder_type::blit: encoder = m_blit_encoder; break;
		case encoder_type::compute: encoder = m_compute_encoder; break;
		case encoder_type::none: return;
		}

		for (auto it = m_debug_groups.rbegin(); it != m_debug_groups.rend(); ++it)
		{
			if (it->target == debug_group_target::encoder && it->owner_id == m_encoder_id)
			{
				[encoder popDebugGroup];
				it->target = debug_group_target::none;
			}
		}
	}

	void command_context::close_command_buffer_debug_groups()
	{
		if (!m_cmd)
		{
			return;
		}

		for (auto it = m_debug_groups.rbegin(); it != m_debug_groups.rend(); ++it)
		{
			if (it->target == debug_group_target::command_buffer && it->owner_id == m_current_sync_id)
			{
				[m_cmd popDebugGroup];
				it->target = debug_group_target::none;
			}
		}
	}

	void command_context::push_debug_group(const char* name)
	{
		debug_group_t group{ debug_group_target::none, 0 };

		if (m_cmd && name)
		{
			@autoreleasepool
			{
				NSString* label = [NSString stringWithUTF8String:name];

				switch (m_encoder_type)
				{
				case encoder_type::render:
					[m_render_encoder pushDebugGroup:label];
					group = { debug_group_target::encoder, m_encoder_id };
					break;
				case encoder_type::blit:
					[m_blit_encoder pushDebugGroup:label];
					group = { debug_group_target::encoder, m_encoder_id };
					break;
				case encoder_type::compute:
					[m_compute_encoder pushDebugGroup:label];
					group = { debug_group_target::encoder, m_encoder_id };
					break;
				case encoder_type::none:
					[m_cmd pushDebugGroup:label];
					group = { debug_group_target::command_buffer, m_current_sync_id };
					break;
				}
			}
		}

		m_debug_groups.push_back(group);
	}

	void command_context::pop_debug_group()
	{
		if (m_debug_groups.empty())
		{
			return;
		}

		const auto group = m_debug_groups.back();
		m_debug_groups.pop_back();

		switch (group.target)
		{
		case debug_group_target::none:
			break;
		case debug_group_target::command_buffer:
			if (m_cmd && group.owner_id == m_current_sync_id)
			{
				[m_cmd popDebugGroup];
			}
			break;
		case debug_group_target::encoder:
			if (group.owner_id == m_encoder_id)
			{
				switch (m_encoder_type)
				{
				case encoder_type::render: [m_render_encoder popDebugGroup]; break;
				case encoder_type::blit: [m_blit_encoder popDebugGroup]; break;
				case encoder_type::compute: [m_compute_encoder popDebugGroup]; break;
				case encoder_type::none: break;
				}
			}
			break;
		}
	}

	u64 command_context::submit(bool wait_for_completion)
	{
		if (!m_cmd)
		{
			// Nothing recorded
			if (wait_for_completion)
			{
				wait(m_last_submitted_sync_id);
			}

			return m_last_submitted_sync_id;
		}

		end_encoding();
		close_command_buffer_debug_groups();

		const u64 sync_id = m_current_sync_id;

		// Let ring buffers record their fences
		for (const auto& hook : m_submit_hooks)
		{
			hook(sync_id);
		}

		// Completion handling. Runs on a Metal-owned thread.
		// Command buffers submitted to the same queue complete in order, so the completed id only moves forward.
		auto handlers = std::make_shared<std::vector<completion_hook_t>>(std::move(m_pending_completion_handlers));
		m_pending_completion_handlers.clear();

		command_context* self = this;
		[m_cmd addCompletedHandler:^(id<MTLCommandBuffer> cb)
		{
			if (cb.status == MTLCommandBufferStatusError)
			{
				const u32 error_count = ++self->m_gpu_errors;
				report_command_buffer_error(cb, sync_id, error_count);
			}

			for (const auto& handler : *handlers)
			{
				handler();
			}

			// Notify while holding the lock. Once the lock is released this block no longer touches the context,
			// which allows destroy() to safely tear it down as soon as the in-flight count reaches zero.
			std::lock_guard lock(self->m_completion_lock);

			if (self->m_completed_sync_id.load() < sync_id)
			{
				self->m_completed_sync_id.store(sync_id);
			}

			self->m_inflight_command_buffers--;
			self->m_completion_cv.notify_all();
		}];

		{
			std::lock_guard lock(m_completion_lock);
			m_inflight_command_buffers++;
		}

		[m_cmd commit];

		m_cmd = nil;
		m_last_submitted_sync_id = sync_id;
		m_current_sync_id++;

		if (wait_for_completion)
		{
			wait(sync_id);
		}

		return sync_id;
	}

	void command_context::present_drawable(id<CAMetalDrawable> drawable)
	{
		ensure(drawable);

		end_encoding();
		[get() presentDrawable:drawable];
	}

	void command_context::add_completion_handler(completion_hook_t handler)
	{
		ensure(handler);

		// Make sure the handler is attached to a command buffer that will actually be committed
		get();
		m_pending_completion_handlers.push_back(std::move(handler));
	}

	void command_context::add_submit_hook(submit_hook_t hook)
	{
		ensure(hook);
		m_submit_hooks.push_back(std::move(hook));
	}

	void command_context::wait(u64 sync_id)
	{
		if (!sync_id || is_complete(sync_id))
		{
			return;
		}

		if (sync_id >= m_current_sync_id)
		{
			if (sync_id > m_current_sync_id)
			{
				rsx_log.error("Metal: Waiting on a command buffer that was never recorded (id=%u, current=%u)", sync_id, m_current_sync_id);
				sync_id = m_current_sync_id;
			}

			if (!m_cmd)
			{
				// Nothing was recorded for the current id, there is nothing to wait for
				return;
			}

			submit();
		}

		if (is_complete(sync_id))
		{
			return;
		}

		stall_trace_scope trace("GPU wait", sync_id);

		std::unique_lock lock(m_completion_lock);
		m_completion_cv.wait(lock, [&]()
		{
			return m_completed_sync_id.load() >= sync_id;
		});
	}

	void command_context::wait_idle()
	{
		if (m_cmd)
		{
			submit();
		}

		wait(m_last_submitted_sync_id);
	}
}
