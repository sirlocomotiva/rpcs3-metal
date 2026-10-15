#pragma once

#include "mtl_common.h"
#include "util/atomic.hpp"

#include <functional>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>

namespace mtl
{
	enum class encoder_type : u8
	{
		none = 0,
		render,
		blit,
		compute
	};

	/**
	 * Command recording context.
	 * Owns the command buffer currently being recorded and the single active encoder.
	 * Only one encoder can be active at a time; requesting a different encoder type ends the active one.
	 * Every command buffer gets a monotonically increasing sync id that can be used to wait on its completion.
	 *
	 * NOTE: Command buffers are created with retained references, so any Metal object referenced by an encoder is kept
	 * alive by Metal until the command buffer completes. Releasing our own references after encoding is always safe.
	 * Memory sub-allocated from ring buffers is NOT protected this way; ring buffers use the sync ids for that.
	 */
	class command_context
	{
	public:
		using submit_hook_t = std::function<void(u64 /*sync_id*/)>;
		using completion_hook_t = std::function<void()>;

	private:
		id<MTLCommandQueue> m_queue = nil;
		id<MTLCommandBuffer> m_cmd = nil;

		encoder_type m_encoder_type = encoder_type::none;
		id<MTLRenderCommandEncoder> m_render_encoder = nil;
		id<MTLBlitCommandEncoder> m_blit_encoder = nil;
		id<MTLComputeCommandEncoder> m_compute_encoder = nil;

		// Incremented every time a render encoder is created. Allows clients to detect that their render pass was interrupted.
		u64 m_render_pass_epoch = 0;

		// Sync ids
		u64 m_current_sync_id = 1;          // Id of the command buffer being recorded
		atomic_t<u64> m_completed_sync_id = 0;
		u64 m_last_submitted_sync_id = 0;

		std::mutex m_completion_lock;
		std::condition_variable m_completion_cv;

		std::vector<submit_hook_t> m_submit_hooks;
		std::vector<completion_hook_t> m_pending_completion_handlers;

		// Error tracking
		atomic_t<u32> m_gpu_errors = 0;

		// Descriptor used for every command buffer (retained references + encoder execution status reporting)
		MTLCommandBufferDescriptor* m_cmd_desc = nil;

		// Debug group bookkeeping. Groups pushed while an encoder is active are attached to that encoder and closed
		// automatically when it ends; groups pushed outside of an encoder are attached to the command buffer and closed on submit.
		enum class debug_group_target : u8
		{
			none = 0,        // Not attached (no command buffer, or already closed implicitly)
			command_buffer,
			encoder
		};

		struct debug_group_t
		{
			debug_group_target target;
			u64 owner_id;    // Sync id of the command buffer or id of the encoder the group was pushed on
		};

		std::vector<debug_group_t> m_debug_groups;
		u64 m_encoder_id = 0;  // Incremented for every encoder created

		// Number of committed command buffers whose completion handler has not finished yet (protected by m_completion_lock).
		// Completion handlers reference this object, destruction waits for this to reach zero.
		u64 m_inflight_command_buffers = 0;

		void begin_command_buffer();
		void close_encoder_debug_groups();
		void close_command_buffer_debug_groups();

	public:
		command_context() = default;
		~command_context();

		command_context(const command_context&) = delete;
		command_context& operator=(const command_context&) = delete;

		void create(id<MTLCommandQueue> queue);
		void destroy();

		// Current command buffer (created on demand)
		id<MTLCommandBuffer> get();

		// Encoders
		id<MTLRenderCommandEncoder> begin_render_pass(MTLRenderPassDescriptor* desc, NSString* label = nil);
		id<MTLBlitCommandEncoder> blit();
		id<MTLComputeCommandEncoder> compute();
		void end_encoding();

		encoder_type active_encoder() const { return m_encoder_type; }
		bool is_render_pass_open() const { return m_encoder_type == encoder_type::render; }
		id<MTLRenderCommandEncoder> render_encoder() const { return m_render_encoder; }
		u64 render_pass_epoch() const { return m_render_pass_epoch; }

		// Debug markers (no-ops when no command buffer is active)
		void push_debug_group(const char* name);
		void pop_debug_group();

		// Submission
		// Commits the current command buffer (if it has any work) and returns its sync id.
		// If there is no recorded work, returns the id of the last submitted command buffer.
		u64 submit(bool wait_for_completion = false);

		// Present a drawable at the end of the current command buffer. Must be followed by submit().
		void present_drawable(id<CAMetalDrawable> drawable);

		// Run a callback once the CURRENT command buffer completes (called from a Metal completion thread).
		void add_completion_handler(completion_hook_t handler);

		// Hooks invoked for every submitted command buffer (on the submitting thread), used by ring buffers.
		void add_submit_hook(submit_hook_t hook);

		// Synchronization
		u64 current_sync_id() const { return m_current_sync_id; }
		u64 last_submitted_sync_id() const { return m_last_submitted_sync_id; }
		u64 last_completed_sync_id() const { return m_completed_sync_id.load(); }
		bool is_complete(u64 sync_id) const { return m_completed_sync_id.load() >= sync_id; }

		// Waits for the command buffer with the given id. Submits the current command buffer if required.
		void wait(u64 sync_id);
		void wait_idle();

		bool has_pending_work() const { return m_cmd != nil; }
		u32 gpu_error_count() const { return m_gpu_errors.load(); }
	};
}
