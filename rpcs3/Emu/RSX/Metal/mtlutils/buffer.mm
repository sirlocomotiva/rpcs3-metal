#include "stdafx.h"
#include "buffer.h"
#include "command_context.h"
#include "device.h"

#include "util/asm.hpp"

#include <array>
#include <cstring>
#include <vector>

namespace mtl
{
	buffer::buffer(usz size, MTLResourceOptions options, const char* label)
		: m_size(size)
	{
		ensure(size, "Zero-sized Metal buffer");

		const auto& caps = get_caps();
		if (caps.max_buffer_length && size > caps.max_buffer_length)
		{
			fmt::throw_exception("Metal: Requested buffer size 0x%x exceeds the device limit of 0x%x bytes", size, caps.max_buffer_length);
		}

		m_buffer = [get_device() newBufferWithLength:size options:options];
		if (!m_buffer)
		{
			fmt::throw_exception("Metal: Failed to allocate a buffer of 0x%x bytes (options=0x%x)", size, static_cast<u64>(options));
		}

		if (label)
		{
			set_label(label);
		}
	}

	buffer::~buffer()
	{
		m_buffer = nil;
	}

	void buffer::set_label(const std::string& label)
	{
		m_buffer.label = to_ns_string(label);
	}

	// Ring buffer

	ring_buffer::~ring_buffer()
	{
		destroy();
	}

	void ring_buffer::create(command_context& ctx, usz size, const char* name, usz guard_size)
	{
		ensure(!m_buffer, "Ring buffer already created");

		m_label = name ? name : "<unnamed>";
		m_ctx = &ctx;
		m_fences.clear();
		m_retired_buffers.clear();
		m_stall_count = 0;
		m_grow_count = 0;
		m_pending_alignment = 0;

		// Name storage must outlive the heap, data_heap only keeps the pointer
		rsx::data_heap::init(size, m_label.c_str(), guard_size);

		// CPU writes, GPU reads. Shared storage is directly visible to the GPU on unified memory systems.
		m_buffer = std::make_unique<buffer>(size, MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked, m_label.c_str());

		// Fence the data recorded into every submitted command buffer
		m_hook_token = std::make_shared<ring_buffer*>(this);
		ctx.add_submit_hook([token = m_hook_token](u64 sync_id)
		{
			if (ring_buffer* heap = *token)
			{
				heap->on_submit(sync_id);
			}
		});
	}

	void ring_buffer::destroy()
	{
		if (m_hook_token)
		{
			// The hook stays registered with the command context but becomes a no-op
			*m_hook_token = nullptr;
			m_hook_token.reset();
		}

		// Command buffers retain the MTLBuffer, so in-flight work is not affected by releasing our reference
		m_buffer.reset();
		m_retired_buffers.clear();
		m_fences.clear();
		m_ctx = nullptr;
	}

	bool ring_buffer::fits(usz aligned_size, usz alignment) const
	{
		return fits(aligned_size, alignment, m_get_pos);
	}

	bool ring_buffer::fits(usz aligned_size, usz alignment, usz get_pos) const
	{
		const usz aligned_put_pos = utils::align(m_put_pos, alignment);

		if (get_pos >= m_put_pos)
		{
			// Free space is the contiguous range [put, get]
			return aligned_put_pos + aligned_size < get_pos;
		}

		// Free space is [put, end) + [0, get]
		if (aligned_put_pos + aligned_size < m_size)
		{
			return true;
		}

		// Allocations that do not fit at the end of the heap restart at offset 0 (see data_heap::alloc)
		return aligned_size <= get_pos;
	}

	usz ring_buffer::get_effective_get_pos() const
	{
		usz get_pos = m_get_pos;

		if (m_ctx)
		{
			const u64 completed = m_ctx->last_completed_sync_id();
			for (const auto& fence : m_fences)
			{
				if (fence.sync_id > completed)
				{
					break;
				}

				get_pos = fence.put_pos;
			}
		}

		return get_pos;
	}

	void ring_buffer::retire()
	{
		if (!m_ctx)
		{
			return;
		}

		const u64 completed = m_ctx->last_completed_sync_id();
		while (!m_fences.empty() && m_fences.front().sync_id <= completed)
		{
			set_get_pos(m_fences.front().put_pos);
			m_fences.pop_front();
		}

		while (!m_retired_buffers.empty() && m_retired_buffers.front().sync_id <= completed)
		{
			m_retired_buffers.pop_front();
		}
	}

	void ring_buffer::on_submit(u64 sync_id)
	{
		retire();

		// Everything allocated so far (up to put - 1) is referenced by the command buffer being submitted
		const usz put_pos = get_current_put_pos_minus_one();
		const usz last_fenced_pos = m_fences.empty() ? m_get_pos : m_fences.back().put_pos;

		if (put_pos == last_fenced_pos)
		{
			// Nothing was allocated from this heap since the previous submission
			return;
		}

		m_fences.push_back({ sync_id, put_pos });
	}

	bool ring_buffer::grow(usz size)
	{
		ensure(m_ctx && m_buffer, "Ring buffer used before creation");

		// Alignment of the request. Requests coming from the base allocator (static_alloc, bulk_allocator) assume the worst case.
		const usz alignment = m_pending_alignment ? m_pending_alignment : max_alignment;

		retire();

		if (fits(size, alignment))
		{
			return true;
		}

		// Waiting for the GPU only helps if the space held by already submitted command buffers is enough
		if (!m_fences.empty() && fits(size, alignment, m_fences.back().put_pos))
		{
			if (m_stall_count++ == 0)
			{
				rsx_log.warning("[%s] Ring buffer is full, waiting for the GPU to release memory (requested=%u, size=%u). Further stalls are not reported.",
					m_name, size, m_size);
			}

			while (!fits(size, alignment))
			{
				ensure(!m_fences.empty());

				// Wait for the oldest in-flight command buffer and reclaim its space
				m_ctx->wait(m_fences.front().sync_id);
				retire();
			}

			return true;
		}

		// The command buffer being recorded holds too much of the heap (or the request is larger than the heap).
		// Submitting from inside an allocation would break the caller's encoder state, so enlarge the heap like VK does.
		return enlarge(size, alignment);
	}

	bool ring_buffer::enlarge(usz aligned_size, usz alignment)
	{
		const usz old_size = m_size;
		const u64 device_limit = get_caps().max_buffer_length;
		const usz heap_limit = device_limit ? std::min<usz>(max_heap_size, static_cast<usz>(device_limit)) : max_heap_size;

		// The new space starts at the end of the old heap. Keep room for the alignment padding and the guard region.
		const usz required_size = old_size + aligned_size + alignment + m_min_guard_size;
		usz new_size = utils::align(required_size, heap_growth_step);

		if (new_size > heap_limit)
		{
			if (required_size > heap_limit)
			{
				rsx_log.fatal("[%s] Ring buffer cannot be enlarged to hold %u more bytes (alignment %u): size=%u, used=%u, limit=%u",
					m_name, aligned_size, alignment, old_size, used_bytes(), heap_limit);
				return false;
			}

			new_size = heap_limit;
		}

		auto new_buffer = std::make_unique<buffer>(new_size, MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked, m_label.c_str());

		// Data written since the last submission may not have been bound yet; keep it at the same offsets.
		// Older data is only read by command buffers that were already submitted, those keep using the old buffer.
		const usz first_used = m_fences.empty() ? m_get_pos : m_fences.back().put_pos;
		const usz begin = (first_used + 1) % old_size;

		const auto copy_range = [&](usz start, usz end)
		{
			if (end > start)
			{
				std::memcpy(new_buffer->map(start), m_buffer->map(start), end - start);
			}
		};

		if (begin <= m_put_pos)
		{
			copy_range(begin, m_put_pos);
		}
		else
		{
			copy_range(begin, old_size);
			copy_range(0, m_put_pos);
		}

		// Encoded commands reference the old buffer and Metal retains it for them. Our reference is kept until the command buffer
		// being recorded completes so that pointers returned by map() before the switch remain valid.
		m_retired_buffers.push_back({ m_ctx->current_sync_id(), std::move(m_buffer) });
		m_buffer = std::move(new_buffer);

		// New layout: everything below the put position is reserved until the next submission completes.
		// Previous fences describe positions in the old buffer and are meaningless now.
		m_fences.clear();
		m_size = new_size;
		m_get_pos = new_size - 1;

		if (m_pending_alignment)
		{
			// Allocation from alloc<>(), which re-reads the put position after growing: continue in the new space.
			// This also keeps the data at [begin, old_size) reserved when the pending data wraps around the old heap.
			m_put_pos = old_size;
		}

		// Otherwise the request comes from the base allocator (bulk_allocator), which keeps using the put position computed before
		// growing, so the put position must not move. [0, put) stays reserved and [put, new_size) is free and large enough.
		// Pending data that wrapped around the end of the old heap is copied but no longer reserved (VK discards it entirely).

		m_grow_count++;
		rsx_log.warning("[%s] Ring buffer enlarged from %u to %u bytes (request of %u bytes, %u enlargements so far)",
			m_name, old_size, new_size, aligned_size, m_grow_count);
		return true;
	}

	usz ring_buffer::used_bytes() const
	{
		const usz get_pos = get_effective_get_pos();

		if (m_put_pos > get_pos)
		{
			// Used region is (get, put)
			return m_put_pos - get_pos - 1;
		}

		// Free region is [put, get]
		return m_size - (get_pos - m_put_pos + 1);
	}

	bool ring_buffer::is_critical() const
	{
		if (!m_buffer)
		{
			return false;
		}

		const usz used = used_bytes();
		return (used * 4 > m_size * 3) || (used + m_min_guard_size >= m_size);
	}

	// Scratch memory handling. Use double-buffered resource to significantly cut down on GPU stalls (same as VK)
	namespace
	{
		struct scratch_buffer_pool_t
		{
			std::array<std::unique_ptr<buffer>, 2> scratch_buffers;
			u32 current_index = 0;

			std::unique_ptr<buffer>& get_buf()
			{
				auto& ret = scratch_buffers[current_index];
				current_index ^= 1;
				return ret;
			}
		};

		struct shared_scratch_entry_t
		{
			std::unique_ptr<buffer> buf;
			u64 last_use = 0;   // Sync id of the last command buffer that may access the buffer
		};

		// Shared scratch buffers are written/read by the CPU, so a buffer can only be reused once the GPU is done with it.
		// Keep a small pool and hand out buffers that are not referenced by any pending command buffer.
		constexpr usz c_max_shared_scratch_buffers = 8;

		scratch_buffer_pool_t g_scratch_buffer_pool;
		std::vector<shared_scratch_entry_t> g_shared_scratch_buffers;
		bool g_shared_scratch_overflow_reported = false;

		u64 get_scratch_allocation_size(u64 min_required_size, u64 previous_size)
		{
			// Choose optimal size. Grow geometrically to avoid reallocating every time the requirement creeps up.
			u64 alloc_size = utils::align<u64>(min_required_size, 0x100000);

			if (previous_size)
			{
				alloc_size = std::max(alloc_size, previous_size * 2);
			}

			if (const u64 limit = get_caps().max_buffer_length; limit)
			{
				// Never exceed the device limit because of the rounding/growth policy (the request itself is validated on creation)
				alloc_size = std::max(std::min(alloc_size, limit), min_required_size);
			}

			return alloc_size;
		}
	}

	buffer* get_scratch_buffer(command_context& cmd, u64 min_required_size, bool zero_memory)
	{
		auto& scratch_buffer = g_scratch_buffer_pool.get_buf();
		bool is_new = false;
		u64 previous_size = 0;

		if (scratch_buffer && scratch_buffer->size() < min_required_size)
		{
			// Scratch heap cannot fit requirements. Discard it and allocate a new one.
			// Command buffers retain the old buffer until the GPU is done with it.
			previous_size = scratch_buffer->size();
			scratch_buffer.reset();
		}

		if (!scratch_buffer)
		{
			const u64 alloc_size = get_scratch_allocation_size(min_required_size, previous_size);
			scratch_buffer = std::make_unique<buffer>(alloc_size, MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeTracked, "Scratch buffer");
			is_new = true;
		}

		buffer* buf = scratch_buffer.get();

		if (is_new || zero_memory)
		{
			// Zero-initialize the allocated memory
			const u64 zero_length = is_new ? buf->size() : std::min<u64>(utils::align<u64>(min_required_size, 4), buf->size());
			[cmd.blit() fillBuffer:buf->handle() range:NSMakeRange(0, zero_length) value:0];
		}

		return buf;
	}

	buffer* get_shared_scratch_buffer(command_context& cmd, u64 min_required_size)
	{
		ensure(min_required_size, "Zero-sized scratch request");

		const u64 current_id = cmd.current_sync_id();

		shared_scratch_entry_t* best_fit = nullptr;      // Smallest idle buffer that is large enough
		shared_scratch_entry_t* idle_too_small = nullptr; // Idle buffer that can be replaced
		shared_scratch_entry_t* oldest_busy = nullptr;    // Busy buffer with the oldest last use

		for (auto& entry : g_shared_scratch_buffers)
		{
			const bool idle = entry.last_use < current_id && cmd.is_complete(entry.last_use);

			if (idle)
			{
				if (entry.buf->size() >= min_required_size)
				{
					if (!best_fit || entry.buf->size() < best_fit->buf->size())
					{
						best_fit = &entry;
					}
				}
				else if (!idle_too_small || entry.buf->size() < idle_too_small->buf->size())
				{
					idle_too_small = &entry;
				}
			}
			else if (!oldest_busy || entry.last_use < oldest_busy->last_use)
			{
				oldest_busy = &entry;
			}
		}

		shared_scratch_entry_t* result = best_fit;

		if (!result && idle_too_small)
		{
			// Replace an idle buffer that is too small
			const u64 alloc_size = get_scratch_allocation_size(min_required_size, idle_too_small->buf->size());
			idle_too_small->buf = std::make_unique<buffer>(alloc_size, MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked, "Shared scratch buffer");
			result = idle_too_small;
		}

		if (!result && oldest_busy && g_shared_scratch_buffers.size() >= c_max_shared_scratch_buffers &&
			oldest_busy->last_use < current_id)
		{
			// Pool is full and every buffer is in flight. Wait for the oldest submitted one instead of growing the pool.
			cmd.wait(oldest_busy->last_use);

			if (oldest_busy->buf->size() < min_required_size)
			{
				const u64 alloc_size = get_scratch_allocation_size(min_required_size, oldest_busy->buf->size());
				oldest_busy->buf = std::make_unique<buffer>(alloc_size, MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked, "Shared scratch buffer");
			}

			result = oldest_busy;
		}

		if (!result)
		{
			if (g_shared_scratch_buffers.size() >= c_max_shared_scratch_buffers && !g_shared_scratch_overflow_reported)
			{
				// Every pooled buffer is referenced by the command buffer being recorded
				rsx_log.warning("Metal: Shared scratch pool exceeded %u buffers within a single command buffer", c_max_shared_scratch_buffers);
				g_shared_scratch_overflow_reported = true;
			}

			const u64 alloc_size = get_scratch_allocation_size(min_required_size, 0);
			auto& entry = g_shared_scratch_buffers.emplace_back();
			entry.buf = std::make_unique<buffer>(alloc_size, MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked, "Shared scratch buffer");
			result = &entry;
		}

		// The buffer may be accessed by the command buffer being recorded
		result->last_use = current_id;
		return result->buf.get();
	}

	void clear_scratch_resources()
	{
		g_scratch_buffer_pool = {};
		g_shared_scratch_buffers.clear();
		g_shared_scratch_overflow_reported = false;
	}

	namespace
	{
		command_context* g_upload_heap_context = nullptr;
		ring_buffer g_upload_heap;
	}

	void init_upload_heap(command_context& ctx)
	{
		if (g_upload_heap_context && g_upload_heap_context != &ctx)
		{
			g_upload_heap.destroy();
		}

		g_upload_heap_context = &ctx;
	}

	void destroy_upload_heap()
	{
		g_upload_heap.destroy();
		g_upload_heap_context = nullptr;
	}

	ring_buffer* get_upload_heap()
	{
		if (!g_upload_heap.handle())
		{
			ensure(g_upload_heap_context, "The auxiliary upload heap has no command context");
			g_upload_heap.create(*g_upload_heap_context, 64 * 0x100000, "auxiliary upload heap", 0x100000);
		}

		return &g_upload_heap;
	}
}
