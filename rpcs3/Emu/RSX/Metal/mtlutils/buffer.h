#pragma once

#include "mtl_common.h"
#include "Emu/RSX/Common/ring_buffer_helper.h"

#include <deque>
#include <memory>

namespace mtl
{
	class command_context;

	class buffer
	{
		id<MTLBuffer> m_buffer = nil;
		usz m_size = 0;

	public:
		buffer(usz size, MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked, const char* label = nullptr);
		~buffer();

		buffer(const buffer&) = delete;
		buffer& operator=(const buffer&) = delete;

		id<MTLBuffer> handle() const { return m_buffer; }
		usz size() const { return m_size; }

		// Shared/managed storage only
		void* map(usz offset = 0) const { return static_cast<u8*>([m_buffer contents]) + offset; }

		void set_label(const std::string& label);
	};

	/**
	 * Ring buffer backed by a single shared-storage MTLBuffer.
	 * Allocations are fenced using command buffer sync ids: every submission records the current put position,
	 * and space is reclaimed once the GPU finishes the corresponding command buffer.
	 *
	 * Usage pattern (same as the VK/GL backends):
	 *     const auto offset = heap.alloc<256>(size);
	 *     void* dst = heap.map(offset);
	 *     ... write data ...
	 *     [encoder setVertexBuffer:heap.handle() offset:offset atIndex:N];
	 *
	 * When an allocation does not fit, the heap first waits for the GPU to release space used by older command buffers.
	 * If the command buffer being recorded holds too much of the heap for that to help, the heap is enlarged instead
	 * (same as the VK data heaps, up to max_heap_size). Enlarging replaces the MTLBuffer: handle() changes, offsets
	 * returned earlier stay valid in the new buffer (pending data is copied over), and work already encoded keeps using
	 * the old buffer, which is kept alive (with its map() pointers) until the current command buffer completes.
	 * Always fetch handle() after allocating; code that only updates offsets (setVertexBufferOffset) must rebind when
	 * handle() differs from the buffer it bound.
	 */
	class ring_buffer : public rsx::data_heap
	{
		struct fence_t
		{
			u64 sync_id;
			usz put_pos;
		};

		std::unique_ptr<buffer> m_buffer;
		command_context* m_ctx = nullptr;
		std::deque<fence_t> m_fences;
		std::string m_label;

		// The submit hook registered with the command context references this ring buffer through this token.
		// destroy() clears the pointee so that a context outliving the ring buffer never calls into a dead object.
		std::shared_ptr<ring_buffer*> m_hook_token;

		// Alignment of the allocation being serviced by grow() (0 if unknown, e.g. static_alloc or bulk_allocator)
		usz m_pending_alignment = 0;

		// Number of times the CPU had to wait for the GPU to release space
		u64 m_stall_count = 0;

		// Number of times the heap had to be enlarged
		u64 m_grow_count = 0;

		// Buffers replaced by an enlargement, released once the command buffer that was being recorded completes
		struct retired_buffer_t
		{
			u64 sync_id;
			std::unique_ptr<buffer> buf;
		};

		std::deque<retired_buffer_t> m_retired_buffers;

		// Exact free space test for an aligned allocation (stricter than data_heap::can_alloc, never crosses the get position)
		bool fits(usz aligned_size, usz alignment) const;
		bool fits(usz aligned_size, usz alignment, usz get_pos) const;

		// Replaces the buffer with a larger one that can hold an allocation of the given size
		bool enlarge(usz aligned_size, usz alignment);

		// Get position after retiring every fence whose command buffer has completed (without modifying the heap)
		usz get_effective_get_pos() const;

	protected:
		bool grow(usz size) override;

	public:
		// Largest alignment accepted by alloc<>(). grow() assumes this alignment for requests of unknown alignment.
		static constexpr usz max_alignment = 4096;

		// Ring buffers are enlarged in steps of heap_growth_step, up to max_heap_size (same limits as the VK data heaps)
		static constexpr usz heap_growth_step = 64 * 0x100000;
		static constexpr usz max_heap_size = 1024 * 0x100000;

		ring_buffer() = default;
		~ring_buffer();

		ring_buffer(const ring_buffer&) = delete;
		ring_buffer& operator=(const ring_buffer&) = delete;

		// Hides rsx::data_heap::alloc. Validates free space with the exact alignment before deferring to the base allocator
		// so that grow() knows the alignment of the request (the base implementation only passes the aligned size).
		template <int Alignment>
		usz alloc(usz size)
		{
			static_assert(Alignment > 0 && static_cast<usz>(Alignment) <= max_alignment && (Alignment & (Alignment - 1)) == 0,
				"Ring buffer allocations must use a power of two alignment no larger than max_alignment");

			const usz alloc_size = utils::align(size, static_cast<usz>(Alignment));
			if (!fits(alloc_size, Alignment))
			{
				m_pending_alignment = Alignment;
				const bool success = grow(alloc_size);
				m_pending_alignment = 0;

				if (!success)
				{
					fmt::throw_exception("[%s] Working buffer not big enough, buffer_length=%d requested=%d guard=%d",
						m_name, m_size, size, m_min_guard_size);
				}
			}

			return rsx::data_heap::alloc<Alignment>(size);
		}

		void create(command_context& ctx, usz size, const char* name, usz guard_size = 0x10000);
		void destroy();

		id<MTLBuffer> handle() const { return m_buffer->handle(); }
		void* map(usz offset) const { return m_buffer->map(offset); }

		// Allocate and return {mapped pointer, offset}
		template <int Alignment>
		std::pair<void*, u32> alloc_map(usz size)
		{
			const usz offset = alloc<Alignment>(size);
			return { map(offset), static_cast<u32>(offset) };
		}

		// Called by command_context for every submitted command buffer
		void on_submit(u64 sync_id);

		// Reclaim space used by completed command buffers
		void retire();

		// Returns true if the heap is more than 3/4 full (including in-flight data)
		bool is_critical() const;
		usz used_bytes() const;
	};

	/**
	 * Scratch memory used as an intermediate for copies and compute conversions (mirrors vk::get_scratch_buffer).
	 * Private storage, at least min_required_size bytes. Contents are undefined between uses.
	 * Hazard tracking makes reuse within a command buffer safe; growing replaces the buffer (the old one stays alive
	 * until the GPU is done with it since command buffers retain their resources).
	 */
	buffer* get_scratch_buffer(command_context& cmd, u64 min_required_size, bool zero_memory = false);

	// CPU-visible (shared) variant for readbacks/uploads that must be accessed by the host
	buffer* get_shared_scratch_buffer(command_context& cmd, u64 min_required_size);

	void clear_scratch_resources();

	/**
	 * Auxiliary CPU-visible upload ring (mirrors vk::get_upload_heap, 64 MiB, grows on demand).
	 * The renderer binds it to its command context with init_upload_heap() on startup; get_upload_heap() creates it lazily.
	 */
	void init_upload_heap(command_context& ctx);
	void destroy_upload_heap();
	ring_buffer* get_upload_heap();
}
