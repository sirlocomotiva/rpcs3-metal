#include "stdafx.h"
#include "MTLGSRender.h"

#include "Emu/RSX/Common/BufferUtils.h"
#include "Emu/RSX/RSXOffload.h"
#include "Emu/RSX/rsx_methods.h"
#include "util/bless.hpp"

#include <span>

// Metal port of VKVertexBuffers.cpp.
//
// Differences from VK:
// - Vertex attribute streams are R8Uint texture buffers (mtl::texel_buffer_view) aliasing a window of the attribute ring.
//   Window starts must be aligned to caps.texture_buffer_alignment and windows are limited to caps.max_texture_buffer_width.
// - Metal always enables primitive restart for indexed draws with the fixed value 0xFFFF / 0xFFFFFFFF.
//   write_index_array_data_to_buffer already rewrites RSX restart indices to those values, but a genuine 0xFFFF index in a
//   16-bit stream would restart the primitive on Metal; such draws are promoted to 32-bit indices.
// - Index buffer offsets must be 4-byte aligned. Multi-range 16-bit draws whose sub-ranges start at odd index offsets are
//   promoted to 32-bit indices as well.
// - Ring buffer space is recycled per command buffer, so the vertex cache is only valid within one command buffer.

namespace mtl
{
	texel_buffer_view::texel_buffer_view(id<MTLBuffer> buffer_, MTLPixelFormat format_, u32 offset_, u32 range_)
		: buffer(buffer_)
		, format(format_)
		, offset(offset_)
		, range(range_)
	{
		ensure(buffer, "Texel buffer view created without a buffer");
		ensure(range > 0, "Texel buffer view created with an empty range");

		const auto& caps = mtl::get_caps();
		const u32 texel_size = mtl::get_format_texel_size(format);

		ensure(texel_size && (range % texel_size) == 0, "Texel buffer view range is not a multiple of the texel size");
		ensure(caps.texture_buffer_alignment == 0 || (offset % caps.texture_buffer_alignment) == 0, "Texel buffer view offset is not aligned");
		ensure(static_cast<u64>(offset) + range <= static_cast<u64>(buffer.length), "Texel buffer view exceeds the buffer");

		const u32 width = range / texel_size;
		ensure(width <= caps.max_texture_buffer_width, "Texel buffer view exceeds the maximum texture buffer width");

		@autoreleasepool
		{
			MTLTextureDescriptor* desc = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:format
				width:width
				resourceOptions:buffer.resourceOptions
				usage:MTLTextureUsageShaderRead];

			value = [buffer newTextureWithDescriptor:desc offset:offset bytesPerRow:range];
		}

		if (!value)
		{
			fmt::throw_exception("Failed to create a texture buffer view (format=%u, offset=0x%x, range=0x%x, buffer length=0x%llx)",
				static_cast<u32>(format), offset, range, static_cast<u64>(buffer.length));
		}
	}
}

namespace
{
	std::tuple<u32, std::tuple<u64, MTLIndexType>> generate_emulating_index_buffer(
		const rsx::draw_clause& clause, u32 vertex_count,
		mtl::ring_buffer& m_index_buffer_ring_info)
	{
		const u32 index_count = get_index_count(clause.primitive, vertex_count);
		const u32 upload_size = index_count * sizeof(u32);

		const u64 offset_in_index_buffer = m_index_buffer_ring_info.alloc<256>(upload_size);
		void* buf = m_index_buffer_ring_info.map(offset_in_index_buffer);

		g_fxo->get<rsx::dma_manager>().emulate_as_indexed(buf, clause.primitive, vertex_count);

		return std::make_tuple(
			index_count, std::make_tuple(offset_in_index_buffer, MTLIndexTypeUInt32));
	}

	struct vertex_input_state
	{
		MTLPrimitiveType native_primitive_type;
		bool index_rebase;
		u32 min_index;
		u32 max_index;
		u32 vertex_draw_count;
		u32 vertex_index_offset;
		std::optional<std::tuple<u64, MTLIndexType>> index_info;
	};

	MTLIndexType get_index_type(rsx::index_array_type type)
	{
		switch (type)
		{
		case rsx::index_array_type::u32:
			return MTLIndexTypeUInt32;
		case rsx::index_array_type::u16:
			return MTLIndexTypeUInt16;
		}
		fmt::throw_exception("Invalid index array type (%u)", static_cast<u8>(type));
	}

	struct draw_command_visitor
	{
		draw_command_visitor(mtl::ring_buffer& index_buffer_ring_info, rsx::vertex_input_layout& layout, rsx::simple_array<u8>& scratch_mem)
			: m_index_buffer_ring_info(index_buffer_ring_info)
			, m_vertex_layout(layout)
			, m_scratch_mem(scratch_mem)
		{
		}

		vertex_input_state operator()(const rsx::draw_array_command& /*command*/)
		{
			const auto [prims, primitives_emulated] = mtl::get_appropriate_topology(rsx::method_registers.current_draw_clause.primitive);
			const u32 vertex_count = rsx::method_registers.current_draw_clause.get_elements_count();
			const u32 min_index = rsx::method_registers.current_draw_clause.min_index();
			const u32 max_index = (min_index + vertex_count) - 1;

			if (primitives_emulated)
			{
				u32 index_count;
				std::optional<std::tuple<u64, MTLIndexType>> index_info;

				std::tie(index_count, index_info) =
					generate_emulating_index_buffer(rsx::method_registers.current_draw_clause,
						vertex_count, m_index_buffer_ring_info);

				return{ prims, false, min_index, max_index, index_count, 0, index_info };
			}

			return{ prims, false, min_index, max_index, vertex_count, 0, {} };
		}

		vertex_input_state operator()(const rsx::draw_indexed_array_command& command)
		{
			const auto& draw_clause = rsx::method_registers.current_draw_clause;
			const auto primitive = draw_clause.primitive;
			const auto [prims, primitives_emulated] = mtl::get_appropriate_topology(primitive);
			const bool restart_index_enabled = rsx::method_registers.restart_index_enabled();
			const u32 restart_index = rsx::method_registers.restart_index();
			const bool emulate_restart = restart_index_enabled && mtl::emulate_primitive_restart(primitive);

			rsx::index_array_type index_type = draw_clause.is_immediate_draw ?
				rsx::index_array_type::u32 :
				rsx::method_registers.index_type();

			const u32 element_count = draw_clause.get_elements_count();
			std::span<const std::byte> raw_index_buffer = command.raw_index_buffer;

			// Metal requires 4-byte aligned index buffer offsets. Sub-ranges of a multi-range draw are addressed by offsetting the
			// index buffer, which breaks that rule for 16-bit indices starting at an odd index. Use 32-bit indices for those.
			if (index_type == rsx::index_array_type::u16 &&
				!draw_clause.is_trivial_instanced_draw &&
				!draw_clause.is_single_draw())
			{
				u32 first_index = 0;
				for (const auto& range : draw_clause.get_subranges())
				{
					if (first_index & 1)
					{
						raw_index_buffer = widen_index_buffer(raw_index_buffer);
						index_type = rsx::index_array_type::u32;
						break;
					}

					first_index += get_index_count(primitive, range.count);
				}
			}

			u64 offset_in_index_buffer = 0;
			u32 min_index = 0, max_index = 0, index_count = 0;

			const auto upload_indices = [&]()
			{
				const u32 type_size = get_index_type_size(index_type);

				index_count = element_count;
				if (primitives_emulated)
					index_count = get_index_count(primitive, index_count);
				u32 upload_size = index_count * type_size;

				if (emulate_restart) upload_size *= 2;

				offset_in_index_buffer = m_index_buffer_ring_info.alloc<64>(upload_size);
				void* buf = m_index_buffer_ring_info.map(offset_in_index_buffer);

				std::span<std::byte> dst;
				stx::single_ptr<std::byte[]> tmp;
				if (emulate_restart)
				{
					tmp = stx::make_single<std::byte[], false, 64>(upload_size);
					dst = std::span<std::byte>(tmp.get(), upload_size);
				}
				else
				{
					dst = std::span<std::byte>(static_cast<std::byte*>(buf), upload_size);
				}

				/**
				* Upload index (and expands it if primitive type is not natively supported).
				*/
				std::tie(min_index, max_index, index_count) = write_index_array_data_to_buffer(
					dst,
					raw_index_buffer, index_type,
					primitive,
					restart_index_enabled,
					restart_index,
					[](auto prim) { return !mtl::is_primitive_native(prim); });

				if (min_index > max_index ||
					(min_index == max_index && primitive != rsx::primitive_type::points))
				{
					// empty set, do not draw
					return false;
				}

				if (emulate_restart)
				{
					if (index_type == rsx::index_array_type::u16)
					{
						index_count = rsx::remove_restart_index(static_cast<u16*>(buf), reinterpret_cast<u16*>(tmp.get()), index_count, u16{umax});
					}
					else
					{
						index_count = rsx::remove_restart_index(static_cast<u32*>(buf), reinterpret_cast<u32*>(tmp.get()), index_count, u32{umax});
					}
				}

				return true;
			};

			if (!upload_indices())
			{
				// empty set, do not draw
				return{ prims, false, 0, 0, 0, 0, {} };
			}

			if (index_type == rsx::index_array_type::u16 && max_index == 0xFFFF)
			{
				// 0xFFFF is a real vertex index here (restart markers are excluded from min/max), but Metal would treat it as a
				// primitive restart. Re-upload with 32-bit indices; restart markers become 0xFFFFFFFF.
				raw_index_buffer = widen_index_buffer(raw_index_buffer);
				index_type = rsx::index_array_type::u32;

				if (!upload_indices())
				{
					return{ prims, false, 0, 0, 0, 0, {} };
				}
			}

			std::optional<std::tuple<u64, MTLIndexType>> index_info =
				std::make_tuple(offset_in_index_buffer, get_index_type(index_type));

			const auto index_offset = rsx::method_registers.vertex_data_base_index();
			return {prims, true, min_index, max_index, index_count, index_offset, index_info};
		}

		vertex_input_state operator()(const rsx::draw_inlined_array& /*command*/)
		{
			auto &draw_clause = rsx::method_registers.current_draw_clause;
			const auto [prims, primitives_emulated] = mtl::get_appropriate_topology(draw_clause.primitive);

			const auto stream_length = rsx::method_registers.current_draw_clause.inline_vertex_array.size();
			const u32 vertex_count = u32(stream_length * sizeof(u32)) / m_vertex_layout.interleaved_blocks[0]->attribute_stride;

			if (!primitives_emulated)
			{
				return{ prims, false, 0, vertex_count - 1, vertex_count, 0, {} };
			}

			u32 index_count;
			std::optional<std::tuple<u64, MTLIndexType>> index_info;
			std::tie(index_count, index_info) = generate_emulating_index_buffer(draw_clause, vertex_count, m_index_buffer_ring_info);
			return{ prims, false, 0, vertex_count - 1, index_count, 0, index_info };
		}

	private:
		// Expands big-endian 16-bit guest indices to big-endian 32-bit indices in the scratch memory.
		// Restart semantics are preserved: write_index_array_data_to_buffer only matches the RSX restart index against the
		// index values, and a restart index above 0xFFFF never matches a 16-bit stream in either representation.
		std::span<const std::byte> widen_index_buffer(std::span<const std::byte> src)
		{
			const usz count = src.size() / sizeof(u16);
			const be_t<u16>* src_indices = utils::bless<const be_t<u16>>(src.data());

			m_scratch_mem.resize(static_cast<u32>(count * sizeof(u32)));
			std::byte* dst_bytes = reinterpret_cast<std::byte*>(m_scratch_mem.data());
			be_t<u32>* dst_indices = utils::bless<be_t<u32>>(dst_bytes);

			for (usz i = 0; i < count; ++i)
			{
				dst_indices[i] = static_cast<u32>(static_cast<u16>(src_indices[i]));
			}

			return std::span<const std::byte>(dst_bytes, count * sizeof(u32));
		}

		mtl::ring_buffer& m_index_buffer_ring_info;
		rsx::vertex_input_layout& m_vertex_layout;
		rsx::simple_array<u8>& m_scratch_mem;
	};
}

mtl::vertex_upload_info MTLGSRender::upload_vertex_data()
{
	draw_command_visitor visitor(m_index_buffer_ring_info, m_vertex_layout, m_scratch_mem);
	auto result = std::visit(visitor, m_draw_processor.get_draw_command(rsx::method_registers));

	if (!result.vertex_draw_count)
	{
		// Empty index set, nothing to draw (emit_geometry aborts on a zero draw count)
		return { result.native_primitive_type, 0, 0, 0, 0, 0, umax, umax, {} };
	}

	const u32 vertex_count = (result.max_index - result.min_index) + 1;
	u32 vertex_base = result.min_index;
	u32 index_base = 0;

	if (result.index_rebase)
	{
		vertex_base = rsx::get_index_from_base(vertex_base, rsx::method_registers.vertex_data_base_index());
		index_base = result.min_index;
	}

	// Cached ranges reference attribute ring offsets. Those are only stable within the command buffer that wrote them:
	// space of completed command buffers is recycled and an enlarged ring only carries over the pending data.
	if (const u64 sync_id = m_cmd.current_sync_id(); m_vertex_cache_sync_id != sync_id)
	{
		m_vertex_cache->purge();
		m_vertex_cache_sync_id = sync_id;
	}

	//Do actual vertex upload
	auto required = calculate_memory_requirements(m_vertex_layout, vertex_base, vertex_count);
	u32 persistent_range_base = -1, volatile_range_base = -1;
	usz persistent_offset = -1, volatile_offset = -1;

	if (required.first > 0)
	{
		//Check if cacheable
		//Only data in the 'persistent' block may be cached
		//TODO: make vertex cache keep local data beyond frame boundaries and hook notify command
		bool in_cache = false;
		bool to_store = false;
		u32  storage_address = -1;

		m_frame_stats.vertex_cache_request_count++;

		if (m_vertex_layout.interleaved_blocks.size() == 1 &&
			rsx::method_registers.current_draw_clause.command != rsx::draw_command::inlined_array)
		{
			const auto data_offset = (vertex_base * m_vertex_layout.interleaved_blocks[0]->attribute_stride);
			storage_address = m_vertex_layout.interleaved_blocks[0]->real_offset_address + data_offset;

			if (auto cached = m_vertex_cache->find_vertex_range(storage_address, required.first))
			{
				ensure(cached->local_address == storage_address);

				in_cache = true;
				persistent_range_base = cached->offset_in_heap;
			}
			else
			{
				to_store = true;
			}
		}

		if (!in_cache)
		{
			m_frame_stats.vertex_cache_miss_count++;

			persistent_offset = static_cast<u32>(m_attrib_ring_info.alloc<256>(required.first));
			persistent_range_base = static_cast<u32>(persistent_offset);

			if (to_store)
			{
				//store ref in vertex cache
				m_vertex_cache->store_range(storage_address, required.first, static_cast<u32>(persistent_offset));
			}
		}
	}

	if (required.second > 0)
	{
		volatile_offset = static_cast<u32>(m_attrib_ring_info.alloc<256>(required.second));
		volatile_range_base = static_cast<u32>(volatile_offset);
	}

	// NOTE: Map only after all allocations are done. An allocation may enlarge the ring, which replaces the buffer.

	//Write all the data once if possible
	if (required.first && required.second && volatile_offset > persistent_offset)
	{
		//Do this once for both to save time on map/unmap cycles
		const usz volatile_offset_in_block = volatile_offset - persistent_offset;

		void *block_mapping = m_attrib_ring_info.map(persistent_offset);
		m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base, vertex_count, block_mapping, static_cast<char*>(block_mapping) + volatile_offset_in_block);
	}
	else
	{
		if (required.first > 0 && persistent_offset != umax)
		{
			void *persistent_mapping = m_attrib_ring_info.map(persistent_offset);
			m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base, vertex_count, persistent_mapping, nullptr);
		}

		if (required.second > 0)
		{
			void *volatile_mapping = m_attrib_ring_info.map(volatile_offset);
			m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base, vertex_count, nullptr, volatile_mapping);
		}
	}

	// Texture buffer windows over the attribute ring. A window is reused while the data fits in it and the ring buffer was not
	// replaced (enlarged) since the window was created (VK: heap_changed interrupt + buffer_view::in_range).
	const auto& caps = mtl::get_caps();
	const mtl::buffer_handle attrib_heap = m_attrib_ring_info.handle();
	const u32 view_alignment = std::max<u32>(caps.texture_buffer_alignment, 1u);
	const u32 max_view_size = m_texbuffer_view_size
		? std::min(m_texbuffer_view_size, caps.max_texture_buffer_width)
		: caps.max_texture_buffer_width;

	const auto update_stream_window = [&](std::unique_ptr<mtl::texel_buffer_view>& storage, u32& range_base, u32 required_size)
	{
		if (storage &&
			storage->buffer == attrib_heap &&
			storage->in_range(range_base, required_size, range_base))
		{
			return;
		}

		// Window starts must be aligned to the texture buffer alignment, the data may start a few bytes into the window
		const u32 window_start = range_base - (range_base % view_alignment);
		const u32 window_delta = range_base - window_start;

		//View 64M blocks at a time (different drivers will only allow a fixed viewable heap size, 64M should be safe)
		u64 view_size = std::min<u64>(max_view_size, m_attrib_ring_info.size() - window_start);
		view_size -= (view_size % view_alignment);

		ensure(view_size >= u64{ window_delta } + required_size, "Vertex data does not fit in a texture buffer window");

		storage = std::make_unique<mtl::texel_buffer_view>(attrib_heap, MTLPixelFormatR8Uint, window_start, static_cast<u32>(view_size));
		range_base = window_delta;
	};

	if (persistent_range_base != umax)
	{
		update_stream_window(m_persistent_attribute_storage, persistent_range_base, required.first);
	}

	if (volatile_range_base != umax)
	{
		update_stream_window(m_volatile_attribute_storage, volatile_range_base, required.second);
	}

	return{ result.native_primitive_type,                 // Primitive
			result.vertex_draw_count,                     // Vertex count
			vertex_count,                                 // Allocated vertex count
			vertex_base,                                  // First vertex in stream
			index_base,                                   // Index of vertex at data location 0
			result.vertex_index_offset,                   // Index offset
			persistent_range_base, volatile_range_base,   // Binding range
			result.index_info };                          // Index buffer info
}
