#include "stdafx.h"
#include "mtl_common.h"

#include <chrono>
#include <cstdlib>

namespace mtl
{
	// Metal only supports points, lines, line strips, triangles and triangle strips.
	// Everything else is expanded into indexed lists by the RSX BufferUtils helpers when is_primitive_native() returns false:
	// - line_loop -> line strip with the first index appended at the end
	// - triangle_fan / polygon -> triangle list
	// - quads -> triangle list
	// Quad strips have the same vertex ordering as triangle strips and are drawn natively.
	bool is_primitive_native(rsx::primitive_type prim)
	{
		switch (prim)
		{
		case rsx::primitive_type::points:
		case rsx::primitive_type::lines:
		case rsx::primitive_type::line_strip:
		case rsx::primitive_type::triangles:
		case rsx::primitive_type::triangle_strip:
		case rsx::primitive_type::quad_strip:
			return true;
		case rsx::primitive_type::line_loop:
		case rsx::primitive_type::triangle_fan:
		case rsx::primitive_type::quads:
		case rsx::primitive_type::polygon:
			return false;
		}

		fmt::throw_exception("Unsupported primitive topology 0x%x", static_cast<u8>(prim));
	}

	MTLPrimitiveType get_primitive_type(rsx::primitive_type prim)
	{
		switch (prim)
		{
		case rsx::primitive_type::points:
			return MTLPrimitiveTypePoint;
		case rsx::primitive_type::lines:
			return MTLPrimitiveTypeLine;
		case rsx::primitive_type::line_loop:
			// Emulated: index buffer is a line strip closed by repeating the first vertex
			return MTLPrimitiveTypeLineStrip;
		case rsx::primitive_type::line_strip:
			return MTLPrimitiveTypeLineStrip;
		case rsx::primitive_type::triangles:
			return MTLPrimitiveTypeTriangle;
		case rsx::primitive_type::triangle_strip:
		case rsx::primitive_type::quad_strip:
			return MTLPrimitiveTypeTriangleStrip;
		case rsx::primitive_type::triangle_fan:
		case rsx::primitive_type::quads:
		case rsx::primitive_type::polygon:
			// Emulated: expanded to a triangle list
			return MTLPrimitiveTypeTriangle;
		}

		fmt::throw_exception("Unsupported primitive topology 0x%x", static_cast<u8>(prim));
	}

	MTLPrimitiveTopologyClass get_primitive_topology_class(rsx::primitive_type prim)
	{
		switch (get_primitive_type(prim))
		{
		case MTLPrimitiveTypePoint:
			return MTLPrimitiveTopologyClassPoint;
		case MTLPrimitiveTypeLine:
		case MTLPrimitiveTypeLineStrip:
			return MTLPrimitiveTopologyClassLine;
		case MTLPrimitiveTypeTriangle:
		case MTLPrimitiveTypeTriangleStrip:
			return MTLPrimitiveTopologyClassTriangle;
		}

		fmt::throw_exception("Unreachable");
	}

	u64 get_stall_trace_threshold_us()
	{
		static const u64 s_threshold_us = []() -> u64
		{
			const char* value = std::getenv("RPCS3_METAL_TRACE_STALLS");
			return value ? std::strtoull(value, nullptr, 10) * 1000 : 0;
		}();

		return s_threshold_us;
	}

	static u64 get_stall_trace_time_us()
	{
		return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	}

	namespace
	{
		thread_local bool g_tls_stall_trace_disabled = false;
	}

	void disable_stall_trace_on_current_thread()
	{
		g_tls_stall_trace_disabled = true;
	}

	stall_trace_scope::stall_trace_scope(const char* what, u64 arg)
		: m_what(what), m_arg(arg)
	{
		if (!g_tls_stall_trace_disabled && get_stall_trace_threshold_us())
		{
			m_start_us = get_stall_trace_time_us();
		}
	}

	stall_trace_scope::~stall_trace_scope()
	{
		if (!m_start_us)
		{
			return;
		}

		const u64 elapsed_us = get_stall_trace_time_us() - m_start_us;
		if (elapsed_us >= get_stall_trace_threshold_us())
		{
			rsx_log.warning("Metal stall: %s (%u) took %.2f ms", m_what, m_arg, elapsed_us / 1000.);
		}
	}
}
