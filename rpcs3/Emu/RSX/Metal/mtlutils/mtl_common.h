#pragma once

// Common header for the native Metal backend.
// All Metal backend sources are Objective-C++ (.mm) compiled with ARC; this header must never be
// included from plain C++ translation units. Use MTLRenderer.h for the C++-facing entry points.

#if !defined(__OBJC__)
#error "Metal backend headers can only be included from Objective-C++ sources"
#endif

#if !__has_feature(objc_arc)
#error "Metal backend sources must be compiled with -fobjc-arc"
#endif

// CarbonCore (pulled in by Foundation) declares a global 'Vector128' union which clashes with the
// Vector128 concept from util/v128.hpp. Rename the Apple one while importing the frameworks.
#pragma push_macro("Vector128")
#undef Vector128
#define Vector128 CarbonCore_Vector128
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#undef Vector128
#pragma pop_macro("Vector128")

#include "util/types.hpp"
#include "util/logs.hpp"
#include "Utilities/StrFmt.h"
#include "Emu/RSX/gcm_enums.h"

#include <string>
#include <string_view>

namespace mtl
{
	// Aspect flags (matches the layout of the GL/VK backends for easy porting)
	enum image_aspect : u32
	{
		color = 1,
		depth = 2,
		stencil = 4
	};

	// Special remap encodings
	enum : u32
	{
		MTL_REMAP_IDENTITY = 0xCAFEBABE,          // Special view encoding to return an identity image view
		MTL_REMAP_VIEW_MULTISAMPLED = 0xDEADBEEF, // Special encoding for multisampled images; returns a multisampled image view
	};

	// Resource binding layout shared between the shader decompilers, the shader compiler and the renderer.
	// Metal exposes per-stage argument tables, these are the slot assignments used by every RSX program.
	namespace binding
	{
		// Vertex stage buffers
		enum vertex_buffer_slot : u32
		{
			vs_context_buffer = 0,        // VertexContextBuffer (scale/offset, clip config, etc)
			vs_layout_buffer = 1,         // VertexLayoutBuffer (attribute descriptors)
			vs_constants_buffer = 2,      // VertexConstantsBuffer (468 x vec4)
			vs_texture_params_buffer = 3, // Vertex texture parameters
			vs_instancing_lut_buffer = 4, // Instanced constants indirection table
			vs_instancing_data_buffer = 5,// Instanced constants storage
		};

		// Vertex stage textures
		enum vertex_texture_slot : u32
		{
			vs_texture0 = 0,              // vtex0..vtex3 use slots 0-3 (samplers 0-3)
			vs_persistent_stream = 4,     // usamplerBuffer persistent_input_stream (texture_buffer<uint>)
			vs_volatile_stream = 5,       // usamplerBuffer volatile_input_stream (texture_buffer<uint>)
		};

		// Fragment stage buffers
		enum fragment_buffer_slot : u32
		{
			fs_state_buffer = 0,          // FragmentStateBuffer
			fs_constants_buffer = 1,      // FragmentConstantsBuffer
			fs_texture_params_buffer = 2, // TextureParametersBuffer
			fs_stipple_buffer = 3,        // RasterizerHeap (polygon stipple pattern)
			fs_blend_params_buffer = 4,   // Programmable blending parameters
		};

		// Fragment stage textures
		enum fragment_texture_slot : u32
		{
			fs_texture0 = 0,              // tex0..tex15 use slots 0-15 (samplers 0-15)
			fs_stencil_mirror0 = 16,      // texN_stencil use slots 16-31 (constexpr samplers)
			fs_depth_input = 32,          // frag_depth (constexpr sampler)
		};

		// Overlay/compute helper passes
		enum helper_slot : u32
		{
			helper_params_buffer = 0,     // Push-constant style parameter block for helper passes
			helper_texture0 = 0,
			helper_vertex_buffer = 30,    // Vertex buffer slot used for stage_in vertex data of helper passes (vertex descriptor layout 30)
		};

		constexpr u32 max_buffer_slots = 31;
	}

	inline NSString* to_ns_string(std::string_view str)
	{
		return [[NSString alloc] initWithBytes:str.data() length:str.size() encoding:NSUTF8StringEncoding];
	}

	inline std::string from_ns_string(NSString* str)
	{
		return str ? std::string([str UTF8String]) : std::string{};
	}

	inline std::string describe_error(NSError* error)
	{
		return error ? from_ns_string([error localizedDescription]) : std::string("<no error>");
	}

	// Primitive helpers
	bool is_primitive_native(rsx::primitive_type prim);
	MTLPrimitiveType get_primitive_type(rsx::primitive_type prim);
	MTLPrimitiveTopologyClass get_primitive_topology_class(rsx::primitive_type prim);

	// Debugging aid: RPCS3_METAL_TRACE_STALLS=<ms> logs blocking GPU waits and synchronous compiles that take longer than <ms>.
	// Returns the threshold in microseconds, or 0 when tracing is disabled.
	u64 get_stall_trace_threshold_us();

	// Turns stall tracing off for the calling thread, for threads that never block the renderer
	void disable_stall_trace_on_current_thread();

	// Reports the scope if it took longer than the stall trace threshold
	class stall_trace_scope
	{
		const char* m_what;
		u64 m_start_us = 0;
		u64 m_arg;

	public:
		stall_trace_scope(const char* what, u64 arg = 0);
		~stall_trace_scope();

		stall_trace_scope(const stall_trace_scope&) = delete;
		stall_trace_scope& operator=(const stall_trace_scope&) = delete;
	};
}
