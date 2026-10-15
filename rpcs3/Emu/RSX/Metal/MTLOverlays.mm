#include "stdafx.h"
#include "MTLOverlays.h"

#include "../Overlays/overlays.h"
#include "../Program/RSXOverlay.h"

#include "util/asm.hpp"
#include "util/fnv_hash.hpp"
#include "Utilities/StrUtil.h"
#include "Utilities/stereo_config.h"

#include "Emu/Cell/timers.hpp"

#include <initializer_list>

namespace mtl
{
	// Overlay pass registry (same scheme as vk::g_overlay_passes)
	std::unordered_map<u32, std::unique_ptr<mtl::overlay_pass>> g_overlay_passes;

	namespace
	{
		// Applies textual replacements to a shared GLSL snippet. Every pattern must be present so that a change to the
		// snippets is caught early instead of silently producing a shader with the wrong resource layout.
		std::string patch_glsl(std::string src, std::initializer_list<std::pair<std::string_view, std::string_view>> replacements)
		{
			for (const auto& [from, to] : replacements)
			{
				if (src.find(from) == std::string::npos)
				{
					fmt::throw_exception("Metal overlays: shader patch target '%s' not found", std::string(from));
				}

				src = fmt::replace_all(src, from, to);
			}

			return src;
		}

		mtl::shader_module compile_overlay_shader(const std::string& src, mtl::shader_stage stage, const char* label)
		{
			mtl::shader_compile_options options{};
			options.label = label;

			auto result = mtl::compile_glsl(src, stage, options);
			if (!result)
			{
				fmt::throw_exception("Metal overlays: Failed to compile %s shader '%s'\n%s",
					stage == mtl::shader_stage::vertex ? "vertex" : "fragment", label, src);
			}

			return result;
		}

		MTLPrimitiveTopologyClass get_topology_class(MTLPrimitiveType type)
		{
			switch (type)
			{
			case MTLPrimitiveTypePoint:
				return MTLPrimitiveTopologyClassPoint;
			case MTLPrimitiveTypeLine:
			case MTLPrimitiveTypeLineStrip:
				return MTLPrimitiveTopologyClassLine;
			case MTLPrimitiveTypeTriangle:
			case MTLPrimitiveTypeTriangleStrip:
				return MTLPrimitiveTopologyClassTriangle;
			default:
				return MTLPrimitiveTopologyClassUnspecified;
			}
		}

		// Generic fullscreen quad vertex shader without texture coordinates (gl_VertexIndex driven, triangle strip)
		const char* s_positions_only_vs =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"\n"
			"void main()\n"
			"{\n"
			"	vec2 positions[] = {vec2(-1., -1.), vec2(1., -1.), vec2(-1., 1.), vec2(1., 1.)};\n"
			"	gl_Position = vec4(positions[gl_VertexIndex % 4], 0., 1.);\n"
			"}\n";

		// Fullscreen quad with texture coordinates remapped to a normalized source rectangle (u0, v0, u1, v1)
		const char* s_src_rect_vs =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"layout(std140, set=0, binding=0) uniform static_data{ vec4 src_rect; };\n"
			"layout(location=0) out vec2 tc0;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	vec2 positions[] = {vec2(-1., -1.), vec2(1., -1.), vec2(-1., 1.), vec2(1., 1.)};\n"
			"	vec2 coords[] = {vec2(0., 0.), vec2(1., 0.), vec2(0., 1.), vec2(1., 1.)};\n"
			"	gl_Position = vec4(positions[gl_VertexIndex % 4], 0., 1.);\n"
			"	tc0 = mix(src_rect.xy, src_rect.zw, coords[gl_VertexIndex % 4]);\n"
			"}\n";
	}

	// --------------------------------------------------------------------------------------------
	// overlay_render_target
	// --------------------------------------------------------------------------------------------

	overlay_render_target overlay_render_target::from(const mtl::image* img, u32 level, u32 layer)
	{
		ensure(img && img->value);
		ensure(level < img->mipmaps());

		overlay_render_target result{};
		result.texture = img->value;
		result.format = img->format();
		result.level = level;

		if (img->type() == MTLTextureType3D)
		{
			result.depth_plane = layer;
		}
		else
		{
			ensure(layer < img->layers());
			result.slice = layer;
		}

		result.width = std::max(img->width() >> level, 1u);
		result.height = std::max(img->height() >> level, 1u);
		result.samples = std::max<u8>(img->samples(), 1);
		return result;
	}

	overlay_render_target overlay_render_target::from(const mtl::image_view* view)
	{
		ensure(view && view->value);

		const mtl::image* img = view->image();
		if (img && view->swizzle != mtl::default_component_map)
		{
			// Swizzled views are not renderable. Render into the parent image instead.
			if (view->format() == img->format())
			{
				return from(img, view->base_level, view->base_layer);
			}

			// Reinterpreting view: build an unswizzled view with the same format (the render target keeps it alive)
			const bool is_3d = (img->type() == MTLTextureType3D);
			const MTLTextureType type = is_3d
				? MTLTextureType3D
				: (img->samples() > 1 ? MTLTextureType2DMultisample : MTLTextureType2D);

			id<MTLTexture> tex = [img->value newTextureViewWithPixelFormat:view->format()
				textureType:type
				levels:NSMakeRange(view->base_level, 1)
				slices:NSMakeRange(is_3d ? 0 : view->base_layer, 1)];

			if (!tex)
			{
				fmt::throw_exception("Metal overlays: Failed to create a renderable view (format 0x%x -> 0x%x)",
					static_cast<u32>(img->format()), static_cast<u32>(view->format()));
			}

			return from(tex, 0, is_3d ? view->base_layer : 0);
		}

		overlay_render_target result{};
		result.texture = view->value;
		result.format = view->format();
		result.level = 0;
		result.slice = 0;
		result.depth_plane = 0;

		if (img)
		{
			result.width = std::max(img->width() >> view->base_level, 1u);
			result.height = std::max(img->height() >> view->base_level, 1u);
			result.samples = std::max<u8>(img->samples(), 1);
		}
		else
		{
			result.width = static_cast<u32>(view->value.width);
			result.height = static_cast<u32>(view->value.height);
			result.samples = static_cast<u8>(std::max<NSUInteger>(view->value.sampleCount, 1));
		}

		return result;
	}

	overlay_render_target overlay_render_target::from(id<MTLTexture> tex, u32 level, u32 layer)
	{
		ensure(tex);
		ensure(level < tex.mipmapLevelCount);

		overlay_render_target result{};
		result.texture = tex;
		result.format = tex.pixelFormat;
		result.level = level;

		if (tex.textureType == MTLTextureType3D)
		{
			result.depth_plane = layer;
		}
		else
		{
			result.slice = layer;
		}

		result.width = std::max(static_cast<u32>(tex.width) >> level, 1u);
		result.height = std::max(static_cast<u32>(tex.height) >> level, 1u);
		result.samples = static_cast<u8>(std::max<NSUInteger>(tex.sampleCount, 1));
		return result;
	}

	u32 overlay_render_target::aspect() const
	{
		return mtl::get_aspect_flags(format);
	}

	mtl::render_pass_formats overlay_render_target::get_formats() const
	{
		mtl::render_pass_formats result{};
		const u32 aspect_mask = aspect();

		if (aspect_mask & mtl::image_aspect::color)
		{
			result.color_formats[0] = static_cast<u32>(format);
			result.color_count = 1;
		}
		else
		{
			if (aspect_mask & mtl::image_aspect::depth)
			{
				result.depth_format = static_cast<u32>(format);
			}

			if (aspect_mask & mtl::image_aspect::stencil)
			{
				result.stencil_format = static_cast<u32>(format);
			}
		}

		result.sample_count = std::max<u8>(samples, 1);
		return result;
	}

	// --------------------------------------------------------------------------------------------
	// overlay_pipeline_state
	// --------------------------------------------------------------------------------------------

	overlay_pipeline_state::overlay_pipeline_state()
	{
		for (auto& att : pipeline.att_state)
		{
			att.enable = 0;
			att.write_mask = static_cast<u8>(MTLColorWriteMaskAll);
			att.src_rgb = static_cast<u8>(MTLBlendFactorOne);
			att.dst_rgb = static_cast<u8>(MTLBlendFactorZero);
			att.src_alpha = static_cast<u8>(MTLBlendFactorOne);
			att.dst_alpha = static_cast<u8>(MTLBlendFactorZero);
			att.op_rgb = static_cast<u8>(MTLBlendOperationAdd);
			att.op_alpha = static_cast<u8>(MTLBlendOperationAdd);
		}

		pipeline.attachment_count = 1;
		pipeline.primitive_type = static_cast<u8>(MTLPrimitiveTypeTriangleStrip);
		pipeline.alpha_to_coverage = 0;
		pipeline.reserved = 0;

		ds.depth_test = 0;
		ds.depth_write = 0;
		ds.depth_func = static_cast<u8>(MTLCompareFunctionAlways);
		ds.stencil_test = 0;

		for (auto face : { &ds.front, &ds.back })
		{
			face->func = static_cast<u8>(MTLCompareFunctionAlways);
			face->fail_op = static_cast<u8>(MTLStencilOperationKeep);
			face->zfail_op = static_cast<u8>(MTLStencilOperationKeep);
			face->pass_op = static_cast<u8>(MTLStencilOperationKeep);
			face->read_mask = 0xFF;
			face->write_mask = 0xFF;
			face->reserved[0] = face->reserved[1] = 0;
		}

		stencil_reference = 0;
	}

	void overlay_pipeline_state::set_attachment_count(u32 count)
	{
		ensure(count <= 4);
		pipeline.attachment_count = static_cast<u8>(count);
	}

	void overlay_pipeline_state::set_color_mask(int index, bool r, bool g, bool b, bool a)
	{
		u8 mask = 0;
		if (a) mask |= static_cast<u8>(MTLColorWriteMaskAlpha);
		if (b) mask |= static_cast<u8>(MTLColorWriteMaskBlue);
		if (g) mask |= static_cast<u8>(MTLColorWriteMaskGreen);
		if (r) mask |= static_cast<u8>(MTLColorWriteMaskRed);

		set_color_write_mask(index, mask);
	}

	void overlay_pipeline_state::set_color_write_mask(int index, u8 mask)
	{
		ensure(index >= 0 && index < 4);
		pipeline.att_state[index].write_mask = mask;
	}

	void overlay_pipeline_state::set_depth_mask(bool enable)
	{
		ds.depth_write = enable ? 1 : 0;
	}

	void overlay_pipeline_state::set_stencil_mask(u32 mask)
	{
		ds.front.write_mask = static_cast<u8>(mask);
		ds.back.write_mask = static_cast<u8>(mask);
	}

	void overlay_pipeline_state::enable_depth_test(MTLCompareFunction op)
	{
		ds.depth_test = 1;
		ds.depth_func = static_cast<u8>(op);
	}

	void overlay_pipeline_state::disable_depth_test()
	{
		ds.depth_test = 0;
		ds.depth_write = 0;
		ds.depth_func = static_cast<u8>(MTLCompareFunctionAlways);
	}

	void overlay_pipeline_state::enable_blend(int mrt_index,
		MTLBlendFactor src_factor_rgb, MTLBlendFactor src_factor_a,
		MTLBlendFactor dst_factor_rgb, MTLBlendFactor dst_factor_a,
		MTLBlendOperation equation_rgb, MTLBlendOperation equation_a)
	{
		ensure(mrt_index >= 0 && mrt_index < 4);
		auto& att = pipeline.att_state[mrt_index];
		att.src_rgb = static_cast<u8>(src_factor_rgb);
		att.src_alpha = static_cast<u8>(src_factor_a);
		att.dst_rgb = static_cast<u8>(dst_factor_rgb);
		att.dst_alpha = static_cast<u8>(dst_factor_a);
		att.op_rgb = static_cast<u8>(equation_rgb);
		att.op_alpha = static_cast<u8>(equation_a);
		att.enable = 1;
	}

	void overlay_pipeline_state::disable_blend(int mrt_index)
	{
		ensure(mrt_index >= 0 && mrt_index < 4);
		pipeline.att_state[mrt_index].enable = 0;
	}

	void overlay_pipeline_state::enable_stencil_test(MTLStencilOperation fail, MTLStencilOperation zfail, MTLStencilOperation pass,
		MTLCompareFunction func, u32 func_mask, u32 ref)
	{
		ds.front.fail_op = static_cast<u8>(fail);
		ds.front.zfail_op = static_cast<u8>(zfail);
		ds.front.pass_op = static_cast<u8>(pass);
		ds.front.func = static_cast<u8>(func);
		ds.front.read_mask = static_cast<u8>(func_mask);
		ds.back = ds.front;

		ds.stencil_test = 1;
		stencil_reference = ref & 0xFF;
	}

	void overlay_pipeline_state::disable_stencil_test()
	{
		ds.stencil_test = 0;

		for (auto face : { &ds.front, &ds.back })
		{
			face->func = static_cast<u8>(MTLCompareFunctionAlways);
			face->fail_op = static_cast<u8>(MTLStencilOperationKeep);
			face->zfail_op = static_cast<u8>(MTLStencilOperationKeep);
			face->pass_op = static_cast<u8>(MTLStencilOperationKeep);
			face->read_mask = 0xFF;
			face->write_mask = 0xFF;
		}

		stencil_reference = 0;
	}

	void overlay_pipeline_state::set_primitive_type(MTLPrimitiveType type)
	{
		pipeline.primitive_type = static_cast<u8>(type);
	}

	void overlay_pipeline_state::set_alpha_to_coverage(bool enable)
	{
		pipeline.alpha_to_coverage = enable ? 1 : 0;
	}

	// --------------------------------------------------------------------------------------------
	// overlay_vertex_heap
	// --------------------------------------------------------------------------------------------

	std::pair<id<MTLBuffer>, u32> overlay_vertex_heap::alloc(mtl::command_context& cmd, const void* data, u32 size)
	{
		ensure(data && size);

		const u32 aligned_size = utils::align(size, 256u);
		chunk_t* target = nullptr;

		if (m_current < m_chunks.size() &&
			(m_chunks[m_current].put + aligned_size) <= m_chunks[m_current].size)
		{
			// Appending after the last write is always safe
			target = &m_chunks[m_current];
		}
		else
		{
			// Look for an idle chunk that can be rewound
			for (usz i = 0; i < m_chunks.size(); ++i)
			{
				auto& chunk = m_chunks[i];
				if (chunk.size >= aligned_size && cmd.is_complete(chunk.sync_id))
				{
					chunk.put = 0;
					target = &chunk;
					m_current = i;
					break;
				}
			}

			if (!target)
			{
				chunk_t chunk{};
				chunk.size = std::max(default_chunk_size, aligned_size);
				chunk.buffer = [mtl::get_device() newBufferWithLength:chunk.size
					options:(MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined)];

				if (!chunk.buffer)
				{
					fmt::throw_exception("Metal overlays: Failed to allocate %u bytes of vertex memory", chunk.size);
				}

				chunk.buffer.label = @"Overlay vertex heap";

				m_chunks.push_back(chunk);
				m_current = m_chunks.size() - 1;
				target = &m_chunks.back();
			}
		}

		const u32 offset = target->put;
		std::memcpy(static_cast<u8*>([target->buffer contents]) + offset, data, size);

		target->put += aligned_size;
		target->sync_id = cmd.current_sync_id();
		return { target->buffer, offset };
	}

	void overlay_vertex_heap::trim(usz max_chunks)
	{
		max_chunks = std::max<usz>(max_chunks, 1);
		if (m_chunks.size() <= max_chunks)
		{
			return;
		}

		// Command buffers retain the buffers they reference, dropping our references is always safe
		if (m_current < m_chunks.size())
		{
			chunk_t current = m_chunks[m_current];
			m_chunks.clear();
			m_chunks.push_back(current);
			m_current = 0;
		}
		else
		{
			m_chunks.clear();
			m_current = umax;
		}
	}

	void overlay_vertex_heap::destroy()
	{
		m_chunks.clear();
		m_current = umax;
	}

	// --------------------------------------------------------------------------------------------
	// overlay_pass
	// --------------------------------------------------------------------------------------------

	overlay_pass::overlay_pass()
	{
		// Override-able defaults
		renderpass_config.set_primitive_type(MTLPrimitiveTypeTriangleStrip);
	}

	overlay_pass::~overlay_pass()
	{
		m_vao.destroy();
	}

	u64 overlay_pass::get_pipeline_key(const mtl::render_pass_formats& formats)
	{
		// Normalize the formats so that uninitialized padding never affects the key
		mtl::render_pass_formats key_formats{};
		std::memcpy(key_formats.color_formats, formats.color_formats, sizeof(key_formats.color_formats));
		key_formats.depth_format = formats.depth_format;
		key_formats.stencil_format = formats.stencil_format;
		key_formats.color_count = formats.color_count;
		key_formats.sample_count = std::max<u8>(formats.sample_count, 1);

		u64 key = rpcs3::hash_struct(renderpass_config.pipeline);
		key = rpcs3::hash64(key, static_cast<u64>(rpcs3::hash_struct(key_formats)));
		key = rpcs3::hash64(key, get_pipeline_variant());
		return key;
	}

	id<MTLFunction> overlay_pass::get_vertex_function()
	{
		if (!m_vertex_shader)
		{
			m_vertex_shader = compile_overlay_shader(vs_src, mtl::shader_stage::vertex, "overlay_pass_vs");
		}

		return m_vertex_shader.function;
	}

	id<MTLFunction> overlay_pass::get_fragment_function()
	{
		if (!m_fragment_shader)
		{
			m_fragment_shader = compile_overlay_shader(fs_src, mtl::shader_stage::fragment, "overlay_pass_fs");
		}

		return m_fragment_shader.function;
	}

	MTLVertexDescriptor* overlay_pass::get_vertex_descriptor()
	{
		if (!m_use_vertex_input)
		{
			return nil;
		}

		// Single vec4 attribute with a 16 byte stride (same layout as the VK overlay VAO)
		MTLVertexDescriptor* desc = [MTLVertexDescriptor vertexDescriptor];
		desc.attributes[0].format = MTLVertexFormatFloat4;
		desc.attributes[0].offset = 0;
		desc.attributes[0].bufferIndex = mtl::binding::helper_vertex_buffer;
		desc.layouts[mtl::binding::helper_vertex_buffer].stride = 16;
		desc.layouts[mtl::binding::helper_vertex_buffer].stepFunction = MTLVertexStepFunctionPerVertex;
		desc.layouts[mtl::binding::helper_vertex_buffer].stepRate = 1;
		return desc;
	}

	mtl::program* overlay_pass::build_pipeline(u64 storage_key, const mtl::render_pass_formats& formats)
	{
		MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
		desc.label = @"Overlay pass";
		desc.vertexFunction = get_vertex_function();
		desc.fragmentFunction = get_fragment_function();
		desc.vertexDescriptor = get_vertex_descriptor();
		compiled = true;

		const auto& state = renderpass_config.pipeline;

		for (u32 index = 0; index < 4; ++index)
		{
			const auto format = static_cast<MTLPixelFormat>(formats.color_formats[index]);
			MTLRenderPipelineColorAttachmentDescriptor* att = desc.colorAttachments[index];
			att.pixelFormat = format;

			if (format == MTLPixelFormatInvalid)
			{
				continue;
			}

			if (index >= state.attachment_count)
			{
				// Not written by the shader, leave the attachment untouched
				att.writeMask = MTLColorWriteMaskNone;
				att.blendingEnabled = NO;
				continue;
			}

			const auto& blend = state.att_state[index];
			att.writeMask = static_cast<MTLColorWriteMask>(blend.write_mask);
			att.blendingEnabled = blend.enable ? YES : NO;

			if (blend.enable)
			{
				att.sourceRGBBlendFactor = static_cast<MTLBlendFactor>(blend.src_rgb);
				att.destinationRGBBlendFactor = static_cast<MTLBlendFactor>(blend.dst_rgb);
				att.sourceAlphaBlendFactor = static_cast<MTLBlendFactor>(blend.src_alpha);
				att.destinationAlphaBlendFactor = static_cast<MTLBlendFactor>(blend.dst_alpha);
				att.rgbBlendOperation = static_cast<MTLBlendOperation>(blend.op_rgb);
				att.alphaBlendOperation = static_cast<MTLBlendOperation>(blend.op_alpha);
			}
		}

		desc.depthAttachmentPixelFormat = static_cast<MTLPixelFormat>(formats.depth_format);
		desc.stencilAttachmentPixelFormat = static_cast<MTLPixelFormat>(formats.stencil_format);
		desc.rasterSampleCount = std::max<u8>(formats.sample_count, 1);
		desc.alphaToCoverageEnabled = state.alpha_to_coverage ? YES : NO;
		desc.inputPrimitiveTopology = get_topology_class(renderpass_config.primitive_type());

		auto compiler = mtl::get_pipe_compiler();
		auto program = compiler->compile(desc, mtl::pipe_compiler::COMPILE_INLINE);

		if (!program || !program->render_pipeline)
		{
			fmt::throw_exception("Metal overlays: Failed to create the pipeline state of an overlay pass");
		}

		auto result = program.get();
		m_program_cache[storage_key] = std::move(program);

		return result;
	}

	mtl::program* overlay_pass::load_program(mtl::command_context& cmd, const mtl::render_pass_formats& formats, const std::vector<mtl::image_view*>& src)
	{
		mtl::program* program = nullptr;
		const auto key = get_pipeline_key(formats);

		auto found = m_program_cache.find(key);
		if (found != m_program_cache.end())
			program = found->second.get();
		else
			program = build_pipeline(key, formats);

		if (!src.empty() && (!m_sampler || m_active_sampler_filter != m_sampler_filter))
		{
			mtl::sampler_desc desc{};
			desc.address_s = MTLSamplerAddressModeClampToEdge;
			desc.address_t = MTLSamplerAddressModeClampToEdge;
			desc.address_r = MTLSamplerAddressModeClampToEdge;
			desc.min_filter = m_sampler_filter;
			desc.mag_filter = m_sampler_filter;
			desc.mip_filter = MTLSamplerMipFilterNotMipmapped;
			desc.border_color = MTLSamplerBorderColorOpaqueBlack;
			desc.min_lod = 0.f;
			desc.max_lod = 0.f;

			m_sampler = mtl::get_sampler_cache().get(desc);
			m_active_sampler_filter = m_sampler_filter;
		}

		id<MTLRenderCommandEncoder> encoder = cmd.render_encoder();
		ensure(encoder);

		[encoder setRenderPipelineState:program->render_pipeline];
		[encoder setDepthStencilState:mtl::get_depth_stencil_state(renderpass_config.ds)];
		[encoder setStencilReferenceValue:renderpass_config.stencil_reference];

		// Reset dynamic state that could have been left behind by the owner of the render pass
		[encoder setCullMode:MTLCullModeNone];
		[encoder setTriangleFillMode:MTLTriangleFillModeFill];
		[encoder setDepthClipMode:MTLDepthClipModeClip];
		[encoder setDepthBias:0.f slopeScale:0.f clamp:0.f];
		[encoder setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];

		update_uniforms(cmd, program);

		for (u32 n = 0; n < src.size(); ++n)
		{
			if (!src[n])
			{
				continue;
			}

			const u32 location = sampler_location(n);
			[encoder setFragmentTexture:src[n]->value atIndex:location];

			if (location < mtl::first_constexpr_sampler_slot)
			{
				[encoder setFragmentSamplerState:m_sampler atIndex:location];
			}
		}

		if (m_use_vertex_input)
		{
			ensure(m_vao_buffer);
			[encoder setVertexBuffer:m_vao_buffer offset:m_vao_offset atIndex:mtl::binding::helper_vertex_buffer];
		}

		return program;
	}

	void overlay_pass::create(const mtl::render_device& dev)
	{
		if (!initialized)
		{
			m_device = &dev;
			initialized = true;
		}
	}

	void overlay_pass::destroy()
	{
		if (initialized)
		{
			m_vertex_shader = {};
			m_fragment_shader = {};
			m_program_cache.clear();
			m_sampler = nil;
			m_null_views.clear();
			m_vao.destroy();
			m_vao_buffer = nil;
			m_vao_offset = 0;

			compiled = false;
			initialized = false;
		}
	}

	void overlay_pass::free_resources()
	{
		// Vertex memory is recycled automatically, only bound the amount of memory kept around
		m_vao.trim(2);
	}

	mtl::image_view* overlay_pass::get_null_image_view(MTLTextureType type)
	{
		ensure(type == MTLTextureType2D || type == MTLTextureType2DArray);

		auto& entry = m_null_views[static_cast<u32>(type)];
		if (!entry.second)
		{
			entry.first = std::make_unique<mtl::image>(type, MTLPixelFormatRGBA8Unorm, 1, 1, 1, 1, 1, 1,
				MTLTextureUsageShaderRead, RSX_FORMAT_CLASS_COLOR, MTLStorageModeShared);

			const u32 zero = 0;
			[entry.first->value replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 slice:0 withBytes:&zero bytesPerRow:4 bytesPerImage:0];

			entry.second = std::make_unique<mtl::image_view>(entry.first.get(), MTLPixelFormatInvalid, type,
				mtl::default_component_map, mtl::image_aspect::color, 0, 1, 0, 1);
		}

		return entry.second.get();
	}

	void overlay_pass::emit_geometry(mtl::command_context& cmd, mtl::program* /*program*/)
	{
		[cmd.render_encoder() drawPrimitives:renderpass_config.primitive_type()
			vertexStart:first_vertex
			vertexCount:num_drawable_elements];
	}

	void overlay_pass::set_up_viewport(mtl::command_context& cmd, u32 x, u32 y, u32 w, u32 h)
	{
		id<MTLRenderCommandEncoder> encoder = cmd.render_encoder();

		MTLViewport vp{};
		vp.originX = static_cast<f64>(x);
		vp.originY = static_cast<f64>(y);
		vp.width = static_cast<f64>(w);
		vp.height = static_cast<f64>(h);
		vp.znear = 0.;
		vp.zfar = 1.;
		[encoder setViewport:vp];

		// Metal requires the scissor to lie within the attachments
		MTLScissorRect scissor = { x, y, w, h };
		if (m_target_width && m_target_height)
		{
			scissor.x = std::min<NSUInteger>(scissor.x, m_target_width);
			scissor.y = std::min<NSUInteger>(scissor.y, m_target_height);
			scissor.width = std::min<NSUInteger>(scissor.width, m_target_width - scissor.x);
			scissor.height = std::min<NSUInteger>(scissor.height, m_target_height - scissor.y);
		}

		[encoder setScissorRect:scissor];
	}

	id<MTLRenderCommandEncoder> overlay_pass::begin_render_pass(mtl::command_context& cmd, const overlay_render_target& target, MTLLoadAction load)
	{
		ensure(target.texture);

		MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor renderPassDescriptor];
		const u32 aspect = target.aspect();

		if (aspect & mtl::image_aspect::color)
		{
			MTLRenderPassColorAttachmentDescriptor* att = desc.colorAttachments[0];
			att.texture = target.texture;
			att.level = target.level;
			att.slice = target.slice;
			att.depthPlane = target.depth_plane;
			att.loadAction = load;
			att.storeAction = MTLStoreActionStore;
			att.clearColor = MTLClearColorMake(0., 0., 0., 0.);
		}
		else
		{
			if (aspect & mtl::image_aspect::depth)
			{
				MTLRenderPassDepthAttachmentDescriptor* att = desc.depthAttachment;
				att.texture = target.texture;
				att.level = target.level;
				att.slice = target.slice;
				att.depthPlane = target.depth_plane;
				att.loadAction = load;
				att.storeAction = MTLStoreActionStore;
				att.clearDepth = 1.;
			}

			if (aspect & mtl::image_aspect::stencil)
			{
				MTLRenderPassStencilAttachmentDescriptor* att = desc.stencilAttachment;
				att.texture = target.texture;
				att.level = target.level;
				att.slice = target.slice;
				att.depthPlane = target.depth_plane;
				att.loadAction = load;
				att.storeAction = MTLStoreActionStore;
				att.clearStencil = 0;
			}
		}

		m_target_width = target.width;
		m_target_height = target.height;

		return cmd.begin_render_pass(desc, @"Overlay pass");
	}

	void overlay_pass::run_in_pass(mtl::command_context& cmd, const areau& viewport, const mtl::render_pass_formats& formats, const std::vector<mtl::image_view*>& src)
	{
		ensure(cmd.is_render_pass_open());

		const u32 x = std::min(viewport.x1, viewport.x2);
		const u32 y = std::min(viewport.y1, viewport.y2);
		const u32 w = viewport.width();
		const u32 h = viewport.height();

		if (!w || !h)
		{
			return;
		}

		if (m_target_width && m_target_height && (x >= m_target_width || y >= m_target_height))
		{
			// Fully outside of the attachment
			return;
		}

		auto program = load_program(cmd, formats, src);
		set_up_viewport(cmd, x, y, w, h);
		emit_geometry(cmd, program);
	}

	void overlay_pass::run(mtl::command_context& cmd, const areau& viewport, const overlay_render_target& target, const std::vector<mtl::image_view*>& src, MTLLoadAction load)
	{
		begin_render_pass(cmd, target, load);
		run_in_pass(cmd, viewport, target.get_formats(), src);
		cmd.end_encoding();
	}

	void overlay_pass::run(mtl::command_context& cmd, const areau& viewport, mtl::image* target, const std::vector<mtl::image_view*>& src, MTLLoadAction load)
	{
		run(cmd, viewport, overlay_render_target::from(target), src, load);
	}

	void overlay_pass::run(mtl::command_context& cmd, const areau& viewport, mtl::image* target, mtl::image_view* src, MTLLoadAction load)
	{
		std::vector<mtl::image_view*> views = { src };
		run(cmd, viewport, target, views, load);
	}

	void overlay_pass::run(mtl::command_context& cmd, const areau& viewport, mtl::image_view* target, const std::vector<mtl::image_view*>& src, MTLLoadAction load)
	{
		run(cmd, viewport, overlay_render_target::from(target), src, load);
	}

	void overlay_pass::run(mtl::command_context& cmd, const areau& viewport, mtl::image_view* target, mtl::image_view* src, MTLLoadAction load)
	{
		std::vector<mtl::image_view*> views = { src };
		run(cmd, viewport, target, views, load);
	}

	// --------------------------------------------------------------------------------------------
	// depth_convert_pass
	// --------------------------------------------------------------------------------------------

	depth_convert_pass::depth_convert_pass()
	{
		vs_src = s_src_rect_vs;

		fs_src =
			"#version 420\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"layout(set=1, binding=0) uniform sampler2D fs0;\n"
			"layout(location=0) in vec2 tc0;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	vec4 rgba_in = texture(fs0, tc0);\n"
			"	gl_FragDepth = rgba_in.w * 0.99609 + rgba_in.x * 0.00389 + rgba_in.y * 0.00002;\n"
			"}\n";

		// Encoded depth must not be interpolated
		m_sampler_filter = MTLSamplerMinMagFilterNearest;

		renderpass_config.set_attachment_count(0);
		renderpass_config.set_depth_mask(true);
		renderpass_config.enable_depth_test(MTLCompareFunctionAlways);
	}

	void depth_convert_pass::update_uniforms(mtl::command_context& cmd, mtl::program* /*program*/)
	{
		[cmd.render_encoder() setVertexBytes:src_rect length:vertex_push_constants_size atIndex:mtl::binding::helper_params_buffer];
	}

	void depth_convert_pass::run(mtl::command_context& cmd, const areai& src_area, const areai& dst_area, mtl::image_view* src, mtl::image* dst, MTLLoadAction load)
	{
		ensure(src && dst);
		auto real_src = src->image();
		ensure(real_src);

		const f32 src_w = static_cast<f32>(std::max(real_src->width() >> src->base_level, 1u));
		const f32 src_h = static_cast<f32>(std::max(real_src->height() >> src->base_level, 1u));

		src_rect[0] = static_cast<f32>(src_area.x1) / src_w;
		src_rect[1] = static_cast<f32>(src_area.y1) / src_h;
		src_rect[2] = static_cast<f32>(src_area.x2) / src_w;
		src_rect[3] = static_cast<f32>(src_area.y2) / src_h;

		overlay_pass::run(cmd, static_cast<areau>(dst_area), dst, src, load);
	}

	// --------------------------------------------------------------------------------------------
	// rgba8_unorm_rg16_sfloat_convert_pass
	// --------------------------------------------------------------------------------------------

	rgba8_unorm_rg16_sfloat_convert_pass::rgba8_unorm_rg16_sfloat_convert_pass()
	{
		vs_src = s_positions_only_vs;

		fs_src =
			"#version 420\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"layout(set=1, binding=0) uniform sampler2D fs0;\n"
			"layout(location=0) out vec4 ocol;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	uint value = packUnorm4x8(texelFetch(fs0, ivec2(gl_FragCoord.xy), 0).zyxw);\n"
			"	ocol = vec4(unpackHalf2x16(value), 0., 0.);\n"
			"}\n";

		m_sampler_filter = MTLSamplerMinMagFilterNearest;

		renderpass_config.set_attachment_count(1);
		renderpass_config.set_color_mask(0, true, true, true, true);
		renderpass_config.set_depth_mask(false);
	}

	void rgba8_unorm_rg16_sfloat_convert_pass::run(mtl::command_context& cmd, const areau& viewport, mtl::image* target, mtl::image_view* src, MTLLoadAction load)
	{
		overlay_pass::run(cmd, viewport, target, src, load);
	}

	// --------------------------------------------------------------------------------------------
	// ui_overlay_renderer
	// --------------------------------------------------------------------------------------------

	ui_overlay_renderer::ui_overlay_renderer()
		: m_texture_type(rsx::overlays::texture_sampling_mode::none)
	{
		vs_src =
		#include "../Program/GLSLSnippets/OverlayRenderVS.glsl"
		;

		fs_src =
		#include "../Program/GLSLSnippets/OverlayRenderFS.glsl"
		;

		vs_src = patch_glsl(vs_src,
		{
			{ "%preprocessor", "// %preprocessor" },
			{ "%push_block", "std140, set=0, binding=0" }
		});

		// NOTE: %push_block_offset must be replaced before %push_block (prefix)
		fs_src = patch_glsl(fs_src,
		{
			{ "%preprocessor", "// %preprocessor" },
			{ "%push_block_offset", "" },
			{ "%push_block", "std140, set=0, binding=0" },
			{ "layout(set=0, binding=0) uniform sampler2D fs0", "layout(set=1, binding=0) uniform sampler2D fs0" },
			{ "layout(set=0, binding=1) uniform sampler2DArray fs1", "layout(set=1, binding=1) uniform sampler2DArray fs1" }
		});

		// 2 input textures
		m_num_usable_samplers = 2;

		// Vertices are fetched from binding::helper_vertex_buffer
		m_use_vertex_input = true;

		renderpass_config.set_attachment_count(1);
		renderpass_config.set_color_mask(0, true, true, true, true);
		renderpass_config.set_depth_mask(false);
		renderpass_config.enable_blend(0,
			MTLBlendFactorSourceAlpha, MTLBlendFactorZero,
			MTLBlendFactorOneMinusSourceAlpha, MTLBlendFactorOne,
			MTLBlendOperationAdd, MTLBlendOperationAdd);
	}

	void ui_overlay_renderer::upload_simple_texture(mtl::image* tex, mtl::command_context& cmd,
		mtl::ring_buffer& upload_heap, u32 w, u32 h, u32 layers, bool font, const void* pixel_src)
	{
		const u32 pitch = (font) ? w : w * 4;
		const u32 layer_size = pitch * h;
		const u32 data_size = layer_size * layers;

		if (!data_size)
		{
			return;
		}

		const auto offset = upload_heap.alloc<512>(data_size);
		const auto addr = upload_heap.map(offset);

		if (pixel_src)
			std::memcpy(addr, pixel_src, data_size);
		else
			std::memset(addr, 0, data_size);

		id<MTLBlitCommandEncoder> blit = cmd.blit();
		const bool is_array = (tex->type() == MTLTextureType2DArray);

		for (u32 layer = 0; layer < layers; ++layer)
		{
			[blit copyFromBuffer:upload_heap.handle()
				sourceOffset:(offset + static_cast<usz>(layer) * layer_size)
				sourceBytesPerRow:pitch
				sourceBytesPerImage:(is_array ? layer_size : 0)
				sourceSize:MTLSizeMake(w, h, 1)
				toTexture:tex->value
				destinationSlice:layer
				destinationLevel:0
				destinationOrigin:MTLOriginMake(0, 0, 0)];
		}
	}

	mtl::image_view* ui_overlay_renderer::upload_simple_texture(mtl::command_context& cmd,
		mtl::ring_buffer& upload_heap, u64 key, u32 w, u32 h, u32 layers, bool font, bool temp, const void* pixel_src, u32 owner_uid)
	{
		const MTLPixelFormat format = (font) ? MTLPixelFormatR8Unorm : MTLPixelFormatBGRA8Unorm;
		const u32 num_layers = std::max(layers, 1u);
		const MTLTextureType type = (num_layers > 1) ? MTLTextureType2DArray : MTLTextureType2D;

		auto tex = std::make_unique<mtl::image>(type, format, std::max(w, 1u), std::max(h, 1u), 1, 1, num_layers, 1,
			MTLTextureUsageShaderRead, RSX_FORMAT_CLASS_COLOR);

		upload_simple_texture(tex.get(), cmd, upload_heap, w, h, layers, font, pixel_src);

		auto view = std::make_unique<mtl::image_view>(tex.get(), MTLPixelFormatInvalid, type,
			mtl::default_component_map, mtl::image_aspect::color);

		auto result = view.get();

		if (!temp || font)
			view_cache[key] = std::move(view);
		else
			temp_view_cache[key] = std::move(view);

		if (font)
			font_cache[key] = std::move(tex);
		else if (!temp)
			resources.push_back(std::move(tex));
		else
			temp_image_cache[key] = std::make_pair(owner_uid, std::move(tex));

		return result;
	}

	void ui_overlay_renderer::init(mtl::command_context& cmd, mtl::ring_buffer& upload_heap)
	{
		rsx::overlays::resource_config configuration;
		configuration.load_files();

		u64 storage_key = 1;

		for (const auto &res : configuration.texture_raw_data)
		{
			upload_simple_texture(cmd, upload_heap, storage_key++, res->w, res->h, 1, false, false, res->get_data(), umax);
		}

		configuration.free_resources();
	}

	void ui_overlay_renderer::destroy()
	{
		temp_view_cache.clear();
		temp_image_cache.clear();

		view_cache.clear();
		resources.clear();
		font_cache.clear();

		m_vertex_scratch.clear();
		m_vertex_scratch.shrink_to_fit();

		overlay_pass::destroy();
	}

	void ui_overlay_renderer::remove_temp_resources(u32 key)
	{
		std::vector<u64> keys_to_remove;
		for (const auto& temp_image : temp_image_cache)
		{
			if (temp_image.second.first == key)
			{
				keys_to_remove.push_back(temp_image.first);
			}
		}

		// Command buffers retain the textures they reference, so these can be released immediately
		for (const auto& _key : keys_to_remove)
		{
			temp_view_cache.erase(_key);
			temp_image_cache.erase(_key);
		}
	}

	mtl::image_view* ui_overlay_renderer::find_font(const rsx::overlays::font* font, mtl::command_context& cmd, mtl::ring_buffer& upload_heap)
	{
		const auto image_size = font->get_glyph_data_dimensions();

		u64 key = reinterpret_cast<u64>(font);
		auto found = view_cache.find(key);
		if (found != view_cache.end())
		{
			if (const auto raw = found->second->image();
				image_size.width == raw->width() &&
				image_size.height == raw->height() &&
				std::max(image_size.depth, 1u) == raw->layers())
			{
				return found->second.get();
			}

			view_cache.erase(key);
			font_cache.erase(key);
		}

		// Create font resource
		const std::vector<u8>& bytes = font->get_glyph_data();

		return upload_simple_texture(cmd, upload_heap, key, image_size.width, image_size.height, image_size.depth,
				true, false, bytes.data(), umax);
	}

	mtl::image_view* ui_overlay_renderer::find_temp_image(const rsx::overlays::image_info_base* desc, mtl::command_context& cmd, mtl::ring_buffer& upload_heap, u32 owner_uid)
	{
		const bool dirty = std::exchange(desc->dirty, false);
		const u64 key = reinterpret_cast<u64>(desc);

		auto cached = temp_view_cache.find(key);
		if (cached != temp_view_cache.end())
		{
			mtl::image_view* view = cached->second.get();

			if (dirty)
			{
				upload_simple_texture(view->image(), cmd, upload_heap, desc->w, desc->h, 1, false, desc->get_data());
			}

			return view;
		}

		return upload_simple_texture(cmd, upload_heap, key, desc->w, desc->h, 1,
				false, true, desc->get_data(), owner_uid);
	}

	void ui_overlay_renderer::update_uniforms(mtl::command_context& cmd, mtl::program* /*program*/)
	{
		// Vertex block (std140)
		// 00: vec4 ui_scale;
		// 16: vec4 albedo;
		// 32: vec4 viewport;
		// 48: vec4 clip_bounds;
		// 64: uint vertex_config;
		//
		// Fragment block (std140)
		// 00: uint fragment_config;
		// 04: float timestamp;
		// 08: float blur_intensity;
		// 16: vec4 sdf_params;
		// 32: vec4 sdf_origin;
		// 48: vec4 sdf_border_color;

		id<MTLRenderCommandEncoder> encoder = cmd.render_encoder();

		// 1. Vertex config
		usz pos = 0;
		std::array<f32, vertex_push_constants_size / sizeof(f32)> vertex_buf{};

		write_to_ptr(vertex_buf, pos, m_scale_offset.rgba);
		pos += sizeof(m_scale_offset.rgba) / sizeof(f32);
		write_to_ptr(vertex_buf, pos, m_color.rgba);
		pos += sizeof(m_color.rgba) / sizeof(f32);

		vertex_buf[pos++] = m_viewport.width;
		vertex_buf[pos++] = m_viewport.height;
		vertex_buf[pos++] = m_viewport.x;
		vertex_buf[pos++] = m_viewport.y;

		vertex_buf[pos++] = m_clip_region.x1;
		vertex_buf[pos++] = m_clip_region.y1;
		vertex_buf[pos++] = m_clip_region.x2;
		vertex_buf[pos++] = m_clip_region.y2;

		rsx::overlays::vertex_options vert_opts {};
		const auto vert_config = vert_opts
			.disable_vertex_snap(m_disable_vertex_snap)
			.get();
		vertex_buf[pos++] = std::bit_cast<f32>(vert_config);

		ensure(pos == 17);
		[encoder setVertexBytes:vertex_buf.data() length:sizeof(vertex_buf) atIndex:mtl::binding::helper_params_buffer];

		// 2. Fragment stuff
		rsx::overlays::fragment_options frag_opts {};
		const auto frag_config = frag_opts
			.texture_mode(m_texture_type)
			.clip_fragments(m_clip_enabled)
			.pulse_glow(m_pulse_glow)
			.set_sdf(m_sdf_config.func)
			.get();

		pos = 0;
		std::array<f32, fragment_push_constants_size / sizeof(f32)> fragment_buf{};

		fragment_buf[pos++] = std::bit_cast<f32>(frag_config);
		fragment_buf[pos++] = m_time;
		fragment_buf[pos++] = m_blur_strength;
		fragment_buf[pos++] = 0.f; // std140 padding before sdf_params
		fragment_buf[pos++] = m_sdf_config.hx;
		fragment_buf[pos++] = m_sdf_config.hy;
		fragment_buf[pos++] = m_sdf_config.br;
		fragment_buf[pos++] = m_sdf_config.bw;
		fragment_buf[pos++] = m_sdf_config.cx;
		fragment_buf[pos++] = m_sdf_config.cy;
		fragment_buf[pos++] = 0.f;
		fragment_buf[pos++] = 0.f;

		write_to_ptr(fragment_buf, pos, m_sdf_config.border_color.rgba);
		pos += sizeof(m_sdf_config.border_color.rgba) / sizeof(f32);

		ensure(pos == fragment_buf.size());
		[encoder setFragmentBytes:fragment_buf.data() length:sizeof(fragment_buf) atIndex:mtl::binding::helper_params_buffer];
	}

	void ui_overlay_renderer::set_primitive_type(rsx::overlays::primitive_type type)
	{
		m_current_primitive_type = type;

		switch (type)
		{
			case rsx::overlays::primitive_type::quad_list:
			case rsx::overlays::primitive_type::triangle_fan:
				// Expanded into triangle lists, see upload_command_vertices
				renderpass_config.set_primitive_type(MTLPrimitiveTypeTriangle);
				break;
			case rsx::overlays::primitive_type::triangle_strip:
				renderpass_config.set_primitive_type(MTLPrimitiveTypeTriangleStrip);
				break;
			case rsx::overlays::primitive_type::line_list:
				renderpass_config.set_primitive_type(MTLPrimitiveTypeLine);
				break;
			case rsx::overlays::primitive_type::line_strip:
				renderpass_config.set_primitive_type(MTLPrimitiveTypeLineStrip);
				break;
			default:
				fmt::throw_exception("Unexpected primitive type %d", static_cast<s32>(type));
		}
	}

	void ui_overlay_renderer::upload_command_vertices(mtl::command_context& cmd, const std::vector<vertex>& verts)
	{
		const std::vector<vertex>* source = &verts;

		switch (m_current_primitive_type)
		{
		case rsx::overlays::primitive_type::quad_list:
		{
			// Emulate quads with triangle pairs (VK draws disjoint 4-vertex triangle strips)
			const usz num_quads = verts.size() / 4;
			m_vertex_scratch.clear();
			m_vertex_scratch.reserve(num_quads * 6);

			for (usz n = 0; n < num_quads; ++n)
			{
				const vertex* quad = &verts[n * 4];
				m_vertex_scratch.push_back(quad[0]);
				m_vertex_scratch.push_back(quad[1]);
				m_vertex_scratch.push_back(quad[2]);
				m_vertex_scratch.push_back(quad[2]);
				m_vertex_scratch.push_back(quad[1]);
				m_vertex_scratch.push_back(quad[3]);
			}

			source = &m_vertex_scratch;
			break;
		}
		case rsx::overlays::primitive_type::triangle_fan:
		{
			// No triangle fans in Metal
			m_vertex_scratch.clear();

			if (verts.size() >= 3)
			{
				m_vertex_scratch.reserve((verts.size() - 2) * 3);

				for (usz n = 1; n + 1 < verts.size(); ++n)
				{
					m_vertex_scratch.push_back(verts[0]);
					m_vertex_scratch.push_back(verts[n]);
					m_vertex_scratch.push_back(verts[n + 1]);
				}
			}

			source = &m_vertex_scratch;
			break;
		}
		default:
			break;
		}

		num_drawable_elements = static_cast<u32>(source->size());
		first_vertex = 0;

		if (num_drawable_elements)
		{
			upload_vertex_data(cmd, source->data(), num_drawable_elements);
		}
	}

	void ui_overlay_renderer::run(mtl::command_context& cmd, const areau& viewport, const overlay_render_target& target,
		mtl::ring_buffer& upload_heap, rsx::overlays::overlay& ui, MTLLoadAction load)
	{
		ui.set_render_viewport(
			static_cast<u16>(std::min<u32>(viewport.width(), std::numeric_limits<u16>::max())),
			static_cast<u16>(std::min<u32>(viewport.height(), std::numeric_limits<u16>::max()))
		);
		m_scale_offset = color4f(ui.get_virtual_width(), ui.get_virtual_height(), 1.f, 1.f);
		m_viewport = { { static_cast<f32>(viewport.x1), static_cast<f32>(viewport.y1) }, { static_cast<f32>(viewport.width()), static_cast<f32>(viewport.height()) } };

		if (ui.status_flags & rsx::overlays::status_bits::invalidate_image_cache)
		{
			remove_temp_resources(ui.uid);
			ui.status_flags.clear(rsx::overlays::status_bits::invalidate_image_cache);
		}

		const auto compiled_resources = ui.get_compiled();
		const auto& draw_commands = compiled_resources.draw_commands;

		// 1. Resolve the image resources first. Uploads are recorded on the blit encoder and would interrupt the render pass.
		std::vector<mtl::image_view*> command_sources(draw_commands.size(), nullptr);

		for (usz i = 0; i < draw_commands.size(); ++i)
		{
			const auto& config = draw_commands[i].config;

			switch (config.texture_ref)
			{
			case rsx::overlays::image_resource_id::game_icon:
			case rsx::overlays::image_resource_id::backbuffer:
				// Not backed by an image resource (same as VK): drawn untextured
			case rsx::overlays::image_resource_id::none:
				break;
			case rsx::overlays::image_resource_id::font_file:
				command_sources[i] = find_font(config.font_ref, cmd, upload_heap);
				break;
			case rsx::overlays::image_resource_id::raw_image:
				command_sources[i] = find_temp_image(static_cast<const rsx::overlays::image_info_base*>(config.external_data_ref), cmd, upload_heap, ui.uid);
				break;
			default:
				if (auto found = view_cache.find(config.texture_ref); found != view_cache.end())
				{
					command_sources[i] = found->second.get();
				}
				break;
			}
		}

		// 2. Record all draws into a single render pass
		if (!draw_commands.empty())
		{
			std::vector<mtl::image_view*> image_views
			{
				get_null_image_view(MTLTextureType2D),
				get_null_image_view(MTLTextureType2DArray)
			};

			const auto formats = target.get_formats();
			begin_render_pass(cmd, target, load);

			for (usz i = 0; i < draw_commands.size(); ++i)
			{
				const auto& command = draw_commands[i];

				set_primitive_type(command.config.primitives);
				upload_command_vertices(cmd, command.verts);

				if (!num_drawable_elements)
				{
					continue;
				}

				m_time = command.config.get_sinus_value();
				m_texture_type = rsx::overlays::texture_sampling_mode::texture2D;
				m_color = command.config.color;
				m_pulse_glow = command.config.pulse_glow;
				m_blur_strength = static_cast<f32>(command.config.blur_strength) * 0.01f;
				m_clip_enabled = command.config.clip_region;
				m_clip_region = command.config.clip_rect;
				m_disable_vertex_snap = command.config.disable_vertex_snap;

				m_sdf_config = command.config.sdf_config;
				m_sdf_config.transform(static_cast<areaf>(viewport), { m_scale_offset.x, m_scale_offset.y });

				mtl::image_view* src = command_sources[i];
				switch (command.config.texture_ref)
				{
				case rsx::overlays::image_resource_id::game_icon:
				case rsx::overlays::image_resource_id::backbuffer:
					// Not backed by an image resource (same as VK): drawn untextured
				case rsx::overlays::image_resource_id::none:
					m_texture_type = rsx::overlays::texture_sampling_mode::none;
					break;
				case rsx::overlays::image_resource_id::font_file:
					m_texture_type = src->image()->layers() == 1
						? rsx::overlays::texture_sampling_mode::font2D
						: rsx::overlays::texture_sampling_mode::font3D;
					break;
				default:
					break;
				}

				if (src)
				{
					const int res_id = src->image()->layers() > 1 ? 1 : 0;
					image_views[res_id] = src;
				}

				run_in_pass(cmd, viewport, formats, image_views);
			}

			cmd.end_encoding();
		}

		ui.update(get_system_time());
	}

	void ui_overlay_renderer::run(mtl::command_context& cmd, const areau& viewport, mtl::image* target,
		mtl::ring_buffer& upload_heap, rsx::overlays::overlay& ui, MTLLoadAction load)
	{
		run(cmd, viewport, overlay_render_target::from(target), upload_heap, ui, load);
	}

	// --------------------------------------------------------------------------------------------
	// attachment_clear_pass
	// --------------------------------------------------------------------------------------------

	attachment_clear_pass::attachment_clear_pass()
	{
		// regs[0] = clear color, regs[1].x = clear depth
		vs_src =
			"#version 450\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"layout(std140, set=0, binding=0) uniform static_data{ vec4 regs[2]; };\n"
			"layout(location=0) out vec4 color;\n"
			"\n"
			"void main()\n"
			"{\n"
			"	vec2 positions[] = {vec2(-1., -1.), vec2(1., -1.), vec2(-1., 1.), vec2(1., 1.)};\n"
			"	color = regs[0];\n"
			"	gl_Position = vec4(positions[gl_VertexIndex % 4], regs[1].x, 1.);\n"
			"}\n";

		// Fragment shaders are generated per number of color outputs (see get_fragment_function)

		// Disable samplers
		m_num_usable_samplers = 0;

		renderpass_config.set_depth_mask(false);
		renderpass_config.set_color_mask(0, true, true, true, true);
		renderpass_config.set_attachment_count(1);
	}

	id<MTLFunction> attachment_clear_pass::get_fragment_function()
	{
		const u32 count = renderpass_config.pipeline.attachment_count;
		ensure(count <= 4);

		auto& variant = m_fragment_variants[count];
		if (!variant)
		{
			std::string src =
				"#version 420\n"
				"#extension GL_ARB_separate_shader_objects : enable\n"
				"layout(location=0) in vec4 color;\n";

			for (u32 n = 0; n < count; ++n)
			{
				src += fmt::format("layout(location=%u) out vec4 out_color%u;\n", n, n);
			}

			src += "\nvoid main()\n{\n";

			for (u32 n = 0; n < count; ++n)
			{
				src += fmt::format("	out_color%u = color;\n", n);
			}

			src += "}\n";

			variant = compile_overlay_shader(src, mtl::shader_stage::fragment, "attachment_clear_fs");
		}

		return variant.function;
	}

	void attachment_clear_pass::update_uniforms(mtl::command_context& cmd, mtl::program* /*program*/)
	{
		f32 data[8];
		data[0] = clear_color.r;
		data[1] = clear_color.g;
		data[2] = clear_color.b;
		data[3] = clear_color.a;
		data[4] = clear_depth;
		data[5] = 0.f;
		data[6] = 0.f;
		data[7] = 0.f;

		static_assert(sizeof(data) == vertex_push_constants_size);
		[cmd.render_encoder() setVertexBytes:data length:vertex_push_constants_size atIndex:mtl::binding::helper_params_buffer];
	}

	void attachment_clear_pass::set_up_viewport(mtl::command_context& cmd, u32 /*x*/, u32 /*y*/, u32 /*w*/, u32 /*h*/)
	{
		// The quad only has to cover the scissor region. The framebuffer dimensions are not known in-pass.
		id<MTLRenderCommandEncoder> encoder = cmd.render_encoder();

		MTLViewport vp{};
		vp.originX = static_cast<f64>(region.x);
		vp.originY = static_cast<f64>(region.y);
		vp.width = static_cast<f64>(region.width);
		vp.height = static_cast<f64>(region.height);
		vp.znear = 0.;
		vp.zfar = 1.;
		[encoder setViewport:vp];

		[encoder setScissorRect:region];
	}

	void attachment_clear_pass::destroy()
	{
		for (auto& variant : m_fragment_variants)
		{
			variant = {};
		}

		overlay_pass::destroy();
	}

	void attachment_clear_pass::run(mtl::command_context& cmd, const mtl::render_pass_formats& formats, const MTLScissorRect& rect, u32 clearmask, color4f color)
	{
		u8 mask = 0;
		if (clearmask & 0x10) mask |= static_cast<u8>(MTLColorWriteMaskRed);
		if (clearmask & 0x20) mask |= static_cast<u8>(MTLColorWriteMaskGreen);
		if (clearmask & 0x40) mask |= static_cast<u8>(MTLColorWriteMaskBlue);
		if (clearmask & 0x80) mask |= static_cast<u8>(MTLColorWriteMaskAlpha);

		clear_info info{};
		info.color = color;

		for (u32 index = 0; index < std::min<u32>(formats.color_count, 4); ++index)
		{
			info.color_write_mask[index] = mask;
		}

		colormask =
		{
			(mask & MTLColorWriteMaskRed) ? 1.f : 0.f,
			(mask & MTLColorWriteMaskGreen) ? 1.f : 0.f,
			(mask & MTLColorWriteMaskBlue) ? 1.f : 0.f,
			(mask & MTLColorWriteMaskAlpha) ? 1.f : 0.f
		};

		run(cmd, formats, rect, info);
	}

	void attachment_clear_pass::run(mtl::command_context& cmd, const mtl::render_pass_formats& formats, const MTLScissorRect& rect, const clear_info& info)
	{
		ensure(cmd.is_render_pass_open());

		if (!rect.width || !rect.height)
		{
			return;
		}

		region = rect;

		// In-pass: the attachment dimensions are unknown, the caller guarantees that rect lies within the attachments
		m_target_width = 0;
		m_target_height = 0;

		// Color
		u32 color_count = 0;
		bool write_color = false;

		for (u32 index = 0; index < 4; ++index)
		{
			if (formats.color_formats[index] != static_cast<u32>(MTLPixelFormatInvalid))
			{
				color_count = index + 1;
			}
		}

		for (u32 index = 0; index < 4; ++index)
		{
			const u8 mask = (index < color_count && formats.color_formats[index] != static_cast<u32>(MTLPixelFormatInvalid))
				? info.color_write_mask[index]
				: 0;

			renderpass_config.set_color_write_mask(index, mask);
			renderpass_config.disable_blend(index);
			write_color |= (mask != 0);
		}

		renderpass_config.set_attachment_count(write_color ? color_count : 0);

		// Depth
		const bool write_depth = info.clear_depth && formats.depth_format != static_cast<u32>(MTLPixelFormatInvalid);
		if (write_depth)
		{
			renderpass_config.enable_depth_test(MTLCompareFunctionAlways);
			renderpass_config.set_depth_mask(true);
		}
		else
		{
			renderpass_config.disable_depth_test();
		}

		// Stencil
		const bool write_stencil = info.clear_stencil && info.stencil_write_mask && formats.stencil_format != static_cast<u32>(MTLPixelFormatInvalid);
		if (write_stencil)
		{
			renderpass_config.enable_stencil_test(
				MTLStencilOperationReplace, MTLStencilOperationReplace, MTLStencilOperationReplace, // Always replace
				MTLCompareFunctionAlways,                                                            // Always pass
				0xFF,                                                                                // Full write-through
				info.stencil);                                                                       // Clear value

			renderpass_config.set_stencil_mask(info.stencil_write_mask);
		}
		else
		{
			renderpass_config.disable_stencil_test();
		}

		if (!write_color && !write_depth && !write_stencil)
		{
			return;
		}

		clear_color = info.color;
		clear_depth = std::clamp(info.depth, 0.f, 1.f);

		const areau viewport =
		{
			static_cast<u32>(rect.x),
			static_cast<u32>(rect.y),
			static_cast<u32>(rect.x + rect.width),
			static_cast<u32>(rect.y + rect.height)
		};

		run_in_pass(cmd, viewport, formats, {});
	}

	// --------------------------------------------------------------------------------------------
	// stencil_clear_pass
	// --------------------------------------------------------------------------------------------

	stencil_clear_pass::stencil_clear_pass()
	{
		vs_src = s_positions_only_vs;

		// No color outputs, color attachments of the open pass are masked off
		fs_src =
			"#version 420\n"
			"#extension GL_ARB_separate_shader_objects : enable\n"
			"\n"
			"void main()\n"
			"{\n"
			"}\n";

		m_num_usable_samplers = 0;

		renderpass_config.set_attachment_count(0);
		renderpass_config.set_depth_mask(false);
	}

	void stencil_clear_pass::set_up_viewport(mtl::command_context& cmd, u32 /*x*/, u32 /*y*/, u32 /*w*/, u32 /*h*/)
	{
		id<MTLRenderCommandEncoder> encoder = cmd.render_encoder();

		MTLViewport vp{};
		vp.originX = static_cast<f64>(region.x);
		vp.originY = static_cast<f64>(region.y);
		vp.width = static_cast<f64>(region.width);
		vp.height = static_cast<f64>(region.height);
		vp.znear = 0.;
		vp.zfar = 1.;
		[encoder setViewport:vp];

		[encoder setScissorRect:region];
	}

	void stencil_clear_pass::run(mtl::command_context& cmd, const mtl::render_pass_formats& formats, const MTLScissorRect& rect, u32 stencil_clear, u32 stencil_write_mask)
	{
		ensure(cmd.is_render_pass_open());
		ensure(formats.stencil_format != static_cast<u32>(MTLPixelFormatInvalid));

		if (!rect.width || !rect.height)
		{
			return;
		}

		region = rect;
		m_target_width = 0;
		m_target_height = 0;

		// Stencil setup. Replace all pixels in the scissor region with stencil_clear with the correct write mask.
		renderpass_config.enable_stencil_test(
			MTLStencilOperationReplace, MTLStencilOperationReplace, MTLStencilOperationReplace, // Always replace
			MTLCompareFunctionAlways,                                                            // Always pass
			0xFF,                                                                                // Full write-through
			stencil_clear);                                                                      // Write active bit

		renderpass_config.set_stencil_mask(stencil_write_mask);
		renderpass_config.disable_depth_test();

		// Leave all color attachments untouched
		renderpass_config.set_attachment_count(0);

		const areau viewport =
		{
			static_cast<u32>(rect.x),
			static_cast<u32>(rect.y),
			static_cast<u32>(rect.x + rect.width),
			static_cast<u32>(rect.y + rect.height)
		};

		run_in_pass(cmd, viewport, formats, {});
	}

	// --------------------------------------------------------------------------------------------
	// video_out_calibration_pass
	// --------------------------------------------------------------------------------------------

	video_out_calibration_pass::video_out_calibration_pass()
	{
		vs_src =
		#include "../Program/GLSLSnippets/GenericVSPassthrough.glsl"
		;

		fs_src =
		#include "../Program/GLSLSnippets/VideoOutCalibrationPass.glsl"
		;

		fs_src = patch_glsl(fs_src,
		{
			{ "%sampler_binding", "x" },
			{ "%set_decorator", "set=1" },
			{ "layout(push_constant) uniform static_data", "layout(std140, set=0, binding=0) uniform static_data" }
		});

		renderpass_config.set_depth_mask(false);
		renderpass_config.set_color_mask(0, true, true, true, true);
		renderpass_config.set_attachment_count(1);

		m_num_usable_samplers = 2;
	}

	void video_out_calibration_pass::update_uniforms(mtl::command_context& cmd, mtl::program* /*program*/)
	{
		static_assert(sizeof(config.data) == fragment_push_constants_size);
		[cmd.render_encoder() setFragmentBytes:config.data length:fragment_push_constants_size atIndex:mtl::binding::helper_params_buffer];
	}

	void video_out_calibration_pass::run(mtl::command_context& cmd, const areau& viewport, const overlay_render_target& target,
		const rsx::simple_array<mtl::viewable_image*>& src, f32 gamma, bool limited_rgb,
		bool stereo_enabled, MTLLoadAction load)
	{
		static stereo_config stereo_cfg = stereo_config(true);
		stereo_cfg.update_from_config(stereo_enabled);
		const auto& matrices = stereo_cfg.matrices();

		config.gamma = gamma;
		config.limit_range = limited_rgb ? 1 : 0;
		config.stereo_display_mode = static_cast<u8>(stereo_cfg.stereo_mode());
		config.stereo_image_count = std::min(::size32(src), 2u);

		for (u32 i = 0; i < 3; i++)
		{
			std::memcpy(config.left_anaglyph_matrix[i].rgba, matrices.left[i].rgb, sizeof(matrices.left[i].rgb));
			std::memcpy(config.right_anaglyph_matrix[i].rgba, matrices.right[i].rgb, sizeof(matrices.right[i].rgb));
		}

		std::vector<mtl::image_view*> views;
		views.reserve(2);

		for (auto& img : src)
		{
			views.push_back(img->get_view(rsx::default_remap_vector.with_encoding(MTL_REMAP_IDENTITY)));
		}

		while (views.size() < 2)
		{
			views.push_back(get_null_image_view(MTLTextureType2D));
		}

		overlay_pass::run(cmd, viewport, target, views, load);
	}

	void video_out_calibration_pass::run(mtl::command_context& cmd, const areau& viewport, mtl::image* target,
		const rsx::simple_array<mtl::viewable_image*>& src, f32 gamma, bool limited_rgb,
		bool stereo_enabled, MTLLoadAction load)
	{
		run(cmd, viewport, overlay_render_target::from(target), src, gamma, limited_rgb, stereo_enabled, load);
	}

	// --------------------------------------------------------------------------------------------
	// blit_pass
	// --------------------------------------------------------------------------------------------

	blit_pass::blit_pass()
	{
		vs_src = s_src_rect_vs;

		// Fragment shaders are selected per blit_mode (see get_fragment_function)
		m_num_usable_samplers = 2;
	}

	id<MTLFunction> blit_pass::get_fragment_function()
	{
		const u32 index = static_cast<u32>(m_mode);
		ensure(index < std::size(m_fragment_variants));

		auto& variant = m_fragment_variants[index];
		if (variant)
		{
			return variant.function;
		}

		std::string src;

		switch (m_mode)
		{
		case blit_mode::color:
			src =
				"#version 450\n"
				"#extension GL_ARB_separate_shader_objects : enable\n"
				"layout(set=1, binding=0) uniform sampler2D fs0;\n"
				"layout(location=0) in vec2 tc0;\n"
				"layout(location=0) out vec4 ocol;\n"
				"\n"
				"void main()\n"
				"{\n"
				"	ocol = textureLod(fs0, tc0, 0.);\n"
				"}\n";
			break;
		case blit_mode::depth:
			src =
				"#version 450\n"
				"#extension GL_ARB_separate_shader_objects : enable\n"
				"layout(set=1, binding=0) uniform sampler2D fs0;\n"
				"layout(location=0) in vec2 tc0;\n"
				"\n"
				"void main()\n"
				"{\n"
				"	gl_FragDepth = textureLod(fs0, tc0, 0.).x;\n"
				"}\n";
			break;
		case blit_mode::stencil:
			src =
				"#version 450\n"
				"#extension GL_ARB_separate_shader_objects : enable\n"
				"#extension GL_ARB_shader_stencil_export : enable\n"
				"layout(set=1, binding=0) uniform usampler2D fs0;\n"
				"layout(location=0) in vec2 tc0;\n"
				"\n"
				"void main()\n"
				"{\n"
				"	ivec2 size = textureSize(fs0, 0);\n"
				"	ivec2 coord = clamp(ivec2(floor(tc0 * vec2(size))), ivec2(0), size - 1);\n"
				"	gl_FragStencilRefARB = int(texelFetch(fs0, coord, 0).x & 0xFFu);\n"
				"}\n";
			break;
		case blit_mode::depth_stencil:
			src =
				"#version 450\n"
				"#extension GL_ARB_separate_shader_objects : enable\n"
				"#extension GL_ARB_shader_stencil_export : enable\n"
				"layout(set=1, binding=0) uniform sampler2D fs0;\n"
				"layout(set=1, binding=1) uniform usampler2D fs1;\n"
				"layout(location=0) in vec2 tc0;\n"
				"\n"
				"void main()\n"
				"{\n"
				"	ivec2 size = textureSize(fs1, 0);\n"
				"	ivec2 coord = clamp(ivec2(floor(tc0 * vec2(size))), ivec2(0), size - 1);\n"
				"	gl_FragDepth = textureLod(fs0, tc0, 0.).x;\n"
				"	gl_FragStencilRefARB = int(texelFetch(fs1, coord, 0).x & 0xFFu);\n"
				"}\n";
			break;
		}

		variant = compile_overlay_shader(src, mtl::shader_stage::fragment, "blit_pass_fs");
		return variant.function;
	}

	void blit_pass::update_uniforms(mtl::command_context& cmd, mtl::program* /*program*/)
	{
		[cmd.render_encoder() setVertexBytes:m_src_rect length:vertex_push_constants_size atIndex:mtl::binding::helper_params_buffer];
	}

	void blit_pass::destroy()
	{
		for (auto& variant : m_fragment_variants)
		{
			variant = {};
		}

		m_scratch_image.reset();
		overlay_pass::destroy();
	}

	mtl::image* blit_pass::get_scratch_image(const mtl::image* reference, u32 width, u32 height)
	{
		if (m_scratch_image &&
			m_scratch_image->format() == reference->format() &&
			m_scratch_image->width() >= width &&
			m_scratch_image->height() >= height)
		{
			return m_scratch_image.get();
		}

		u32 new_width = width;
		u32 new_height = height;

		if (m_scratch_image && m_scratch_image->format() == reference->format())
		{
			new_width = std::max(new_width, m_scratch_image->width());
			new_height = std::max(new_height, m_scratch_image->height());
		}

		new_width = utils::align(new_width, 64u);
		new_height = utils::align(new_height, 64u);

		// Command buffers retain the old image if it is still in use
		m_scratch_image = std::make_unique<mtl::image>(MTLTextureType2D, reference->format(), new_width, new_height, 1, 1, 1, 1,
			MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView, reference->format_class());
		m_scratch_image->set_debug_name("Blit pass scratch");

		return m_scratch_image.get();
	}

	void blit_pass::run(mtl::command_context& cmd, mtl::image* src, mtl::image* dst,
		const areai& src_area, const areai& dst_area,
		u32 src_level, u32 src_layer, u32 dst_level, u32 dst_layer,
		bool linear, u32 aspect)
	{
		ensure(src && dst);
		ensure(src->samples() == 1);
		ensure(src->type() != MTLTextureType3D);
		ensure(src_level < src->mipmaps() && dst_level < dst->mipmaps());

		const u32 src_aspect = mtl::get_aspect_flags(src->format());
		const u32 dst_aspect = mtl::get_aspect_flags(dst->format());

		if (!aspect)
		{
			aspect = dst_aspect;
		}

		// Select the operation
		if (aspect & mtl::image_aspect::color)
		{
			ensure(dst_aspect & mtl::image_aspect::color);
			m_mode = blit_mode::color;
		}
		else
		{
			const bool copy_depth = (aspect & mtl::image_aspect::depth) && (dst_aspect & mtl::image_aspect::depth) &&
				(src_aspect & (mtl::image_aspect::depth | mtl::image_aspect::color));
			const bool copy_stencil = (aspect & mtl::image_aspect::stencil) && (dst_aspect & mtl::image_aspect::stencil) &&
				(src_aspect & mtl::image_aspect::stencil);

			if (!copy_depth && !copy_stencil)
			{
				rsx_log.error("Metal: blit_pass: Unsupported transfer 0x%x -> 0x%x (aspect=0x%x)", static_cast<u32>(src->format()), static_cast<u32>(dst->format()), aspect);
				return;
			}

			m_mode = copy_depth
				? (copy_stencil ? blit_mode::depth_stencil : blit_mode::depth)
				: blit_mode::stencil;
		}

		// Normalize the destination rectangle. Flips are moved to the source rectangle.
		areai s = src_area;
		areai d = dst_area;

		if (d.x1 > d.x2)
		{
			std::swap(d.x1, d.x2);
			std::swap(s.x1, s.x2);
		}

		if (d.y1 > d.y2)
		{
			std::swap(d.y1, d.y2);
			std::swap(s.y1, s.y2);
		}

		if (d.x1 == d.x2 || d.y1 == d.y2 || s.x1 == s.x2 || s.y1 == s.y2)
		{
			return;
		}

		// Sampling and rendering into the same subresource is a hazard. Copy the source region out first.
		if (src == dst && src_level == dst_level && src_layer == dst_layer)
		{
			const s32 sx = std::min(s.x1, s.x2);
			const s32 sy = std::min(s.y1, s.y2);
			const s32 sw = std::abs(s.x2 - s.x1);
			const s32 sh = std::abs(s.y2 - s.y1);

			auto scratch = get_scratch_image(src, static_cast<u32>(sw), static_cast<u32>(sh));
			run(cmd, src, scratch, areai{ sx, sy, sx + sw, sy + sh }, areai{ 0, 0, sw, sh }, src_level, src_layer, 0, 0, false, aspect);

			areai scratch_area = { 0, 0, sw, sh };
			if (s.x1 > s.x2) scratch_area.flip_horizontal();
			if (s.y1 > s.y2) scratch_area.flip_vertical();

			run(cmd, scratch, dst, scratch_area, d, 0, 0, dst_level, dst_layer, linear, aspect);
			return;
		}

		const f32 src_w = static_cast<f32>(std::max(src->width() >> src_level, 1u));
		const f32 src_h = static_cast<f32>(std::max(src->height() >> src_level, 1u));
		const s32 dst_w = static_cast<s32>(std::max(dst->width() >> dst_level, 1u));
		const s32 dst_h = static_cast<s32>(std::max(dst->height() >> dst_level, 1u));

		// Clip the destination to the target level and adjust the source rectangle accordingly
		f32 u0 = static_cast<f32>(s.x1), u1 = static_cast<f32>(s.x2);
		f32 v0 = static_cast<f32>(s.y1), v1 = static_cast<f32>(s.y2);

		const s32 cx1 = std::max(d.x1, 0), cx2 = std::min(d.x2, dst_w);
		const s32 cy1 = std::max(d.y1, 0), cy2 = std::min(d.y2, dst_h);

		if (cx1 >= cx2 || cy1 >= cy2)
		{
			return;
		}

		if (cx1 != d.x1 || cx2 != d.x2)
		{
			const f32 scale = (u1 - u0) / static_cast<f32>(d.x2 - d.x1);
			const f32 new_u0 = u0 + static_cast<f32>(cx1 - d.x1) * scale;
			const f32 new_u1 = u0 + static_cast<f32>(cx2 - d.x1) * scale;
			u0 = new_u0;
			u1 = new_u1;
		}

		if (cy1 != d.y1 || cy2 != d.y2)
		{
			const f32 scale = (v1 - v0) / static_cast<f32>(d.y2 - d.y1);
			const f32 new_v0 = v0 + static_cast<f32>(cy1 - d.y1) * scale;
			const f32 new_v1 = v0 + static_cast<f32>(cy2 - d.y1) * scale;
			v0 = new_v0;
			v1 = new_v1;
		}

		m_src_rect[0] = u0 / src_w;
		m_src_rect[1] = v0 / src_h;
		m_src_rect[2] = u1 / src_w;
		m_src_rect[3] = v1 / src_h;

		// Source views (owned here, the command buffer retains the underlying textures)
		std::vector<std::unique_ptr<mtl::image_view>> owned_views;
		std::vector<mtl::image_view*> views;

		auto make_view = [&](MTLPixelFormat format, u32 view_aspect) -> mtl::image_view*
		{
			owned_views.push_back(std::make_unique<mtl::image_view>(src, format, MTLTextureType2D,
				mtl::default_component_map, view_aspect, src_level, 1, src_layer, 1));
			return owned_views.back().get();
		};

		const u32 depth_view_aspect = (src_aspect & mtl::image_aspect::depth) ? static_cast<u32>(mtl::image_aspect::depth) : static_cast<u32>(mtl::image_aspect::color);

		switch (m_mode)
		{
		case blit_mode::color:
			views.push_back(make_view(src->format(), mtl::image_aspect::color));
			break;
		case blit_mode::depth:
			views.push_back(make_view(src->format(), depth_view_aspect));
			break;
		case blit_mode::stencil:
			views.push_back(make_view(mtl::get_stencil_view_format(src->format()), mtl::image_aspect::stencil));
			break;
		case blit_mode::depth_stencil:
			views.push_back(make_view(src->format(), depth_view_aspect));
			views.push_back(make_view(mtl::get_stencil_view_format(src->format()), mtl::image_aspect::stencil));
			break;
		}

		// Fixed function state
		const bool write_color = (m_mode == blit_mode::color);
		const bool write_depth = (m_mode == blit_mode::depth || m_mode == blit_mode::depth_stencil);
		const bool write_stencil = (m_mode == blit_mode::stencil || m_mode == blit_mode::depth_stencil);

		renderpass_config.set_attachment_count(write_color ? 1 : 0);
		renderpass_config.set_color_mask(0, true, true, true, true);
		renderpass_config.disable_blend(0);

		if (write_depth)
		{
			renderpass_config.enable_depth_test(MTLCompareFunctionAlways);
			renderpass_config.set_depth_mask(true);
		}
		else
		{
			renderpass_config.disable_depth_test();
		}

		if (write_stencil)
		{
			// The reference value is exported by the fragment shader
			renderpass_config.enable_stencil_test(
				MTLStencilOperationReplace, MTLStencilOperationReplace, MTLStencilOperationReplace,
				MTLCompareFunctionAlways, 0xFF, 0);
			renderpass_config.set_stencil_mask(0xFF);
		}
		else
		{
			renderpass_config.disable_stencil_test();
		}

		// Depth and stencil data must never be interpolated
		m_sampler_filter = (write_color && linear) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;

		// Skip loading the attachment contents when every aspect of the target is overwritten
		const bool full_coverage = (cx1 == 0 && cy1 == 0 && cx2 == dst_w && cy2 == dst_h);
		const bool all_aspects = write_color ||
			((!(dst_aspect & mtl::image_aspect::depth) || write_depth) && (!(dst_aspect & mtl::image_aspect::stencil) || write_stencil));
		const MTLLoadAction load = (full_coverage && all_aspects) ? MTLLoadActionDontCare : MTLLoadActionLoad;

		const areau viewport =
		{
			static_cast<u32>(cx1),
			static_cast<u32>(cy1),
			static_cast<u32>(cx2),
			static_cast<u32>(cy2)
		};

		overlay_pass::run(cmd, viewport, overlay_render_target::from(dst, dst_level, dst_layer), views, load);
	}

	void blit_pass::run(mtl::command_context& cmd, mtl::image* src, mtl::image* dst,
		const coord3i& src_rect, const coord3i& dst_rect,
		u32 src_level, u32 src_layer, u32 dst_level, u32 dst_layer,
		bool linear, u32 aspect)
	{
		const areai src_area = { src_rect.x, src_rect.y, src_rect.x + src_rect.width, src_rect.y + src_rect.height };
		const areai dst_area = { dst_rect.x, dst_rect.y, dst_rect.x + dst_rect.width, dst_rect.y + dst_rect.height };
		run(cmd, src, dst, src_area, dst_area, src_level, src_layer, dst_level, dst_layer, linear, aspect);
	}

	// --------------------------------------------------------------------------------------------
	// Global management
	// --------------------------------------------------------------------------------------------

	void reset_overlay_passes()
	{
		for (const auto& p : g_overlay_passes)
		{
			p.second->free_resources();
		}
	}

	void destroy_overlay_passes()
	{
		for (const auto& p : g_overlay_passes)
		{
			p.second->destroy();
		}

		g_overlay_passes.clear();
	}
}
