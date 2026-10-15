#include "stdafx.h"
#include "MTLGSRender.h"

#include "Emu/RSX/Common/BufferUtils.h"
#include "Emu/RSX/rsx_methods.h"
#include "Emu/system_config.h"

#include <bit>

// Metal port of VKDraw.cpp (+ the viewport/scissor helpers of VKGSRender.cpp).
//
// Differences from VK:
// - All dynamic state (viewport, scissor, depth bias, stencil references, blend color, cull mode, winding, depth clip,
//   fill mode, visibility counting) and all resource bindings are written to mtl::render_encoder_state, which pushes the
//   dirty parts into the main render encoder right before the draw. Metal encoder state does not survive the encoder, so
//   the whole state is re-applied whenever the main pass is reopened (begin_render_pass invalidates it).
// - Depth/stencil state (including the stencil masks, which are dynamic in VK) lives in an immutable
//   MTLDepthStencilState looked up from m_depth_stencil_properties.
// - Draw ordering: everything that may record blits, compute or helper passes (texture uploads, temporary subresources,
//   surface barriers, vertex/index uploads) happens before the main pass is (re)opened, then the state is applied and the
//   draw is encoded. Feedback loops are resolved by ending the pass (render_target::texture_barrier).
// - There are no image layouts, so validate_image_layout_for_read_access reduces to the feedback loop barrier.
// - No hardware depth bounds test, no line width, no conditional rendering (rsx::thread CPU fallback).

namespace mtl
{
	namespace
	{
		template <bool Fragment, typename Bindings>
		void apply_stage_bindings(id<MTLRenderCommandEncoder> encoder, Bindings& stage)
		{
			// Full buffer rebinds (set*Buffer / set*Bytes)
			for (u32 mask = stage.buffer_dirty; mask; mask &= (mask - 1))
			{
				const u32 slot = std::countr_zero(mask);
				const auto& binding = stage.buffers[slot];

				if (binding.bytes)
				{
					if (!binding.length)
					{
						// Empty block, nothing to upload
						continue;
					}

					if constexpr (Fragment)
					{
						[encoder setFragmentBytes:binding.bytes length:binding.length atIndex:slot];
					}
					else
					{
						[encoder setVertexBytes:binding.bytes length:binding.length atIndex:slot];
					}

					continue;
				}

				if constexpr (Fragment)
				{
					[encoder setFragmentBuffer:binding.buffer offset:binding.offset atIndex:slot];
				}
				else
				{
					[encoder setVertexBuffer:binding.buffer offset:binding.offset atIndex:slot];
				}
			}

			// Same buffer, new offset (the buffer was bound to this encoder before)
			for (u32 mask = (stage.buffer_offset_dirty & ~stage.buffer_dirty); mask; mask &= (mask - 1))
			{
				const u32 slot = std::countr_zero(mask);
				const auto& binding = stage.buffers[slot];

				if constexpr (Fragment)
				{
					[encoder setFragmentBufferOffset:binding.offset atIndex:slot];
				}
				else
				{
					[encoder setVertexBufferOffset:binding.offset atIndex:slot];
				}
			}

			for (u64 mask = stage.texture_dirty; mask; mask &= (mask - 1))
			{
				const u32 slot = std::countr_zero(mask);
				if (slot >= Bindings::max_textures)
				{
					break;
				}

				if constexpr (Fragment)
				{
					[encoder setFragmentTexture:stage.textures[slot] atIndex:slot];
				}
				else
				{
					[encoder setVertexTexture:stage.textures[slot] atIndex:slot];
				}
			}

			for (u32 mask = stage.sampler_dirty; mask; mask &= (mask - 1))
			{
				const u32 slot = std::countr_zero(mask);
				if (slot >= Bindings::max_samplers)
				{
					break;
				}

				if constexpr (Fragment)
				{
					[encoder setFragmentSamplerState:stage.samplers[slot] atIndex:slot];
				}
				else
				{
					[encoder setVertexSamplerState:stage.samplers[slot] atIndex:slot];
				}
			}

			stage.buffer_dirty = 0;
			stage.buffer_offset_dirty = 0;
			stage.texture_dirty = 0;
			stage.sampler_dirty = 0;
		}
	}

	void render_encoder_state::apply(id<MTLRenderCommandEncoder> encoder)
	{
		ensure(encoder, "Render encoder state applied without an encoder");

		if (dirty)
		{
			if ((dirty & dirty_pipeline) && pipeline)
			{
				[encoder setRenderPipelineState:pipeline];
			}

			if ((dirty & dirty_depth_stencil) && depth_stencil)
			{
				[encoder setDepthStencilState:depth_stencil];
			}

			if (dirty & dirty_viewport)
			{
				[encoder setViewport:viewport];
			}

			if (dirty & dirty_scissor)
			{
				[encoder setScissorRect:scissor];
			}

			if (dirty & dirty_cull_mode)
			{
				[encoder setCullMode:cull_mode];
			}

			if (dirty & dirty_front_face)
			{
				[encoder setFrontFacingWinding:front_face];
			}

			if (dirty & dirty_depth_bias)
			{
				[encoder setDepthBias:depth_bias slopeScale:depth_bias_slope clamp:depth_bias_clamp];
			}

			if (dirty & dirty_stencil_ref)
			{
				[encoder setStencilFrontReferenceValue:stencil_ref_front backReferenceValue:stencil_ref_back];
			}

			if (dirty & dirty_blend_color)
			{
				[encoder setBlendColorRed:blend_color[0] green:blend_color[1] blue:blend_color[2] alpha:blend_color[3]];
			}

			if (dirty & dirty_depth_clip)
			{
				[encoder setDepthClipMode:depth_clip_mode];
			}

			if (dirty & dirty_fill_mode)
			{
				[encoder setTriangleFillMode:fill_mode];
			}

			if (dirty & dirty_visibility)
			{
				[encoder setVisibilityResultMode:visibility_mode offset:visibility_offset];
			}

			dirty = 0;
		}

		if (vs.is_dirty())
		{
			apply_stage_bindings<false>(encoder, vs);
		}

		if (fs.is_dirty())
		{
			apply_stage_bindings<true>(encoder, fs);
		}
	}
}

namespace
{
	// vk::get_view_type equivalent for placeholder textures
	MTLTextureType get_null_texture_type(rsx::texture_dimension_extended type, bool multisampled)
	{
		if (multisampled)
		{
			return MTLTextureType2DMultisample;
		}

		switch (type)
		{
		case rsx::texture_dimension_extended::texture_dimension_1d:
			return MTLTextureType1D;
		case rsx::texture_dimension_extended::texture_dimension_2d:
			return MTLTextureType2D;
		case rsx::texture_dimension_extended::texture_dimension_cubemap:
			return MTLTextureTypeCube;
		case rsx::texture_dimension_extended::texture_dimension_3d:
			return MTLTextureType3D;
		default: fmt::throw_exception("Unreachable");
		}
	}

	// Packs a border color into the RSX ARGB8 layout expected by mtl::get_border_color (inverse of rsx::decode_border_color)
	u32 encode_border_color(const color4f& color)
	{
		const auto to_unorm8 = [](f32 value) -> u32
		{
			return static_cast<u32>(std::clamp(value, 0.f, 1.f) * 255.f + 0.5f);
		};

		return (to_unorm8(color.a) << 24) | (to_unorm8(color.r) << 16) | (to_unorm8(color.g) << 8) | to_unorm8(color.b);
	}

	// Metal requires the scissor rectangle to lie within the render area
	MTLScissorRect clamp_scissor(const MTLScissorRect& scissor, u32 width, u32 height)
	{
		if (!width || !height)
		{
			// Render area unknown, nothing to clamp against
			return scissor;
		}

		MTLScissorRect result;
		result.x = std::min<NSUInteger>(scissor.x, width);
		result.y = std::min<NSUInteger>(scissor.y, height);
		result.width = std::min<NSUInteger>(scissor.width, width - result.x);
		result.height = std::min<NSUInteger>(scissor.height, height - result.y);
		return result;
	}
}

bool MTLGSRender::check_heap_status(const mtl::ring_buffer& heap)
{
	if (!heap.is_critical())
	{
		return false;
	}

	handle_heap_critical();
	return true;
}

void MTLGSRender::handle_heap_critical()
{
	m_profiler.start();
	mtl::stall_trace_scope trace("Heap critical");

	mtl::frame_context_t *target_frame = nullptr;
	if (!m_queued_frames.empty())
	{
		if (m_current_frame != &m_aux_frame_context)
		{
			target_frame = m_queued_frames.front();
		}
	}

	if (target_frame == nullptr)
	{
		// No frame to wait on, drain the queue. Every ring buffer reclaims its space once the work has completed.
		flush_command_queue(true);
		m_vertex_cache->purge();

		m_current_frame->reset_heap_ptrs();
		m_last_heap_sync_time = rsx::get_shared_tag();
	}
	else
	{
		// Wait for the oldest queued frame, releasing everything it used
		frame_context_cleanup(target_frame);
	}

	m_frame_stats.flip_time += m_profiler.duration();
}

mtl::image_view* MTLGSRender::get_null_texture(MTLTextureType type, bool depth)
{
	if (depth &&
		type != MTLTextureType2D &&
		type != MTLTextureTypeCube &&
		type != MTLTextureType2DMultisample)
	{
		// Metal has no 1D or 3D depth textures
		depth = false;
	}

	auto& null_texture = m_null_textures[get_null_texture_key(type, depth)];
	if (!null_texture)
	{
		const u32 layers = (type == MTLTextureTypeCube) ? 6 : 1;
		const bool multisampled = (type == MTLTextureType2DMultisample);
		const u8 samples = multisampled ? 4 : 1;

		if (depth)
		{
			// Placeholder for shadow samplers and stencil mirrors (X32_Stencil8 view). Depth 0, stencil 0.
			null_texture = std::make_unique<mtl::viewable_image>(
				type,
				MTLPixelFormatDepth32Float_Stencil8,
				1, 1, 1, 1, layers, samples,
				MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView,
				RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32,
				MTLStorageModePrivate);

			mtl::clear_depth_stencil_image(m_cmd, null_texture.get(), 0.f, 0);
		}
		else if (multisampled)
		{
			// Multisampled textures can only be initialized on the GPU
			null_texture = std::make_unique<mtl::viewable_image>(
				type,
				MTLPixelFormatBGRA8Unorm,
				1, 1, 1, 1, 1, samples,
				MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget,
				RSX_FORMAT_CLASS_COLOR,
				MTLStorageModePrivate);

			mtl::clear_color_image(m_cmd, null_texture.get(), color4f(0.f));
		}
		else
		{
			// Transparent black like vk::null_image_view. Shared storage, zero-filled from the CPU.
			null_texture = std::make_unique<mtl::viewable_image>(
				type,
				MTLPixelFormatBGRA8Unorm,
				1, 1, 1, 1, layers, samples,
				MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView,
				RSX_FORMAT_CLASS_COLOR,
				MTLStorageModeShared);

			const u32 zero = 0;
			const u32 slices = static_cast<u32>([null_texture->value arrayLength] * ((type == MTLTextureTypeCube) ? 6 : 1));
			for (u32 slice = 0; slice < slices; ++slice)
			{
				[null_texture->value replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1)
					mipmapLevel:0
					slice:slice
					withBytes:&zero
					bytesPerRow:4
					bytesPerImage:4];
			}
		}
	}

	return null_texture->get_view(rsx::default_remap_vector.with_encoding(mtl::MTL_REMAP_IDENTITY));
}

void MTLGSRender::apply_encoder_state()
{
	const mtl::render_encoder_handle encoder = begin_render_pass();
	ensure(encoder, "The main render pass could not be opened");

	m_encoder_state.apply(encoder);
}

void MTLGSRender::update_draw_state()
{
	m_profiler.start();

	const auto& regs = rsx::method_registers;
	const auto& formats = get_render_pass_formats();
	auto ds = m_rtts.m_bound_depth_stencil.second;

	if (regs.blend_enabled_mask())
	{
		// Update blend constants
		const auto blend_colors = rsx::get_constant_blend_colors();
		m_encoder_state.set_blend_color(blend_colors[0], blend_colors[1], blend_colors[2], blend_colors[3]);
	}

	// Depth/stencil state object (VK: pipeline depth-stencil state + dynamic stencil masks)
	mtl::depth_stencil_desc ds_desc{};

	if (formats.depth_format != MTLPixelFormatInvalid && regs.depth_test_enabled())
	{
		// NOTE: Like stencil, depth write is meaningless without depth test
		ds_desc.depth_test = 1;
		ds_desc.depth_write = regs.depth_write_enabled() ? 1 : 0;
		ds_desc.depth_func = static_cast<u8>(mtl::get_compare_func(regs.depth_func()));
	}

	if (regs.stencil_test_enabled())
	{
		auto& front = ds_desc.front;
		front.func = static_cast<u8>(mtl::get_compare_func(regs.stencil_func()));
		front.fail_op = static_cast<u8>(mtl::get_stencil_op(regs.stencil_op_fail()));
		front.zfail_op = static_cast<u8>(mtl::get_stencil_op(regs.stencil_op_zfail()));
		front.pass_op = static_cast<u8>(mtl::get_stencil_op(regs.stencil_op_zpass()));
		front.read_mask = regs.stencil_func_mask();
		front.write_mask = regs.stencil_mask();

		if (regs.two_sided_stencil_test_enabled())
		{
			auto& back = ds_desc.back;
			back.func = static_cast<u8>(mtl::get_compare_func(regs.back_stencil_func()));
			back.fail_op = static_cast<u8>(mtl::get_stencil_op(regs.back_stencil_op_fail()));
			back.zfail_op = static_cast<u8>(mtl::get_stencil_op(regs.back_stencil_op_zfail()));
			back.pass_op = static_cast<u8>(mtl::get_stencil_op(regs.back_stencil_op_zpass()));
			back.read_mask = regs.back_stencil_func_mask();
			back.write_mask = regs.back_stencil_mask();
		}
		else
		{
			ds_desc.back = front;
		}

		if (ds && ds->samples() > 1 && !(ds->stencil_init_flags & 0xFF00))
		{
			constexpr u8 keep = static_cast<u8>(MTLStencilOperationKeep);
			if (ds_desc.front.fail_op != keep ||
				ds_desc.front.zfail_op != keep ||
				ds_desc.front.pass_op != keep ||
				ds_desc.back.fail_op != keep ||
				ds_desc.back.zfail_op != keep ||
				ds_desc.back.pass_op != keep)
			{
				// Toggle bit 9 to signal require full bit-wise transfer
				ds->stencil_init_flags |= (1 << 8);
			}
		}

		if (formats.stencil_format == MTLPixelFormatInvalid)
		{
			// No stencil attachment, the test is a no-op
			ds_desc.front = {};
			ds_desc.back = {};
		}
		else
		{
			ds_desc.stencil_test = 1;
		}
	}

	if (!m_depth_stencil_state || !(ds_desc == m_depth_stencil_properties))
	{
		m_depth_stencil_properties = ds_desc;
		m_depth_stencil_state = mtl::get_depth_stencil_state(ds_desc);
	}

	m_encoder_state.set_depth_stencil_state(m_depth_stencil_state);

	if (ds_desc.stencil_test)
	{
		const u32 front_ref = regs.stencil_func_ref();
		const u32 back_ref = regs.two_sided_stencil_test_enabled() ? regs.back_stencil_func_ref() : front_ref;
		m_encoder_state.set_stencil_reference(front_ref, back_ref);
	}

	// Rasterizer state
	m_encoder_state.set_cull_mode(regs.cull_face_enabled() ? mtl::get_cull_face(regs.cull_face_mode()) : MTLCullModeNone);
	m_encoder_state.set_front_face(mtl::get_front_face(regs.front_face_mode()));

	// Depth clip and clamp are emulated in the fragment shader when depth range emulation is active
	const bool depth_clamp =
		(current_fragment_program.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_RANGE) ||
		regs.depth_clamp_enabled() ||
		!regs.depth_clip_enabled();
	m_encoder_state.set_depth_clip_mode(depth_clamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip);

	// Polygon mode. Only one mode can be active at a time, so pick the visible face (same as GL).
	const bool show_back = regs.cull_face_enabled() && regs.cull_face_mode() == rsx::cull_face::front;
	m_encoder_state.set_fill_mode(mtl::get_polygon_mode(show_back ? regs.polygon_mode_back() : regs.polygon_mode_front()));

	if (regs.poly_offset_fill_enabled())
	{
		// offset_bias is the constant factor, multiplied by the implementation factor R
		// offst_scale is the slope factor, multiplied by the triangle slope factor M
		// Metal has no fixed-point 24-bit depth formats, so the VK NVIDIA D24 correction never applies.
		m_encoder_state.set_depth_bias(regs.poly_offset_bias(), regs.poly_offset_scale(), 0.f);
	}
	else
	{
		// Zero bias value - disables depth bias
		m_encoder_state.set_depth_bias(0.f, 0.f, 0.f);
	}

	// The depth bounds test is emulated in the fragment shader (RSX_SHADER_CONTROL_EMULATE_DEPTH_BOUNDS, see load_program_env)

	bind_viewport();

	get_cb_flags() &= ~mtl::command_buffer_flags::cb_reload_dynamic_state;
	m_graphics_state.clear(rsx::pipeline_state::polygon_offset_state_dirty | rsx::pipeline_state::depth_bounds_state_dirty);
	m_frame_stats.setup_time += m_profiler.duration();
}

void MTLGSRender::load_texture_env()
{
	// Load textures
	auto check_surface_cache_sampler_valid = [&](auto descriptor, const auto& tex)
	{
		return !m_texture_cache.test_if_descriptor_expired(m_cmd, m_rtts, descriptor, tex);
	};

	const auto& caps = mtl::get_caps();

	std::lock_guard lock(m_sampler_mutex);

	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			continue;
		}

		if (!fs_sampler_state[i])
		{
			fs_sampler_state[i] = std::make_unique<mtl::texture_cache::sampled_image_descriptor>();
		}

		auto sampler_state = static_cast<mtl::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());
		const auto& tex = rsx::method_registers.fragment_textures[i];
		const auto previous_format_class = fs_sampler_state[i]->format_class;

		if (!m_samplers_dirty &&
			!m_textures_dirty[i] &&
			check_surface_cache_sampler_valid(sampler_state, tex))
		{
			continue;
		}

		const bool is_sampler_dirty = m_textures_dirty[i];
		m_textures_dirty[i] = false;

		if (!tex.enabled())
		{
			*sampler_state = {};
			continue;
		}

		// Relieve upload heap pressure. Uploads of the previous textures are already recorded, nothing else is allocated yet.
		check_heap_status(m_texture_upload_buffer_ring_info);

		*sampler_state = m_texture_cache.upload_texture(m_cmd, tex, m_rtts);
		if (!sampler_state->validate())
		{
			continue;
		}

		if (!is_sampler_dirty)
		{
			if (sampler_state->format_class != previous_format_class)
			{
				// Host details changed but RSX is not aware
				m_graphics_state |= rsx::fragment_program_state_dirty;
			}

			if (sampler_state->format_ex)
			{
				// Nothing to change, use cached sampler
				continue;
			}
		}

		sampler_state->format_ex = tex.format_ex();

		if (sampler_state->format_ex.texel_remap_control &&
			sampler_state->image_handle &&
			sampler_state->upload_context == rsx::texture_upload_context::shader_read &&
			(current_fp_metadata.bx2_texture_reads_mask & (1u << i)) == 0 &&
			!g_cfg.video.disable_hardware_texel_remapping) [[ unlikely ]]
		{
			// Check if we need to override the view format
			const MTLPixelFormat mtl_format = sampler_state->image_handle->format();
			MTLPixelFormat format_override = mtl_format;
			rsx::flags32_t flags_to_erase = 0u;
			rsx::flags32_t host_flags_to_set = 0u;

			if (sampler_state->format_ex.hw_SNORM_possible())
			{
				format_override = mtl::get_compatible_snorm_format(mtl_format);
				flags_to_erase = rsx::texture_control_bits::SEXT_MASK;
				host_flags_to_set = rsx::RSX_HOST_FORMAT_FEATURE_SNORM;
			}
			else if (sampler_state->format_ex.hw_SRGB_possible())
			{
				format_override = mtl::get_compatible_srgb_format(mtl_format);
				flags_to_erase = rsx::texture_control_bits::GAMMA_CTRL_MASK;
				host_flags_to_set = rsx::RSX_HOST_FORMAT_FEATURE_SRGB;
			}

			if (format_override != MTLPixelFormatInvalid && format_override != mtl_format)
			{
				sampler_state->image_handle = sampler_state->image_handle->as(format_override);
				sampler_state->format_ex.texel_remap_control &= (~flags_to_erase);
				sampler_state->format_ex.host_features |= host_flags_to_set;
			}
		}

		mtl::sampler_desc desc{};
		f32 min_lod = 0.f, max_lod = 0.f;
		f32 lod_bias = 0.f;

		const u32 texture_format = sampler_state->format_ex.format();

		if (texture_format >= CELL_GCM_TEXTURE_DEPTH24_D8 && texture_format <= CELL_GCM_TEXTURE_DEPTH16_FLOAT)
		{
			desc.compare_enabled = true;
			desc.compare_func = mtl::get_compare_func(tex.zfunc(), true);
		}

		desc.max_anisotropy = static_cast<u8>(std::min<u32>(mtl::max_aniso(tex.max_aniso()), caps.max_anisotropy));
		desc.address_s = mtl::mtl_wrap_mode(tex.wrap_s());
		desc.address_t = mtl::mtl_wrap_mode(tex.wrap_t());
		desc.address_r = mtl::mtl_wrap_mode(tex.wrap_r());

		// NOTE: Metal border colors go through the view swizzle (verified on Apple GPUs), so the border color is given
		// in the stored (unswizzled) channel order, like VK on devices without require_border_color_remap.
		// Only the transparent black, opaque black and opaque white presets exist.
		const bool sext_conv_required = (sampler_state->format_ex.texel_remap_control & rsx::texture_control_bits::SEXT_MASK) != 0;
		desc.border_color = MTLSamplerBorderColorOpaqueBlack;

		if (rsx::is_border_clamped_texture(tex))
		{
			auto color_value = rsx::decode_border_color(tex.border_color(sext_conv_required));
			if (sampler_state->format_ex.host_snorm_format_active())
			{
				// Convert the border color in host space (2N - 1)
				// HW does the conversion in integer space as (x - 128) / 127 which introduces a biasing error.
				const float bias_v = 128.f / 255.f;
				const float scale_v = 255.f / 127.f;

				color4f scale{ 1.f }, bias{ 0.f };
				const auto snorm_mask = tex.argb_signed();
				if (snorm_mask & 1) { scale.a = scale_v; bias.a = -bias_v; }
				if (snorm_mask & 2) { scale.r = scale_v; bias.r = -bias_v; }
				if (snorm_mask & 4) { scale.g = scale_v; bias.g = -bias_v; }
				if (snorm_mask & 8) { scale.b = scale_v; bias.b = -bias_v; }
				color_value = (color_value + bias) * scale;
			}

			desc.border_color = mtl::get_border_color(encode_border_color(color_value));
		}

		// Check if non-point filtering can even be used on this format
		bool can_sample_linear;
		if (sampler_state->format_class == RSX_FORMAT_CLASS_COLOR) [[likely]]
		{
			// Most PS3-like formats can be linearly filtered without problem
			// Exclude textures that require SNORM conversion however
			can_sample_linear = !sext_conv_required;
		}
		else if (sampler_state->format_class != rsx::classify_format(texture_format) &&
			(texture_format == CELL_GCM_TEXTURE_A8R8G8B8 || texture_format == CELL_GCM_TEXTURE_D8R8G8B8))
		{
			// Depth format redirected to BGRA8 resample stage. Do not filter to avoid bits leaking
			can_sample_linear = false;
		}
		else
		{
			// Not all formats support linear filtering (e.g. 32-bit float formats on some GPUs)
			const MTLPixelFormat mtl_format = sampler_state->image_handle ? sampler_state->image_handle->image()->format() :
				mtl::get_compatible_sampler_format(mtl::get_formats_support(), sampler_state->external_subresource_desc.gcm_format);

			can_sample_linear = mtl::is_linear_filtering_supported(mtl_format);
		}

		const auto mipmap_count = tex.get_exact_mipmap_count();
		auto min_filter = mtl::get_min_filter(tex.min_filter());
		MTLSamplerMinMagFilter mag_filter;

		if (can_sample_linear)
		{
			mag_filter = mtl::get_mag_filter(tex.mag_filter());
		}
		else
		{
			mag_filter = MTLSamplerMinMagFilterNearest;
			min_filter.filter = MTLSamplerMinMagFilterNearest;
			min_filter.mipmap_mode = MTLSamplerMipFilterNearest;
		}

		if (min_filter.sample_mipmaps && mipmap_count > 1)
		{
			f32 actual_mipmaps;
			if (sampler_state->upload_context == rsx::texture_upload_context::shader_read)
			{
				actual_mipmaps = static_cast<f32>(mipmap_count);
			}
			else if (sampler_state->external_subresource_desc.op != rsx::deferred_request_command::nop)
			{
				actual_mipmaps = sampler_state->external_subresource_desc.exact_mip_count();
			}
			else
			{
				actual_mipmaps = 1.f;
			}

			if (actual_mipmaps > 1.f)
			{
				min_lod = tex.min_lod();
				max_lod = tex.max_lod();
				lod_bias = tex.bias();

				min_lod = std::min(min_lod, actual_mipmaps - 1.f);
				max_lod = std::min(max_lod, actual_mipmaps - 1.f);

				if (min_filter.mipmap_mode == MTLSamplerMipFilterNearest)
				{
					// Round to nearest 0.5 to work around some broken games
					// Sampler parameters cannot be dynamically changed on Metal either, leading to many permutations
					lod_bias = std::floor(lod_bias * 2.f + 0.5f) * 0.5f;
				}
			}
			else
			{
				min_lod = max_lod = lod_bias = 0.f;
				min_filter.mipmap_mode = MTLSamplerMipFilterNearest;
			}
		}

		desc.min_filter = min_filter.filter;
		desc.mag_filter = mag_filter;
		desc.mip_filter = min_filter.sample_mipmaps ? min_filter.mipmap_mode : MTLSamplerMipFilterNotMipmapped;
		desc.min_lod = min_lod;
		desc.max_lod = max_lod;
		desc.lod_bias = lod_bias;
		desc.normalized_coords = true;

		if (fs_sampler_handles[i] && fs_sampler_descs[i] == desc)
		{
			continue;
		}

		fs_sampler_handles[i] = mtl::get_sampler_cache().get(desc);
		fs_sampler_descs[i] = desc;
	}

	for (u32 textures_ref = current_vp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			continue;
		}

		if (!vs_sampler_state[i])
		{
			vs_sampler_state[i] = std::make_unique<mtl::texture_cache::sampled_image_descriptor>();
		}

		auto sampler_state = static_cast<mtl::texture_cache::sampled_image_descriptor*>(vs_sampler_state[i].get());
		const auto& tex = rsx::method_registers.vertex_textures[i];
		const auto previous_format_class = sampler_state->format_class;

		if (!m_samplers_dirty &&
			!m_vertex_textures_dirty[i] &&
			check_surface_cache_sampler_valid(sampler_state, tex))
		{
			continue;
		}

		const bool is_sampler_dirty = m_vertex_textures_dirty[i];
		m_vertex_textures_dirty[i] = false;

		if (!rsx::method_registers.vertex_textures[i].enabled())
		{
			*sampler_state = {};
			continue;
		}

		check_heap_status(m_texture_upload_buffer_ring_info);

		*sampler_state = m_texture_cache.upload_texture(m_cmd, tex, m_rtts);
		if (!sampler_state->validate())
		{
			continue;
		}

		if (!is_sampler_dirty)
		{
			if (sampler_state->format_class != previous_format_class)
			{
				// Host details changed but RSX is not aware
				m_graphics_state |= rsx::vertex_program_state_dirty;
			}

			if (vs_sampler_handles[i])
			{
				continue;
			}
		}

		mtl::sampler_desc desc{};
		desc.normalized_coords = !(tex.format() & CELL_GCM_TEXTURE_UN);
		desc.min_lod = tex.min_lod();
		desc.max_lod = tex.max_lod();
		desc.lod_bias = 0.f;
		desc.max_anisotropy = 1;
		desc.address_s = mtl::mtl_wrap_mode(tex.wrap_s());
		desc.address_t = mtl::mtl_wrap_mode(tex.wrap_t());
		desc.address_r = MTLSamplerAddressModeRepeat;
		desc.min_filter = MTLSamplerMinMagFilterNearest;
		desc.mag_filter = MTLSamplerMinMagFilterNearest;
		desc.mip_filter = MTLSamplerMipFilterNearest;

		desc.border_color = rsx::is_border_clamped_texture(tex)
			? mtl::get_border_color(encode_border_color(rsx::decode_border_color(tex.border_color(false))))
			: MTLSamplerBorderColorOpaqueBlack;

		if (vs_sampler_handles[i] && vs_sampler_descs[i] == desc)
		{
			continue;
		}

		vs_sampler_handles[i] = mtl::get_sampler_cache().get(desc);
		vs_sampler_descs[i] = desc;
	}

	m_samplers_dirty.store(false);
}

bool MTLGSRender::bind_texture_env()
{
	bool out_of_memory = false;

	const mtl::sampler_state_handle default_sampler = mtl::get_sampler_cache().get(mtl::sampler_desc{});
	const auto& fs_texture_state = current_fragment_program.texture_state;

	// Stencil mirrors are usampler declarations; the placeholder is the stencil aspect of a null depth-stencil texture
	const auto get_null_stencil_view = [this](MTLTextureType type) -> mtl::image_view*
	{
		if (type != MTLTextureTypeCube && type != MTLTextureType2DMultisample)
		{
			// Depth surfaces (the only redirected sources) are 2D
			type = MTLTextureType2D;
		}

		auto root_image = static_cast<mtl::viewable_image*>(get_null_texture(type, true)->image());
		return root_image->get_view(rsx::default_remap_vector, mtl::image_aspect::stencil);
	};

	// VK validate_image_layout_for_read_access: a cyclic reference must see the attachment contents written so far.
	// Normally the texture cache already inserted the barrier, but that is only guaranteed if the attachment state never
	// changes outside of RTT rebind interrupts (Draw 1 (cyclic) -> barrier -> Draw 2 -> Draw 3 (cyclic again, no new data)).
	const auto validate_feedback_loop = [this](mtl::image_view* view, const mtl::texture_cache::sampled_image_descriptor* sampler_state)
	{
		if (sampler_state->upload_context != rsx::texture_upload_context::framebuffer_storage ||
			!sampler_state->is_cyclic_reference)
		{
			return;
		}

		if (auto rtt = mtl::try_as_rtt(view->image());
			rtt && !rtt->is_feedback_loop_active())
		{
			rtt->texture_barrier(m_cmd);
		}
	};

	// Respect the program's reflection data: every texture the program declares must be bound
	const u32 fs_textures_mask = current_fp_metadata.referenced_textures_mask | (m_program ? m_program->fs_texture_mask : 0u);
	const u32 fs_stencil_mask = fs_texture_state.redirected_textures | (m_program ? m_program->fs_stencil_mirror_mask : 0u);

	for (u32 textures_ref = fs_textures_mask, i = 0; textures_ref && i < rsx::limits::fragment_textures_count; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			// Unused TIU
			continue;
		}

		const u32 mask = (1u << i);
		mtl::image_view* view = nullptr;
		auto sampler_state = static_cast<mtl::texture_cache::sampled_image_descriptor*>(fs_sampler_state[i].get());

		if ((current_fp_metadata.referenced_textures_mask & mask) &&
			sampler_state &&
			rsx::method_registers.fragment_textures[i].enabled() &&
			sampler_state->validate())
		{
			if (view = sampler_state->image_handle; !view)
			{
				//Requires update, copy subresource
				if (!(view = m_texture_cache.create_temporary_subresource(m_cmd, sampler_state->external_subresource_desc)))
				{
					out_of_memory = true;
				}
			}
			else
			{
				validate_feedback_loop(view, sampler_state);
			}
		}

		if (view) [[likely]]
		{
			m_encoder_state.fs.set_texture(mtl::binding::fs_texture0 + i, view->value);
			m_encoder_state.fs.set_sampler(i, fs_sampler_handles[i] ? fs_sampler_handles[i] : default_sampler);

			if (fs_stencil_mask & mask)
			{
				// Stencil mirror required
				auto root_image = static_cast<mtl::viewable_image*>(view->image());
				mtl::image_view* stencil_view = (root_image->aspect() & mtl::image_aspect::stencil)
					? root_image->get_view(rsx::default_remap_vector, mtl::image_aspect::stencil)
					: get_null_stencil_view(root_image->value.textureType);

				m_encoder_state.fs.set_texture(mtl::binding::fs_stencil_mirror0 + i, stencil_view->value);
			}
		}
		else
		{
			const MTLTextureType view_type = get_null_texture_type(
				current_fragment_program.get_texture_dimension(i),
				!!(fs_texture_state.multisampled_textures & mask));

			// Shadow samplers are depth textures in MSL
			const bool depth = !!(fs_texture_state.shadow_textures & mask);

			m_encoder_state.fs.set_texture(mtl::binding::fs_texture0 + i, get_null_texture(view_type, depth)->value);
			m_encoder_state.fs.set_sampler(i, default_sampler);

			if (fs_stencil_mask & mask)
			{
				m_encoder_state.fs.set_texture(mtl::binding::fs_stencil_mirror0 + i, get_null_stencil_view(view_type)->value);
			}
		}
	}

	const auto& vs_texture_state = current_vertex_program.texture_state;
	const u32 vs_textures_mask = current_vp_metadata.referenced_textures_mask | (m_program ? m_program->vs_texture_mask : 0u);

	for (u32 textures_ref = vs_textures_mask, i = 0; textures_ref && i < rsx::limits::vertex_textures_count; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			// Unused TIU
			continue;
		}

		const u32 mask = (1u << i);
		const auto bind_null_texture = [&]()
		{
			const MTLTextureType view_type = get_null_texture_type(
				current_vertex_program.get_texture_dimension(i),
				!!(vs_texture_state.multisampled_textures & mask));

			m_encoder_state.vs.set_texture(mtl::binding::vs_texture0 + i, get_null_texture(view_type)->value);
			m_encoder_state.vs.set_sampler(i, default_sampler);
		};

		if (!(current_vp_metadata.referenced_textures_mask & mask) ||
			!rsx::method_registers.vertex_textures[i].enabled() ||
			!vs_sampler_state[i])
		{
			bind_null_texture();
			continue;
		}

		auto sampler_state = static_cast<mtl::texture_cache::sampled_image_descriptor*>(vs_sampler_state[i].get());
		auto image_ptr = sampler_state->image_handle;

		if (!image_ptr && sampler_state->validate())
		{
			if (!(image_ptr = m_texture_cache.create_temporary_subresource(m_cmd, sampler_state->external_subresource_desc)))
			{
				out_of_memory = true;
			}
		}

		if (!image_ptr)
		{
			rsx_log.error("Texture upload failed to vtexture index %d. Binding null sampler.", i);
			bind_null_texture();
			continue;
		}

		validate_feedback_loop(image_ptr, sampler_state);

		m_encoder_state.vs.set_texture(mtl::binding::vs_texture0 + i, image_ptr->value);
		m_encoder_state.vs.set_sampler(i, vs_sampler_handles[i] ? vs_sampler_handles[i] : default_sampler);
	}

	if ((current_fragment_program.ctrl & (RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE | RSX_SHADER_CONTROL_EMULATE_DEPTH_BOUNDS)) ||
		(m_program && m_program->fs_uses_depth_input))
	{
		mtl::image_view* depth_view = nullptr;

		if (auto ds = m_rtts.m_bound_depth_stencil.second)
		{
			depth_view = ds->get_view(rsx::default_remap_vector, mtl::image_aspect::depth);
		}

		if (!depth_view) [[unlikely]]
		{
			rsx_log.error("Fragment program reads the depth buffer, but no depth surface is bound. Binding a placeholder.");

			const bool msaa = !!(current_fragment_program.ctrl & RSX_SHADER_CONTROL_ROP_MULTISAMPLED);
			depth_view = get_null_texture(msaa ? MTLTextureType2DMultisample : MTLTextureType2D, true);
		}

		m_encoder_state.fs.set_texture(mtl::binding::fs_depth_input, depth_view->value);
	}

	if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING)
	{
		// Color attachments are read with framebuffer fetch ([[color(n)]]); no bindings or barriers are required.
		ensure(current_fragment_program.mrt_buffers_count == m_draw_buffers.size());
	}

	return out_of_memory;
}

void MTLGSRender::set_viewport()
{
	const auto [clip_width, clip_height] = rsx::apply_resolution_scale<true>(
		resolution_scaling_config,
		rsx::method_registers.surface_clip_width(), rsx::method_registers.surface_clip_height());

	const auto zclip_near = rsx::method_registers.clip_min();
	const auto zclip_far = rsx::method_registers.clip_max();

	//NOTE: The scale_offset matrix already has viewport matrix factored in
	m_viewport.originX = 0.;
	m_viewport.originY = 0.;
	m_viewport.width = clip_width;
	m_viewport.height = clip_height;

	if (backend_config.supports_extended_depth_range)
	{
		m_viewport.znear = zclip_near;
		m_viewport.zfar = zclip_far;
	}
	else
	{
		m_viewport.znear = 0.;
		m_viewport.zfar = 1.;
	}

	get_cb_flags() |= mtl::command_buffer_flags::cb_reload_dynamic_state;
	m_graphics_state.clear(rsx::pipeline_state::zclip_config_state_dirty);
}

void MTLGSRender::set_scissor(bool clip_viewport)
{
	areau scissor;
	if (get_scissor(scissor, clip_viewport))
	{
		m_scissor.width = scissor.width();
		m_scissor.height = scissor.height();
		m_scissor.x = scissor.x1;
		m_scissor.y = scissor.y1;

		get_cb_flags() |= mtl::command_buffer_flags::cb_reload_dynamic_state;
	}
}

void MTLGSRender::bind_viewport()
{
	if (m_graphics_state & rsx::pipeline_state::zclip_config_state_dirty)
	{
		if (backend_config.supports_extended_depth_range)
		{
			m_viewport.znear = rsx::method_registers.clip_min();
			m_viewport.zfar = rsx::method_registers.clip_max();
		}

		m_graphics_state.clear(rsx::pipeline_state::zclip_config_state_dirty);
	}

	m_encoder_state.set_viewport(m_viewport);

	// An empty rectangle is never pushed to the encoder (emit_geometry skips such draws)
	if (const auto scissor = clamp_scissor(m_scissor, m_draw_fbo.width, m_draw_fbo.height);
		scissor.width && scissor.height)
	{
		m_encoder_state.set_scissor(scissor);
	}
}

void MTLGSRender::emit_geometry(u32 sub_index)
{
	auto &draw_call = rsx::method_registers.current_draw_clause;
	m_profiler.start();

	const rsx::flags32_t vertex_state_mask = rsx::vertex_base_changed | rsx::vertex_arrays_changed;
	const rsx::flags32_t state_flags = (sub_index == 0) ? rsx::vertex_arrays_changed : draw_call.execute_pipeline_dependencies(m_ctx);

	if (state_flags & rsx::vertex_arrays_changed)
	{
		m_draw_processor.analyse_inputs_interleaved(m_vertex_layout, current_vp_metadata);
	}
	else if (state_flags & rsx::vertex_base_changed)
	{
		// Rebase vertex bases instead of
		for (auto& info : m_vertex_layout.interleaved_blocks)
		{
			info->vertex_range.second = 0;
			const auto vertex_base_offset = rsx::method_registers.vertex_data_base_offset();
			info->real_offset_address = rsx::get_address(rsx::get_vertex_offset_from_base(vertex_base_offset, info->base_offset), info->memory_location);
		}
	}
	else
	{
		// Discard cached results
		for (auto& info : m_vertex_layout.interleaved_blocks)
		{
			info->vertex_range.second = 0;
		}
	}

	if ((state_flags & vertex_state_mask) && !m_vertex_layout.validate())
	{
		// No vertex inputs enabled
		// Execute remainining pipeline barriers with NOP draw
		do
		{
			draw_call.execute_pipeline_dependencies(m_ctx);
		}
		while (draw_call.next());

		draw_call.end();
		return;
	}

	// Programs data is dependent on vertex state
	auto upload_info = upload_vertex_data();
	if (!upload_info.vertex_draw_count)
	{
		// Malformed vertex setup; abort
		return;
	}

	m_frame_stats.vertex_upload_time += m_profiler.duration();

	// Faults are allowed during vertex upload. Ensure consistent CB state after uploads.
	if (m_occlusion_query_active && m_active_query_info &&
		(get_cb_flags() & mtl::command_buffer_flags::cb_load_occluson_task)) [[unlikely]]
	{
		u32 occlusion_id = m_occlusion_query_manager->allocate_query(m_cmd);
		if (occlusion_id == umax)
		{
			// Force flush
			rsx_log.warning("[Performance Warning] Out of free occlusion slots. Forcing hard sync.");
			ZCULL_control::sync(this);

			occlusion_id = m_occlusion_query_manager->allocate_query(m_cmd);
		}

		if (occlusion_id == umax)
		{
			//rsx_log.error("Occlusion pool overflow");
			if (m_current_task) m_current_task->result = 1;

			// Nothing can be counted in this command buffer, do not retry on every draw
			get_cb_flags() &= ~mtl::command_buffer_flags::cb_load_occluson_task;
		}
		else
		{
			// Before starting a query, match the pass scope on drivers that require it (VK_1_0 rules in VK)
			if (mtl::use_strict_query_scopes() && is_main_pass_open())
			{
				close_render_pass();
				emergency_query_cleanup(&m_cmd);
			}

			// Begin query
			m_occlusion_query_manager->begin_query(m_cmd, occlusion_id);

			auto& data = m_occlusion_map[m_active_query_info->driver_handle];
			data.indices.push_back(occlusion_id);
			data.set_sync_command_buffer(m_cmd);

			// Counted from the next main pass segment on (see begin_render_pass)
			m_visibility_slot = occlusion_id;
			m_visibility_slot_pass_epoch = 0;

			auto& cb_flags = get_cb_flags();
			cb_flags &= ~mtl::command_buffer_flags::cb_load_occluson_task;
			cb_flags |= (mtl::command_buffer_flags::cb_has_occlusion_task | mtl::command_buffer_flags::cb_has_open_query);
		}
	}

	// Vertex attribute streams
	if (!null_buffer_view) [[unlikely]]
	{
		if (!null_buffer)
		{
			null_buffer = std::make_unique<mtl::buffer>(32, MTLResourceStorageModeShared | MTLResourceHazardTrackingModeTracked, "null buffer");
			std::memset(null_buffer->map(), 0, 32);
		}

		null_buffer_view = std::make_unique<mtl::texel_buffer_view>(null_buffer->handle(), MTLPixelFormatR8Uint, 0, static_cast<u32>(std::min<usz>(null_buffer->size(), 32)));
	}

	auto& vs = m_encoder_state.vs;
	auto& fs = m_encoder_state.fs;

	vs.set_texture(mtl::binding::vs_persistent_stream, (m_persistent_attribute_storage ? m_persistent_attribute_storage : null_buffer_view)->value);
	vs.set_texture(mtl::binding::vs_volatile_stream, (m_volatile_attribute_storage ? m_volatile_attribute_storage : null_buffer_view)->value);

	// Update vertex fetch parameters
	update_vertex_env(sub_index, upload_info);

	// Uniform blocks. Ring buffer handles are fetched after all allocations for this draw (a ring may have been enlarged).
	vs.set_bytes(mtl::binding::vs_layout_buffer, m_vertex_layout_data.data(), ::size32(m_vertex_layout_data));
	vs.set_buffer(mtl::binding::vs_constants_buffer, m_transform_constants_ring_info.handle(), m_xform_constants_dynamic_offset);

	if (current_vertex_program.ctrl & RSX_SHADER_CONTROL_INSTANCED_CONSTANTS)
	{
		const mtl::buffer_handle instancing_heap = m_instancing_buffer_ring_info.handle();
		vs.set_buffer(mtl::binding::vs_instancing_lut_buffer, instancing_heap, m_instancing_indirection_offset);
		vs.set_buffer(mtl::binding::vs_instancing_data_buffer, instancing_heap, m_instancing_constants_offset);
	}

	if (m_current_draw.subdraw_id == 0)
	{
		// Program environment, constant for the whole draw call
		vs.set_bytes(mtl::binding::vs_context_buffer, m_vertex_env_data.data(), ::size32(m_vertex_env_data));

		fs.set_bytes(mtl::binding::fs_state_buffer, m_fragment_env_data.data(), ::size32(m_fragment_env_data));
		fs.set_bytes(mtl::binding::fs_texture_params_buffer, m_fragment_texture_params_data.data(), ::size32(m_fragment_texture_params_data));
		fs.set_bytes(mtl::binding::fs_stipple_buffer, m_raster_env_data.data(), ::size32(m_raster_env_data));

		if (m_fragment_constants_inline)
		{
			if (const u32 constants_size = static_cast<u32>(m_fragment_constants_data.size()))
			{
				fs.set_bytes(mtl::binding::fs_constants_buffer, m_fragment_constants_data.data(), constants_size);
			}
		}
		else
		{
			fs.set_buffer(mtl::binding::fs_constants_buffer, m_fragment_constants_ring_info.handle(), m_fragment_constants_dynamic_offset);
		}

		if (current_fragment_program.ctrl & RSX_SHADER_CONTROL_PROGRAMMABLE_BLENDING)
		{
			fs.set_bytes(mtl::binding::fs_blend_params_buffer, m_blend_params_data.data(), ::size32(m_blend_params_data));
		}
	}

	const bool reload_state = (!m_current_draw.subdraw_id++);
	if (reload_state || (get_cb_flags() & mtl::command_buffer_flags::cb_reload_dynamic_state))
	{
		update_draw_state();
	}

	m_encoder_state.set_pipeline(m_program->render_pipeline);

	m_frame_stats.setup_time += m_profiler.duration();

	// Draws that cannot produce any fragment
	const auto scissor = clamp_scissor(m_scissor, m_draw_fbo.width, m_draw_fbo.height);
	if (!scissor.width || !scissor.height)
	{
		return;
	}

	if (rsx::method_registers.cull_face_enabled() &&
		mtl::cull_face_rejects_all_polygons(rsx::method_registers.cull_face_mode()) &&
		draw_call.classify_mode() == rsx::primitive_class::polygon)
	{
		// FRONT_AND_BACK culling (no Metal equivalent): every polygon is rejected
		return;
	}

	// Open the main pass (reapplies the whole encoder state if it was interrupted) and set up visibility counting.
	// Counting is encoder state: a query slot relocated by begin_render_pass is picked up here.
	const mtl::render_encoder_handle encoder = begin_render_pass();
	ensure(encoder, "The main render pass could not be opened");

	{
		MTLVisibilityResultMode visibility_mode = MTLVisibilityResultModeDisabled;
		u32 visibility_offset = 0;

		if (m_occlusion_query_active && m_active_query_info &&
			(get_cb_flags() & mtl::command_buffer_flags::cb_has_open_query))
		{
			u32 slot = m_visibility_slot;
			if (slot == umax)
			{
				const auto& indices = m_occlusion_map[m_active_query_info->driver_handle].indices;
				slot = indices.empty() ? umax : indices.back();
			}

			if (slot != umax)
			{
				visibility_mode = m_occlusion_query_manager->result_mode();
				visibility_offset = mtl::query_pool_manager::get_slot_offset(slot);

				if (!m_visibility_slot_pass_epoch)
				{
					m_visibility_slot_pass_epoch = m_draw_fbo.pass_epoch;
				}
			}
		}

		m_encoder_state.set_visibility_result_mode(visibility_mode, visibility_offset);
	}

	m_encoder_state.apply(encoder);

	const MTLPrimitiveType primitive = upload_info.primitive;

	if (!upload_info.index_info)
	{
		if (draw_call.is_trivial_instanced_draw)
		{
			[encoder drawPrimitives:primitive vertexStart:0 vertexCount:upload_info.vertex_draw_count instanceCount:draw_call.pass_count()];
		}
		else if (draw_call.is_single_draw())
		{
			[encoder drawPrimitives:primitive vertexStart:0 vertexCount:upload_info.vertex_draw_count instanceCount:1];
		}
		else
		{
			u32 vertex_offset = 0;
			const auto& subranges = draw_call.get_subranges();
			for (const auto& range : subranges)
			{
				if (range.count)
				{
					[encoder drawPrimitives:primitive vertexStart:vertex_offset vertexCount:range.count instanceCount:1];
				}

				vertex_offset += range.count;
			}
		}
	}
	else
	{
		const MTLIndexType index_type = std::get<1>(*upload_info.index_info);
		const u64 offset = std::get<0>(*upload_info.index_info);
		const u32 index_size = (index_type == MTLIndexTypeUInt16) ? 2 : 4;
		const mtl::buffer_handle index_buffer = m_index_buffer_ring_info.handle();

		const auto draw_indexed = [&](u32 first_index, u32 index_count, u32 instance_count)
		{
			const u64 index_offset = offset + u64{ first_index } * index_size;
			ensure((index_offset % 4) == 0, "Index buffer offsets must be 4-byte aligned");

			[encoder drawIndexedPrimitives:primitive
				indexCount:index_count
				indexType:index_type
				indexBuffer:index_buffer
				indexBufferOffset:index_offset
				instanceCount:instance_count];
		};

		if (draw_call.is_trivial_instanced_draw)
		{
			draw_indexed(0, upload_info.vertex_draw_count, draw_call.pass_count());
		}
		else if (rsx::method_registers.current_draw_clause.is_single_draw())
		{
			draw_indexed(0, upload_info.vertex_draw_count, 1);
		}
		else
		{
			u32 vertex_offset = 0;
			const auto& subranges = draw_call.get_subranges();
			for (const auto& range : subranges)
			{
				const auto count = get_index_count(draw_call.primitive, range.count);
				if (count)
				{
					draw_indexed(vertex_offset, count, 1);
				}

				vertex_offset += count;
			}
		}
	}

	m_frame_stats.draw_exec_time += m_profiler.duration();
}

void MTLGSRender::begin()
{
	rsx::thread::begin();

	if (skip_current_frame ||
		cond_render_ctrl.disable_rendering())
	{
		return;
	}

	if (m_graphics_state.test(rsx::depth_bounds_state_dirty) &&
		m_graphics_state.test(rsx::rtt_config_contested) &&
		!m_framebuffer_layout.zeta_address &&
		rsx::method_registers.depth_bounds_test_enabled() &&
		(rsx::method_registers.depth_bounds_min() > 0.f || rsx::method_registers.depth_bounds_max() < 1.f)) [[unlikely]]
	{
		// The layout dropped the unused Z buffer, then the depth bounds test was enabled. The depth bounds registers do not
		// re-evaluate the framebuffer layout, so rebuild it: layout setup keeps the Z buffer bound for an active bounds test.
		m_graphics_state.set(rsx::rtt_config_dirty);
	}

	init_buffers(rsx::framebuffer_creation_context::context_draw);

	if (m_graphics_state & rsx::pipeline_state::invalidate_pipeline_bits)
	{
		// Shaders need to be reloaded.
		m_prev_program = m_program;
		m_program = nullptr;
	}
}

void MTLGSRender::end()
{
	if (should_skip_draw())
	{
		execute_nop_draw();
		rsx::thread::end();
		return;
	}

	m_profiler.start();

	// Check for frame resource status here because it is possible for an async flip to happen between begin/end
	if (m_current_frame->flags & frame_context_state::dirty) [[unlikely]]
	{
		check_present_status();

		if (m_current_frame->swap_sync_id) [[unlikely]]
		{
			// Borrow time by using the auxilliary context
			m_aux_frame_context.grab_resources(*m_current_frame);
			m_current_frame = &m_aux_frame_context;
		}

		ensure(!m_current_frame->swap_sync_id);

		m_current_frame->flags &= ~frame_context_state::dirty;
	}

	// Relieve ring buffer pressure before anything is allocated for this draw. A hard sync lets every ring reclaim the
	// space of completed work, which is only safe while no allocation of the upcoming draw is pending.
	for (const mtl::ring_buffer* heap : {
		&m_attrib_ring_info,
		&m_index_buffer_ring_info,
		&m_transform_constants_ring_info,
		&m_fragment_constants_ring_info,
		&m_instancing_buffer_ring_info,
		&m_texture_upload_buffer_ring_info })
	{
		if (check_heap_status(*heap))
		{
			break;
		}
	}

	analyse_current_rsx_pipeline();

	m_frame_stats.setup_time += m_profiler.duration();

	load_texture_env();
	m_frame_stats.textures_upload_time += m_profiler.duration();

	if (!load_program())
	{
		// Program is not ready, skip drawing this
		std::this_thread::yield();
		execute_nop_draw();
		// m_rtts.on_write(); - breaks games for obvious reasons
		rsx::thread::end();
		return;
	}

	if (current_fragment_program.ctrl & (RSX_SHADER_CONTROL_EMULATE_DEPTH_COMPARE | RSX_SHADER_CONTROL_EMULATE_DEPTH_BOUNDS))
	{
		// The fragment shader reads the bound depth buffer (frag_depth). End the pass so that the reads see the stored
		// attachment contents. (VK does this in load_texture_env and also regenerates the render pass key; the main pass
		// is simply reopened on Metal.) Done once the program is known to be ready: a skipped draw must not leave the
		// cyclic tracker counting a barrier for a draw that never ran.
		// A chain of reads stays in the open pass while no draw writes depth (render_target::post_texture_barrier).
		auto ds = ensure(m_rtts.m_bound_depth_stencil.second, "Invalid FS export configuration.");
		ds->texture_barrier(m_cmd);

		static bool s_depth_bounds_write_reported = false;
		if (!s_depth_bounds_write_reported &&
			(current_fragment_program.ctrl & RSX_SHADER_CONTROL_EMULATE_DEPTH_BOUNDS) &&
			rsx::method_registers.depth_test_enabled() && rsx::method_registers.depth_write_enabled()) [[unlikely]]
		{
			// Hardware tests each fragment against the current Z, including Z written earlier in the same draw.
			// The shader reads the Z buffer as a texture, which only holds the values from before this draw.
			s_depth_bounds_write_reported = true;
			rsx_log.todo("Metal: Depth bounds test with depth writes enabled. Primitives of the same draw are tested against the Z buffer from before the draw. Further occurrences are not reported.");
		}
	}

	// Load program execution environment
	load_program_env();
	m_frame_stats.setup_time += m_profiler.duration();

	// Apply write memory barriers
	if (auto ds = m_rtts.m_bound_depth_stencil.second)
	{
		ds->write_barrier(m_cmd);

		if (m_graphics_state.test(rsx::zeta_address_cyclic_barrier) &&
			ds->is_feedback_loop_active())
		{
			// We actually need to end the pass as a minimum. Without this, early-Z optimizations in following draws
			// will clobber reads from previous draws and cause flickering.
			ds->reset_surface_counters();

			// Start a new main pass for the next draw
			invalidate_render_pass();
		}
	}

	for (auto &rtt : m_rtts.m_bound_render_targets)
	{
		if (auto surface = std::get<1>(rtt))
		{
			surface->write_barrier(m_cmd);
		}
	}

	m_graphics_state.clear(rsx::zeta_address_cyclic_barrier);

	m_frame_stats.setup_time += m_profiler.duration();

	// Now bind the shader resources. It is important that this takes place after the barriers so that we don't end up with stale descriptors
	for (int retry = 0; retry < 3; ++retry)
	{
		if (retry > 0 && m_samplers_dirty) [[ unlikely ]]
		{
			// Reload texture env if referenced objects were invalidated during OOM handling.
			load_texture_env();

			// Do not trust fragment/vertex texture state after a texture state reset.
			// NOTE: We don't want to change the program - it's too late for that now. We just need to harmonize the state.
			m_graphics_state |= rsx::vertex_program_state_dirty | rsx::fragment_program_state_dirty;
			get_current_fragment_program(fs_sampler_state);
			get_current_vertex_program(vs_sampler_state);
			m_graphics_state.clear(rsx::pipeline_state::invalidate_pipeline_bits);
		}

		const bool out_of_memory = bind_texture_env();
		if (!out_of_memory)
		{
			break;
		}

		// Handle OOM
		if (!on_vram_exhausted(rsx::problem_severity::fatal))
		{
			// It is not possible to free memory. Just use placeholder textures. Can cause graphics glitches but shouldn't crash otherwise
			break;
		}
	}

	m_texture_cache.release_uncached_temporary_subresources();
	m_frame_stats.textures_upload_time += m_profiler.duration();

	// Final heap check (VK: vertex storage heaps). Nothing has been allocated from these rings for this draw yet.
	for (const mtl::ring_buffer* heap : { &m_attrib_ring_info, &m_index_buffer_ring_info })
	{
		if (check_heap_status(*heap))
		{
			break;
		}
	}

	u32 sub_index = 0;               // RSX subdraw ID
	m_current_draw.subdraw_id = 0;   // Host subdraw ID. Invalid RSX subdraws do not increment this value

	if (m_graphics_state & rsx::pipeline_state::invalidate_vk_dynamic_state)
	{
		get_cb_flags() |= mtl::command_buffer_flags::cb_reload_dynamic_state;
	}

	auto& draw_call = rsx::method_registers.current_draw_clause;
	draw_call.begin();
	do
	{
		emit_geometry(sub_index++);

		if (draw_call.is_trivial_instanced_draw)
		{
			// We already completed. End the draw.
			draw_call.end();
		}
	}
	while (draw_call.next());

	m_rtts.on_write(m_framebuffer_layout.color_write_enabled, m_framebuffer_layout.zeta_write_enabled);

	rsx::thread::end();
}
