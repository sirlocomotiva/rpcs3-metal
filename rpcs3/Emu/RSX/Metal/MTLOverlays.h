#pragma once

#include "Emu/RSX/Metal/mtlutils/mtl_common.h"
#include "Emu/RSX/Metal/mtlutils/buffer.h"
#include "Emu/RSX/Metal/mtlutils/command_context.h"
#include "Emu/RSX/Metal/mtlutils/device.h"
#include "Emu/RSX/Metal/mtlutils/image.h"
#include "Emu/RSX/Metal/mtlutils/pipeline.h"
#include "Emu/RSX/Metal/mtlutils/sampler.h"
#include "Emu/RSX/Metal/mtlutils/shader.h"

#include "../Common/simple_array.hpp"
#include "../Overlays/overlay_controls.h"

#include "Emu/IdManager.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * Overlay (helper graphics) passes. Port of VKOverlays.h.
 *
 * Differences with the VK backend:
 * - There are no render pass / framebuffer objects. Passes that own their render pass (overlay_pass::run) open an
 *   MTLRenderPassDescriptor on the target (color or depth/stencil attachments are picked from the target pixel format),
 *   draw and end the encoder. This implicitly ends any render pass the renderer had open (see guide section 9.3).
 * - In-pass helpers (attachment_clear_pass, stencil_clear_pass) draw into the currently open cmd.render_encoder() and
 *   take the open pass' mtl::render_pass_formats to build a compatible pipeline. They clobber the pipeline, depth-stencil
 *   state, stencil reference, viewport, scissor, cull/fill/depth-clip/depth-bias state and the visibility result mode;
 *   the caller must re-apply its own state afterwards.
 * - Push constants are replaced by std140 parameter blocks at set=0, binding=0 bound with setVertexBytes/setFragmentBytes
 *   (mtl::binding::helper_params_buffer). Textures live at set=1, binding=N with samplers from mtl::get_sampler_cache().
 * - Vertex data (ui_overlay_renderer only) is fetched through a vertex descriptor from mtl::binding::helper_vertex_buffer.
 *   Metal has no triangle fans and no quads, so those are expanded into triangle lists on the CPU.
 * - Every vertex shader goes through mtl::compile_glsl which flips Y, so all VK viewport conventions carry over.
 */

namespace rsx
{
	namespace overlays
	{
		enum class texture_sampling_mode;
		struct overlay;
	}
}

namespace mtl
{
	class render_target;

	/**
	 * Attachment used by the passes that open their own render pass (replaces vk::framebuffer).
	 * For color formats the texture is bound as color attachment 0, otherwise as depth and/or stencil attachment.
	 */
	struct overlay_render_target
	{
		id<MTLTexture> texture = nil;
		MTLPixelFormat format = MTLPixelFormatInvalid;
		u32 level = 0;          // Mip level
		u32 slice = 0;          // Array layer / cube face
		u32 depth_plane = 0;    // 3D textures only
		u32 width = 0;          // Dimensions of the selected mip level
		u32 height = 0;
		u8 samples = 1;

		// Wrap level/layer of an image
		static overlay_render_target from(const mtl::image* img, u32 level = 0, u32 layer = 0);

		// Wrap the first level/layer of a view. Swizzled views cannot be rendered to in Metal, so for those the parent
		// image is used directly (the view must then have the same pixel format as the image).
		static overlay_render_target from(const mtl::image_view* view);

		// Wrap a raw texture (e.g. a CAMetalDrawable texture)
		static overlay_render_target from(id<MTLTexture> tex, u32 level = 0, u32 layer = 0);

		mtl::render_pass_formats get_formats() const;
		u32 aspect() const;
	};

	/**
	 * Fixed function state of an overlay pass (subset of vk::graphics_pipeline_state, same method names).
	 * pipeline_state is baked into the MTLRenderPipelineState (and hashed as raw bytes for the pipeline cache),
	 * ds is turned into an MTLDepthStencilState, stencil_reference is dynamic encoder state.
	 */
	struct overlay_pipeline_state
	{
		struct pipeline_state_t
		{
			mtl::blend_attachment_state att_state[4];
			u8 attachment_count;   // Number of color outputs written by the fragment shader
			u8 primitive_type;     // MTLPrimitiveType
			u8 alpha_to_coverage;
			u8 reserved;
		};

		pipeline_state_t pipeline{};
		mtl::depth_stencil_desc ds{};
		u32 stencil_reference = 0;

		overlay_pipeline_state();

		void set_attachment_count(u32 count);
		void set_color_mask(int index, bool r, bool g, bool b, bool a);
		void set_color_write_mask(int index, u8 mask); // Raw MTLColorWriteMask
		void set_depth_mask(bool enable);
		void set_stencil_mask(u32 mask);
		void enable_depth_test(MTLCompareFunction op);
		void disable_depth_test();
		void enable_blend(int mrt_index,
			MTLBlendFactor src_factor_rgb, MTLBlendFactor src_factor_a,
			MTLBlendFactor dst_factor_rgb, MTLBlendFactor dst_factor_a,
			MTLBlendOperation equation_rgb, MTLBlendOperation equation_a);
		void disable_blend(int mrt_index);
		void enable_stencil_test(MTLStencilOperation fail, MTLStencilOperation zfail, MTLStencilOperation pass,
			MTLCompareFunction func, u32 func_mask, u32 ref);
		void disable_stencil_test();
		void set_primitive_type(MTLPrimitiveType type);
		void set_alpha_to_coverage(bool enable);

		MTLPrimitiveType primitive_type() const { return static_cast<MTLPrimitiveType>(pipeline.primitive_type); }
	};

	/**
	 * Small CPU-visible linear allocator for helper vertex data (replaces the overlay VAO data_heap).
	 * Chunks are shared-storage MTLBuffers that are only rewound once the GPU has finished every command buffer that
	 * referenced them (tracked with command_context sync ids). Appending past the last write is always safe.
	 */
	class overlay_vertex_heap
	{
		struct chunk_t
		{
			id<MTLBuffer> buffer = nil;
			u32 size = 0;
			u32 put = 0;
			u64 sync_id = 0;
		};

		std::vector<chunk_t> m_chunks;
		usz m_current = umax;

	public:
		static constexpr u32 default_chunk_size = 0x40000; // 256K

		// Copies size bytes into the heap. Returns the buffer and the byte offset of the copy.
		std::pair<id<MTLBuffer>, u32> alloc(mtl::command_context& cmd, const void* data, u32 size);

		// Keep at most max_chunks chunks around (the GPU keeps released buffers alive while in use)
		void trim(usz max_chunks = 1);
		void destroy();
	};

	struct overlay_pass
	{
		mtl::shader_module m_vertex_shader;
		mtl::shader_module m_fragment_shader;

		MTLSamplerMinMagFilter m_sampler_filter = MTLSamplerMinMagFilterLinear;
		u32 m_num_usable_samplers = 1;
		bool m_use_vertex_input = false;   // VS reads 'layout(location=0) in vec4' (16 byte stride) from binding::helper_vertex_buffer

		std::unordered_map<u64, std::unique_ptr<mtl::program>> m_program_cache;
		id<MTLSamplerState> m_sampler = nil;
		MTLSamplerMinMagFilter m_active_sampler_filter = MTLSamplerMinMagFilterLinear;
		overlay_vertex_heap m_vao;
		const mtl::render_device* m_device = nullptr;

		// Fallback textures for unused sampler slots (vk::null_image_view)
		std::unordered_map<u32, std::pair<std::unique_ptr<mtl::image>, std::unique_ptr<mtl::image_view>>> m_null_views;

		std::string vs_src;
		std::string fs_src;

		overlay_pipeline_state renderpass_config;

		bool initialized = false;
		bool compiled = false;

		u32 num_drawable_elements = 4;
		u32 first_vertex = 0;

		id<MTLBuffer> m_vao_buffer = nil;
		u32 m_vao_offset = 0;

		// Bounds used to clamp the scissor (dimensions of the attachment of the pass we opened; 0 = unknown)
		u32 m_target_width = 0;
		u32 m_target_height = 0;

		overlay_pass();
		virtual ~overlay_pass();

		overlay_pass(const overlay_pass&) = delete;
		overlay_pass& operator=(const overlay_pass&) = delete;

		u64 get_pipeline_key(const mtl::render_pass_formats& formats);

		// Extra pipeline key bits for passes that switch shaders at runtime
		virtual u64 get_pipeline_variant() const { return 0; }

		// Encoder state. Called with the target render encoder active (cmd.render_encoder()).
		virtual void update_uniforms(mtl::command_context& /*cmd*/, mtl::program* /*program*/) {}

		// Shader functions (compiled on first use from vs_src / fs_src)
		virtual id<MTLFunction> get_vertex_function();
		virtual id<MTLFunction> get_fragment_function();
		virtual MTLVertexDescriptor* get_vertex_descriptor();

		int sampler_location(int index) const { return index; }

		template <typename T>
		void upload_vertex_data(mtl::command_context& cmd, const T* data, u32 count)
		{
			const auto [buf, offset] = m_vao.alloc(cmd, data, count * sizeof(T));
			m_vao_buffer = buf;
			m_vao_offset = offset;
		}

		mtl::program* build_pipeline(u64 storage_key, const mtl::render_pass_formats& formats);
		mtl::program* load_program(mtl::command_context& cmd, const mtl::render_pass_formats& formats, const std::vector<mtl::image_view*>& src);

		virtual void create(const mtl::render_device& dev);
		virtual void destroy();

		void free_resources();

		// 1x1 zero texture usable as a dummy binding (2D or 2DArray)
		mtl::image_view* get_null_image_view(MTLTextureType type);

		virtual void emit_geometry(mtl::command_context& cmd, mtl::program* program);

		virtual void set_up_viewport(mtl::command_context& cmd, u32 x, u32 y, u32 w, u32 h);

		// Opens a render pass on target (storeAction = Store) and returns the encoder
		id<MTLRenderCommandEncoder> begin_render_pass(mtl::command_context& cmd, const overlay_render_target& target, MTLLoadAction load);

		// Draws into the render pass currently open on cmd
		void run_in_pass(mtl::command_context& cmd, const areau& viewport, const mtl::render_pass_formats& formats, const std::vector<mtl::image_view*>& src);

		// Opens a render pass on the target, draws and ends the pass
		void run(mtl::command_context& cmd, const areau& viewport, const overlay_render_target& target, const std::vector<mtl::image_view*>& src, MTLLoadAction load = MTLLoadActionLoad);
		void run(mtl::command_context& cmd, const areau& viewport, mtl::image* target, const std::vector<mtl::image_view*>& src, MTLLoadAction load = MTLLoadActionLoad);
		void run(mtl::command_context& cmd, const areau& viewport, mtl::image* target, mtl::image_view* src, MTLLoadAction load = MTLLoadActionLoad);
		void run(mtl::command_context& cmd, const areau& viewport, mtl::image_view* target, const std::vector<mtl::image_view*>& src, MTLLoadAction load = MTLLoadActionLoad);
		void run(mtl::command_context& cmd, const areau& viewport, mtl::image_view* target, mtl::image_view* src, MTLLoadAction load = MTLLoadActionLoad);
	};

	// Deprecated in VK (ARGB8 -> D24 reinterpretation), kept for completeness: writes depth from an RGBA8 encoded source
	struct depth_convert_pass : public overlay_pass
	{
		f32 src_rect[4] = { 0.f, 0.f, 1.f, 1.f }; // Normalized source rectangle (u0, v0, u1, v1)

		static constexpr u32 vertex_push_constants_size = 16;

		depth_convert_pass();

		void update_uniforms(mtl::command_context& cmd, mtl::program* program) override;

		void run(mtl::command_context& cmd, const areai& src_area, const areai& dst_area, mtl::image_view* src, mtl::image* dst, MTLLoadAction load = MTLLoadActionLoad);
	};

	// Reinterprets RGBA8_UNORM texel data as RG16_SFLOAT (1:1 pixel mapping, texelFetch at gl_FragCoord)
	struct rgba8_unorm_rg16_sfloat_convert_pass : public overlay_pass
	{
		rgba8_unorm_rg16_sfloat_convert_pass();

		void run(mtl::command_context& cmd, const areau& viewport, mtl::image* target, mtl::image_view* src, MTLLoadAction load = MTLLoadActionLoad);
	};

	struct ui_overlay_renderer : public overlay_pass
	{
		f32 m_time = 0.f;
		f32 m_blur_strength = 0.f;
		color4f m_scale_offset;
		color4f m_color;
		bool m_pulse_glow = false;
		bool m_clip_enabled = false;
		bool m_disable_vertex_snap = false;
		rsx::overlays::texture_sampling_mode m_texture_type;
		areaf m_clip_region;
		coordf m_viewport;

		rsx::overlays::compiled_resource::sdf_config_t m_sdf_config{};

		std::vector<std::unique_ptr<mtl::image>> resources;
		std::unordered_map<u64, std::unique_ptr<mtl::image>> font_cache;
		std::unordered_map<u64, std::unique_ptr<mtl::image_view>> view_cache;
		std::unordered_map<u64, std::pair<u32, std::unique_ptr<mtl::image>>> temp_image_cache;
		std::unordered_map<u64, std::unique_ptr<mtl::image_view>> temp_view_cache;
		rsx::overlays::primitive_type m_current_primitive_type = rsx::overlays::primitive_type::quad_list;
		std::vector<vertex> m_vertex_scratch; // Primitive expansion scratch

		// std140 parameter block sizes (vertex: Configuration, fragment: FragmentConfiguration)
		static constexpr u32 vertex_push_constants_size = 80;
		static constexpr u32 fragment_push_constants_size = 64;

		ui_overlay_renderer();

		void upload_simple_texture(mtl::image* tex, mtl::command_context& cmd,
			mtl::ring_buffer& upload_heap, u32 w, u32 h, u32 layers, bool font, const void* pixel_src);

		mtl::image_view* upload_simple_texture(mtl::command_context& cmd,
			mtl::ring_buffer& upload_heap, u64 key, u32 w, u32 h, u32 layers, bool font, bool temp, const void* pixel_src, u32 owner_uid);

		void init(mtl::command_context& cmd, mtl::ring_buffer& upload_heap);

		void destroy() override;

		void remove_temp_resources(u32 key);

		mtl::image_view* find_font(const rsx::overlays::font* font, mtl::command_context& cmd, mtl::ring_buffer& upload_heap);
		mtl::image_view* find_temp_image(const rsx::overlays::image_info_base* desc, mtl::command_context& cmd, mtl::ring_buffer& upload_heap, u32 owner_uid);

		void update_uniforms(mtl::command_context& cmd, mtl::program* program) override;

		void set_primitive_type(rsx::overlays::primitive_type type);

		// Uploads the vertices of a draw command, expanding quad lists and triangle fans into triangle lists
		void upload_command_vertices(mtl::command_context& cmd, const std::vector<vertex>& verts);

		// All draw commands of the overlay are recorded into a single render pass on target.
		// Texture uploads are recorded before the pass is opened.
		void run(mtl::command_context& cmd, const areau& viewport, const overlay_render_target& target,
			mtl::ring_buffer& upload_heap, rsx::overlays::overlay& ui, MTLLoadAction load = MTLLoadActionLoad);
		void run(mtl::command_context& cmd, const areau& viewport, mtl::image* target,
			mtl::ring_buffer& upload_heap, rsx::overlays::overlay& ui, MTLLoadAction load = MTLLoadActionLoad);
	};

	/**
	 * In-pass clear of the attachments of the currently open render pass (replaces vkCmdClearAttachments and
	 * vk::attachment_clear_pass). Draws a quad covering rect with the requested write masks.
	 */
	struct attachment_clear_pass : public overlay_pass
	{
		struct clear_info
		{
			u8 color_write_mask[4] = {};   // MTLColorWriteMask per color attachment (0 = attachment untouched)
			color4f color = { 0.f, 0.f, 0.f, 0.f };

			bool clear_depth = false;
			f32 depth = 1.f;

			bool clear_stencil = false;
			u8 stencil = 0;
			u8 stencil_write_mask = 0xFF;
		};

		color4f clear_color = { 0.f, 0.f, 0.f, 0.f };
		color4f colormask = { 1.f, 1.f, 1.f, 1.f };
		f32 clear_depth = 1.f;
		MTLScissorRect region = {};

		mtl::shader_module m_fragment_variants[5]; // Indexed by the number of color outputs

		static constexpr u32 vertex_push_constants_size = 32;
		static_assert(vertex_push_constants_size == (sizeof(clear_color) + sizeof(colormask)));

		attachment_clear_pass();

		id<MTLFunction> get_fragment_function() override;

		void update_uniforms(mtl::command_context& cmd, mtl::program* program) override;

		void set_up_viewport(mtl::command_context& cmd, u32 x, u32 y, u32 w, u32 h) override;

		void destroy() override;

		// Mirrors vk::attachment_clear_pass::run: clears all color attachments of the open pass inside rect.
		// clearmask uses the RSX clear bits (0x10 = R, 0x20 = G, 0x40 = B, 0x80 = A).
		void run(mtl::command_context& cmd, const mtl::render_pass_formats& formats, const MTLScissorRect& rect, u32 clearmask, color4f color);

		// Generic in-pass clear of color/depth/stencil (vkCmdClearAttachments with a single rect)
		void run(mtl::command_context& cmd, const mtl::render_pass_formats& formats, const MTLScissorRect& rect, const clear_info& info);
	};

	// In-pass partial stencil clear (honours the stencil write mask)
	struct stencil_clear_pass : public overlay_pass
	{
		MTLScissorRect region = {};

		stencil_clear_pass();

		void set_up_viewport(mtl::command_context& cmd, u32 x, u32 y, u32 w, u32 h) override;

		void run(mtl::command_context& cmd, const mtl::render_pass_formats& formats, const MTLScissorRect& rect, u32 stencil_clear, u32 stencil_write_mask);
	};

	struct video_out_calibration_pass : public overlay_pass
	{
		union config_t
		{
			struct
			{
				float gamma;
				int   limit_range;
				int   stereo_display_mode;
				int   stereo_image_count;
				color4_base<float> left_anaglyph_matrix[3];
				color4_base<float> right_anaglyph_matrix[3];
			};

			float data[(
				sizeof(gamma) +
				sizeof(limit_range) +
				sizeof(stereo_display_mode) +
				sizeof(stereo_image_count) +
				sizeof(left_anaglyph_matrix) +
				sizeof(right_anaglyph_matrix)
				) / sizeof(float)];
		}
		config = {};

		static constexpr u32 fragment_push_constants_size = 112;

		video_out_calibration_pass();

		void update_uniforms(mtl::command_context& cmd, mtl::program* /*program*/) override;

		void run(mtl::command_context& cmd, const areau& viewport, const overlay_render_target& target,
			const rsx::simple_array<mtl::viewable_image*>& src, f32 gamma, bool limited_rgb,
			bool stereo_enabled, MTLLoadAction load = MTLLoadActionLoad);
		void run(mtl::command_context& cmd, const areau& viewport, mtl::image* target,
			const rsx::simple_array<mtl::viewable_image*>& src, f32 gamma, bool limited_rgb,
			bool stereo_enabled, MTLLoadAction load = MTLLoadActionLoad);
	};

	/**
	 * Scaled image copy (Metal has no vkCmdBlitImage). Renders a textured quad into dst.
	 * - color: color -> color with nearest or linear filtering (float/unorm formats)
	 * - depth: writes gl_FragDepth from the source (depth formats, or the red channel of a color source)
	 * - stencil: stencil export (gl_FragStencilRefARB -> [[stencil]]) from a stencil view (mtl::get_stencil_view_format)
	 *   of a depth-stencil source
	 * - aspect == 0 copies every aspect of dst. Unsupported aspect combinations are logged and skipped.
	 * dst needs MTLTextureUsageRenderTarget; src needs MTLTextureUsageShaderRead (+ PixelFormatView for stencil).
	 * Depth and stencil are always copied with nearest filtering.
	 * Rects may be flipped (x1 > x2 / y1 > y2, or negative width/height for coord3i). Copies between overlapping regions of
	 * the same subresource go through a scratch image. Source must be single-sampled and not 3D.
	 */
	struct blit_pass : public overlay_pass
	{
		enum class blit_mode : u32
		{
			color = 0,
			depth = 1,
			stencil = 2,
			depth_stencil = 3
		};

		blit_mode m_mode = blit_mode::color;
		f32 m_src_rect[4] = { 0.f, 0.f, 1.f, 1.f }; // Normalized source rectangle (u0, v0, u1, v1)
		mtl::shader_module m_fragment_variants[4];   // Indexed by blit_mode
		std::unique_ptr<mtl::image> m_scratch_image;

		static constexpr u32 vertex_push_constants_size = 16;

		blit_pass();

		id<MTLFunction> get_fragment_function() override;
		u64 get_pipeline_variant() const override { return static_cast<u64>(m_mode); }

		void update_uniforms(mtl::command_context& cmd, mtl::program* program) override;

		void destroy() override;

		mtl::image* get_scratch_image(const mtl::image* reference, u32 width, u32 height);

		// aspect: mtl::image_aspect bits to copy (color, or depth and/or stencil)
		void run(mtl::command_context& cmd, mtl::image* src, mtl::image* dst,
			const areai& src_area, const areai& dst_area,
			u32 src_level, u32 src_layer, u32 dst_level, u32 dst_layer,
			bool linear, u32 aspect);

		// Same with VK-style rects (negative width/height = flipped). z/depth must describe a single slice.
		void run(mtl::command_context& cmd, mtl::image* src, mtl::image* dst,
			const coord3i& src_rect, const coord3i& dst_rect,
			u32 src_level, u32 src_layer, u32 dst_level, u32 dst_layer,
			bool linear, u32 aspect);
	};

	// Overlay pass registry (same scheme as vk::g_overlay_passes)
	extern std::unordered_map<u32, std::unique_ptr<mtl::overlay_pass>> g_overlay_passes;

	template<class T>
	T* get_overlay_pass()
	{
		const u32 index = stx::typeindex<id_manager::typeinfo, T>();
		auto& e = g_overlay_passes[index];

		if (!e)
		{
			e = std::make_unique<T>();
			e->create(*mtl::get_current_device());
		}

		return static_cast<T*>(e.get());
	}

	// Per-frame cleanup (vk::reset_overlay_passes): releases transient resources, keeps pipelines and cached textures
	void reset_overlay_passes();

	// Destroys every overlay pass (the g_overlay_passes part of vk::destroy_global_resources)
	void destroy_overlay_passes();
}
