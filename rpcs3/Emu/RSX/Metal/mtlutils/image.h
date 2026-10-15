#pragma once

#include "mtl_common.h"
#include "Emu/RSX/Common/TextureUtils.h"
#include "Emu/RSX/Utils/color_utils.hpp"

#include <array>
#include <memory>
#include <unordered_map>

//using enum rsx::format_class;
using namespace ::rsx::format_class_;

namespace mtl
{
	class image_view;

	constexpr MTLTextureSwizzleChannels default_component_map = { MTLTextureSwizzleRed, MTLTextureSwizzleGreen, MTLTextureSwizzleBlue, MTLTextureSwizzleAlpha };

	inline bool operator == (const MTLTextureSwizzleChannels& a, const MTLTextureSwizzleChannels& b)
	{
		return a.red == b.red && a.green == b.green && a.blue == b.blue && a.alpha == b.alpha;
	}

	inline bool operator != (const MTLTextureSwizzleChannels& a, const MTLTextureSwizzleChannels& b)
	{
		return !(a == b);
	}

	struct image_info
	{
		MTLTextureType type = MTLTextureType2D;
		MTLPixelFormat format = MTLPixelFormatInvalid;
		u32 width = 1;
		u32 height = 1;
		u32 depth = 1;     // Depth of 3D textures only
		u32 mipmaps = 1;
		u32 layers = 1;    // Array layers. Cubemaps report 6 layers.
		u8 samples = 1;
		MTLTextureUsage usage = MTLTextureUsageShaderRead;
		MTLStorageMode storage = MTLStorageModePrivate;
	};

	/**
	 * Wrapper around an MTLTexture. Mirrors vk::image so that the shared texture cache / surface store
	 * logic can be ported with minimal changes. Metal tracks hazards automatically, so there are no layouts.
	 */
	class image
	{
		u32 m_aspect = 0;
		rsx::format_class m_format_class = RSX_FORMAT_CLASS_UNDEFINED;
		std::string m_debug_name;

	protected:
		image() = default;
		void create_impl();

	public:
		id<MTLTexture> value = nil;
		image_info info{};

		// The swizzle applied to all views to make the native host format behave like the RSX format
		MTLTextureSwizzleChannels native_component_map = default_component_map;

		image(
			MTLTextureType type,
			MTLPixelFormat format,
			u32 width, u32 height, u32 depth,
			u32 mipmaps, u32 layers,
			u8 samples,
			MTLTextureUsage usage,
			rsx::format_class format_class = RSX_FORMAT_CLASS_UNDEFINED,
			MTLStorageMode storage = MTLStorageModePrivate);

		// Wrap an existing texture (e.g a drawable). The texture is retained.
		explicit image(id<MTLTexture> texture, rsx::format_class format_class = RSX_FORMAT_CLASS_COLOR);

		virtual ~image();

		image(const image&) = delete;
		image(image&&) = delete;

		// Identifiers
		id<MTLTexture> handle() const { return value; }

		// Properties
		u32 width() const { return info.width; }
		u32 height() const { return info.height; }
		u32 depth() const { return info.depth; }
		u32 mipmaps() const { return info.mipmaps; }
		u32 layers() const { return info.layers; }
		u8 samples() const { return info.samples; }
		MTLPixelFormat format() const { return info.format; }
		MTLTextureType type() const { return info.type; }
		u32 aspect() const { return m_aspect; }
		rsx::format_class format_class() const { return m_format_class; }
		void set_format_class(rsx::format_class fc) { m_format_class = fc; }

		// Approximate memory footprint in bytes
		u64 memory_size() const;

		// Debug utils
		void set_debug_name(const std::string& name);
		const std::string& debug_name() const { return m_debug_name; }
	};

	/**
	 * Texture view. Metal views are regular MTLTexture objects created with newTextureViewWithPixelFormat:...
	 */
	class image_view
	{
		std::unordered_map<u32, std::unique_ptr<image_view>> m_subviews;
		image* m_resource = nullptr;
		image_view* m_root_view = nullptr;

		void create_impl();

	public:
		id<MTLTexture> value = nil;

		MTLPixelFormat view_format = MTLPixelFormatInvalid;
		MTLTextureType view_type = MTLTextureType2D;
		MTLTextureSwizzleChannels swizzle = default_component_map;
		u32 base_level = 0;
		u32 level_count = 1;
		u32 base_layer = 0;
		u32 layer_count = 1;
		u32 aspect = image_aspect::color;

		// format = Invalid uses the image's format, view_type = 0xFF uses the image's type, level/layer count of 0 means "all"
		image_view(image* resource,
			MTLPixelFormat format = MTLPixelFormatInvalid,
			MTLTextureType view_type = static_cast<MTLTextureType>(0xFF),
			const MTLTextureSwizzleChannels& mapping = default_component_map,
			u32 aspect_mask = 0,
			u32 base_level = 0, u32 level_count = 0,
			u32 base_layer = 0, u32 layer_count = 0);

		~image_view();

		image_view(const image_view&) = delete;
		image_view(image_view&&) = delete;

		// Reinterpret as another (compatible) format. The returned view is owned by this view.
		image_view* as(MTLPixelFormat new_format);

		id<MTLTexture> handle() const { return value; }
		MTLPixelFormat format() const { return view_format; }
		mtl::image* image() const { return m_resource; }

		// Swizzles are native in Metal
		u32 encoded_component_map() const { return 0; }
	};

	class viewable_image : public image
	{
	protected:
		std::unordered_map<u64, std::unique_ptr<image_view>> views;

	public:
		using image::image;

		virtual image_view* get_view(
			const rsx::texture_channel_remap_t& remap,
			u32 aspect_mask = image_aspect::color | image_aspect::depth);

		// Identity view with the native component mapping applied
		image_view* get_view(u32 aspect_mask = image_aspect::color | image_aspect::depth);

		void set_native_component_layout(const MTLTextureSwizzleChannels& new_layout);
	};

	// Helpers
	MTLTextureSwizzleChannels apply_swizzle_remap(const std::array<MTLTextureSwizzle, 4>& base_remap_argb, const rsx::texture_channel_remap_t& remap_vector);
	u32 get_aspect_flags(MTLPixelFormat format);
	bool is_depth_format(MTLPixelFormat format);
	bool is_stencil_format(MTLPixelFormat format);
	bool is_compressed_format(MTLPixelFormat format);
	MTLPixelFormat get_stencil_view_format(MTLPixelFormat format);

	// Size of one texel (or one 4x4 block for BCn formats) in bytes.
	// Combined depth/stencil formats return the size of the depth plane (what a MTLBlitOptionDepthFromDepthStencil copy
	// transfers per texel); use get_aspect_texel_size() for per-aspect buffer layouts.
	u32 get_format_texel_size(MTLPixelFormat format);
	// Size in bytes of one texel of a single aspect as laid out in buffers by blit copies
	// (Depth32Float_Stencil8: depth = 4, stencil = 1; color formats: get_format_texel_size()).
	u32 get_aspect_texel_size(MTLPixelFormat format, u32 aspect);
	// Width/height of a block in texels (4 for BCn, 1 otherwise)
	u32 get_format_block_dim(MTLPixelFormat format);
}
