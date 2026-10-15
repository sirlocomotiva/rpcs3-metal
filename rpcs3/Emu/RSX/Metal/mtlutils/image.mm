#include "stdafx.h"
#include "image.h"
#include "device.h"

namespace mtl
{
	namespace
	{
		constexpr u32 c_max_texture_dimension_1d_2d = 16384;
		constexpr u32 c_max_texture_dimension_cube = 16384;
		constexpr u32 c_max_texture_dimension_3d = 2048;
		constexpr u32 c_max_texture_array_length = 2048;

		bool is_cube_type(MTLTextureType type)
		{
			return type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray;
		}

		bool is_array_type(MTLTextureType type)
		{
			return type == MTLTextureType1DArray || type == MTLTextureType2DArray || type == MTLTextureType2DMultisampleArray;
		}

		const char* describe_texture_type(MTLTextureType type)
		{
			switch (type)
			{
			case MTLTextureType1D: return "1D";
			case MTLTextureType1DArray: return "1DArray";
			case MTLTextureType2D: return "2D";
			case MTLTextureType2DArray: return "2DArray";
			case MTLTextureType2DMultisample: return "2DMultisample";
			case MTLTextureType2DMultisampleArray: return "2DMultisampleArray";
			case MTLTextureTypeCube: return "Cube";
			case MTLTextureTypeCubeArray: return "CubeArray";
			case MTLTextureType3D: return "3D";
			case MTLTextureTypeTextureBuffer: return "TextureBuffer";
			}

			return "Unknown";
		}

		// Promote the requested texture type to the Metal type that can actually hold the requested layers/samples.
		// VK expresses arrays/cubes/MSAA through layer counts and create flags; Metal encodes them in the texture type.
		MTLTextureType get_effective_texture_type(MTLTextureType type, u32 layers, u8 samples)
		{
			switch (type)
			{
			case MTLTextureType1D:
				return (layers > 1) ? MTLTextureType1DArray : MTLTextureType1D;
			case MTLTextureType1DArray:
				return MTLTextureType1DArray;
			case MTLTextureType2D:
				if (samples > 1)
				{
					return (layers > 1) ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
				}
				return (layers > 1) ? MTLTextureType2DArray : MTLTextureType2D;
			case MTLTextureType2DArray:
				return (samples > 1) ? MTLTextureType2DMultisampleArray : MTLTextureType2DArray;
			case MTLTextureType2DMultisample:
				return (layers > 1) ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
			case MTLTextureType2DMultisampleArray:
				return MTLTextureType2DMultisampleArray;
			case MTLTextureTypeCube:
				return (layers > 6) ? MTLTextureTypeCubeArray : MTLTextureTypeCube;
			case MTLTextureTypeCubeArray:
				return MTLTextureTypeCubeArray;
			case MTLTextureType3D:
				return MTLTextureType3D;
			case MTLTextureTypeTextureBuffer:
				break;
			}

			fmt::throw_exception("Unsupported image type %s (0x%x)", describe_texture_type(type), static_cast<u32>(type));
		}
	}

	image::image(
		MTLTextureType type,
		MTLPixelFormat format,
		u32 width, u32 height, u32 depth,
		u32 mipmaps, u32 layers,
		u8 samples,
		MTLTextureUsage usage,
		rsx::format_class format_class,
		MTLStorageMode storage)
	{
		info.type = type;
		info.format = format;
		info.width = width;
		info.height = height;
		info.depth = depth;
		info.mipmaps = mipmaps;
		info.layers = layers;
		info.samples = samples;
		info.usage = usage;
		info.storage = storage;

		create_impl();

		if (format_class == RSX_FORMAT_CLASS_UNDEFINED)
		{
			if (m_aspect != image_aspect::color)
			{
				rsx_log.error("Depth/stencil textures must have format class explicitly declared");
			}
			else
			{
				format_class = RSX_FORMAT_CLASS_COLOR;
			}
		}

		m_format_class = format_class;
	}

	image::image(id<MTLTexture> texture, rsx::format_class format_class)
	{
		ensure(texture);

		value = texture;

		info.type = texture.textureType;
		info.format = texture.pixelFormat;
		info.width = static_cast<u32>(texture.width);
		info.height = static_cast<u32>(texture.height);
		info.depth = static_cast<u32>(texture.depth);
		info.mipmaps = static_cast<u32>(texture.mipmapLevelCount);
		info.layers = static_cast<u32>(texture.arrayLength) * (is_cube_type(texture.textureType) ? 6u : 1u);
		info.samples = static_cast<u8>(texture.sampleCount);
		info.usage = texture.usage;
		info.storage = texture.storageMode;

		m_aspect = get_aspect_flags(info.format);
		m_format_class = format_class;
	}

	image::~image()
	{
		value = nil;
	}

	void image::create_impl()
	{
		ensure(info.format != MTLPixelFormatInvalid, "Invalid image format");

		info.width = std::max(info.width, 1u);
		info.height = std::max(info.height, 1u);
		info.depth = std::max(info.depth, 1u);
		info.mipmaps = std::max(info.mipmaps, 1u);
		info.layers = std::max(info.layers, 1u);
		info.samples = std::max<u8>(info.samples, 1);

		if (is_cube_type(info.type) && info.layers < 6)
		{
			// Cubemaps always have 6 faces (VK passes layers = 6 together with the cube-compatible flag)
			info.layers = 6;
		}

		info.type = get_effective_texture_type(info.type, info.layers, info.samples);

		u32 array_length = 1;
		u32 longest_dim = 0;
		u32 dim_limit = c_max_texture_dimension_1d_2d;

		switch (info.type)
		{
		case MTLTextureType1D:
		case MTLTextureType1DArray:
			ensure(info.height == 1 && info.depth == 1 && info.samples == 1, "Invalid 1D image configuration");
			longest_dim = info.width;
			array_length = info.layers;
			break;
		case MTLTextureType2D:
		case MTLTextureType2DArray:
		case MTLTextureType2DMultisample:
		case MTLTextureType2DMultisampleArray:
			ensure(info.depth == 1, "2D images cannot have depth");
			longest_dim = std::max(info.width, info.height);
			array_length = is_array_type(info.type) ? info.layers : 1;
			break;
		case MTLTextureTypeCube:
		case MTLTextureTypeCubeArray:
			ensure(info.depth == 1 && info.samples == 1 && (info.layers % 6) == 0, "Invalid cubemap configuration");
			ensure(info.width == info.height, "Cubemap faces must be square");
			longest_dim = info.width;
			dim_limit = c_max_texture_dimension_cube;
			array_length = info.layers / 6;
			break;
		case MTLTextureType3D:
			ensure(info.layers == 1 && info.samples == 1, "Invalid 3D image configuration");
			longest_dim = std::max({ info.width, info.height, info.depth });
			dim_limit = c_max_texture_dimension_3d;
			break;
		default:
			fmt::throw_exception("Unreachable");
		}

		if (longest_dim > dim_limit || array_length > c_max_texture_array_length)
		{
			// Longest dimension exceeds the limit. Can happen when using MSAA + very high resolution scaling
			// Just kill the application at this point.
			fmt::throw_exception(
				"The renderer requested an image larger than the limit allowed for by your GPU hardware. "
				"Turn down your resolution scale and/or disable MSAA to fit within the image budget.");
		}

		if (info.samples > 1)
		{
			// Multisampled textures cannot be mipmapped
			info.mipmaps = 1;
		}

		m_aspect = get_aspect_flags(info.format);

		if (info.storage != MTLStorageModePrivate && info.storage != MTLStorageModeMemoryless &&
			(m_aspect != image_aspect::color || info.samples > 1))
		{
			// Depth/stencil and multisampled textures must live in private memory
			rsx_log.warning("Metal: Storage mode %u is not supported for %s textures, using private storage",
				static_cast<u32>(info.storage), info.samples > 1 ? "multisampled" : "depth/stencil");
			info.storage = MTLStorageModePrivate;
		}

		// Views with a different pixel format or swizzle are created freely by the texture cache and render targets
		info.usage |= MTLTextureUsagePixelFormatView;

		MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
		desc.textureType = info.type;
		desc.pixelFormat = info.format;
		desc.width = info.width;
		desc.height = info.height;
		desc.depth = (info.type == MTLTextureType3D) ? info.depth : 1;
		desc.mipmapLevelCount = info.mipmaps;
		desc.arrayLength = array_length;
		desc.sampleCount = info.samples;
		desc.usage = info.usage;
		desc.storageMode = info.storage;
		desc.hazardTrackingMode = MTLHazardTrackingModeTracked;

		value = [get_device() newTextureWithDescriptor:desc];

		if (!value)
		{
			fmt::throw_exception("Metal: Failed to create %s texture (format=0x%x, %ux%ux%u, mips=%u, layers=%u, samples=%u, usage=0x%x)",
				describe_texture_type(info.type), static_cast<u32>(info.format), info.width, info.height, info.depth,
				info.mipmaps, info.layers, info.samples, static_cast<u64>(info.usage));
		}

		if (!m_debug_name.empty())
		{
			value.label = to_ns_string(m_debug_name);
		}
	}

	u64 image::memory_size() const
	{
		if (!value)
		{
			return 0;
		}

		if (const u64 allocated = static_cast<u64>(value.allocatedSize))
		{
			return allocated;
		}

		// Wrapped textures (e.g drawables) may not report an allocation size, estimate it
		const u32 block_dim = get_format_block_dim(info.format);
		u32 texel_size = get_format_texel_size(info.format);
		if (info.format == MTLPixelFormatDepth32Float_Stencil8)
		{
			// Separate stencil plane
			texel_size += 1;
		}

		u64 total = 0;
		for (u32 level = 0; level < info.mipmaps; ++level)
		{
			const u32 w = std::max(info.width >> level, 1u);
			const u32 h = std::max(info.height >> level, 1u);
			const u32 d = (info.type == MTLTextureType3D) ? std::max(info.depth >> level, 1u) : 1u;
			const u64 blocks_x = (w + block_dim - 1) / block_dim;
			const u64 blocks_y = (h + block_dim - 1) / block_dim;
			total += blocks_x * blocks_y * d * texel_size;
		}

		return total * info.layers * info.samples;
	}

	void image::set_debug_name(const std::string& name)
	{
		m_debug_name = name;

		if (value)
		{
			value.label = to_ns_string(name);
		}
	}

	// Image views

	image_view::image_view(mtl::image* resource,
		MTLPixelFormat format,
		MTLTextureType type,
		const MTLTextureSwizzleChannels& mapping,
		u32 aspect_mask,
		u32 base_level, u32 level_count,
		u32 base_layer, u32 layer_count)
		: m_resource(resource)
	{
		ensure(resource && resource->value, "Cannot create a view of an empty image");

		const u32 resource_aspect = resource->aspect();
		aspect = aspect_mask ? (resource_aspect & aspect_mask) : resource_aspect;
		ensure(aspect, "Image view aspect does not match the image aspect");

		view_format = (format == MTLPixelFormatInvalid) ? resource->format() : format;
		if (format == MTLPixelFormatInvalid && aspect == image_aspect::stencil && is_depth_format(view_format))
		{
			// Stencil-only view of a combined depth/stencil texture
			view_format = get_stencil_view_format(view_format);
		}

		view_type = (static_cast<u32>(type) == 0xFF) ? resource->type() : type;

		this->base_level = base_level;
		this->level_count = level_count ? level_count : (resource->mipmaps() - base_level);
		this->base_layer = base_layer;
		this->layer_count = layer_count ? layer_count : (resource->layers() - base_layer);

		ensure(this->base_level + this->level_count <= resource->mipmaps(), "Image view level range out of bounds");
		ensure(this->base_layer + this->layer_count <= resource->layers(), "Image view layer range out of bounds");

		// Depth and stencil formats do not support swizzling. The decompilers handle depth reads explicitly.
		swizzle = (is_depth_format(view_format) || is_stencil_format(view_format)) ? default_component_map : mapping;

		create_impl();
	}

	image_view::~image_view()
	{
		m_subviews.clear();
		value = nil;
	}

	void image_view::create_impl()
	{
		mtl::image* resource = m_resource;

		const bool is_identical =
			view_format == resource->format() &&
			view_type == resource->type() &&
			base_level == 0 && level_count == resource->mipmaps() &&
			base_layer == 0 && layer_count == resource->layers() &&
			swizzle == default_component_map;

		if (is_identical)
		{
			// The view would be identical to the texture itself
			value = resource->value;
			return;
		}

		value = [resource->value newTextureViewWithPixelFormat:view_format
			textureType:view_type
			levels:NSMakeRange(base_level, level_count)
			slices:NSMakeRange(base_layer, layer_count)
			swizzle:swizzle];

		if (!value)
		{
			fmt::throw_exception("Metal: Failed to create %s view (format=0x%x) of %s texture (format=0x%x), levels=[%u, %u), layers=[%u, %u)",
				describe_texture_type(view_type), static_cast<u32>(view_format), describe_texture_type(resource->type()),
				static_cast<u32>(resource->format()), base_level, base_level + level_count, base_layer, base_layer + layer_count);
		}

		if (const auto& name = resource->debug_name(); !name.empty())
		{
			value.label = to_ns_string(name);
		}
	}

	image_view* image_view::as(MTLPixelFormat new_format)
	{
		if (this->format() == new_format)
		{
			return this;
		}

		auto self = this->m_root_view
			? this->m_root_view
			: this;

		if (auto found = self->m_subviews.find(static_cast<u32>(new_format));
			found != self->m_subviews.end())
		{
			return found->second.get();
		}

		// Create a derived
		auto view = std::make_unique<image_view>(self->m_resource, new_format, self->view_type, self->swizzle, self->aspect,
			self->base_level, self->level_count, self->base_layer, self->layer_count);
		view->m_root_view = self;

		auto ret = view.get();
		self->m_subviews.emplace(static_cast<u32>(new_format), std::move(view));
		return ret;
	}

	// Viewable image

	image_view* viewable_image::get_view(const rsx::texture_channel_remap_t& remap, u32 mask)
	{
		u32 remap_encoding = remap.encoded;
		if (remap_encoding == MTL_REMAP_IDENTITY)
		{
			if (native_component_map == default_component_map)
			{
				remap_encoding = RSX_TEXTURE_REMAP_IDENTITY;
			}
		}

		const u64 storage_key = remap_encoding | (static_cast<u64>(mask) << 32);
		auto found = views.find(storage_key);
		if (found != views.end())
		{
			ensure(found->second->aspect & mask);
			return found->second.get();
		}

		MTLTextureSwizzleChannels real_mapping;
		switch (remap_encoding)
		{
		case MTL_REMAP_IDENTITY:
			real_mapping = default_component_map;
			break;
		case MTL_REMAP_VIEW_MULTISAMPLED:
		case RSX_TEXTURE_REMAP_IDENTITY:
			real_mapping = native_component_map;
			break;
		default:
			real_mapping = mtl::apply_swizzle_remap
			(
				{ native_component_map.alpha, native_component_map.red, native_component_map.green, native_component_map.blue },
				remap
			);
			break;
		}

		const u32 view_aspect = aspect() & mask;
		ensure(view_aspect);

		auto view = std::make_unique<image_view>(this, MTLPixelFormatInvalid, static_cast<MTLTextureType>(0xFF), real_mapping, view_aspect,
			0, info.mipmaps, 0, info.layers);
		auto result = view.get();
		views.emplace(storage_key, std::move(view));
		return result;
	}

	image_view* viewable_image::get_view(u32 aspect_mask)
	{
		return get_view(rsx::default_remap_vector, aspect_mask);
	}

	void viewable_image::set_native_component_layout(const MTLTextureSwizzleChannels& new_layout)
	{
		if (new_layout != native_component_map)
		{
			native_component_map = new_layout;

			// Safely discard existing views. Command buffers retain any texture still referenced by the GPU.
			views.clear();
		}
	}

	// Helpers

	MTLTextureSwizzleChannels apply_swizzle_remap(const std::array<MTLTextureSwizzle, 4>& base_remap_argb, const rsx::texture_channel_remap_t& remap_vector)
	{
		const auto final_mapping = remap_vector.remap(base_remap_argb, MTLTextureSwizzleZero, MTLTextureSwizzleOne);
		return { final_mapping[1], final_mapping[2], final_mapping[3], final_mapping[0] };
	}

	u32 get_aspect_flags(MTLPixelFormat format)
	{
		switch (format)
		{
		default:
			return image_aspect::color;
		case MTLPixelFormatDepth16Unorm:
		case MTLPixelFormatDepth32Float:
			return image_aspect::depth;
		case MTLPixelFormatDepth24Unorm_Stencil8:
		case MTLPixelFormatDepth32Float_Stencil8:
			return image_aspect::depth | image_aspect::stencil;
		case MTLPixelFormatStencil8:
		case MTLPixelFormatX24_Stencil8:
		case MTLPixelFormatX32_Stencil8:
			return image_aspect::stencil;
		}
	}

	bool is_depth_format(MTLPixelFormat format)
	{
		switch (format)
		{
		case MTLPixelFormatDepth16Unorm:
		case MTLPixelFormatDepth32Float:
		case MTLPixelFormatDepth24Unorm_Stencil8:
		case MTLPixelFormatDepth32Float_Stencil8:
			return true;
		default:
			return false;
		}
	}

	bool is_stencil_format(MTLPixelFormat format)
	{
		switch (format)
		{
		case MTLPixelFormatStencil8:
		case MTLPixelFormatDepth24Unorm_Stencil8:
		case MTLPixelFormatDepth32Float_Stencil8:
		case MTLPixelFormatX24_Stencil8:
		case MTLPixelFormatX32_Stencil8:
			return true;
		default:
			return false;
		}
	}

	bool is_compressed_format(MTLPixelFormat format)
	{
		return get_format_block_dim(format) > 1;
	}

	MTLPixelFormat get_stencil_view_format(MTLPixelFormat format)
	{
		switch (format)
		{
		case MTLPixelFormatDepth32Float_Stencil8:
		case MTLPixelFormatX32_Stencil8:
			return MTLPixelFormatX32_Stencil8;
		case MTLPixelFormatDepth24Unorm_Stencil8:
		case MTLPixelFormatX24_Stencil8:
			return MTLPixelFormatX24_Stencil8;
		case MTLPixelFormatStencil8:
			return MTLPixelFormatStencil8;
		default:
			fmt::throw_exception("Format 0x%x has no stencil aspect", static_cast<u32>(format));
		}
	}

	u32 get_format_texel_size(MTLPixelFormat format)
	{
		switch (format)
		{
		// 8-bit
		case MTLPixelFormatA8Unorm:
		case MTLPixelFormatR8Unorm:
		case MTLPixelFormatR8Unorm_sRGB:
		case MTLPixelFormatR8Snorm:
		case MTLPixelFormatR8Uint:
		case MTLPixelFormatR8Sint:
		case MTLPixelFormatStencil8:
		case MTLPixelFormatX24_Stencil8:
		case MTLPixelFormatX32_Stencil8:
			return 1;

		// 16-bit
		case MTLPixelFormatR16Unorm:
		case MTLPixelFormatR16Snorm:
		case MTLPixelFormatR16Uint:
		case MTLPixelFormatR16Sint:
		case MTLPixelFormatR16Float:
		case MTLPixelFormatRG8Unorm:
		case MTLPixelFormatRG8Unorm_sRGB:
		case MTLPixelFormatRG8Snorm:
		case MTLPixelFormatRG8Uint:
		case MTLPixelFormatRG8Sint:
		case MTLPixelFormatB5G6R5Unorm:
		case MTLPixelFormatA1BGR5Unorm:
		case MTLPixelFormatABGR4Unorm:
		case MTLPixelFormatBGR5A1Unorm:
		case MTLPixelFormatDepth16Unorm:
			return 2;

		// 32-bit
		case MTLPixelFormatR32Uint:
		case MTLPixelFormatR32Sint:
		case MTLPixelFormatR32Float:
		case MTLPixelFormatRG16Unorm:
		case MTLPixelFormatRG16Snorm:
		case MTLPixelFormatRG16Uint:
		case MTLPixelFormatRG16Sint:
		case MTLPixelFormatRG16Float:
		case MTLPixelFormatRGBA8Unorm:
		case MTLPixelFormatRGBA8Unorm_sRGB:
		case MTLPixelFormatRGBA8Snorm:
		case MTLPixelFormatRGBA8Uint:
		case MTLPixelFormatRGBA8Sint:
		case MTLPixelFormatBGRA8Unorm:
		case MTLPixelFormatBGRA8Unorm_sRGB:
		case MTLPixelFormatRGB10A2Unorm:
		case MTLPixelFormatRGB10A2Uint:
		case MTLPixelFormatBGR10A2Unorm:
		case MTLPixelFormatRG11B10Float:
		case MTLPixelFormatRGB9E5Float:
		case MTLPixelFormatBGR10_XR:
		case MTLPixelFormatBGR10_XR_sRGB:
		case MTLPixelFormatDepth32Float:
		case MTLPixelFormatDepth24Unorm_Stencil8:
		case MTLPixelFormatDepth32Float_Stencil8: // Depth plane only, the stencil plane is 1 byte per texel
			return 4;

		// 64-bit
		case MTLPixelFormatRG32Uint:
		case MTLPixelFormatRG32Sint:
		case MTLPixelFormatRG32Float:
		case MTLPixelFormatRGBA16Unorm:
		case MTLPixelFormatRGBA16Snorm:
		case MTLPixelFormatRGBA16Uint:
		case MTLPixelFormatRGBA16Sint:
		case MTLPixelFormatRGBA16Float:
		case MTLPixelFormatBGRA10_XR:
		case MTLPixelFormatBGRA10_XR_sRGB:
			return 8;

		// 128-bit
		case MTLPixelFormatRGBA32Uint:
		case MTLPixelFormatRGBA32Sint:
		case MTLPixelFormatRGBA32Float:
			return 16;

		// Block compressed (bytes per 4x4 block)
		case MTLPixelFormatBC1_RGBA:
		case MTLPixelFormatBC1_RGBA_sRGB:
		case MTLPixelFormatBC4_RUnorm:
		case MTLPixelFormatBC4_RSnorm:
		case MTLPixelFormatETC2_RGB8:
		case MTLPixelFormatETC2_RGB8_sRGB:
		case MTLPixelFormatETC2_RGB8A1:
		case MTLPixelFormatETC2_RGB8A1_sRGB:
		case MTLPixelFormatEAC_R11Unorm:
		case MTLPixelFormatEAC_R11Snorm:
			return 8;
		case MTLPixelFormatBC2_RGBA:
		case MTLPixelFormatBC2_RGBA_sRGB:
		case MTLPixelFormatBC3_RGBA:
		case MTLPixelFormatBC3_RGBA_sRGB:
		case MTLPixelFormatBC5_RGUnorm:
		case MTLPixelFormatBC5_RGSnorm:
		case MTLPixelFormatBC6H_RGBFloat:
		case MTLPixelFormatBC6H_RGBUfloat:
		case MTLPixelFormatBC7_RGBAUnorm:
		case MTLPixelFormatBC7_RGBAUnorm_sRGB:
		case MTLPixelFormatEAC_RGBA8:
		case MTLPixelFormatEAC_RGBA8_sRGB:
		case MTLPixelFormatEAC_RG11Unorm:
		case MTLPixelFormatEAC_RG11Snorm:
		case MTLPixelFormatASTC_4x4_LDR:
		case MTLPixelFormatASTC_4x4_sRGB:
		case MTLPixelFormatASTC_4x4_HDR:
			return 16;

		default:
			break;
		}

		fmt::throw_exception("Unexpected MTLPixelFormat 0x%x", static_cast<u32>(format));
	}

	u32 get_aspect_texel_size(MTLPixelFormat format, u32 aspect)
	{
		if (aspect == image_aspect::stencil)
		{
			ensure(is_stencil_format(format), "Format has no stencil aspect");
			return 1;
		}

		// Depth (or color) plane
		return get_format_texel_size(format);
	}

	u32 get_format_block_dim(MTLPixelFormat format)
	{
		switch (format)
		{
		case MTLPixelFormatBC1_RGBA:
		case MTLPixelFormatBC1_RGBA_sRGB:
		case MTLPixelFormatBC2_RGBA:
		case MTLPixelFormatBC2_RGBA_sRGB:
		case MTLPixelFormatBC3_RGBA:
		case MTLPixelFormatBC3_RGBA_sRGB:
		case MTLPixelFormatBC4_RUnorm:
		case MTLPixelFormatBC4_RSnorm:
		case MTLPixelFormatBC5_RGUnorm:
		case MTLPixelFormatBC5_RGSnorm:
		case MTLPixelFormatBC6H_RGBFloat:
		case MTLPixelFormatBC6H_RGBUfloat:
		case MTLPixelFormatBC7_RGBAUnorm:
		case MTLPixelFormatBC7_RGBAUnorm_sRGB:
		case MTLPixelFormatETC2_RGB8:
		case MTLPixelFormatETC2_RGB8_sRGB:
		case MTLPixelFormatETC2_RGB8A1:
		case MTLPixelFormatETC2_RGB8A1_sRGB:
		case MTLPixelFormatEAC_R11Unorm:
		case MTLPixelFormatEAC_R11Snorm:
		case MTLPixelFormatEAC_RG11Unorm:
		case MTLPixelFormatEAC_RG11Snorm:
		case MTLPixelFormatEAC_RGBA8:
		case MTLPixelFormatEAC_RGBA8_sRGB:
		case MTLPixelFormatASTC_4x4_LDR:
		case MTLPixelFormatASTC_4x4_sRGB:
		case MTLPixelFormatASTC_4x4_HDR:
			return 4;
		default:
			return 1;
		}
	}
}
