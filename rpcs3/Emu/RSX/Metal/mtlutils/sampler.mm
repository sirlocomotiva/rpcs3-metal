#include "stdafx.h"
#include "sampler.h"
#include "device.h"

#include "Emu/RSX/Utils/color_utils.hpp"
#include "util/atomic.hpp"

#include <bit>

namespace mtl
{
	namespace
	{
		sampler_cache g_sampler_cache;

		atomic_t<bool> g_inexact_border_color_reported = false;
		atomic_t<bool> g_lod_bias_unsupported_reported = false;

		inline u64 hash_combine(u64 seed, u64 value)
		{
			return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
		}

		bool is_valid_unnormalized_address_mode(MTLSamplerAddressMode mode)
		{
			return mode == MTLSamplerAddressModeClampToEdge ||
				mode == MTLSamplerAddressModeClampToZero ||
				mode == MTLSamplerAddressModeClampToBorderColor;
		}

		// Metal has strict rules for samplers using pixel coordinates (VK's unnormalizedCoordinates has the same ones,
		// but Metal enforces them through the validation layer instead of silently ignoring the extra state).
		sampler_desc sanitize(const sampler_desc& desc)
		{
			sampler_desc result = desc;

			if (!result.normalized_coords)
			{
				result.mag_filter = result.min_filter;
				result.mip_filter = MTLSamplerMipFilterNotMipmapped;
				result.max_anisotropy = 1;
				result.compare_enabled = false;
				result.min_lod = 0.f;
				result.max_lod = 0.f;
				result.lod_bias = 0.f;

				for (auto* mode : { &result.address_s, &result.address_t, &result.address_r })
				{
					if (!is_valid_unnormalized_address_mode(*mode))
					{
						*mode = MTLSamplerAddressModeClampToEdge;
					}
				}
			}

			result.max_anisotropy = std::clamp<u8>(result.max_anisotropy, 1, static_cast<u8>(std::min<u32>(get_caps().max_anisotropy, 16)));

			if (result.max_lod < result.min_lod)
			{
				result.max_lod = result.min_lod;
			}

			return result;
		}
	}

	u64 sampler_desc::hash() const
	{
		u64 result = 0;
		result = hash_combine(result, static_cast<u64>(address_s) | (static_cast<u64>(address_t) << 8) | (static_cast<u64>(address_r) << 16));
		result = hash_combine(result, static_cast<u64>(min_filter) | (static_cast<u64>(mag_filter) << 8) | (static_cast<u64>(mip_filter) << 16));
		result = hash_combine(result, static_cast<u64>(border_color) | (static_cast<u64>(compare_func) << 8));
		result = hash_combine(result, static_cast<u64>(compare_enabled) | (static_cast<u64>(normalized_coords) << 1) | (static_cast<u64>(max_anisotropy) << 8));
		result = hash_combine(result, static_cast<u64>(std::bit_cast<u32>(min_lod)) | (static_cast<u64>(std::bit_cast<u32>(max_lod)) << 32));
		result = hash_combine(result, std::bit_cast<u32>(lod_bias));
		return result;
	}

	bool sampler_desc::operator == (const sampler_desc& other) const
	{
		// Floats are compared bitwise to stay consistent with hash()
		return address_s == other.address_s &&
			address_t == other.address_t &&
			address_r == other.address_r &&
			min_filter == other.min_filter &&
			mag_filter == other.mag_filter &&
			mip_filter == other.mip_filter &&
			border_color == other.border_color &&
			compare_func == other.compare_func &&
			compare_enabled == other.compare_enabled &&
			normalized_coords == other.normalized_coords &&
			max_anisotropy == other.max_anisotropy &&
			std::bit_cast<u32>(min_lod) == std::bit_cast<u32>(other.min_lod) &&
			std::bit_cast<u32>(max_lod) == std::bit_cast<u32>(other.max_lod) &&
			std::bit_cast<u32>(lod_bias) == std::bit_cast<u32>(other.lod_bias);
	}

	id<MTLSamplerState> sampler_cache::get(const sampler_desc& desc)
	{
		const u64 key = desc.hash();

		std::lock_guard lock(m_lock);

		auto& bucket = m_cache[key];
		for (const auto& [cached_desc, state] : bucket)
		{
			if (cached_desc == desc)
			{
				return state;
			}
		}

		const sampler_desc info = sanitize(desc);

		MTLSamplerDescriptor* sampler_info = [MTLSamplerDescriptor new];
		sampler_info.sAddressMode = info.address_s;
		sampler_info.tAddressMode = info.address_t;
		sampler_info.rAddressMode = info.address_r;
		sampler_info.minFilter = info.min_filter;
		sampler_info.magFilter = info.mag_filter;
		sampler_info.mipFilter = info.mip_filter;
		sampler_info.maxAnisotropy = info.max_anisotropy;
		sampler_info.normalizedCoordinates = info.normalized_coords ? YES : NO;
		sampler_info.lodMinClamp = info.min_lod;
		sampler_info.lodMaxClamp = info.max_lod;
		sampler_info.lodAverage = NO;
		sampler_info.borderColor = info.border_color;
		sampler_info.supportArgumentBuffers = NO;

		if (info.compare_enabled)
		{
			sampler_info.compareFunction = info.compare_func;
		}

		if (info.lod_bias != 0.f)
		{
			if (@available(macOS 26.0, *))
			{
				sampler_info.lodBias = info.lod_bias;
			}
			else if (!g_lod_bias_unsupported_reported.exchange(true))
			{
				rsx_log.warning("Metal: Sampler LOD bias requires macOS 26 or newer and will be ignored");
			}
		}

		id<MTLSamplerState> state = [get_device() newSamplerStateWithDescriptor:sampler_info];
		if (!state)
		{
			fmt::throw_exception("Metal: Failed to create sampler state (min=%u, mag=%u, mip=%u, address=[%u, %u, %u], aniso=%u, normalized=%d, compare=%d)",
				static_cast<u32>(info.min_filter), static_cast<u32>(info.mag_filter), static_cast<u32>(info.mip_filter),
				static_cast<u32>(info.address_s), static_cast<u32>(info.address_t), static_cast<u32>(info.address_r),
				info.max_anisotropy, info.normalized_coords, info.compare_enabled);
		}

		// Cache under the original description so that identical requests hit without sanitizing again
		bucket.emplace_back(desc, state);
		return state;
	}

	void sampler_cache::clear()
	{
		std::lock_guard lock(m_lock);
		m_cache.clear();
	}

	sampler_cache& get_sampler_cache()
	{
		return g_sampler_cache;
	}

	MTLSamplerBorderColor get_border_color(u32 rsx_argb8)
	{
		switch (rsx_argb8)
		{
		case 0x00000000:
			return MTLSamplerBorderColorTransparentBlack;
		case 0xFFFFFFFF:
			return MTLSamplerBorderColorOpaqueWhite;
		case 0xFF000000:
			return MTLSamplerBorderColorOpaqueBlack;
		default:
			break;
		}

		// Metal only has the 3 preset border colors (no VK_EXT_custom_border_color equivalent).
		// Use the same fallback heuristic as VK on devices without custom border color support.
		const color4f color = rsx::decode_border_color(rsx_argb8);

		MTLSamplerBorderColor result;
		if ((color.r + color.g + color.b) > 1.35f)
		{
			// If color elements are brighter than roughly 0.5 average, use white border
			result = MTLSamplerBorderColorOpaqueWhite;
		}
		else if (color.a > 0.5f)
		{
			result = MTLSamplerBorderColorOpaqueBlack;
		}
		else
		{
			result = MTLSamplerBorderColorTransparentBlack;
		}

		if (!g_inexact_border_color_reported.exchange(true))
		{
			rsx_log.warning("Metal: Border color 0x%08x has no exact Metal equivalent, using the closest preset (%s). Further occurrences are not reported.",
				rsx_argb8,
				result == MTLSamplerBorderColorOpaqueWhite ? "opaque white" : (result == MTLSamplerBorderColorOpaqueBlack ? "opaque black" : "transparent black"));
		}

		return result;
	}
}
