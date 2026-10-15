#pragma once

#include "mtl_common.h"

#include <unordered_map>
#include <mutex>

namespace mtl
{
	struct sampler_desc
	{
		MTLSamplerAddressMode address_s = MTLSamplerAddressModeClampToEdge;
		MTLSamplerAddressMode address_t = MTLSamplerAddressModeClampToEdge;
		MTLSamplerAddressMode address_r = MTLSamplerAddressModeClampToEdge;
		MTLSamplerMinMagFilter min_filter = MTLSamplerMinMagFilterNearest;
		MTLSamplerMinMagFilter mag_filter = MTLSamplerMinMagFilterNearest;
		MTLSamplerMipFilter mip_filter = MTLSamplerMipFilterNotMipmapped;
		MTLSamplerBorderColor border_color = MTLSamplerBorderColorTransparentBlack;
		MTLCompareFunction compare_func = MTLCompareFunctionNever;
		bool compare_enabled = false;
		bool normalized_coords = true;
		u8 max_anisotropy = 1;
		float min_lod = 0.f;
		float max_lod = 1000.f;
		float lod_bias = 0.f;

		u64 hash() const;
		bool operator == (const sampler_desc& other) const;
	};

	/**
	 * Thread-safe cache of immutable MTLSamplerState objects.
	 * Metal samplers are immutable and cheap to keep around, so they are never released until shutdown.
	 */
	class sampler_cache
	{
		std::unordered_map<u64, std::vector<std::pair<sampler_desc, id<MTLSamplerState>>>> m_cache;
		std::mutex m_lock;

	public:
		id<MTLSamplerState> get(const sampler_desc& desc);
		void clear();
	};

	sampler_cache& get_sampler_cache();

	// Map an RSX border color to the closest Metal preset
	MTLSamplerBorderColor get_border_color(u32 rsx_argb8);
}
