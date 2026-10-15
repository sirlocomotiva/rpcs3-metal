#include "stdafx.h"
#include "MTLRenderer.h"
#include "MTLGSRender.h"

#include "Utilities/Thread.h"
#include "util/fixed_typemap.hpp"

namespace mtl
{
	void create_renderer(utils::serial* ar)
	{
		g_fxo->init<rsx::thread, named_thread<MTLGSRender>>(ar);
	}
}
