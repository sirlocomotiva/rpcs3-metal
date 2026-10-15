#pragma once

// C++-only entry point for the Metal renderer (the rest of the backend is Objective-C++)
namespace utils
{
	struct serial;
}

namespace mtl
{
	void create_renderer(utils::serial* ar);
}
