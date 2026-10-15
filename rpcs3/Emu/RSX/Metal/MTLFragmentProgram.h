#pragma once
#include "Emu/RSX/Program/FragmentProgramDecompiler.h"
#include "Emu/RSX/Program/GLSLTypes.h"
#include "MTLCommonDecompiler.h"

#include <array>

class MTLFragmentDecompilerThread : public FragmentProgramDecompiler
{
	std::string& m_shader;
	[[maybe_unused]] ParamArray& m_parrDummy;
	class MTLFragmentProgram *mtl_prog;
	glsl::shader_properties m_shader_props{};

public:
	MTLFragmentDecompilerThread(std::string& shader, ParamArray& parr, const RSXFragmentProgram &prog, u32& size, class MTLFragmentProgram& dst)
		: FragmentProgramDecompiler(prog, size)
		, m_shader(shader)
		, m_parrDummy(parr)
		, mtl_prog(&dst)
	{
	}

	void Task();

protected:
	std::string getFloatTypeName(usz elementCount) override;
	std::string getHalfTypeName(usz elementCount) override;
	std::string getFunction(FUNCTION f) override;
	std::string compareFunction(COMPARE f, std::string_view Op0, std::string_view Op1) override;

	void insertHeader(std::stringstream &OS) override;
	void insertInputs(std::stringstream &OS) override;
	void insertOutputs(std::stringstream &OS) override;
	void insertConstants(std::stringstream &OS) override;
	void insertGlobalFunctions(std::stringstream &OS) override;
	void insertMainStart(std::stringstream &OS) override;
	void insertMainEnd(std::stringstream &OS) override;
};

/** Storage for a Fragment Program in the process of recompilation.
 *  Decompilation and compilation are thread-safe (the shader cache loader runs them on worker threads).
 */
class MTLFragmentProgram
{
public:
	MTLFragmentProgram();
	~MTLFragmentProgram();

	ParamArray parr;
	mtl::deferred_function_ptr handle;
	u32 id = 0;
	mtl::decompiled_shader shader;

	// Variant without depth writes; Metal requires a depth attachment for fragment functions that write depth.
	// Only compiled for programs that export depth (writes_depth).
	mtl::deferred_function_ptr handle_no_depth;
	mtl::decompiled_shader shader_no_depth;
	bool writes_depth = false;
	std::vector<u32> constant_offsets;

	std::array<u32, 4> output_color_masks{ {} };

	// Reflection (copied into mtl::program by the pipeline builder)
	u32 referenced_textures_mask = 0;     // Bit N set if texN is declared by the program
	u32 stencil_mirror_mask = 0;          // Bit N set if texN_stencil is declared by the program
	bool uses_depth_input = false;        // frag_depth is declared (depth compare emulation)
	bool uses_framebuffer_fetch = false;  // frag_src_N inputs are declared (programmable blending)

	/**
	 * Decompile a fragment shader located in the PS3's Memory.  This function operates synchronously.
	 * @param prog RSXShaderProgram specifying the location and size of the shader in memory
	 */
	void Decompile(const RSXFragmentProgram& prog);

	/** Compile the decompiled fragment shader into a Metal function. */
	void Compile();

private:
	/** Deletes the shader and any stored information */
	void Delete();
};
