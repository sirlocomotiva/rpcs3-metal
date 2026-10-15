#pragma once
#include "Emu/RSX/Program/VertexProgramDecompiler.h"
#include "Emu/RSX/Program/program_util.h"
#include "MTLCommonDecompiler.h"

struct MTLVertexDecompilerThread : public VertexProgramDecompiler
{
	std::string &m_shader;
	class MTLVertexProgram *mtl_prog;

protected:
	std::string getFloatTypeName(usz elementCount) override;
	std::string getIntTypeName(usz elementCount) override;
	std::string getFunction(FUNCTION f) override;
	std::string compareFunction(COMPARE f, std::string_view Op0, std::string_view Op1, bool scalar) override;

	void insertHeader(std::stringstream& OS) override;
	void insertInputs(std::stringstream& OS, const std::vector<ParamType>& inputs) override;
	void insertConstants(std::stringstream& OS, const std::vector<ParamType>& constants) override;
	void insertOutputs(std::stringstream& OS, const std::vector<ParamType>& outputs) override;
	void insertMainStart(std::stringstream& OS) override;
	void insertMainEnd(std::stringstream& OS) override;

	const RSXVertexProgram &rsx_vertex_program;
public:
	MTLVertexDecompilerThread(const RSXVertexProgram& prog, std::string& shader, ParamArray&, class MTLVertexProgram& dst)
		: VertexProgramDecompiler(prog)
		, m_shader(shader)
		, mtl_prog(&dst)
		, rsx_vertex_program(prog)
	{
	}

	void Task();
};

class MTLVertexProgram : public rsx::VertexProgramBase
{
public:
	MTLVertexProgram();
	~MTLVertexProgram();

	ParamArray parr;
	mtl::deferred_function_ptr handle;
	mtl::decompiled_shader shader;

	// Reflection (copied into mtl::program by the pipeline builder)
	u32 referenced_textures_mask = 0;   // Bit N set if vtexN is declared by the program

	void Decompile(const RSXVertexProgram& prog);
	void Compile();

	u32 Flags() const { return m_ctrl; }

private:
	void Delete();

	u32 m_ctrl = 0u;
};
