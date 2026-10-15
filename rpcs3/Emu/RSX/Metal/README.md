# Metal renderer

Native Metal backend for macOS. It is a port of the Vulkan backend: the RSX-facing logic (surface cache, texture cache,
program cache, draw/flip flow) mirrors `VK/`, and the Vulkan-specific layers (descriptors, render pass objects, layouts,
barriers, swapchain) are replaced by their Metal equivalents.

Select it with *Configuration → GPU → Renderer → Metal*, or `Renderer: Metal` under `Video:` in a custom config.
Prebuilt Apple Silicon builds are on the [releases page](https://github.com/sirlocomotiva/rpcs3-metal/releases/latest).

## Building

`USE_METAL` is on by default on macOS. Building needs the macOS 26 SDK (Xcode 26) or newer. The app runs on macOS 15
or newer, and uses features of later releases when the running system has them. SPIRV-Cross is fetched at configure
time when no system copy is found.

Release builds come from [`metal-mac.yml`](../../../../.github/workflows/metal-mac.yml), which runs RPCS3's macOS CI
script (`.ci/build-mac.sh`) with MacPorts dependencies. Development builds use Homebrew:

```bash
brew install cmake ninja llvm@22 qt sdl3 molten-vk vulkan-headers vulkan-loader
git clone --recurse-submodules -b metal-renderer https://github.com/sirlocomotiva/rpcs3-metal.git
cd rpcs3-metal
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$(brew --prefix llvm@22);$(brew --prefix qt)" \
    -DLLVM_DIR="$(brew --prefix llvm@22)/lib/cmake/llvm" \
    -DSTATIC_LINK_LLVM=ON -DUSE_SYSTEM_MVK=ON -DUSE_SYSTEM_OPENCV=OFF \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0
cmake --build build
```

The app is written to `build/bin/rpcs3.app`.

## Layout

| Files | Contents |
| --- | --- |
| `MTLRenderer.h/.mm` | C++-only entry point (`mtl::create_renderer`); everything else is Objective-C++ (ARC) |
| `MTLGSRender.*` | Renderer thread: setup, render targets, render pass, clears, submission and sync, queries, program loading |
| `MTLDraw.mm`, `MTLVertexBuffers.mm` | Draw state, textures/samplers, vertex and index upload |
| `MTLPresent.mm` | Flip, `CAMetalLayer` presentation, overlays, screenshots |
| `MTLTextureCache.*`, `MTLRenderTargets.*`, `MTLTexture.*` | Texture and surface caches, uploads, copies, scaled blits |
| `MTL{Vertex,Fragment}Program.*`, `MTLCommonDecompiler.*`, `MTLProgramBuffer.*` | RSX shader decompilers and program/pipeline cache |
| `MTLCompute.*`, `MTLOverlays.*`, `MTLResolveHelper.*`, `MTLUpscaler.*` | Compute helpers, helper graphics passes, MSAA (un)resolve, output scaling |
| `mtlutils/` | Device and capabilities, command context, ring buffers, images, samplers, shader compiler, pipeline compiler |

## Shaders and pipelines

RSX programs are decompiled to Vulkan-flavoured GLSL, compiled to SPIR-V with glslang, translated to MSL 3.2 with
SPIRV-Cross and compiled with `newLibraryWithSource`. Every entry point is named `main0`.

Bindings follow the descriptor set layout of the generated GLSL:
* Set 0, binding N maps to `[[buffer(N)]]`.
* Set 1, binding N maps to `[[texture(N)]]`, with `[[sampler(N)]]` below 16 and constexpr samplers above.
* Set 2 holds input attachments, read with framebuffer fetch (`[[color(N)]]`).

The RSX thread only decompiles programs to GLSL. Translation to MSL, the `MTLLibrary` and the pipeline state are built
on the `RSX.W` worker threads, the first time a pipeline links the stage. Draws whose pipeline is not ready yet are
skipped, the same as the asynchronous shader recompiler mode. The shader interpreter is not available, so the
interpreter modes fall back to asynchronous compilation.

Two Metal rules shape pipeline creation:
* A vertex function that writes `[[point_size]]` is only valid with the point or unspecified topology class. RSX vertex
  programs always write the point size, so only point draws declare their class.
* A fragment function that writes depth needs a depth attachment. Fragment programs that export depth also get a
  variant without the export (`MTL_NO_DEPTH_EXPORT`), which is linked into pipelines with no depth attachment.

Metal has no depth bounds test, so it is done in the fragment shader. When the test is enabled with bounds that can
reject anything, the fragment program is compiled with `RSX_SHADER_CONTROL_EMULATE_DEPTH_BOUNDS`. It reads the stored
depth from the bound Z buffer (per sample on MSAA surfaces) and discards fragments outside the bounds, which are passed
in the fragment state buffer. Like depth compare emulation, the first draw that reads the Z buffer ends the render pass,
so that the stored contents are visible to the shader. Later reads stay in the same pass until a draw or a clear writes
depth or stencil.

## Metal features in use

* **MetalFX spatial upscaling** when the output scaling mode is FSR.
* **`MTLBinaryArchive`** in each game's cache directory (`shaders_cache/pipelines/metal/archive_v1.00/`), so pipelines
  from earlier sessions load without recompiling. Each session that compiles new pipelines writes them to a new archive
  layer (`NNNNN.bin`) on a background queue: 30 seconds after the first one, then again for later ones, and when the
  renderer closes. Saved pipelines survive a crash or a killed process, and the write started at close is finished
  before the process exits. Loaded layers are only read. Metal can fail to re-serialize a loaded archive with new
  pipelines added, so they are never re-saved. Up to 32 layers are kept. Once the limit is reached, the oldest layer
  is retired: pipelines found in it are copied into the session's layer, and it is deleted after that layer is written.
  `shouldMaximizeConcurrentCompilation` is enabled on the device.
* **Framebuffer fetch** for RSX programmable blending, with no texture copies of the attachments.
* **Unified memory.** Ring buffers and readback (DMA) buffers use shared storage, so GPU→guest flushes copy directly out
  of mapped memory.
* **Native BC1–BC3 (DXT) textures** when the GPU supports them.
* **Accumulating visibility results** (`MTLVisibilityResultTypeAccumulate`) for occlusion queries when available.
* **`CAMetalLayer` presentation.** `displaySyncEnabled` follows the VSync setting. The drawable count (2 or 3) follows
  the VSync and frame limiter settings, like the Vulkan swapchain image count.
* **macOS Game Mode.** The app bundle declares the games category and `LSSupportsGameMode`, so playing in fullscreen
  raises CPU and GPU scheduling priority.

## Debugging

Metal's own switches work as usual: `MTL_HUD_ENABLED=1` (performance HUD) and `MTL_DEBUG_LAYER=1` (API validation; add
`MTL_DEBUG_LAYER_ERROR_MODE=nslog` to keep running on errors). The backend adds these environment variables:

| Variable | Effect |
| --- | --- |
| `RPCS3_METAL_CAPTURE=N` | Captures presented frame N into a `.gputrace` (requires `MTL_CAPTURE_ENABLED=1`) |
| `RPCS3_METAL_TRACE_STALLS=ms` | Logs GPU waits, hard syncs and synchronous shader/pipeline compiles that take at least `ms` |
| `RPCS3_METAL_TRACE_FLIP=N` | Logs the flip source every N flips |
| `RPCS3_METAL_SCREENSHOT_EVERY=N` | Saves a screenshot every N flips |
| `RPCS3_METAL_DEBUG_CLEAR=1` / `2` | Clears the output to magenta (`1`), or presents only the magenta clear (`2`) |

## Known limitations

* No last-vertex provoking convention, so flat shading falls back to smooth interpolation. Vulkan does the same without
  `VK_EXT_provoking_vertex`.
* No logic operations. `logic_copy`, by far the most common, renders correctly; other operations are reported once and
  ignored.
* Depth outside [0, 1] is only available through *Emulate extended depth range*.
* The emulated depth bounds test reads the Z buffer from before the draw. With depth writes enabled, primitives of the
  same draw are not tested against each other's depth, unlike the hardware test. This is reported once in the log.
* No hardware conditional rendering. The RSX CPU fallback is used.
