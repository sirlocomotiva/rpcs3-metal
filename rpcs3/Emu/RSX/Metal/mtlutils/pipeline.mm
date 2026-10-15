#include "stdafx.h"
#include "pipeline.h"
#include "device.h"

#include "Utilities/File.h"
#include "Utilities/Thread.h"
#include "util/sysinfo.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <shared_mutex>
#include <vector>

namespace mtl
{
	namespace
	{
		// Global list of worker threads
		std::unique_ptr<named_thread_group<pipe_compiler>> g_pipe_compilers;
		int g_num_pipe_compilers = 0;
		atomic_t<int> g_compiler_index{};

		// Set while the renderer shuts the workers down: queued jobs are dropped instead of compiled
		atomic_t<bool> g_discard_pipe_compiler_jobs = false;

		/**
		 * Pipeline binary archives, stored as layers in one directory per game.
		 * Archives loaded from disk are only used for lookups, they are never serialized again: Metal fails to re-serialize
		 * some loaded archives once new pipelines are added ("Failed to generate machO ... expecting 'fragment' stage").
		 * Pipelines compiled during the session are recorded and written to a new layer built from scratch, in the
		 * background shortly after they are compiled, so that a session that does not end cleanly keeps them too.
		 * Writing a layer past the limit deletes the oldest layers. Pipelines this session found in those layers are
		 * recorded too, so that the pipelines still in use survive.
		 */
		constexpr u32 max_binary_archive_layers = 32;
		constexpr s64 binary_archive_flush_delay_ns = 30'000'000'000; // Time from the first unsaved pipeline to the write
		constexpr s64 binary_archive_exit_timeout_ns = 30'000'000'000; // Longest wait for the last writes when the process exits

		struct binary_archive_t
		{
			std::mutex lock;
			bool open = false;
			u64 generation = 0; // Incremented on close, invalidates flushes scheduled by an earlier session
			id<MTLDevice> device = nil;

			NSArray<id<MTLBinaryArchive>>* layers = nil;          // Immutable, used for lookups
			NSArray<id<MTLBinaryArchive>>* evicted_layers = nil;  // Oldest layers, deleted when the session layer is written
			std::vector<std::string> evicted_paths;
			NSMutableArray<MTLRenderPipelineDescriptor*>* render_descs = nil;   // Recorded during this session
			NSMutableArray<MTLComputePipelineDescriptor*>* compute_descs = nil;
			bool has_new_pipelines = false; // A pipeline was compiled. Pipelines found in evicted layers alone do not cause a write.
			u32 saved_count = 0;           // Recorded pipelines included in the session layer on disk
			bool flush_scheduled = false;
			std::string session_layer_path;

			atomic_t<u64> hits = 0;
			atomic_t<u64> misses = 0;
			atomic_t<bool> add_error_reported = false;
			atomic_t<bool> save_error_reported = false;

			dispatch_queue_t queue = nil; // Serial, runs all layer writes
		};

		// Never destroyed: a delayed write may still fire while the process exits
		binary_archive_t& g_binary_archive = *new binary_archive_t;

		struct binary_archive_write_job
		{
			id<MTLDevice> device = nil;
			std::string path;
			NSArray<MTLRenderPipelineDescriptor*>* render_descs = nil;
			NSArray<MTLComputePipelineDescriptor*>* compute_descs = nil;
			std::vector<std::string> evicted_paths; // Deleted once the layer is written
		};

		// Depth-stencil states
		std::shared_mutex g_depth_stencil_lock;
		std::unordered_map<u64, std::vector<std::pair<depth_stencil_desc, id<MTLDepthStencilState>>>> g_depth_stencil_states;

		bool is_integer_format(MTLPixelFormat format)
		{
			switch (format)
			{
			case MTLPixelFormatR8Uint:
			case MTLPixelFormatR8Sint:
			case MTLPixelFormatR16Uint:
			case MTLPixelFormatR16Sint:
			case MTLPixelFormatRG8Uint:
			case MTLPixelFormatRG8Sint:
			case MTLPixelFormatR32Uint:
			case MTLPixelFormatR32Sint:
			case MTLPixelFormatRG16Uint:
			case MTLPixelFormatRG16Sint:
			case MTLPixelFormatRGBA8Uint:
			case MTLPixelFormatRGBA8Sint:
			case MTLPixelFormatRGB10A2Uint:
			case MTLPixelFormatRG32Uint:
			case MTLPixelFormatRG32Sint:
			case MTLPixelFormatRGBA16Uint:
			case MTLPixelFormatRGBA16Sint:
			case MTLPixelFormatRGBA32Uint:
			case MTLPixelFormatRGBA32Sint:
				return true;
			default:
				return false;
			}
		}

		std::string describe_function(id<MTLFunction> function)
		{
			return function ? from_ns_string(function.name) + "/" + (function.label ? from_ns_string(function.label) : std::string("?")) : std::string("<none>");
		}

		std::string describe_render_descriptor(MTLRenderPipelineDescriptor* desc)
		{
			std::string result = fmt::format("vs=%s, fs=%s, samples=%u, topology=%u, a2c=%d, a2o=%d, depth=0x%x, stencil=0x%x",
				describe_function(desc.vertexFunction), describe_function(desc.fragmentFunction),
				static_cast<u32>(desc.rasterSampleCount), static_cast<u32>(desc.inputPrimitiveTopology),
				!!desc.alphaToCoverageEnabled, !!desc.alphaToOneEnabled,
				static_cast<u32>(desc.depthAttachmentPixelFormat), static_cast<u32>(desc.stencilAttachmentPixelFormat));

			for (u32 i = 0; i < 8; ++i)
			{
				MTLRenderPipelineColorAttachmentDescriptor* att = desc.colorAttachments[i];
				if (att.pixelFormat == MTLPixelFormatInvalid)
				{
					continue;
				}

				fmt::append(result, ", color%u={format=0x%x, mask=0x%x, blend=%d", i, static_cast<u32>(att.pixelFormat),
					static_cast<u32>(att.writeMask), !!att.blendingEnabled);

				if (att.blendingEnabled)
				{
					fmt::append(result, " (rgb: %u %u op %u, alpha: %u %u op %u)",
						static_cast<u32>(att.sourceRGBBlendFactor), static_cast<u32>(att.destinationRGBBlendFactor), static_cast<u32>(att.rgbBlendOperation),
						static_cast<u32>(att.sourceAlphaBlendFactor), static_cast<u32>(att.destinationAlphaBlendFactor), static_cast<u32>(att.alphaBlendOperation));
				}

				result += "}";
			}

			return result;
		}

		struct archive_lookup_state
		{
			NSArray<id<MTLBinaryArchive>>* layers = nil;
			NSArray<id<MTLBinaryArchive>>* evicted_layers = nil;
			bool recording = false;
		};

		archive_lookup_state get_archive_lookup_state()
		{
			std::lock_guard lock(g_binary_archive.lock);
			return { g_binary_archive.layers, g_binary_archive.evicted_layers, g_binary_archive.open };
		}

		void report_archive_add_failure(NSError* error)
		{
			if (!g_binary_archive.add_error_reported.exchange(true))
			{
				rsx_log.warning("Metal: Failed to add a pipeline to the binary archive: %s. Further failures are not reported.", describe_error(error));
			}
		}

		// Builds an archive from scratch with the job's pipelines, writes it to the job's path and deletes the evicted
		// layers. Runs on the archive queue.
		bool write_binary_archive_layer(const binary_archive_write_job& job)
		{
			@autoreleasepool
			{
				const auto start = std::chrono::steady_clock::now();

				NSError* error = nil;
				id<MTLBinaryArchive> archive = [job.device newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&error];
				if (!archive)
				{
					rsx_log.error("Metal: Failed to create a pipeline binary archive: %s", describe_error(error));
					return false;
				}

				for (MTLRenderPipelineDescriptor* desc in job.render_descs)
				{
					if (![archive addRenderPipelineFunctionsWithDescriptor:desc error:&error])
					{
						report_archive_add_failure(error);
					}
				}

				for (MTLComputePipelineDescriptor* desc in job.compute_descs)
				{
					if (![archive addComputePipelineFunctionsWithDescriptor:desc error:&error])
					{
						report_archive_add_failure(error);
					}
				}

				// Serialization compiles the GPU binaries. Write to a temporary file so that a valid layer is never left half written.
				const std::string temp_path = job.path + ".tmp";
				if (![archive serializeToURL:[NSURL fileURLWithPath:to_ns_string(temp_path)] error:&error])
				{
					if (!g_binary_archive.save_error_reported.exchange(true))
					{
						rsx_log.error("Metal: Failed to save pipeline binary archive layer '%s': %s", job.path, describe_error(error));
					}

					fs::remove_file(temp_path);
					return false;
				}

				if (!fs::rename(temp_path, job.path, true))
				{
					rsx_log.error("Metal: Failed to move pipeline binary archive layer to '%s' (%s)", job.path, fs::g_tls_error);
					fs::remove_file(temp_path);
					return false;
				}

				rsx_log.notice("Metal: Saved %u pipeline(s) to binary archive layer '%s' in %.1f ms",
					static_cast<u32>(job.render_descs.count + job.compute_descs.count), job.path,
					std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());

				for (const auto& path : job.evicted_paths)
				{
					if (fs::remove_file(path))
					{
						rsx_log.notice("Metal: Deleted the oldest pipeline binary archive layer '%s'", path);
					}
				}

				return true;
			}
		}

		// Takes the session's pipelines for a layer write, if the session compiled pipelines that are not on disk yet.
		// The caller holds the lock.
		bool get_binary_archive_write_job(binary_archive_write_job& job)
		{
			const u32 count = static_cast<u32>(g_binary_archive.render_descs.count + g_binary_archive.compute_descs.count);

			if (!g_binary_archive.has_new_pipelines || count == g_binary_archive.saved_count)
			{
				return false;
			}

			job.device = g_binary_archive.device;
			job.path = g_binary_archive.session_layer_path;
			job.render_descs = [g_binary_archive.render_descs copy];
			job.compute_descs = [g_binary_archive.compute_descs copy];
			job.evicted_paths = g_binary_archive.evicted_paths;
			return true;
		}

		// Writes every pipeline recorded during the session to the session layer, if any were added since the last write.
		// Runs on the archive queue. Writes scheduled by an earlier session (other generation) are ignored.
		void flush_binary_archive(u64 generation)
		{
			binary_archive_write_job job;

			{
				std::lock_guard lock(g_binary_archive.lock);

				if (!g_binary_archive.open || g_binary_archive.generation != generation)
				{
					return;
				}

				g_binary_archive.flush_scheduled = false;

				if (!get_binary_archive_write_job(job))
				{
					return;
				}
			}

			if (write_binary_archive_layer(job))
			{
				std::lock_guard lock(g_binary_archive.lock);

				if (g_binary_archive.generation == generation)
				{
					g_binary_archive.saved_count = std::max<u32>(g_binary_archive.saved_count, static_cast<u32>(job.render_descs.count + job.compute_descs.count));
					g_binary_archive.evicted_paths.clear();
				}
			}
		}

		// Records a pipeline for the session layer: compiled during this session, or found in an evicted layer.
		// It is written to the session layer after a short delay.
		void record_pipeline(MTLRenderPipelineDescriptor* render_desc, MTLComputePipelineDescriptor* compute_desc, bool compiled)
		{
			std::lock_guard lock(g_binary_archive.lock);

			if (!g_binary_archive.open)
			{
				return;
			}

			// The copies keep the shader functions alive until the session ends
			if (render_desc)
			{
				MTLRenderPipelineDescriptor* copy = [render_desc copy];
				copy.binaryArchives = nil;
				[g_binary_archive.render_descs addObject:copy];
			}

			if (compute_desc)
			{
				MTLComputePipelineDescriptor* copy = [compute_desc copy];
				copy.binaryArchives = nil;
				[g_binary_archive.compute_descs addObject:copy];
			}

			g_binary_archive.has_new_pipelines |= compiled;

			if (g_binary_archive.has_new_pipelines && !g_binary_archive.flush_scheduled)
			{
				g_binary_archive.flush_scheduled = true;

				const u64 generation = g_binary_archive.generation;
				dispatch_after(dispatch_time(DISPATCH_TIME_NOW, binary_archive_flush_delay_ns), g_binary_archive.queue, ^{
					flush_binary_archive(generation);
				});
			}
		}

		// Loads a pipeline binary from the archives without compiling. Returns nil on a miss.
		id<MTLRenderPipelineState> find_render_pipeline_state(id<MTLDevice> device, MTLRenderPipelineDescriptor* desc, NSArray<id<MTLBinaryArchive>>* archives)
		{
			desc.binaryArchives = archives;

			NSError* error = nil;
			id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:desc
				options:MTLPipelineOptionFailOnBinaryArchiveMiss
				reflection:nil
				error:&error];

			desc.binaryArchives = nil;
			return state;
		}

		id<MTLComputePipelineState> find_compute_pipeline_state(id<MTLDevice> device, MTLComputePipelineDescriptor* desc, NSArray<id<MTLBinaryArchive>>* archives)
		{
			desc.binaryArchives = archives;

			NSError* error = nil;
			id<MTLComputePipelineState> state = [device newComputePipelineStateWithDescriptor:desc
				options:MTLPipelineOptionFailOnBinaryArchiveMiss
				reflection:nil
				error:&error];

			desc.binaryArchives = nil;
			return state;
		}

		id<MTLRenderPipelineState> create_render_pipeline_state(id<MTLDevice> device, MTLRenderPipelineDescriptor* desc, NSError** error)
		{
			const auto archive = get_archive_lookup_state();

			if (archive.evicted_layers.count)
			{
				if (id<MTLRenderPipelineState> state = find_render_pipeline_state(device, desc, archive.evicted_layers))
				{
					g_binary_archive.hits++;
					record_pipeline(desc, nil, false);
					return state;
				}
			}

			if (archive.layers.count)
			{
				if (id<MTLRenderPipelineState> state = find_render_pipeline_state(device, desc, archive.layers))
				{
					g_binary_archive.hits++;
					return state;
				}
			}

			// Archive miss, compile from source and record the result
			id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:desc error:error];

			if (state && archive.recording)
			{
				g_binary_archive.misses++;
				record_pipeline(desc, nil, true);
			}

			return state;
		}

		id<MTLComputePipelineState> create_compute_pipeline_state(id<MTLDevice> device, MTLComputePipelineDescriptor* desc, NSError** error)
		{
			const auto archive = get_archive_lookup_state();

			if (archive.evicted_layers.count)
			{
				if (id<MTLComputePipelineState> state = find_compute_pipeline_state(device, desc, archive.evicted_layers))
				{
					g_binary_archive.hits++;
					record_pipeline(nil, desc, false);
					return state;
				}
			}

			if (archive.layers.count)
			{
				if (id<MTLComputePipelineState> state = find_compute_pipeline_state(device, desc, archive.layers))
				{
					g_binary_archive.hits++;
					return state;
				}
			}

			id<MTLComputePipelineState> state = [device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:error];

			if (state && archive.recording)
			{
				g_binary_archive.misses++;
				record_pipeline(nil, desc, true);
			}

			return state;
		}

		int decay_num_worker_threads(int num_worker_threads)
		{
			if (num_worker_threads <= 0)
			{
				// Select a conservative but modern default for async pipeline compilation (same heuristic as VK)
				const auto hw_threads = utils::get_thread_count();
				if (hw_threads >= 24)
				{
					num_worker_threads = 12;
				}
				else if (hw_threads >= 16)
				{
					num_worker_threads = 8;
				}
				else if (hw_threads > 12)
				{
					num_worker_threads = 6;
				}
				else if (hw_threads > 8)
				{
					num_worker_threads = 4;
				}
				else if (hw_threads == 8)
				{
					num_worker_threads = 2;
				}
				else
				{
					num_worker_threads = 1;
				}

				rsx_log.notice("Async pipeline compiler auto-selected %d worker(s) for %u host thread(s).",
					num_worker_threads, hw_threads);
			}

			return num_worker_threads;
		}
	}

	void render_pass_formats::apply(MTLRenderPipelineDescriptor* desc) const
	{
		// The format array is authoritative: unused slots hold MTLPixelFormatInvalid
		for (u32 i = 0; i < 4; ++i)
		{
			desc.colorAttachments[i].pixelFormat = static_cast<MTLPixelFormat>(color_formats[i]);
		}

		desc.depthAttachmentPixelFormat = static_cast<MTLPixelFormat>(depth_format);
		desc.stencilAttachmentPixelFormat = static_cast<MTLPixelFormat>(stencil_format);
		desc.rasterSampleCount = std::max<u32>(sample_count, 1);
	}

	MTLRenderPipelineDescriptor* make_pipeline_descriptor(const pipeline_props& props, id<MTLFunction> vs, id<MTLFunction> fs)
	{
		MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
		desc.vertexFunction = vs;
		desc.fragmentFunction = fs;

		// Vertex data is fetched manually from texture buffers by the RSX vertex programs
		desc.vertexDescriptor = nil;

		render_pass_formats::from(props).apply(desc);

		for (u32 i = 0; i < 4; ++i)
		{
			const auto format = static_cast<MTLPixelFormat>(props.color_formats[i]);
			if (format == MTLPixelFormatInvalid)
			{
				continue;
			}

			const auto& blend = props.blend[i];
			MTLRenderPipelineColorAttachmentDescriptor* att = desc.colorAttachments[i];
			att.writeMask = static_cast<MTLColorWriteMask>(blend.write_mask);

			// Integer render targets cannot be blended (VK silently ignores blending for them)
			att.blendingEnabled = (blend.enable && !is_integer_format(format)) ? YES : NO;

			if (att.blendingEnabled)
			{
				att.sourceRGBBlendFactor = static_cast<MTLBlendFactor>(blend.src_rgb);
				att.destinationRGBBlendFactor = static_cast<MTLBlendFactor>(blend.dst_rgb);
				att.sourceAlphaBlendFactor = static_cast<MTLBlendFactor>(blend.src_alpha);
				att.destinationAlphaBlendFactor = static_cast<MTLBlendFactor>(blend.dst_alpha);
				att.rgbBlendOperation = static_cast<MTLBlendOperation>(blend.op_rgb);
				att.alphaBlendOperation = static_cast<MTLBlendOperation>(blend.op_alpha);
			}
		}

		desc.rasterSampleCount = std::max<u32>(props.sample_count, 1);
		desc.alphaToCoverageEnabled = props.alpha_to_coverage ? YES : NO;
		desc.alphaToOneEnabled = props.alpha_to_one ? YES : NO;
		// Vertex functions writing [[point_size]] are only valid with the point (or an unspecified) topology class
		desc.inputPrimitiveTopology = (props.topology_class == MTLPrimitiveTopologyClassPoint)
			? MTLPrimitiveTopologyClassPoint
			: MTLPrimitiveTopologyClassUnspecified;
		return desc;
	}

	// Pipe compiler

	void pipe_compiler::operator()()
	{
		// Compiles here never block the renderer
		disable_stall_trace_on_current_thread();

		const auto run_jobs = [this]()
		{
			for (auto&& job : m_work_queue.pop_all())
			{
				if (g_discard_pipe_compiler_jobs) [[unlikely]]
				{
					// The renderer is shutting down. Deferred jobs also compile their shader stages, so finishing them
					// would delay the emulator stop by seconds. Nothing waits for the results.
					continue;
				}

				@autoreleasepool
				{
					auto compiled = int_compile(job);

					if (job.callback)
					{
						job.callback(compiled);
					}
					else if (compiled)
					{
						rsx_log.error("Metal: Deferred pipeline compiled without a completion callback, the result is discarded");
					}
				}
			}
		};

		while (thread_ctrl::state() != thread_state::aborting)
		{
			run_jobs();
			thread_ctrl::wait_on(m_work_queue);
		}

		// The pool is replaced by a bigger one (resize_pipe_compiler). A dropped job would leave its pipeline missing
		// from the program cache for good, so complete the jobs queued so far.
		run_jobs();
	}

	std::unique_ptr<program> pipe_compiler::int_compile(const pipe_compiler_job& job)
	{
		id<MTLDevice> device = get_device();
		ensure(device, "Pipeline compilation requires an active Metal device");

		@autoreleasepool
		{
			NSError* error = nil;

			if (job.cs)
			{
				MTLComputePipelineDescriptor* desc = [MTLComputePipelineDescriptor new];
				desc.computeFunction = job.cs;

				id<MTLComputePipelineState> state = create_compute_pipeline_state(device, desc, &error);
				if (!state)
				{
					rsx_log.error("Metal: Failed to create compute pipeline (cs=%s): %s", describe_function(job.cs), describe_error(error));
					return {};
				}

				// NOTE: local_size is left at {1, 1, 1}; the caller fills it from the shader_module
				return std::make_unique<program>(state);
			}

			MTLRenderPipelineDescriptor* desc = nil;
			if (job.desc)
			{
				// Work on a private copy: binaryArchives is modified during the lookup and callers may reuse their descriptor
				desc = [job.desc copy];
			}
			else
			{
				// RSX program stages are compiled on first use, usually here on a worker thread
				id<MTLFunction> vs = job.vs ? job.vs->get() : nil;
				id<MTLFunction> fs = job.fs ? job.fs->get() : nil;

				if (!vs || !fs)
				{
					// The shader compiler already logged the failure with its source
					rsx_log.error("Metal: Cannot link render pipeline, a shader stage failed to compile");
					return {};
				}

				desc = make_pipeline_descriptor(job.props, vs, fs);
			}

			id<MTLRenderPipelineState> state = create_render_pipeline_state(device, desc, &error);
			if (!state)
			{
				rsx_log.error("Metal: Failed to create render pipeline: %s\nDescriptor: %s", describe_error(error), describe_render_descriptor(desc));
				return {};
			}

			return std::make_unique<program>(state);
		}
	}

	std::unique_ptr<program> pipe_compiler::compile(
		const pipeline_props& props,
		const deferred_function_ptr& vs,
		const deferred_function_ptr& fs,
		op_flags flags,
		callback_t callback)
	{
		pipe_compiler_job job{};
		job.props = props;
		job.vs = vs;
		job.fs = fs;
		job.flags = flags;

		if (flags & COMPILE_INLINE)
		{
			stall_trace_scope trace("Inline pipeline compile");
			return int_compile(job);
		}

		job.callback = std::move(callback);
		m_work_queue.push(std::move(job));
		return {};
	}

	std::unique_ptr<program> pipe_compiler::compile(
		MTLRenderPipelineDescriptor* desc,
		op_flags flags,
		callback_t callback)
	{
		ensure(desc);

		pipe_compiler_job job{};
		job.flags = flags;

		if (flags & COMPILE_INLINE)
		{
			stall_trace_scope trace("Inline pipeline compile");
			job.desc = desc;
			return int_compile(job);
		}

		// Deferred: snapshot the descriptor so that the caller can keep modifying its own
		job.desc = [desc copy];
		job.callback = std::move(callback);
		m_work_queue.push(std::move(job));
		return {};
	}

	std::unique_ptr<program> pipe_compiler::compile(
		id<MTLFunction> cs,
		op_flags flags,
		callback_t callback)
	{
		ensure(cs);

		pipe_compiler_job job{};
		job.cs = cs;
		job.flags = flags;

		if (flags & COMPILE_INLINE)
		{
			stall_trace_scope trace("Inline pipeline compile");
			return int_compile(job);
		}

		job.callback = std::move(callback);
		m_work_queue.push(std::move(job));
		return {};
	}

	void initialize_pipe_compiler(int num_worker_threads)
	{
		num_worker_threads = decay_num_worker_threads(num_worker_threads);

		ensure(num_worker_threads >= 1);
		ensure(get_current_device()); // "Cannot initialize pipe compiler before creating a device"

		// Create the thread pool
		g_pipe_compilers = std::make_unique<named_thread_group<pipe_compiler>>("RSX.W", num_worker_threads);
		g_num_pipe_compilers = num_worker_threads;
	}

	void resize_pipe_compiler(int num_worker_threads)
	{
		num_worker_threads = decay_num_worker_threads(num_worker_threads);
		if (g_pipe_compilers && static_cast<u32>(num_worker_threads) <= g_pipe_compilers->size())
		{
			// Just lie about how many compilers we have
			g_num_pipe_compilers = num_worker_threads;
			return;
		}

		// We need a bigger thread pool. Very unlikely but provided for correctness.
		g_pipe_compilers.reset();
		g_num_pipe_compilers = 0;

		initialize_pipe_compiler(num_worker_threads);
	}

	void destroy_pipe_compiler()
	{
		g_discard_pipe_compiler_jobs = true;
		g_pipe_compilers.reset();
		g_num_pipe_compilers = 0;
		g_discard_pipe_compiler_jobs = false;
	}

	pipe_compiler* get_pipe_compiler()
	{
		ensure(g_pipe_compilers);
		int thread_index = g_compiler_index++;

		return g_pipe_compilers.get()->begin() + (thread_index % g_num_pipe_compilers);
	}

	// Binary archive

	namespace
	{
		// Runs at process exit: saves the open session and waits for the queued writes, which would otherwise be cut off
		void finish_binary_archive_writes()
		{
			close_pipeline_binary_archive();

			dispatch_queue_t queue = nil;
			{
				std::lock_guard lock(g_binary_archive.lock);
				queue = g_binary_archive.queue;
			}

			dispatch_semaphore_t done = dispatch_semaphore_create(0);
			dispatch_async(queue, ^{
				dispatch_semaphore_signal(done);
			});

			dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, binary_archive_exit_timeout_ns));
		}
	}

	void open_pipeline_binary_archive(const std::string& dir_path)
	{
		ensure(!dir_path.empty());

		const auto dev = get_current_device();
		ensure(dev && dev->get(), "Cannot open a pipeline binary archive without a device");

		// Close (and save) any previously opened archive
		close_pipeline_binary_archive();

		if (!dev->caps().supports_binary_archives)
		{
			rsx_log.notice("Metal: Pipeline binary archives are not supported on this device");
			return;
		}

		if (dev->caps().supports_shader_validation)
		{
			// Shader validation instruments every pipeline, so archived binaries are useless. Loading an archive written
			// without validation also crashes inside Metal (MTLGPUDebugDevice newBinaryArchiveWithDescriptor).
			rsx_log.notice("Metal: Shader validation is enabled, pipeline binary archives in '%s' will not be used", dir_path);
			return;
		}

		std::string dir = dir_path;
		if (!dir.ends_with('/'))
		{
			dir += '/';
		}

		if (!fs::is_dir(dir) && !fs::create_path(dir))
		{
			rsx_log.error("Metal: Failed to create directory '%s' for pipeline binary archives (%s)", dir, fs::g_tls_error);
			return;
		}

		dispatch_queue_t queue = nil;
		{
			std::lock_guard lock(g_binary_archive.lock);

			if (!g_binary_archive.queue)
			{
				g_binary_archive.queue = dispatch_queue_create("net.rpcs3.metal.binary_archive",
					dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));

				std::atexit(finish_binary_archive_writes);
			}

			queue = g_binary_archive.queue;
		}

		// Wait for the last write of an earlier session, it can add a layer to this directory
		dispatch_sync(queue, ^{});

		@autoreleasepool
		{
			// Layers are named after their index, a higher index is newer
			std::vector<std::pair<u32, std::string>> files;

			for (const auto& entry : fs::dir(dir))
			{
				if (entry.is_directory)
				{
					continue;
				}

				if (entry.name.ends_with(".tmp") || entry.name.ends_with(".metallib"))
				{
					// Left over by a write that was interrupted (Metal serializes to a temporary .metallib file first)
					fs::remove_file(dir + entry.name);
					continue;
				}

				const std::string_view name = entry.name;
				u32 index = 0;

				if (!name.ends_with(".bin") || name.size() <= 4)
				{
					continue;
				}

				if (const auto [ptr, ec] = std::from_chars(name.data(), name.data() + name.size() - 4, index);
					ec != std::errc{} || ptr != name.data() + name.size() - 4)
				{
					continue;
				}

				files.emplace_back(index, entry.name);
			}

			std::sort(files.begin(), files.end());

			id<MTLDevice> device = dev->get();
			NSMutableArray<id<MTLBinaryArchive>>* layers = [NSMutableArray array];
			std::vector<std::string> layer_paths;

			for (const auto& [index, name] : files)
			{
				MTLBinaryArchiveDescriptor* desc = [MTLBinaryArchiveDescriptor new];
				desc.url = [NSURL fileURLWithPath:to_ns_string(dir + name)];

				NSError* error = nil;
				if (id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:desc error:&error])
				{
					[layers addObject:archive];
					layer_paths.push_back(dir + name);
				}
				else
				{
					// Corrupt or incompatible (e.g. after an OS update). Its pipelines are compiled and saved again.
					rsx_log.warning("Metal: Discarding pipeline binary archive layer '%s': %s", dir + name, describe_error(error));
					fs::remove_file(dir + name);
				}
			}

			// If this session writes a layer, the oldest layers go to stay within the limit
			const usz evict_count = layers.count >= max_binary_archive_layers ? layers.count + 1 - max_binary_archive_layers : 0;
			const NSRange evicted_range = NSMakeRange(0, evict_count);
			const NSRange kept_range = NSMakeRange(evict_count, layers.count - evict_count);

			const u32 session_index = files.empty() ? 0 : files.back().first + 1;

			std::lock_guard lock(g_binary_archive.lock);

			g_binary_archive.open = true;
			g_binary_archive.device = device;
			g_binary_archive.layers = [layers subarrayWithRange:kept_range];
			g_binary_archive.evicted_layers = evict_count ? [layers subarrayWithRange:evicted_range] : nil;
			g_binary_archive.evicted_paths.assign(layer_paths.begin(), layer_paths.begin() + evict_count);
			g_binary_archive.render_descs = [NSMutableArray array];
			g_binary_archive.compute_descs = [NSMutableArray array];
			g_binary_archive.has_new_pipelines = false;
			g_binary_archive.saved_count = 0;
			g_binary_archive.flush_scheduled = false;
			g_binary_archive.session_layer_path = fmt::format("%s%05u.bin", dir, session_index);
			g_binary_archive.hits = 0;
			g_binary_archive.misses = 0;
			g_binary_archive.add_error_reported = false;
			g_binary_archive.save_error_reported = false;

			rsx_log.notice("Metal: Using %u pipeline binary archive layer(s) in '%s'", static_cast<u32>(layers.count), dir);
		}
	}

	void close_pipeline_binary_archive()
	{
		binary_archive_write_job job;
		bool write = false;
		dispatch_queue_t queue = nil;

		{
			std::lock_guard lock(g_binary_archive.lock);

			if (!g_binary_archive.open)
			{
				return;
			}

			write = get_binary_archive_write_job(job);
			queue = g_binary_archive.queue;

			rsx_log.notice("Metal: Closed pipeline binary archives (layers: %u, hits: %u, misses: %u, unsaved: %u)",
				static_cast<u32>(g_binary_archive.layers.count + g_binary_archive.evicted_layers.count), g_binary_archive.hits.load(),
				g_binary_archive.misses.load(), write ? static_cast<u32>(job.render_descs.count + job.compute_descs.count) - g_binary_archive.saved_count : 0);

			// Delayed writes still queued for this session become no-ops
			g_binary_archive.generation++;
			g_binary_archive.open = false;
			g_binary_archive.device = nil;
			g_binary_archive.layers = nil;
			g_binary_archive.evicted_layers = nil;
			g_binary_archive.evicted_paths.clear();
			g_binary_archive.render_descs = nil;
			g_binary_archive.compute_descs = nil;
			g_binary_archive.has_new_pipelines = false;
			g_binary_archive.saved_count = 0;
			g_binary_archive.flush_scheduled = false;
			g_binary_archive.session_layer_path.clear();
		}

		if (write)
		{
			// Write what is left in the background, after any write in progress. Opening an archive waits for it.
			dispatch_async(queue, ^{
				write_binary_archive_layer(job);
			});
		}
	}

	bool is_pipeline_binary_archive_open()
	{
		std::lock_guard lock(g_binary_archive.lock);
		return g_binary_archive.open;
	}

	// Depth-stencil states

	id<MTLDepthStencilState> get_depth_stencil_state(const depth_stencil_desc& desc)
	{
		const u64 key = rpcs3::hash_struct(desc);

		{
			std::shared_lock lock(g_depth_stencil_lock);

			if (const auto found = g_depth_stencil_states.find(key); found != g_depth_stencil_states.end())
			{
				for (const auto& [cached_desc, state] : found->second)
				{
					if (cached_desc == desc)
					{
						return state;
					}
				}
			}
		}

		std::unique_lock lock(g_depth_stencil_lock);

		auto& bucket = g_depth_stencil_states[key];
		for (const auto& [cached_desc, state] : bucket)
		{
			// Another thread may have created it in the meantime
			if (cached_desc == desc)
			{
				return state;
			}
		}

		@autoreleasepool
		{
			MTLDepthStencilDescriptor* info = [MTLDepthStencilDescriptor new];

			// Same semantics as VK: no depth test means no depth writes either
			info.depthCompareFunction = desc.depth_test ? static_cast<MTLCompareFunction>(desc.depth_func) : MTLCompareFunctionAlways;
			info.depthWriteEnabled = (desc.depth_test && desc.depth_write) ? YES : NO;

			auto make_stencil_descriptor = [](const depth_stencil_desc::stencil_face& face)
			{
				MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
				stencil.stencilCompareFunction = static_cast<MTLCompareFunction>(face.func);
				stencil.stencilFailureOperation = static_cast<MTLStencilOperation>(face.fail_op);
				stencil.depthFailureOperation = static_cast<MTLStencilOperation>(face.zfail_op);
				stencil.depthStencilPassOperation = static_cast<MTLStencilOperation>(face.pass_op);
				stencil.readMask = face.read_mask;
				stencil.writeMask = face.write_mask;
				return stencil;
			};

			if (desc.stencil_test)
			{
				info.frontFaceStencil = make_stencil_descriptor(desc.front);
				info.backFaceStencil = make_stencil_descriptor(desc.back);
			}
			else
			{
				info.frontFaceStencil = nil;
				info.backFaceStencil = nil;
			}

			id<MTLDepthStencilState> state = [get_device() newDepthStencilStateWithDescriptor:info];
			if (!state)
			{
				fmt::throw_exception("Metal: Failed to create depth-stencil state (depth_test=%u, depth_write=%u, func=%u, stencil_test=%u)",
					desc.depth_test, desc.depth_write, desc.depth_func, desc.stencil_test);
			}

			bucket.emplace_back(desc, state);
			return state;
		}
	}

	void clear_depth_stencil_states()
	{
		std::unique_lock lock(g_depth_stencil_lock);
		g_depth_stencil_states.clear();
	}
}
