#include "runner.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

#include <omp.h>

#include "../image/image.hpp"

namespace stipple {

namespace {

std::string deriveVariantPath(const std::string& base, const std::string& suffix) {
    std::filesystem::path p(base);
    std::filesystem::path dir = p.parent_path();
    std::string stem = p.stem().string();
    std::string ext = p.extension().string();
    if (stem.empty()) stem = "stipple";
    if (ext.empty()) ext = ".png";
    std::filesystem::path result = dir.empty() ? std::filesystem::path(stem + "_" + suffix + ext)
                                                : dir / (stem + "_" + suffix + ext);
    return result.string();
}

std::string deriveVariantAnimationPath(const std::string& base, const std::string& suffix) {
    std::filesystem::path p(base);
    std::filesystem::path dir = p.parent_path();
    std::string stem = p.stem().string();
    if (stem.empty()) stem = "stipple";
    std::filesystem::path result = dir.empty()
                                        ? std::filesystem::path(stem + "_" + suffix + "_animation.gif")
                                        : dir / (stem + "_" + suffix + "_animation.gif");
    return result.string();
}

std::string buildConfigString() {
#ifdef NDEBUG
    return "Release (-O3)";
#else
    return "Debug";
#endif
}

bool saveOrReport(const std::string& path, const Image& img, std::string& errorOut) {
    std::string err;
    if (!saveImagePng(path, img, err)) {
        errorOut = "Error: failed to write output image (" + err + ")";
        return false;
    }
    return true;
}

}  // namespace

bool isGpuBuildAvailable() {
#ifdef STIPPLE_CUDA_ENABLED
    return true;
#else
    return false;
#endif
}

bool isGpuRuntimeAvailable() {
#ifdef STIPPLE_CUDA_ENABLED
    return gpuAvailable();
#else
    return false;
#endif
}

std::string deriveAnimationPath(const std::string& inputPath, const std::string& outputDir) {
    std::filesystem::path in(inputPath);
    std::string stem = in.stem().string();
    if (stem.empty()) stem = "stipple";
    return (std::filesystem::path(outputDir) / (stem + "_animation.gif")).string();
}

SingleRunOutcome runOnce(Mode mode, const std::vector<float>& weights, const Params& params,
                          const std::vector<Point>& initial, int threads, float radius,
                          const std::string& outputPath, const AnimationOptions& animation,
                          const IterationObserver* progressObserver) {
    SingleRunOutcome outcome;

    // Points only, one vector per iteration; rendering is deferred until the
    // algorithm finishes so rasterizing frames never pollutes the timed loop.
    std::vector<std::vector<Point>> capturedStates;
    bool frameCapExceeded = false;

    IterationObserver observer = [&](int idx, const std::vector<Point>& pts) {
        if (animation.enabled) {
            // Cap keeps a long run from holding thousands of frames in
            // memory; capture stops at the limit rather than failing the run.
            if (static_cast<int>(capturedStates.size()) >= animation.maxFrames) {
                frameCapExceeded = true;
            } else {
                capturedStates.push_back(pts);
            }
        }
        if (progressObserver) (*progressObserver)(idx, pts);
    };
    const IterationObserver* observerPtr = (animation.enabled || progressObserver) ? &observer : nullptr;

    if (mode == Mode::Serial) {
        outcome.result = runSerial(weights, params, initial, observerPtr);
        std::ostringstream oss;
        oss << "Serial: " << outcome.result.iterationsRun << " iteration(s), "
            << (outcome.result.converged ? "converged" : "hit max iterations") << ", "
            << outcome.result.elapsedMs << " ms";
        outcome.summaryLine = oss.str();
    } else if (mode == Mode::Cpu) {
        outcome.result = runCpuParallel(weights, params, initial, threads, observerPtr);
        const int threadsUsed = omp_get_max_threads();
        std::ostringstream oss;
        oss << "Parallel CPU (" << threadsUsed << " threads): " << outcome.result.iterationsRun
            << " iteration(s), " << (outcome.result.converged ? "converged" : "hit max iterations") << ", "
            << outcome.result.elapsedMs << " ms";
        outcome.summaryLine = oss.str();
    } else if (mode == Mode::CpuSimd) {
        if (!simdAvx2Available()) {
            outcome.error = "Error: AVX2 is not supported by this CPU";
            return outcome;
        }
        outcome.result = runCpuParallelSimd(weights, params, initial, threads, observerPtr);
        const int threadsUsed = omp_get_max_threads();
        std::ostringstream oss;
        oss << "Parallel CPU + SIMD (AVX2, " << threadsUsed << " threads): "
            << outcome.result.iterationsRun << " iteration(s), "
            << (outcome.result.converged ? "converged" : "hit max iterations") << ", "
            << outcome.result.elapsedMs << " ms";
        outcome.summaryLine = oss.str();
    } else if (mode == Mode::Gpu) {
#ifdef STIPPLE_CUDA_ENABLED
        if (!gpuAvailable()) {
            outcome.error = "Error: CUDA device is unavailable";
            return outcome;
        }
        try {
            GpuTiming timing;
            outcome.result = runGpu(weights, params, initial, &timing, observerPtr);
            std::ostringstream oss;
            oss << "GPU (" << gpuDeviceName() << "): " << outcome.result.iterationsRun << " iteration(s), "
                << (outcome.result.converged ? "converged" : "hit max iterations")
                << ", kernel time: " << timing.kernelMs << " ms, end-to-end: " << timing.endToEndMs << " ms";
            outcome.summaryLine = oss.str();
        } catch (const std::exception& e) {
            outcome.error = std::string("Error: ") + e.what();
            return outcome;
        }
#else
        outcome.error =
            "Error: CUDA device is unavailable (this binary was built without CUDA support -- "
            "install the CUDA Toolkit and rebuild)";
        return outcome;
#endif
    } else {
        outcome.error = "Error: benchmark mode must go through runBenchmarkAll, not runOnce";
        return outcome;
    }

    const Image out = renderStipple(params.width, params.height, outcome.result.points, radius);
    std::string saveErr;
    if (!saveOrReport(outputPath, out, saveErr)) {
        outcome.error = saveErr;
        return outcome;
    }
    outcome.outputPath = outputPath;

    if (animation.enabled) {
        if (capturedStates.empty()) {
            outcome.animationWarning =
                "no iterations ran, so no animation frames were captured; animation not written";
        } else {
            std::vector<Image> frames;
            frames.reserve(capturedStates.size());
            for (const std::vector<Point>& pts : capturedStates) {
                frames.push_back(renderStipple(params.width, params.height, pts, radius));
            }

            std::string animErr;
            if (!saveAnimationGif(animation.outputPath, frames, animation.frameDelayMs,
                                   animation.loopCount, animErr)) {
                // The stipple itself already saved successfully; report this
                // as an animation-specific failure rather than discard the run.
                outcome.error = "Error: failed to write animation '" + animation.outputPath + "' (" +
                                animErr + ")";
                return outcome;
            }
            outcome.animationPath = animation.outputPath;
            outcome.animationFrames = static_cast<int>(frames.size());
            if (frameCapExceeded) {
                outcome.animationWarning = "iteration count exceeded the " +
                                            std::to_string(animation.maxFrames) +
                                            "-frame animation cap; the animation shows the first " +
                                            std::to_string(frames.size()) + " iterations only";
            }
        }
    }

    outcome.ok = true;
    return outcome;
}

BenchmarkOutcome runBenchmarkAll(const std::vector<float>& weights, const Params& params,
                                  const std::vector<Point>& initial, int threads, int repeats,
                                  float radius, const std::string& outputBasePath,
                                  const std::string& inputPath, const AnimationOptions& animation) {
    BenchmarkOutcome bo;

    // Renders and writes the GIF for one mode, if animation is enabled and at
    // least one frame was captured. Frames are only ever captured on a mode's
    // *last* repeat (see the capture-observer wiring in each block below), so
    // this never affects the timings that feed mean/min/max/stddev.
    auto saveAnimationIfCaptured = [&](const std::string& label, const std::string& suffix,
                                        const std::vector<std::vector<Point>>& capturedStates,
                                        bool frameCapExceeded) {
        if (!animation.enabled) return;
        if (capturedStates.empty()) {
            bo.warnings.push_back(label +
                                   ": no iterations ran, so no animation frames were captured; "
                                   "animation not written.");
            return;
        }
        std::vector<Image> frames;
        frames.reserve(capturedStates.size());
        for (const std::vector<Point>& pts : capturedStates) {
            frames.push_back(renderStipple(params.width, params.height, pts, radius));
        }
        const std::string animPath = deriveVariantAnimationPath(outputBasePath, suffix);
        std::string animErr;
        if (saveAnimationGif(animPath, frames, animation.frameDelayMs, animation.loopCount, animErr)) {
            bo.animationPaths.push_back(animPath);
            if (frameCapExceeded) {
                bo.warnings.push_back(label + " animation covers only the first " +
                                       std::to_string(frames.size()) + " iterations (" +
                                       std::to_string(animation.maxFrames) + "-frame cap).");
            }
        } else {
            bo.warnings.push_back(label + ": failed to write animation (" + animErr + ").");
        }
    };

    {
        std::vector<double> times;
        Result last;
        std::vector<std::vector<Point>> capturedStates;
        bool frameCapExceeded = false;
        IterationObserver observer = [&](int, const std::vector<Point>& pts) {
            if (static_cast<int>(capturedStates.size()) >= animation.maxFrames) {
                frameCapExceeded = true;
                return;
            }
            capturedStates.push_back(pts);
        };
        for (int rep = 0; rep < repeats; ++rep) {
            const bool captureThis = animation.enabled && rep == repeats - 1;
            last = runSerial(weights, params, initial, captureThis ? &observer : nullptr);
            times.push_back(last.elapsedMs);
        }
        bo.runs.push_back(summarize("Serial CPU", times, last.iterationsRun, last.converged, last.points));
        const Image out = renderStipple(params.width, params.height, last.points, radius);
        const std::string path = deriveVariantPath(outputBasePath, "serial");
        std::string saveErr;
        if (saveOrReport(path, out, saveErr)) {
            bo.outputPaths.push_back(path);
        } else {
            bo.warnings.push_back(saveErr);
        }
        saveAnimationIfCaptured("Serial", "serial", capturedStates, frameCapExceeded);
    }

    int threadsUsed = 0;
    {
        std::vector<double> times;
        Result last;
        std::vector<std::vector<Point>> capturedStates;
        bool frameCapExceeded = false;
        IterationObserver observer = [&](int, const std::vector<Point>& pts) {
            if (static_cast<int>(capturedStates.size()) >= animation.maxFrames) {
                frameCapExceeded = true;
                return;
            }
            capturedStates.push_back(pts);
        };
        for (int rep = 0; rep < repeats; ++rep) {
            const bool captureThis = animation.enabled && rep == repeats - 1;
            last = runCpuParallel(weights, params, initial, threads, captureThis ? &observer : nullptr);
            times.push_back(last.elapsedMs);
        }
        threadsUsed = omp_get_max_threads();
        bo.runs.push_back(summarize("Parallel CPU (OpenMP)", times, last.iterationsRun, last.converged,
                                     last.points));
        const Image out = renderStipple(params.width, params.height, last.points, radius);
        const std::string path = deriveVariantPath(outputBasePath, "cpu");
        std::string saveErr;
        if (saveOrReport(path, out, saveErr)) {
            bo.outputPaths.push_back(path);
        } else {
            bo.warnings.push_back(saveErr);
        }
        saveAnimationIfCaptured("Parallel CPU", "cpu", capturedStates, frameCapExceeded);
    }

    if (simdAvx2Available()) {
        std::vector<double> times;
        Result last;
        std::vector<std::vector<Point>> capturedStates;
        bool frameCapExceeded = false;
        IterationObserver observer = [&](int, const std::vector<Point>& pts) {
            if (static_cast<int>(capturedStates.size()) >= animation.maxFrames) {
                frameCapExceeded = true;
                return;
            }
            capturedStates.push_back(pts);
        };
        for (int rep = 0; rep < repeats; ++rep) {
            const bool captureThis = animation.enabled && rep == repeats - 1;
            last = runCpuParallelSimd(weights, params, initial, threads, captureThis ? &observer : nullptr);
            times.push_back(last.elapsedMs);
        }
        bo.runs.push_back(summarize("Parallel CPU + SIMD (AVX2)", times, last.iterationsRun,
                                     last.converged, last.points));
        const Image out = renderStipple(params.width, params.height, last.points, radius);
        const std::string path = deriveVariantPath(outputBasePath, "simd");
        std::string saveErr;
        if (saveOrReport(path, out, saveErr)) {
            bo.outputPaths.push_back(path);
        } else {
            bo.warnings.push_back(saveErr);
        }
        saveAnimationIfCaptured("SIMD", "simd", capturedStates, frameCapExceeded);
    } else {
        bo.warnings.push_back("This CPU does not support AVX2; SIMD benchmark skipped.");
    }

    std::string gpuModel;
#ifdef STIPPLE_CUDA_ENABLED
    if (gpuAvailable()) {
        try {
            std::vector<double> times;
            Result last;
            std::vector<std::vector<Point>> capturedStates;
            bool frameCapExceeded = false;
            IterationObserver observer = [&](int, const std::vector<Point>& pts) {
                if (static_cast<int>(capturedStates.size()) >= animation.maxFrames) {
                    frameCapExceeded = true;
                    return;
                }
                capturedStates.push_back(pts);
            };
            for (int rep = 0; rep < repeats; ++rep) {
                const bool captureThis = animation.enabled && rep == repeats - 1;
                GpuTiming t;
                last = runGpu(weights, params, initial, &t, captureThis ? &observer : nullptr);
                times.push_back(t.kernelMs);
            }
            gpuModel = gpuDeviceName();
            bo.runs.push_back(summarize("GPU (CUDA, kernel-only)", times, last.iterationsRun,
                                         last.converged, last.points));
            const Image out = renderStipple(params.width, params.height, last.points, radius);
            const std::string path = deriveVariantPath(outputBasePath, "gpu");
            std::string saveErr;
            if (saveOrReport(path, out, saveErr)) {
                bo.outputPaths.push_back(path);
            } else {
                bo.warnings.push_back(saveErr);
            }
            saveAnimationIfCaptured("GPU", "gpu", capturedStates, frameCapExceeded);
        } catch (const std::exception& e) {
            bo.warnings.push_back(std::string("GPU benchmark failed (") + e.what() +
                                   "); continuing without it.");
        }
    } else {
        bo.warnings.push_back("CUDA device unavailable; skipping GPU benchmark.");
    }
#else
    bo.warnings.push_back(
        "This binary was built without CUDA support; GPU benchmark skipped. Install "
        "the CUDA Toolkit and rebuild for GPU numbers.");
#endif

    bo.meta.inputPath = inputPath;
    bo.meta.width = params.width;
    bo.meta.height = params.height;
    bo.meta.numPoints = params.numPoints;
    bo.meta.maxIterations = params.maxIterations;
    bo.meta.repeats = repeats;
    bo.meta.epsilon = params.epsilon;
    bo.meta.seed = params.seed;
    bo.meta.cpuThreads = threadsUsed;
    bo.meta.cpuModel = getCpuBrandString();
    bo.meta.gpuModel = gpuModel;
    bo.meta.compilerInfo = getCompilerInfo();
    bo.meta.buildConfig = buildConfigString();

    bo.report = formatReport(bo.meta, bo.runs);

    // Written next to the PNGs (same directory as outputBasePath) rather than
    // a separate docs/ folder, so a benchmark run only ever touches one
    // output tree.
    std::filesystem::path reportDir = std::filesystem::path(outputBasePath).parent_path();
    if (reportDir.empty()) reportDir = "output";
    std::error_code ec;
    std::filesystem::create_directories(reportDir, ec);
    bo.reportPath = (reportDir / "benchmark.md").string();
    std::ofstream reportFile(bo.reportPath);
    reportFile << bo.report;
    reportFile.close();

    bo.ok = true;
    return bo;
}

} 
