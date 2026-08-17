#pragma once

#include <string>
#include <vector>

#include "../algorithm/stippling.hpp"
#include "../benchmark/benchmark.hpp"
#include "cli.hpp"

namespace stipple {

// Optional animation capture, shared by runOnce and runBenchmarkAll. Disabled
// by default. In runBenchmarkAll, frames are only ever captured on each
// implementation's last repeat, so the repeats that feed the reported timing
// stats carry none of this cost.
struct AnimationOptions {
    bool enabled = false;
    std::string outputPath;      // e.g. output/DoesHeKnow_animation.gif
    int frameDelayMs = 120;      // per-frame display time (GIF rounds to 10 ms steps)
    int loopCount = 0;           // 0 = loop forever
    int maxFrames = 300;         // safety cap; see runner.cpp for why
};

struct SingleRunOutcome {
    bool ok = false;
    std::string error;        // set when !ok
    Result result;
    std::string summaryLine;  // human-readable one-line summary
    std::string outputPath;   // where the rendered image was saved (when ok)

    // Animation results (only meaningful when AnimationOptions::enabled).
    std::string animationPath;        // written GIF, empty if none
    int animationFrames = 0;          // frames actually captured and written
    std::string animationWarning;     // non-fatal animation problem, run still succeeded
};

// progressObserver, when non-null, is invoked once per completed iteration
// alongside animation capture (if enabled) -- used by the GUI to drive a
// progress bar without paying for frame capture.
SingleRunOutcome runOnce(Mode mode, const std::vector<float>& weights, const Params& params,
                          const std::vector<Point>& initial, int threads, float radius,
                          const std::string& outputPath,
                          const AnimationOptions& animation = AnimationOptions{},
                          const IterationObserver* progressObserver = nullptr);

// Default animation path for an input image: output/<input-stem>_animation.gif
std::string deriveAnimationPath(const std::string& inputPath, const std::string& outputDir);

struct BenchmarkOutcome {
    bool ok = false;
    std::string error;
    std::vector<RunStats> runs;
    BenchmarkMeta meta;
    std::string report;                    // formatted markdown
    std::string reportPath;                 // where `report` was written (benchmark.md next to outputPaths)
    std::vector<std::string> outputPaths;   // one per mode that actually ran
    std::vector<std::string> animationPaths;  // one per mode, only when animation.enabled
    std::vector<std::string> warnings;      // e.g. "GPU unavailable, skipped"
};

// animation, when enabled, captures one GIF per implementation that actually
// runs -- captured only on each implementation's last repeat, so it costs
// nothing on the repeats whose timings feed mean/min/max/stddev.
BenchmarkOutcome runBenchmarkAll(const std::vector<float>& weights, const Params& params,
                                  const std::vector<Point>& initial, int threads, int repeats,
                                  float radius, const std::string& outputBasePath,
                                  const std::string& inputPath,
                                  const AnimationOptions& animation = AnimationOptions{});

bool isGpuBuildAvailable();
bool isGpuRuntimeAvailable();

}
