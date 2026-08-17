#pragma once

#include <cstdint>
#include <string>

namespace stipple {

enum class Mode { Serial, Cpu, CpuSimd, Gpu, Benchmark };

struct Args {
    std::string inputPath;
    std::string outputPath;
    int numPoints = 0;
    int maxIterations = 0;
    float epsilon = 0.0f;
    Mode mode = Mode::Serial;
    int threads = 0;      // <=0 means "leave OpenMP default"
    uint32_t seed = 42;
    float radius = -1.0f;  // <0 means "compute a default from image size/point count"
    int repeats = 1;       // benchmark repetitions

    // Animation: capture one frame per Lloyd iteration and write an animated
    // GIF. In benchmark mode this produces one GIF per implementation that
    // actually ran, captured only on each implementation's last repeat so
    // repeats before it stay exactly as fast as without --animate.
    bool animate = false;
    std::string animationPath;    // single-run modes only; empty => output/<input-stem>_animation.gif
    int animationDelayMs = 120;   // per-frame display time
};

struct ParseResult {
    bool ok = false;
    bool helpRequested = false;
    std::string error;  // set when !ok && !helpRequested
    Args args;
};

ParseResult parseArgs(int argc, char** argv);

void printHelp(const char* programName);

const char* modeToString(Mode mode);

}
