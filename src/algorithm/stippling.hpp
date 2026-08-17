#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace stipple {

struct Point {
    float x;
    float y;
};

constexpr float kMinClusterWeight = 1e-6f;

struct Params {
    int width = 0;
    int height = 0;
    int numPoints = 0;
    int maxIterations = 0;
    float epsilon = 0.0f;
    uint32_t seed = 42;
};

struct Result {
    std::vector<Point> points;
    int iterationsRun = 0;
    bool converged = false;
    double elapsedMs = 0.0;
};

std::vector<Point> initializePoints(const std::vector<float>& weights, int width, int height,
                                     int numPoints, uint32_t seed);

using IterationObserver = std::function<void(int iterationIndex, const std::vector<Point>& points)>;

Result runSerial(const std::vector<float>& weights, const Params& params,
                  std::vector<Point> initialPoints, const IterationObserver* observer = nullptr);

Result runCpuParallel(const std::vector<float>& weights, const Params& params,
                       std::vector<Point> initialPoints, int numThreads,
                       const IterationObserver* observer = nullptr);

bool simdAvx2Available();

Result runCpuParallelSimd(const std::vector<float>& weights, const Params& params,
                           std::vector<Point> initialPoints, int numThreads,
                           const IterationObserver* observer = nullptr);

#ifdef STIPPLE_CUDA_ENABLED

struct GpuTiming {
    double kernelMs = 0.0;    // sum of per-iteration kernel time only (cudaEvent-measured)
    double endToEndMs = 0.0;  // kernelMs + one-time H2D upload + final D2H download
};

bool gpuAvailable();
std::string gpuDeviceName();

Result runGpu(const std::vector<float>& weights, const Params& params,
              std::vector<Point> initialPoints, GpuTiming* timingOut = nullptr,
              const IterationObserver* observer = nullptr);

#endif  // STIPPLE_CUDA_ENABLED

}
