#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../algorithm/stippling.hpp"

namespace stipple {

struct RunStats {
    std::string label;
    std::vector<double> timesMs;
    double meanMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    double stddevMs = 0.0;
    int iterationsRun = 0;
    bool converged = false;
    std::vector<Point> finalPoints;
};

RunStats summarize(const std::string& label, std::vector<double> timesMs, int iterationsRun,
                    bool converged, std::vector<Point> finalPoints);

struct BenchmarkMeta {
    std::string inputPath;
    int width = 0;
    int height = 0;
    int numPoints = 0;
    int maxIterations = 0;
    int repeats = 1;
    float epsilon = 0.0f;
    uint32_t seed = 0;
    int cpuThreads = 0;
    std::string cpuModel;
    std::string gpuModel;
    std::string compilerInfo;
    std::string buildConfig;
};

std::string formatReport(const BenchmarkMeta& meta, const std::vector<RunStats>& runs);
float maxPointDelta(const std::vector<Point>& a, const std::vector<Point>& b);

std::string getCpuBrandString();
std::string getCompilerInfo();

}
