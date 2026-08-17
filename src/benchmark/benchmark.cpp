#include "benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__)
#include <cpuid.h>
#endif

namespace stipple {

namespace {

std::string fixed(double v, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

std::string trim(const std::string& s) {
    size_t begin = s.find_first_not_of(" \t");
    size_t end = s.find_last_not_of(" \t");
    if (begin == std::string::npos) return "";
    return s.substr(begin, end - begin + 1);
}

}  // namespace

RunStats summarize(const std::string& label, std::vector<double> timesMs, int iterationsRun,
                    bool converged, std::vector<Point> finalPoints) {
    RunStats s;
    s.label = label;
    s.timesMs = timesMs;
    s.iterationsRun = iterationsRun;
    s.converged = converged;
    s.finalPoints = std::move(finalPoints);

    if (!timesMs.empty()) {
        double sum = 0.0;
        s.minMs = timesMs.front();
        s.maxMs = timesMs.front();
        for (double t : timesMs) {
            sum += t;
            s.minMs = std::min(s.minMs, t);
            s.maxMs = std::max(s.maxMs, t);
        }
        s.meanMs = sum / static_cast<double>(timesMs.size());

        double variance = 0.0;
        for (double t : timesMs) variance += (t - s.meanMs) * (t - s.meanMs);
        variance /= static_cast<double>(timesMs.size());
        s.stddevMs = std::sqrt(variance);
    }
    return s;
}

float maxPointDelta(const std::vector<Point>& a, const std::vector<Point>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float maxD = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float dx = a[i].x - b[i].x;
        const float dy = a[i].y - b[i].y;
        maxD = std::max(maxD, std::sqrt(dx * dx + dy * dy));
    }
    return maxD;
}

std::string getCpuBrandString() {
    uint32_t regs[12] = {0};

#if defined(_MSC_VER)
    int info[4] = {0};
    __cpuid(info, 0x80000000);
    const unsigned int maxExtLeaf = static_cast<unsigned int>(info[0]);
    if (maxExtLeaf >= 0x80000004) {
        __cpuid(reinterpret_cast<int*>(regs + 0), 0x80000002);
        __cpuid(reinterpret_cast<int*>(regs + 4), 0x80000003);
        __cpuid(reinterpret_cast<int*>(regs + 8), 0x80000004);
    }
#elif defined(__GNUC__)
    unsigned int a = 0, b = 0, c = 0, d = 0;
    __get_cpuid(0x80000000, &a, &b, &c, &d);
    if (a >= 0x80000004) {
        __get_cpuid(0x80000002, &regs[0], &regs[1], &regs[2], &regs[3]);
        __get_cpuid(0x80000003, &regs[4], &regs[5], &regs[6], &regs[7]);
        __get_cpuid(0x80000004, &regs[8], &regs[9], &regs[10], &regs[11]);
    }
#endif

    char brand[49] = {0};
    std::memcpy(brand, regs, 48);
    std::string s = trim(std::string(brand));
    return s.empty() ? "unknown CPU" : s;
}

std::string getCompilerInfo() {
#if defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_VER);
#else
    return "unknown compiler";
#endif
}

std::string formatReport(const BenchmarkMeta& meta, const std::vector<RunStats>& runs) {
    std::ostringstream out;
    out << "# Benchmark Results\n\n";

    out << "## Methodology\n\n";
    out << "- Input image: `" << meta.inputPath << "`\n";
    out << "- Image dimensions: " << meta.width << " x " << meta.height << "\n";
    out << "- Points: " << meta.numPoints << "\n";
    out << "- Max iterations: " << meta.maxIterations << "\n";
    out << "- Epsilon: " << meta.epsilon << " px\n";
    out << "- Seed: " << meta.seed << "\n";
    out << "- Repeats per mode: " << meta.repeats << "\n";
    out << "- CPU model: " << meta.cpuModel << "\n";
    out << "- CPU threads used (parallel mode): " << meta.cpuThreads << "\n";
    out << "- GPU model: " << (meta.gpuModel.empty() ? "N/A (not run this session)" : meta.gpuModel) << "\n";
    out << "- Compiler: " << meta.compilerInfo << "\n";
    out << "- Build configuration: " << meta.buildConfig << "\n";
    out << "- Timer: `std::chrono::steady_clock` (monotonic); measures the Lloyd iteration loop only "
           "-- excludes argument parsing, image load/save, and console output\n\n";

    out << "## Timing\n\n";
    out << "| Implementation | Mean (ms) | Min (ms) | Max (ms) | Stddev (ms) | Iterations | Converged |\n";
    out << "|---|---:|---:|---:|---:|---:|:---:|\n";
    for (const auto& r : runs) {
        out << "| " << r.label << " | " << fixed(r.meanMs, 3) << " | " << fixed(r.minMs, 3) << " | "
            << fixed(r.maxMs, 3) << " | " << fixed(r.stddevMs, 3) << " | " << r.iterationsRun << " | "
            << (r.converged ? "yes" : "no (hit max iterations)") << " |\n";
    }

    const RunStats* serial = nullptr;
    for (const auto& r : runs) {
        if (r.label == "Serial CPU") {
            serial = &r;
            break;
        }
    }

    out << "\n## Speedup\n\n";
    out << "| Implementation | Time (ms) | Speedup |\n";
    out << "|---|---:|---:|\n";
    for (const auto& r : runs) {
        const double speedup = (serial && r.meanMs > 0.0) ? serial->meanMs / r.meanMs : 1.0;
        out << "| " << r.label << " | " << fixed(r.meanMs, 3) << " | " << fixed(speedup, 2) << "x |\n";
    }

    if (runs.size() >= 2) {
        out << "\n## Correctness Comparison\n\n";
        for (size_t i = 1; i < runs.size(); ++i) {
            const float delta = maxPointDelta(runs[0].finalPoints, runs[i].finalPoints);
            out << "- Max point-position delta, " << runs[0].label << " vs " << runs[i].label << ": "
                << fixed(delta, 4) << " px\n";
        }
    }

    return out.str();
}

}
