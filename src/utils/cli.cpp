#include "cli.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <system_error>

namespace stipple {

namespace {

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool parseFloat(const std::string& s, float& out) {
    try {
        size_t pos = 0;
        out = std::stof(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

bool parseInt(const std::string& s, int& out) {
    try {
        size_t pos = 0;
        out = std::stoi(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

bool parseUint(const std::string& s, uint32_t& out) {
    try {
        size_t pos = 0;
        unsigned long v = std::stoul(s, &pos);
        if (pos != s.size()) return false;
        out = static_cast<uint32_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

const char* modeToString(Mode mode) {
    switch (mode) {
        case Mode::Serial:
            return "serial";
        case Mode::Cpu:
            return "cpu";
        case Mode::CpuSimd:
            return "simd";
        case Mode::Gpu:
            return "gpu";
        case Mode::Benchmark:
            return "benchmark";
    }
    return "unknown";
}

void printHelp(const char* programName) {
    std::cout
        << "Usage: " << programName
        << " --input <path> --points <n> --iterations <n> --epsilon <v> --output <path> [options]\n\n"
        << "Required:\n"
        << "  --input <path>        Path to the source image (PNG/BMP/JPG/...)\n"
        << "  --points <n>          Number of stippling points (n > 0)\n"
        << "  --iterations <n>      Maximum Lloyd iterations (n > 0)\n"
        << "  --epsilon <value>     Convergence threshold in pixels (value >= 0)\n"
        << "  --output <path>       Output image path (PNG)\n\n"
        << "Options:\n"
        << "  --mode <mode>         serial | cpu | simd | gpu | benchmark (default: serial)\n"
        << "  --threads <n>         OpenMP thread count for cpu/simd/benchmark modes (default: OpenMP default)\n"
        << "  --seed <n>            RNG seed for point initialization (default: 42)\n"
        << "  --radius <value>      Output dot radius in pixels (default: auto)\n"
        << "  --repeats <n>         Benchmark repetitions per mode (default: 1)\n"
        << "  --animate             Save an animated GIF of the point positions, one frame\n"
        << "                        per Lloyd iteration (one GIF per mode in benchmark mode,\n"
        << "                        captured on each mode's last repeat only)\n"
        << "  --animation-output <path>  Where to write it; single-run modes only (default:\n"
        << "                        output/<input>_animation.gif)\n"
        << "  --animation-delay <ms>     Per-frame display time (default: 120, GIF rounds to 10ms)\n"
        << "  --help, -h            Show this help message\n\n"
        << "Example:\n"
        << "  " << programName
        << " --input input/DoesHeKnow.png --points 5000 --iterations 50 \\\n"
        << "      --epsilon 0.2 --output output/stipple.png --mode benchmark\n";
}

ParseResult parseArgs(int argc, char** argv) {
    ParseResult result;
    Args& a = result.args;

    bool haveInput = false, haveOutput = false, havePoints = false, haveIterations = false,
         haveEpsilon = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            result.helpRequested = true;
            result.ok = false;
            return result;
        }

        auto requireValue = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                result.error = std::string("Error: missing value for --") + name;
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--input") {
            const char* v = requireValue("input");
            if (!v) return result;
            a.inputPath = v;
            haveInput = true;
        } else if (arg == "--output") {
            const char* v = requireValue("output");
            if (!v) return result;
            a.outputPath = v;
            haveOutput = true;
        } else if (arg == "--points") {
            const char* v = requireValue("points");
            if (!v) return result;
            int n;
            if (!parseInt(v, n)) {
                result.error = "Error: --points must be an integer";
                return result;
            }
            a.numPoints = n;
            havePoints = true;
        } else if (arg == "--iterations") {
            const char* v = requireValue("iterations");
            if (!v) return result;
            int n;
            if (!parseInt(v, n)) {
                result.error = "Error: --iterations must be an integer";
                return result;
            }
            a.maxIterations = n;
            haveIterations = true;
        } else if (arg == "--epsilon") {
            const char* v = requireValue("epsilon");
            if (!v) return result;
            float f;
            if (!parseFloat(v, f)) {
                result.error = "Error: --epsilon must be a number";
                return result;
            }
            a.epsilon = f;
            haveEpsilon = true;
        } else if (arg == "--mode") {
            const char* v = requireValue("mode");
            if (!v) return result;
            const std::string m = toLower(v);
            if (m == "serial")
                a.mode = Mode::Serial;
            else if (m == "cpu")
                a.mode = Mode::Cpu;
            else if (m == "simd")
                a.mode = Mode::CpuSimd;
            else if (m == "gpu")
                a.mode = Mode::Gpu;
            else if (m == "benchmark")
                a.mode = Mode::Benchmark;
            else {
                result.error = "Error: invalid --mode '" + std::string(v) +
                                "' (expected serial, cpu, simd, gpu, or benchmark)";
                return result;
            }
        } else if (arg == "--threads") {
            const char* v = requireValue("threads");
            if (!v) return result;
            int n;
            if (!parseInt(v, n) || n <= 0) {
                result.error = "Error: --threads must be a positive integer";
                return result;
            }
            a.threads = n;
        } else if (arg == "--seed") {
            const char* v = requireValue("seed");
            if (!v) return result;
            uint32_t n;
            if (!parseUint(v, n)) {
                result.error = "Error: --seed must be a non-negative integer";
                return result;
            }
            a.seed = n;
        } else if (arg == "--radius") {
            const char* v = requireValue("radius");
            if (!v) return result;
            float f;
            if (!parseFloat(v, f) || f <= 0.0f) {
                result.error = "Error: --radius must be a positive number";
                return result;
            }
            a.radius = f;
        } else if (arg == "--repeats") {
            const char* v = requireValue("repeats");
            if (!v) return result;
            int n;
            if (!parseInt(v, n) || n <= 0) {
                result.error = "Error: --repeats must be a positive integer";
                return result;
            }
            a.repeats = n;
        } else if (arg == "--animate") {
            a.animate = true;
        } else if (arg == "--animation-output") {
            const char* v = requireValue("animation-output");
            if (!v) return result;
            a.animationPath = v;
        } else if (arg == "--animation-delay") {
            const char* v = requireValue("animation-delay");
            if (!v) return result;
            int n;
            if (!parseInt(v, n) || n <= 0) {
                result.error = "Error: --animation-delay must be a positive integer (milliseconds)";
                return result;
            }
            a.animationDelayMs = n;
        } else {
            result.error = "Error: unknown argument '" + arg + "'";
            return result;
        }
    }

    if (!haveInput) {
        result.error = "Error: --input is required";
        return result;
    }
    if (!haveOutput) {
        result.error = "Error: --output is required";
        return result;
    }
    if (!havePoints) {
        result.error = "Error: --points is required";
        return result;
    }
    if (!haveIterations) {
        result.error = "Error: --iterations is required";
        return result;
    }
    if (!haveEpsilon) {
        result.error = "Error: --epsilon is required";
        return result;
    }

    if (a.numPoints <= 0) {
        result.error = "Error: number of points must be greater than 0";
        return result;
    }
    if (a.maxIterations <= 0) {
        result.error = "Error: iterations must be greater than 0";
        return result;
    }
    if (a.epsilon < 0.0f) {
        result.error = "Error: epsilon must be non-negative";
        return result;
    }

    std::error_code ec;
    if (!std::filesystem::exists(a.inputPath, ec) || ec) {
        result.error = "Error: input image does not exist";
        return result;
    }

    result.ok = true;
    return result;
}

}
