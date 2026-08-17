#include "interactive.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "../image/image.hpp"
#include "runner.hpp"

namespace stipple {

namespace {

std::string trim(const std::string& s) {
    const size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

bool isQuitCommand(const std::string& text) {
    const std::string t = toLower(trim(text));
    return t == "q" || t == "quit" || t == "exit";
}

bool isSupportedImageFile(const std::filesystem::path& path) {
    static const std::vector<std::string> kSupportedExt = {".png", ".jpg", ".jpeg", ".bmp"};
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec) || ec) return false;
    const std::string ext = toLower(path.extension().string());
    return std::find(kSupportedExt.begin(), kSupportedExt.end(), ext) != kSupportedExt.end();
}

std::vector<std::filesystem::path> listSupportedImages(const std::string& dir) {
    std::vector<std::filesystem::path> result;
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || ec) return result;

    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (isSupportedImageFile(entry.path())) {
            result.push_back(entry.path());
        }
    }
    std::sort(result.begin(), result.end(),
              [](const std::filesystem::path& a, const std::filesystem::path& b) {
                  return a.filename().string() < b.filename().string();
              });
    return result;
}

FolderStatus ensureFolderExists(const std::string& dir, std::string& errorOut) {
    std::error_code ec;
    if (std::filesystem::exists(dir, ec) && !ec) {
        return FolderStatus::Ready;
    }
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        errorOut = ec.message();
        return FolderStatus::CreateFailed;
    }
    return FolderStatus::Created;
}

bool parseIntInRange(const std::string& text, int min, int max, int& out, std::string& errorOut) {
    const std::string trimmed = trim(text);
    if (trimmed.empty()) {
        errorOut = "Input cannot be empty.";
        return false;
    }
    try {
        size_t pos = 0;
        const long value = std::stol(trimmed, &pos);
        if (pos != trimmed.size()) {
            errorOut = "Input must be a whole number (example: 100).";
            return false;
        }
        if (value < min || value > max) {
            errorOut =
                "Input must be between " + std::to_string(min) + " and " + std::to_string(max) + ".";
            return false;
        }
        out = static_cast<int>(value);
        return true;
    } catch (...) {
        errorOut = "Input must be a whole number (example: 100).";
        return false;
    }
}

bool parseFloatInRange(const std::string& text, float min, float max, float& out, std::string& errorOut) {
    const std::string trimmed = trim(text);
    if (trimmed.empty()) {
        errorOut = "Input cannot be empty.";
        return false;
    }
    try {
        size_t pos = 0;
        const float value = std::stof(trimmed, &pos);
        if (pos != trimmed.size()) {
            errorOut = "Input must be a number (example: 0.5).";
            return false;
        }
        if (!std::isfinite(value)) {
            errorOut = "Input must be a finite number (not NaN/Inf).";
            return false;
        }
        if (value < min || value > max) {
            std::ostringstream oss;
            oss << "Input must be between " << min << " and " << max << ".";
            errorOut = oss.str();
            return false;
        }
        out = value;
        return true;
    } catch (...) {
        errorOut = "Input must be a number (example: 0.5).";
        return false;
    }
}

bool promptInt(std::istream& in, std::ostream& out, const std::string& prompt, int min, int max,
               int& result) {
    while (true) {
        out << prompt;
        std::string line;
        if (!std::getline(in, line)) return false;
        if (isQuitCommand(line)) return false;

        std::string error;
        if (parseIntInRange(line, min, max, result, error)) return true;
        out << "  " << error << "\n";
    }
}

bool promptFloat(std::istream& in, std::ostream& out, const std::string& prompt, float min, float max,
                  float& result) {
    while (true) {
        out << prompt;
        std::string line;
        if (!std::getline(in, line)) return false;
        if (isQuitCommand(line)) return false;

        std::string error;
        if (parseFloatInRange(line, min, max, result, error)) return true;
        out << "  " << error << "\n";
    }
}

bool promptImageChoice(std::istream& in, std::ostream& out,
                        const std::vector<std::filesystem::path>& images, size_t& selectedIndex) {
    out << "\nAvailable images in '" << kInteractiveInputDir << "':\n";
    for (size_t i = 0; i < images.size(); ++i) {
        out << "  " << (i + 1) << ". " << images[i].filename().string() << "\n";
    }

    int choice = 0;
    std::ostringstream prompt;
    prompt << "\nSelect an image [1-" << images.size() << ", or 'q' to quit]: ";
    if (!promptInt(in, out, prompt.str(), 1, static_cast<int>(images.size()), choice)) return false;

    selectedIndex = static_cast<size_t>(choice - 1);
    return true;
}

bool promptYesNo(std::istream& in, std::ostream& out, const std::string& prompt, bool defaultYes,
                  bool& result) {
    while (true) {
        out << prompt;
        std::string line;
        if (!std::getline(in, line)) return false;
        if (isQuitCommand(line)) return false;

        const std::string t = toLower(trim(line));
        if (t.empty()) {
            result = defaultYes;
            return true;
        }
        if (t == "y" || t == "yes") {
            result = true;
            return true;
        }
        if (t == "n" || t == "no") {
            result = false;
            return true;
        }
        out << "  Please answer y or n.\n";
    }
}

bool promptMode(std::istream& in, std::ostream& out, Mode& selectedMode) {
    const bool gpuReady = isGpuRuntimeAvailable();
    const bool gpuBuilt = isGpuBuildAvailable();
    const bool simdReady = simdAvx2Available();

    out << "\nSelect execution mode:\n";
    out << "  1. Serial CPU\n";
    out << "  2. Parallel CPU (OpenMP)\n";
    if (simdReady) {
        out << "  3. Parallel CPU + SIMD (AVX2)\n";
    } else {
        out << "  3. Parallel CPU + SIMD (AVX2) -- this CPU does not support AVX2\n";
    }
    if (gpuReady) {
        out << "  4. GPU (CUDA)\n";
    } else if (gpuBuilt) {
        out << "  4. GPU (CUDA) -- no CUDA device detected\n";
    } else {
        out << "  4. GPU (CUDA) -- not available in this build (CUDA Toolkit not found)\n";
    }
    out << "  5. Benchmark all available modes (default)\n";

    while (true) {
        out << "\nEnter choice [1-5, default 5, or 'q' to quit]: ";
        std::string line;
        if (!std::getline(in, line)) return false;
        if (isQuitCommand(line)) return false;

        const std::string trimmed = trim(line);
        if (trimmed.empty()) {
            selectedMode = Mode::Benchmark;
            return true;
        }

        int choice = 0;
        std::string error;
        if (!parseIntInRange(trimmed, 1, 5, choice, error)) {
            out << "  " << error << "\n";
            continue;
        }
        if (choice == 3 && !simdReady) {
            out << "  SIMD is not available on this CPU. Choose another option.\n";
            continue;
        }
        if (choice == 4 && !gpuReady) {
            out << "  GPU is not available right now. Choose another option.\n";
            continue;
        }
        switch (choice) {
            case 1:
                selectedMode = Mode::Serial;
                return true;
            case 2:
                selectedMode = Mode::Cpu;
                return true;
            case 3:
                selectedMode = Mode::CpuSimd;
                return true;
            case 4:
                selectedMode = Mode::Gpu;
                return true;
            default:
                selectedMode = Mode::Benchmark;
                return true;
        }
    }
}

int runInteractive() {
    std::cout << "=== Stipple Me This -- Interactive Mode ===\n";
    std::cout << "(type 'q' at any prompt to quit)\n";

    while (true) {
        std::vector<std::filesystem::path> images;
        while (true) {
            std::string folderErr;
            const FolderStatus status = ensureFolderExists(kInteractiveInputDir, folderErr);
            if (status == FolderStatus::CreateFailed) {
                std::cerr << "Error: could not create folder '" << kInteractiveInputDir << "' ("
                          << folderErr << ")\n";
                return 1;
            }
            if (status == FolderStatus::Created) {
                std::cout << "\nFolder '" << kInteractiveInputDir << "' did not exist -- created it.\n";
            }

            images = listSupportedImages(kInteractiveInputDir);
            if (!images.empty()) break;

            std::cout << "\nNo supported images found in '" << kInteractiveInputDir << "'.\n"
                      << "Supported formats: .png, .jpg, .jpeg, .bmp\n"
                      << "Add image files to that folder, then press Enter to rescan (or type 'q' to "
                         "quit): ";
            std::string line;
            if (!std::getline(std::cin, line)) return 0;
            if (isQuitCommand(line)) return 0;
        }

        size_t selected = 0;
        if (!promptImageChoice(std::cin, std::cout, images, selected)) {
            std::cout << "Cancelled.\n";
            return 0;
        }
        const std::filesystem::path chosenPath = images[selected];

        Image inputImage;
        std::string loadErr;
        if (!loadImage(chosenPath.string(), inputImage, loadErr)) {
            std::cerr << "Error: failed to decode image '" << chosenPath.string() << "' (" << loadErr
                      << ")\n";
            return 1;
        }

        bool uniformFallback = false;
        const std::vector<float> weights = computeWeightMap(inputImage, &uniformFallback);
        if (uniformFallback) {
            std::cout << "Warning: image has no dark pixels; falling back to a uniform point "
                         "distribution.\n";
        }

        int numPoints = 0;
        {
            std::ostringstream prompt;
            prompt << "\nNumber of stippling points [" << kInteractiveMinPoints << "-"
                   << kInteractiveMaxPoints << "]: ";
            if (!promptInt(std::cin, std::cout, prompt.str(), kInteractiveMinPoints,
                            kInteractiveMaxPoints, numPoints)) {
                std::cout << "Cancelled.\n";
                return 0;
            }
        }

        int maxIterations = 0;
        {
            std::ostringstream prompt;
            prompt << "Maximum number of iterations [" << kInteractiveMinIterations << "-"
                   << kInteractiveMaxIterations << "]: ";
            if (!promptInt(std::cin, std::cout, prompt.str(), kInteractiveMinIterations,
                            kInteractiveMaxIterations, maxIterations)) {
                std::cout << "Cancelled.\n";
                return 0;
            }
        }

        float epsilon = 0.0f;
        {
            std::ostringstream prompt;
            prompt << "Epsilon, point-movement convergence threshold in pixels ["
                   << kInteractiveMinEpsilon << "-" << kInteractiveMaxEpsilon << "]: ";
            if (!promptFloat(std::cin, std::cout, prompt.str(), kInteractiveMinEpsilon,
                              kInteractiveMaxEpsilon, epsilon)) {
                std::cout << "Cancelled.\n";
                return 0;
            }
        }

        Mode mode = Mode::Benchmark;
        if (!promptMode(std::cin, std::cout, mode)) {
            std::cout << "Cancelled.\n";
            return 0;
        }

        // In benchmark mode this captures one GIF per implementation, taken
        // only on each implementation's last repeat so it never affects the
        // measured times.
        bool animate = false;
        {
            const std::string prompt = mode == Mode::Benchmark
                                            ? "\nSave an animated GIF for each mode? [y/N]: "
                                            : "\nSave an animated GIF of the iterations? [y/N]: ";
            if (!promptYesNo(std::cin, std::cout, prompt, false, animate)) {
                std::cout << "Cancelled.\n";
                return 0;
            }
        }

        Params params;
        params.width = inputImage.width;
        params.height = inputImage.height;
        params.numPoints = numPoints;
        params.maxIterations = maxIterations;
        params.epsilon = epsilon;
        params.seed = 42;

        const float radius = defaultRadius(params.width, params.height, params.numPoints);
        const std::vector<Point> initial =
            initializePoints(weights, params.width, params.height, params.numPoints, params.seed);

        std::string outputFolderErr;
        if (ensureFolderExists(kInteractiveOutputDir, outputFolderErr) == FolderStatus::CreateFailed) {
            std::cerr << "Error: could not create output folder '" << kInteractiveOutputDir << "' ("
                      << outputFolderErr << ")\n";
            return 1;
        }

        const std::string outputPath =
            std::string(kInteractiveOutputDir) + "/" + chosenPath.stem().string() + "_stipple.png";

        std::cout << "\nRunning Lloyd's Algorithm...\n";

        if (mode == Mode::Benchmark) {
            AnimationOptions benchAnimation;
            benchAnimation.enabled = animate;

            BenchmarkOutcome bo = runBenchmarkAll(weights, params, initial, 0, 1, radius, outputPath,
                                                   chosenPath.string(), benchAnimation);
            for (const std::string& w : bo.warnings) std::cout << "Note: " << w << "\n";
            std::cout << "\n" << bo.report << "\n";
            std::cout << "Benchmark report written to " << bo.reportPath << "\n";
            std::cout << "\nOutput file(s):\n";
            for (const std::string& p : bo.outputPaths) {
                std::error_code ec;
                const auto abs = std::filesystem::absolute(p, ec);
                std::cout << "  - " << (ec ? p : abs.string()) << "\n";
            }
            for (const std::string& p : bo.animationPaths) {
                std::error_code ec;
                const auto abs = std::filesystem::absolute(p, ec);
                std::cout << "  - " << (ec ? p : abs.string()) << "\n";
            }
        } else {
            AnimationOptions animation;
            animation.enabled = animate;
            animation.outputPath =
                deriveAnimationPath(chosenPath.string(), kInteractiveOutputDir);

            SingleRunOutcome outcome =
                runOnce(mode, weights, params, initial, 0, radius, outputPath, animation);
            if (!outcome.ok) {
                std::cerr << outcome.error << "\n";
                return 1;
            }
            std::cout << outcome.summaryLine << "\n";
            if (!outcome.animationWarning.empty()) {
                std::cout << "Note: " << outcome.animationWarning << "\n";
            }
            std::error_code ec;
            const auto abs = std::filesystem::absolute(outcome.outputPath, ec);
            std::cout << "\nOutput saved to: " << (ec ? outcome.outputPath : abs.string()) << "\n";
            if (!outcome.animationPath.empty()) {
                std::error_code aec;
                const auto animAbs = std::filesystem::absolute(outcome.animationPath, aec);
                std::cout << "Animation saved to: "
                          << (aec ? outcome.animationPath : animAbs.string()) << " ("
                          << outcome.animationFrames << " frames)\n";
            }
        }

        std::cout << "\nRun again with another image/parameters? [y/N]: ";
        std::string again;
        if (!std::getline(std::cin, again)) break;
        const std::string a = toLower(trim(again));
        if (a != "y" && a != "yes") break;
        std::cout << "\n";
    }

    std::cout << "\nGoodbye!\n";
    return 0;
}

}
