#include <iostream>

#include "algorithm/stippling.hpp"
#include "image/image.hpp"
#include "utils/cli.hpp"
#include "utils/interactive.hpp"
#include "utils/runner.hpp"

using namespace stipple;

int main(int argc, char** argv) {
    if (argc <= 1) {
        return runInteractive();
    }

    ParseResult parsed = parseArgs(argc, argv);
    if (parsed.helpRequested) {
        printHelp(argc > 0 ? argv[0] : "stipple");
        return 0;
    }
    if (!parsed.ok) {
        std::cerr << parsed.error << "\n";
        return 1;
    }
    const Args args = parsed.args;

    Image inputImage;
    std::string loadErr;
    if (!loadImage(args.inputPath, inputImage, loadErr)) {
        std::cerr << "Error: failed to decode input image (" << loadErr << ")\n";
        return 1;
    }

    bool uniformFallback = false;
    const std::vector<float> weights = computeWeightMap(inputImage, &uniformFallback);
    if (uniformFallback) {
        std::cerr << "Warning: input image has no dark pixels; falling back to a uniform point "
                     "distribution.\n";
    }

    Params params;
    params.width = inputImage.width;
    params.height = inputImage.height;
    params.numPoints = args.numPoints;
    params.maxIterations = args.maxIterations;
    params.epsilon = args.epsilon;
    params.seed = args.seed;

    const float radius =
        args.radius > 0.0f ? args.radius : defaultRadius(params.width, params.height, params.numPoints);

    std::cout << "Loaded " << args.inputPath << " (" << params.width << "x" << params.height << ", "
              << params.numPoints << " points)\n";

    const std::vector<Point> initial =
        initializePoints(weights, params.width, params.height, params.numPoints, params.seed);

    if (args.mode != Mode::Benchmark) {
        AnimationOptions animation;
        animation.enabled = args.animate;
        animation.frameDelayMs = args.animationDelayMs;
        animation.outputPath = args.animationPath.empty()
                                    ? deriveAnimationPath(args.inputPath, "output")
                                    : args.animationPath;

        const SingleRunOutcome outcome = runOnce(args.mode, weights, params, initial, args.threads,
                                                  radius, args.outputPath, animation);
        if (!outcome.ok) {
            std::cerr << outcome.error << "\n";
            return 1;
        }
        std::cout << outcome.summaryLine << "\n";
        std::cout << "Wrote " << outcome.outputPath << "\n";
        if (!outcome.animationWarning.empty()) {
            std::cerr << "Note: " << outcome.animationWarning << "\n";
        }
        if (!outcome.animationPath.empty()) {
            std::cout << "Wrote " << outcome.animationPath << " (" << outcome.animationFrames
                      << " frames)\n";
        }
        return 0;
    }

    AnimationOptions animation;
    animation.enabled = args.animate;
    animation.frameDelayMs = args.animationDelayMs;

    const BenchmarkOutcome bo = runBenchmarkAll(weights, params, initial, args.threads, args.repeats,
                                                 radius, args.outputPath, args.inputPath, animation);
    for (const std::string& w : bo.warnings) std::cerr << "Note: " << w << "\n";
    std::cout << "\n" << bo.report << "\n";
    std::cout << "Report written to " << bo.reportPath << "\n";
    for (const std::string& animPath : bo.animationPaths) {
        std::cout << "Wrote " << animPath << "\n";
    }
    return 0;
}
