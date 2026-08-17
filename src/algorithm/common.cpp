#include "stippling.hpp"

#include <algorithm>
#include <random>

namespace stipple {

std::vector<Point> initializePoints(const std::vector<float>& weights, int width, int height,
                                     int numPoints, uint32_t seed) {
    std::vector<Point> points;
    if (width <= 0 || height <= 0 || numPoints <= 0) {
        return points;
    }
    points.reserve(static_cast<size_t>(numPoints));

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);

    double maxWeight = 0.0;
    for (size_t i = 0; i < pixelCount; ++i) {
        maxWeight = std::max(maxWeight, static_cast<double>(weights[i]));
    }
    const bool allZero = maxWeight <= 0.0;

    std::vector<double> cdf(pixelCount);
    double running = 0.0;
    for (size_t i = 0; i < pixelCount; ++i) {
        const double w = allZero ? 1.0 : std::max(static_cast<double>(weights[i]), 0.0);
        running += w;
        cdf[i] = running;
    }
    const double total = running;

    std::mt19937 rng(seed);
    const double rngRange = static_cast<double>(std::mt19937::max()) + 1.0;
    auto rand01 = [&rng, rngRange]() -> double { return static_cast<double>(rng()) / rngRange; };

    for (int i = 0; i < numPoints; ++i) {
        const double target = rand01() * total;
        auto it = std::upper_bound(cdf.begin(), cdf.end(), target);
        size_t idx = (it == cdf.end()) ? (pixelCount - 1) : static_cast<size_t>(it - cdf.begin());
        const int px = static_cast<int>(idx % static_cast<size_t>(width));
        const int py = static_cast<int>(idx / static_cast<size_t>(width));
        points.push_back(Point{static_cast<float>(px) + 0.5f, static_cast<float>(py) + 0.5f});
    }
    return points;
}

}
