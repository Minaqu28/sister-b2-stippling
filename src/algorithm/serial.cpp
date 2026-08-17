#include "stippling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace stipple {

Result runSerial(const std::vector<float>& weights, const Params& params,
                  std::vector<Point> initialPoints, const IterationObserver* observer) {
    Result result;
    result.points = std::move(initialPoints);

    const int width = params.width;
    const int height = params.height;
    const int numPoints = params.numPoints;
    if (numPoints <= 0 || width <= 0 || height <= 0) {
        return result;
    }

    std::vector<double> sumX(static_cast<size_t>(numPoints));
    std::vector<double> sumY(static_cast<size_t>(numPoints));
    std::vector<double> sumW(static_cast<size_t>(numPoints));

    const auto start = std::chrono::steady_clock::now();
    double observerMs = 0.0;

    int iter = 0;
    bool converged = false;
    for (; iter < params.maxIterations; ++iter) {
        std::fill(sumX.begin(), sumX.end(), 0.0);
        std::fill(sumY.begin(), sumY.end(), 0.0);
        std::fill(sumW.begin(), sumW.end(), 0.0);

        for (int y = 0; y < height; ++y) {
            const float py = static_cast<float>(y) + 0.5f;
            for (int x = 0; x < width; ++x) {
                const float w = weights[static_cast<size_t>(y) * width + x];
                const float px = static_cast<float>(x) + 0.5f;

                float bestDistSq = std::numeric_limits<float>::max();
                int best = 0;
                for (int p = 0; p < numPoints; ++p) {
                    const float dx = px - result.points[p].x;
                    const float dy = py - result.points[p].y;
                    const float d2 = dx * dx + dy * dy;
                    if (d2 < bestDistSq) {
                        bestDistSq = d2;
                        best = p;
                    }
                }
                sumX[best] += static_cast<double>(w) * px;
                sumY[best] += static_cast<double>(w) * py;
                sumW[best] += static_cast<double>(w);
            }
        }

        float maxDisp = 0.0f;
        for (int p = 0; p < numPoints; ++p) {
            const Point old = result.points[p];
            Point next = old;
            if (sumW[p] > kMinClusterWeight) {
                next.x = static_cast<float>(sumX[p] / sumW[p]);
                next.y = static_cast<float>(sumY[p] / sumW[p]);
            }
            next.x = std::clamp(next.x, 0.0f, static_cast<float>(width));
            next.y = std::clamp(next.y, 0.0f, static_cast<float>(height));

            const float dx = next.x - old.x;
            const float dy = next.y - old.y;
            const float disp = std::sqrt(dx * dx + dy * dy);
            maxDisp = std::max(maxDisp, disp);
            result.points[p] = next;
        }

        if (observer) {
            const auto obsStart = std::chrono::steady_clock::now();
            (*observer)(iter, result.points);
            observerMs += std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - obsStart)
                              .count();
        }

        if (maxDisp <= params.epsilon) {
            converged = true;
            ++iter;
            break;
        }
    }

    const auto end = std::chrono::steady_clock::now();
    result.iterationsRun = iter;
    result.converged = converged;
    result.elapsedMs =
        std::chrono::duration<double, std::milli>(end - start).count() - observerMs;
    return result;
}

}
