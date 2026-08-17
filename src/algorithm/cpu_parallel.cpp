#include "stippling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <omp.h>

namespace stipple {
Result runCpuParallel(const std::vector<float>& weights, const Params& params,
                       std::vector<Point> initialPoints, int numThreads,
                       const IterationObserver* observer) {
    Result result;
    result.points = std::move(initialPoints);

    const int width = params.width;
    const int height = params.height;
    const int numPoints = params.numPoints;
    if (numPoints <= 0 || width <= 0 || height <= 0) {
        return result;
    }

    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }
    const int actualThreads = std::max(1, omp_get_max_threads());

    std::vector<std::vector<double>> localSumX(actualThreads, std::vector<double>(numPoints));
    std::vector<std::vector<double>> localSumY(actualThreads, std::vector<double>(numPoints));
    std::vector<std::vector<double>> localSumW(actualThreads, std::vector<double>(numPoints));
    std::vector<double> sumX(numPoints), sumY(numPoints), sumW(numPoints);
    std::vector<float> threadMax(actualThreads, 0.0f);

    const auto start = std::chrono::steady_clock::now();
    double observerMs = 0.0;

    int iter = 0;
    bool converged = false;
    for (; iter < params.maxIterations; ++iter) {
        for (int t = 0; t < actualThreads; ++t) {
            std::fill(localSumX[t].begin(), localSumX[t].end(), 0.0);
            std::fill(localSumY[t].begin(), localSumY[t].end(), 0.0);
            std::fill(localSumW[t].begin(), localSumW[t].end(), 0.0);
        }

        const Point* pts = result.points.data();
        const float* w = weights.data();

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double* lx = localSumX[tid].data();
            double* ly = localSumY[tid].data();
            double* lw = localSumW[tid].data();

#pragma omp for schedule(static)
            for (int y = 0; y < height; ++y) {
                const float py = static_cast<float>(y) + 0.5f;
                for (int x = 0; x < width; ++x) {
                    const float weight = w[static_cast<size_t>(y) * width + x];
                    const float px = static_cast<float>(x) + 0.5f;

                    float bestDistSq = std::numeric_limits<float>::max();
                    int best = 0;
                    for (int p = 0; p < numPoints; ++p) {
                        const float dx = px - pts[p].x;
                        const float dy = py - pts[p].y;
                        const float d2 = dx * dx + dy * dy;
                        if (d2 < bestDistSq) {
                            bestDistSq = d2;
                            best = p;
                        }
                    }
                    lx[best] += static_cast<double>(weight) * px;
                    ly[best] += static_cast<double>(weight) * py;
                    lw[best] += static_cast<double>(weight);
                }
            }
        }

#pragma omp parallel for schedule(static)
        for (int p = 0; p < numPoints; ++p) {
            double sx = 0.0, sy = 0.0, sw = 0.0;
            for (int t = 0; t < actualThreads; ++t) {
                sx += localSumX[t][p];
                sy += localSumY[t][p];
                sw += localSumW[t][p];
            }
            sumX[p] = sx;
            sumY[p] = sy;
            sumW[p] = sw;
        }

        std::fill(threadMax.begin(), threadMax.end(), 0.0f);

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            float localMax = 0.0f;

#pragma omp for schedule(static)
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
                localMax = std::max(localMax, disp);
                result.points[p] = next;
            }
            threadMax[tid] = localMax;
        }

        float maxDisp = 0.0f;
        for (float m : threadMax) maxDisp = std::max(maxDisp, m);

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
