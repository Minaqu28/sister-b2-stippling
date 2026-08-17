#include "stippling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <immintrin.h>
#include <omp.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__)
#include <cpuid.h>
#endif

namespace stipple {

namespace {
inline void nearest8(const float* px8, const float* py8, const Point* points, int numPoints,
                      float* bestDist8, int* bestIdx8) {
    __m256 bestD = _mm256_set1_ps(std::numeric_limits<float>::max());
    __m256i bestI = _mm256_setzero_si256();
    const __m256 vpx = _mm256_loadu_ps(px8);
    const __m256 vpy = _mm256_loadu_ps(py8);

    for (int p = 0; p < numPoints; ++p) {
        const __m256 vqx = _mm256_set1_ps(points[p].x);
        const __m256 vqy = _mm256_set1_ps(points[p].y);
        const __m256 dx = _mm256_sub_ps(vpx, vqx);
        const __m256 dy = _mm256_sub_ps(vpy, vqy);
        const __m256 d2 = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy));
        const __m256 mask = _mm256_cmp_ps(d2, bestD, _CMP_LT_OQ);
        bestD = _mm256_blendv_ps(bestD, d2, mask);
        const __m256i idxVec = _mm256_set1_epi32(p);
        bestI = _mm256_castps_si256(
            _mm256_blendv_ps(_mm256_castsi256_ps(bestI), _mm256_castsi256_ps(idxVec), mask));
    }

    _mm256_storeu_ps(bestDist8, bestD);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(bestIdx8), bestI);
}

}  // namespace

bool simdAvx2Available() {
    static const bool result = [] {
#if defined(_MSC_VER)
        int info1[4];
        __cpuid(info1, 1);
        const bool osUsesXsave = (info1[2] & (1 << 27)) != 0;
        const bool avxSupported = (info1[2] & (1 << 28)) != 0;
        if (!osUsesXsave || !avxSupported) return false;
        const unsigned long long xcr0 = _xgetbv(0);
        if ((xcr0 & 0x6) != 0x6) return false;  // OS saves/restores XMM and YMM state
        int info7[4];
        __cpuidex(info7, 7, 0);
        return (info7[1] & (1 << 5)) != 0;  // AVX2 bit in EBX
#elif defined(__GNUC__)
        unsigned int eax, ebx, ecx, edx;
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        const bool osUsesXsave = (ecx & (1u << 27)) != 0;
        const bool avxSupported = (ecx & (1u << 28)) != 0;
        if (!osUsesXsave || !avxSupported) return false;
        unsigned int xcr0lo, xcr0hi;
        __asm__ __volatile__("xgetbv" : "=a"(xcr0lo), "=d"(xcr0hi) : "c"(0));
        if ((xcr0lo & 0x6) != 0x6) return false;
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1u << 5)) != 0;
#else
        return false;
#endif
    }();
    return result;
}

Result runCpuParallelSimd(const std::vector<float>& weights, const Params& params,
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
    if (!simdAvx2Available()) {
        return runCpuParallel(weights, params, std::move(result.points), numThreads, observer);
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

    const int simdWidth = (width / 8) * 8;

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
                const float py8[8] = {py, py, py, py, py, py, py, py};

                int x = 0;
                for (; x < simdWidth; x += 8) {
                    float px8[8];
                    for (int k = 0; k < 8; ++k) px8[k] = static_cast<float>(x + k) + 0.5f;

                    float bestDist8[8];
                    int bestIdx8[8];
                    nearest8(px8, py8, pts, numPoints, bestDist8, bestIdx8);

                    for (int k = 0; k < 8; ++k) {
                        const float weight = w[static_cast<size_t>(y) * width + x + k];
                        const int best = bestIdx8[k];
                        lx[best] += static_cast<double>(weight) * px8[k];
                        ly[best] += static_cast<double>(weight) * py8[k];
                        lw[best] += static_cast<double>(weight);
                    }
                }
                for (; x < width; ++x) {
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

}  // namespace stipple
