#include "stippling.hpp"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace stipple {

namespace {

void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error during ") + what + ": " +
                                  cudaGetErrorString(err));
    }
}

__device__ inline void atomicMaxNonNegativeFloat(float* addr, float value) {
    atomicMax(reinterpret_cast<int*>(addr), __float_as_int(value));
}

__global__ void assignAndAccumulateKernel(const float* __restrict__ weights,
                                           const Point* __restrict__ points, int numPoints, int width,
                                           int height, double* __restrict__ sumX,
                                           double* __restrict__ sumY, double* __restrict__ sumW) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixelCount = width * height;
    if (idx >= pixelCount) return;

    const int x = idx % width;
    const int y = idx / width;
    const float w = weights[idx];
    const float px = static_cast<float>(x) + 0.5f;
    const float py = static_cast<float>(y) + 0.5f;

    float bestDistSq = FLT_MAX;
    int best = 0;
    for (int p = 0; p < numPoints; ++p) {
        const Point pt = points[p];
        const float dx = px - pt.x;
        const float dy = py - pt.y;
        const float d2 = dx * dx + dy * dy;
        if (d2 < bestDistSq) {
            bestDistSq = d2;
            best = p;
        }
    }

    atomicAdd(&sumX[best], static_cast<double>(w) * static_cast<double>(px));
    atomicAdd(&sumY[best], static_cast<double>(w) * static_cast<double>(py));
    atomicAdd(&sumW[best], static_cast<double>(w));
}

__global__ void computeCentroidsKernel(const double* __restrict__ sumX, const double* __restrict__ sumY,
                                        const double* __restrict__ sumW, Point* points, int numPoints,
                                        int width, int height, float* maxDisplacement) {
    extern __shared__ float sharedMax[];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float disp = 0.0f;

    if (i < numPoints) {
        const Point old = points[i];
        Point next = old;
        const double w = sumW[i];
        if (w > static_cast<double>(kMinClusterWeight)) {
            next.x = static_cast<float>(sumX[i] / w);
            next.y = static_cast<float>(sumY[i] / w);
        }
        next.x = fminf(fmaxf(next.x, 0.0f), static_cast<float>(width));
        next.y = fminf(fmaxf(next.y, 0.0f), static_cast<float>(height));

        const float dx = next.x - old.x;
        const float dy = next.y - old.y;
        disp = sqrtf(dx * dx + dy * dy);
        points[i] = next;
    }

    sharedMax[threadIdx.x] = disp;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedMax[threadIdx.x] = fmaxf(sharedMax[threadIdx.x], sharedMax[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMaxNonNegativeFloat(maxDisplacement, sharedMax[0]);
    }
}

}  // namespace

bool gpuAvailable() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

std::string gpuDeviceName() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return "unknown GPU";
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return "unknown GPU";
    return std::string(prop.name);
}

Result runGpu(const std::vector<float>& weights, const Params& params, std::vector<Point> initialPoints,
              GpuTiming* timingOut, const IterationObserver* observer) {
    Result result;
    result.points = initialPoints;

    const int width = params.width;
    const int height = params.height;
    const int numPoints = params.numPoints;
    if (numPoints <= 0 || width <= 0 || height <= 0) {
        return result;
    }
    const int pixelCount = width * height;

    cudaEvent_t evStart, evEnd, evKernelStart, evKernelEnd;
    checkCuda(cudaEventCreate(&evStart), "create evStart");
    checkCuda(cudaEventCreate(&evEnd), "create evEnd");
    checkCuda(cudaEventCreate(&evKernelStart), "create evKernelStart");
    checkCuda(cudaEventCreate(&evKernelEnd), "create evKernelEnd");

    checkCuda(cudaEventRecord(evStart), "record evStart");

    float* d_weights = nullptr;
    Point* d_points = nullptr;
    double* d_sumX = nullptr;
    double* d_sumY = nullptr;
    double* d_sumW = nullptr;
    float* d_maxDisplacement = nullptr;

    checkCuda(cudaMalloc(&d_weights, sizeof(float) * static_cast<size_t>(pixelCount)), "malloc weights");
    checkCuda(cudaMalloc(&d_points, sizeof(Point) * static_cast<size_t>(numPoints)), "malloc points");
    checkCuda(cudaMalloc(&d_sumX, sizeof(double) * static_cast<size_t>(numPoints)), "malloc sumX");
    checkCuda(cudaMalloc(&d_sumY, sizeof(double) * static_cast<size_t>(numPoints)), "malloc sumY");
    checkCuda(cudaMalloc(&d_sumW, sizeof(double) * static_cast<size_t>(numPoints)), "malloc sumW");
    checkCuda(cudaMalloc(&d_maxDisplacement, sizeof(float)), "malloc maxDisplacement");

    checkCuda(cudaMemcpy(d_weights, weights.data(), sizeof(float) * static_cast<size_t>(pixelCount),
                          cudaMemcpyHostToDevice),
              "H2D weights");
    checkCuda(cudaMemcpy(d_points, initialPoints.data(), sizeof(Point) * static_cast<size_t>(numPoints),
                          cudaMemcpyHostToDevice),
              "H2D points");

    const int pixelBlockSize = 256;
    const int pixelGridSize = (pixelCount + pixelBlockSize - 1) / pixelBlockSize;
    const int pointBlockSize = 256;
    const int pointGridSize = (numPoints + pointBlockSize - 1) / pointBlockSize;
    const size_t sharedBytes = static_cast<size_t>(pointBlockSize) * sizeof(float);

    checkCuda(cudaEventRecord(evKernelStart), "record evKernelStart");
    double observerMs = 0.0;

    int iter = 0;
    bool converged = false;
    for (; iter < params.maxIterations; ++iter) {
        checkCuda(cudaMemset(d_sumX, 0, sizeof(double) * static_cast<size_t>(numPoints)), "memset sumX");
        checkCuda(cudaMemset(d_sumY, 0, sizeof(double) * static_cast<size_t>(numPoints)), "memset sumY");
        checkCuda(cudaMemset(d_sumW, 0, sizeof(double) * static_cast<size_t>(numPoints)), "memset sumW");
        checkCuda(cudaMemset(d_maxDisplacement, 0, sizeof(float)), "memset maxDisplacement");

        assignAndAccumulateKernel<<<pixelGridSize, pixelBlockSize>>>(
            d_weights, d_points, numPoints, width, height, d_sumX, d_sumY, d_sumW);
        checkCuda(cudaGetLastError(), "assignAndAccumulateKernel launch");

        computeCentroidsKernel<<<pointGridSize, pointBlockSize, sharedBytes>>>(
            d_sumX, d_sumY, d_sumW, d_points, numPoints, width, height, d_maxDisplacement);
        checkCuda(cudaGetLastError(), "computeCentroidsKernel launch");

        float maxDisp = 0.0f;
        checkCuda(cudaMemcpy(&maxDisp, d_maxDisplacement, sizeof(float), cudaMemcpyDeviceToHost),
                  "D2H maxDisplacement");

        // Animation only: an extra D2H copy per iteration, timed on the host
        // and subtracted below so it never counts against the GPU's own time.
        if (observer) {
            const auto obsStart = std::chrono::steady_clock::now();
            checkCuda(cudaMemcpy(result.points.data(), d_points,
                                  sizeof(Point) * static_cast<size_t>(numPoints),
                                  cudaMemcpyDeviceToHost),
                      "D2H points (animation frame)");
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

    checkCuda(cudaEventRecord(evKernelEnd), "record evKernelEnd");
    checkCuda(cudaEventSynchronize(evKernelEnd), "sync evKernelEnd");

    result.points.resize(static_cast<size_t>(numPoints));
    checkCuda(cudaMemcpy(result.points.data(), d_points, sizeof(Point) * static_cast<size_t>(numPoints),
                          cudaMemcpyDeviceToHost),
              "D2H points");

    checkCuda(cudaEventRecord(evEnd), "record evEnd");
    checkCuda(cudaEventSynchronize(evEnd), "sync evEnd");

    float kernelMs = 0.0f;
    float endToEndMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&kernelMs, evKernelStart, evKernelEnd), "elapsed kernel");
    checkCuda(cudaEventElapsedTime(&endToEndMs, evStart, evEnd), "elapsed end-to-end");

    kernelMs = static_cast<float>(std::max(0.0, kernelMs - observerMs));
    endToEndMs = static_cast<float>(std::max(0.0, endToEndMs - observerMs));

    cudaFree(d_weights);
    cudaFree(d_points);
    cudaFree(d_sumX);
    cudaFree(d_sumY);
    cudaFree(d_sumW);
    cudaFree(d_maxDisplacement);
    cudaEventDestroy(evStart);
    cudaEventDestroy(evEnd);
    cudaEventDestroy(evKernelStart);
    cudaEventDestroy(evKernelEnd);

    result.iterationsRun = iter;
    result.converged = converged;
    result.elapsedMs = kernelMs;

    if (timingOut) {
        timingOut->kernelMs = kernelMs;
        timingOut->endToEndMs = endToEndMs;
    }

    return result;
}

}
