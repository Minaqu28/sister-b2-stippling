# Benchmark Results

## Methodology

- Input image: `C:\Users\Eduard\Documents\ITB\Seleksi\Sister\BagianB-pt2\sister-b2-stippling\input\DoesHeKnow.png`
- Image dimensions: 960 x 540
- Points: 200000
- Max iterations: 300
- Epsilon: 1 px
- Seed: 42
- Repeats per mode: 1
- CPU model: AMD Ryzen 7 5800H with Radeon Graphics
- CPU threads used (parallel mode): 16
- GPU model: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- Compiler: MSVC 1951
- Build configuration: Release (-O3)
- Timer: `std::chrono::steady_clock` (monotonic); measures the Lloyd iteration loop only -- excludes argument parsing, image load/save, and console output

## Timing

| Implementation | Mean (ms) | Min (ms) | Max (ms) | Stddev (ms) | Iterations | Converged |
|---|---:|---:|---:|---:|---:|:---:|
| Serial CPU | 507826.315 | 507826.315 | 507826.315 | 0.000 | 5 | yes |
| Parallel CPU (OpenMP) | 94679.259 | 94679.259 | 94679.259 | 0.000 | 5 | yes |
| Parallel CPU + SIMD (AVX2) | 12778.043 | 12778.043 | 12778.043 | 0.000 | 5 | yes |
| GPU (CUDA, kernel-only) | 3934.131 | 3934.131 | 3934.131 | 0.000 | 5 | yes |

## Speedup

| Implementation | Time (ms) | Speedup |
|---|---:|---:|
| Serial CPU | 507826.315 | 1.00x |
| Parallel CPU (OpenMP) | 94679.259 | 5.36x |
| Parallel CPU + SIMD (AVX2) | 12778.043 | 39.74x |
| GPU (CUDA, kernel-only) | 3934.131 | 129.08x |

## Correctness Comparison

- Max point-position delta, Serial CPU vs Parallel CPU (OpenMP): 0.0000 px
- Max point-position delta, Serial CPU vs Parallel CPU + SIMD (AVX2): 0.0000 px
- Max point-position delta, Serial CPU vs GPU (CUDA, kernel-only): 0.0000 px
