# Benchmark Results: Real-Time Performance

**Benchmark Run Date:** 2026-09-21 18:58:14 UTC

### Hardware Configuration
- **CPU Model:** 1801           AMD Ryzen 7 5700U with Radeon Graphics
- **CPU Cores:** 8 Physical / 16 Logical
- **System RAM:** 13.85 GB
- **GPU / Accelerator:** CPU SIMD / OpenMP (AMD Ryzen / Intel AVX2 optimized)
- **Operating System:** Windows 10 (10.0.22631)
- **Python:** 3.11.0 | OpenCV: 5.0.0 | NumPy: 2.4.6 | Numba: 0.67.0

### Latency & Throughput Measurements (>= 1,000 Frames Measured)
| Resolution | Frames Tested | Avg Latency (ms) | Min (ms) | Max (ms) | p95 (ms) | p99 (ms) | FPS | Hard Limit (<=36 ms) | Margin Target (p99<=30ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **512x512** | 1000 | **25.22** | 22.22 | **35.67** | 30.59 | **32.31** | **39.6** | **PASSED** | **ACCEPTABLE** |
| **1024x1024** | 1000 | **26.47** | 24.1 | **35.74** | 30.41 | **32.25** | **37.8** | **PASSED** | **ACCEPTABLE** |

### Per-Stage Latency Breakdown (Average ms per Frame)
| Stage Name | 512x512 (ms) | 1024x1024 (ms) | Notes |
| :--- | :---: | :---: | :--- |
| `contrast_enhancement` | 5.448 | 5.243 | 16-bit CLAHE and vessel-guided unsharp mask |
| `coronary_enhancement` | 2.841 | 2.677 | Multi-scale Frangi vesselness with soft gain map |
| `downsample_input` | - | 0.302 | Multiscale working resolution subsampling |
| `ingest_preprocess` | 1.885 | 1.822 | Percentile windowing + lead shutter border detection |
| `log_transform` | 1.656 | 1.622 | Beer-Lambert optical density space mapping |
| `lung_background_suppression` | 5.014 | 4.831 | Laplacian pyramid multi-band attenuation |
| `noise_suppression` | 0.395 | 0.386 | Edge-preserving spatial Gaussian/guided denoise |
| `output_quantization` | 1.277 | 1.238 | Conversion to 16-bit uint16 tensor buffers |
| `rib_suppression` | 1.907 | 1.879 | Large-scale Hessian ridge bone map subtraction |
| `spine_suppression` | 1.84 | 1.796 | Anisotropic vertical spine bone map subtraction |
| `upsample_blend` | - | 1.771 | Full-resolution reconstruction and detail blend |