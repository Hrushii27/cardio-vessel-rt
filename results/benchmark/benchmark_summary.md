# Benchmark Results: Real-Time Performance

**Benchmark Run Date:** 2026-09-21 16:11:03 UTC

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
| **512x512** | 1000 | **16.42** | 14.48 | **26.39** | 19.72 | **23.14** | **60.9** | **PASSED** | **PASSED** |
| **1024x1024** | 1000 | **17.87** | 16.68 | **24.18** | 21.19 | **23.3** | **56.0** | **PASSED** | **PASSED** |

### Per-Stage Latency Breakdown (Average ms per Frame)
| Stage Name | 512x512 (ms) | 1024x1024 (ms) | Notes |
| :--- | :---: | :---: | :--- |
| `contrast_enhancement` | 3.536 | 3.357 | 16-bit CLAHE and vessel-guided unsharp mask |
| `coronary_enhancement` | 1.949 | 1.814 | Multi-scale Frangi vesselness with soft gain map |
| `downsample_input` | 0.578 | 1.537 | Multiscale working resolution subsampling |
| `ingest_preprocess` | 1.524 | 1.457 | Percentile windowing + lead shutter border detection |
| `log_transform` | 0.637 | 0.605 | Beer-Lambert optical density space mapping |
| `lung_background_suppression` | 1.96 | 2.036 | Laplacian pyramid multi-band attenuation |
| `noise_suppression` | 0.266 | 0.254 | Edge-preserving spatial Gaussian/guided denoise |
| `output_quantization` | 0.78 | 0.739 | Conversion to 16-bit uint16 tensor buffers |
| `rib_suppression` | 1.234 | 1.174 | Large-scale Hessian ridge bone map subtraction |
| `spine_suppression` | 1.426 | 1.364 | Anisotropic vertical spine bone map subtraction |
| `upsample_blend` | 0.74 | 1.827 | Full-resolution reconstruction and detail blend |