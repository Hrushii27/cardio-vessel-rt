# Benchmark Results: Real-Time Performance

**Benchmark Run Date:** 2026-09-21 17:25:45 UTC

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
| **512x512** | 1000 | **15.74** | 13.84 | **22.3** | 18.46 | **20.43** | **63.5** | **PASSED** | **PASSED** |
| **1024x1024** | 1000 | **18.93** | 16.4 | **27.85** | 21.31 | **22.95** | **52.8** | **PASSED** | **PASSED** |

### Per-Stage Latency Breakdown (Average ms per Frame)
| Stage Name | 512x512 (ms) | 1024x1024 (ms) | Notes |
| :--- | :---: | :---: | :--- |
| `contrast_enhancement` | 3.425 | 3.586 | 16-bit CLAHE and vessel-guided unsharp mask |
| `coronary_enhancement` | 1.76 | 1.83 | Multi-scale Frangi vesselness with soft gain map |
| `downsample_input` | 0.567 | 1.638 | Multiscale working resolution subsampling |
| `ingest_preprocess` | 1.402 | 1.438 | Percentile windowing + lead shutter border detection |
| `log_transform` | 0.548 | 0.576 | Beer-Lambert optical density space mapping |
| `lung_background_suppression` | 2.169 | 2.309 | Laplacian pyramid multi-band attenuation |
| `noise_suppression` | 0.234 | 0.234 | Edge-preserving spatial Gaussian/guided denoise |
| `output_quantization` | 0.765 | 0.788 | Conversion to 16-bit uint16 tensor buffers |
| `rib_suppression` | 1.158 | 1.174 | Large-scale Hessian ridge bone map subtraction |
| `spine_suppression` | 1.359 | 1.399 | Anisotropic vertical spine bone map subtraction |
| `upsample_blend` | 0.719 | 2.22 | Full-resolution reconstruction and detail blend |