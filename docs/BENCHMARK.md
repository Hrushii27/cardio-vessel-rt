# CardioRT: Benchmark Protocol & Hardware Verification Log

## 1. Compliance Certification

The non-negotiable hard requirement dictates:
> **Maximum processing latency $\le 36.0\text{ ms}$ per frame.**
> Internal target: $p99 \le 30.0\text{ ms}$.

CardioRT **PASSED** both requirements on both target resolutions over $\ge 1,000$ consecutive frames:

| Metric | Target Limit | Measured ($512\times 512$) | Measured ($1024\times 1024$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Average Latency** | N/A | **14.67 ms** | **16.66 ms** | **PASSED** |
| **Minimum Latency** | N/A | **13.59 ms** | **15.47 ms** | **PASSED** |
| **Maximum Latency** | $\le 36.0\text{ ms}$ | **18.67 ms** | **21.66 ms** | **PASSED** |
| **p95 Latency** | N/A | **16.35 ms** | **18.98 ms** | **PASSED** |
| **p99 Latency** | $\le 30.0\text{ ms}$ | **17.48 ms** | **20.64 ms** | **PASSED** |
| **Throughput (FPS)**| $\ge 30\text{ FPS}$ | **68.2 FPS** | **60.0 FPS** | **PASSED** |

---

## 2. Hardware Environment Profile
- **Processor:** AMD Ryzen 7 5700U with Radeon Graphics
- **Physical Cores:** 8
- **Logical Processors:** 16
- **Base Clock:** 1.80 GHz (Max Boost: 4.30 GHz)
- **Installed Memory:** 13.85 GB RAM
- **GPU Accelerator:** Integrated AMD Radeon (Pure CPU execution path)
- **Operating System:** Microsoft Windows 11 Home (Version 10.0.22631)
- **Software Toolchain:** Python 3.11.0, OpenCV 5.0.0, NumPy 2.4.6, Numba 0.67.0

---

## 3. Benchmark Verification Command
To reproduce these exact numbers independently on your hardware:
```bash
python scripts/benchmark.py --frames 1000 --output_dir results/benchmark
```
The script will measure all latencies using `time.perf_counter_ns` with garbage collection disabled, output CSV/JSON logs, generate distribution histograms, and exit with code 0 upon compliance.
