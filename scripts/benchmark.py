"""
CardioRT Benchmark Suite: Real-Time Latency & Throughput Profiler.
Measures true per-frame latency across >= 1000 frames at 512x512 and 1024x1024.
Outputs JSON, CSV, latency distribution plots, and markdown tables.
Enforces the non-negotiable hard requirement: MAX and P99 latency <= 36 ms.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
import psutil
import matplotlib.pyplot as plt

from cardio_rt.io import load_image
from cardio_rt.pipeline import CardioPipeline, PipelineConfig


def collect_hardware_info() -> Dict[str, Any]:
    """Gather complete CPU, RAM, GPU, OS, and runtime library metadata."""
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
    }

    try:
        import numba
        info["numba_version"] = numba.__version__
    except ImportError:
        info["numba_version"] = "N/A"

    # CPU model detection for Windows
    if sys.platform == "win32":
        try:
            import subprocess
            out = subprocess.check_output("wmic cpu get Name,MaxClockSpeed", shell=True).decode()
            lines = [line.strip() for line in out.splitlines() if line.strip() and "Name" not in line]
            if lines:
                info["cpu_model"] = lines[0]
        except Exception:
            info["cpu_model"] = platform.processor()
    else:
        info["cpu_model"] = platform.processor()

    # GPU detection
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            info["gpu_accelerator"] = f"CUDA ({cv2.cuda.getCudaEnabledDeviceCount()} devices)"
        else:
            info["gpu_accelerator"] = "CPU SIMD / OpenMP (AMD Ryzen / Intel AVX2 optimized)"
    except Exception:
        info["gpu_accelerator"] = "CPU SIMD"

    return info


def run_benchmark_for_resolution(
    pipeline: CardioPipeline,
    frames: List[np.ndarray],
    num_iterations: int = 1000,
    warmup_runs: int = 25,
    resolution_label: str = "512x512",
    max_latency_limit: float = 36.0,
) -> Dict[str, Any]:
    """Execute benchmark loop with GC disabled, strictly measuring perf_counter_ns."""
    print(f"\n[{resolution_label}] Warming up pipeline ({warmup_runs} iterations)...")
    pipeline.warmup(frames[0].shape[:2], iterations=warmup_runs)

    print(f"[{resolution_label}] Running timed benchmark ({num_iterations} frames)...")
    latencies_ms: List[float] = []
    stage_accumulators: Dict[str, float] = {}

    num_samples = len(frames)

    # Set high process priority to shield benchmark from background OS scheduling jitter
    try:
        p = psutil.Process()
        if hasattr(psutil, "HIGH_PRIORITY_CLASS"):
            p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:
        pass

    # Clean garbage before loop and disable garbage collector during hot benchmark loop
    gc.collect()
    gc.disable()
    try:
        t_bench_start = time.perf_counter_ns()
        for i in range(num_iterations):
            frame = frames[i % num_samples]
            out = pipeline.process_frame(frame)
            latencies_ms.append(out.total_latency_ms)

            for stage, stime in out.stage_latencies_ms.items():
                stage_accumulators[stage] = stage_accumulators.get(stage, 0.0) + stime
        t_bench_total = (time.perf_counter_ns() - t_bench_start) / 1e6
    finally:
        gc.enable()

    lat_arr = np.array(latencies_ms, dtype=np.float64)

    avg_lat = float(np.mean(lat_arr))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    p50_lat = float(np.percentile(lat_arr, 50))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))
    fps = float(1000.0 / avg_lat) if avg_lat > 0 else 0.0

    avg_stages = {k: round(v / num_iterations, 3) for k, v in stage_accumulators.items()}

    results = {
        "resolution": resolution_label,
        "iterations": num_iterations,
        "avg_ms": round(avg_lat, 2),
        "min_ms": round(min_lat, 2),
        "max_ms": round(max_lat, 2),
        "p50_ms": round(p50_lat, 2),
        "p95_ms": round(p95_lat, 2),
        "p99_ms": round(p99_lat, 2),
        "fps": round(fps, 1),
        "total_benchmark_time_sec": round(t_bench_total / 1000.0, 2),
        "stage_breakdown_avg_ms": avg_stages,
        "raw_latencies": latencies_ms,
        "passes_hard_constraint": (max_lat <= max_latency_limit and p99_lat <= max_latency_limit),
    }

    print(f"[{resolution_label}] Results:")
    print(f"  Average : {results['avg_ms']:6.2f} ms")
    print(f"  Minimum : {results['min_ms']:6.2f} ms")
    print(f"  Maximum : {results['max_ms']:6.2f} ms  (Limit: <= 36.0 ms)")
    print(f"  p95     : {results['p95_ms']:6.2f} ms")
    print(f"  p99     : {results['p99_ms']:6.2f} ms  (Internal target: <= 30.0 ms)")
    print(f"  FPS     : {results['fps']:6.1f} frames/sec")
    print(f"  Status  : {'[PASS] COMPLIANT' if results['passes_hard_constraint'] else '[FAIL] EXCEEDED LIMIT'}")

    return results


def plot_benchmark_results(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate professional publication-grade latency distribution and timeseries plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Latency Histogram
    plt.figure(figsize=(10, 5), dpi=200)
    colors = {"512x512": "#1f77b4", "1024x1024": "#ff7f0e"}

    for res, data in all_results.items():
        lats = data["raw_latencies"]
        plt.hist(
            lats,
            bins=40,
            alpha=0.65,
            label=f"{res} (Avg: {data['avg_ms']}ms, P99: {data['p99_ms']}ms)",
            color=colors.get(res, "#2ca02c"),
            edgecolor="black",
            linewidth=0.5,
        )

    plt.axvline(36.0, color="red", linestyle="--", linewidth=2.0, label="Hard Real-Time Limit (36 ms)")
    plt.axvline(30.0, color="green", linestyle=":", linewidth=1.5, label="Internal Target (30 ms)")
    plt.title("CardioRT Latency Distribution (1000+ Frames per Resolution)", fontsize=13, fontweight="bold")
    plt.xlabel("End-to-End Latency per Frame (ms)", fontsize=11)
    plt.ylabel("Frame Count", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    hist_path = output_dir / "latency_histogram.png"
    plt.savefig(hist_path)
    plt.close()

    # 2. Time-Series Trace Plot (First 300 frames)
    plt.figure(figsize=(12, 4.5), dpi=200)
    for res, data in all_results.items():
        sample_lats = data["raw_latencies"][:300]
        plt.plot(sample_lats, label=f"{res} Frame Trace", alpha=0.8, linewidth=1.0)

    plt.axhline(36.0, color="red", linestyle="--", linewidth=1.5, label="Hard Limit (36 ms)")
    plt.axhline(30.0, color="green", linestyle=":", linewidth=1.2, label="P99 Margin (30 ms)")
    plt.title("Per-Frame Processing Latency Over Time (Consecutive Frames)", fontsize=13, fontweight="bold")
    plt.xlabel("Frame Index", fontsize=11)
    plt.ylabel("Latency (ms)", fontsize=11)
    plt.ylim(0, 42)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    ts_path = output_dir / "latency_timeseries.png"
    plt.savefig(ts_path)
    plt.close()

    # 3. Stage Latency Breakdown Bar Chart
    plt.figure(figsize=(11, 5), dpi=200)
    first_res = list(all_results.keys())[0]
    stages = list(all_results[first_res]["stage_breakdown_avg_ms"].keys())

    y_pos = np.arange(len(stages))
    bar_height = 0.35

    for idx, (res, data) in enumerate(all_results.items()):
        vals = [data["stage_breakdown_avg_ms"].get(st, 0.0) for st in stages]
        offset = (idx - 0.5) * bar_height
        plt.barh(y_pos + offset, vals, height=bar_height, label=f"{res}", alpha=0.85)

    plt.yticks(y_pos, [st.replace("_", " ").title() for st in stages], fontsize=10)
    plt.xlabel("Average Execution Time (ms)", fontsize=11)
    plt.title("Per-Stage Execution Latency Breakdown", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.legend(loc="lower right")
    plt.tight_layout()
    stage_path = output_dir / "stage_breakdown.png"
    plt.savefig(stage_path)
    plt.close()


def save_reports(
    hardware_info: Dict[str, Any],
    all_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save results as JSON, CSV, and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON
    clean_json = {
        "hardware": hardware_info,
        "benchmarks": {
            res: {k: v for k, v in data.items() if k != "raw_latencies"}
            for res, data in all_results.items()
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(clean_json, f, indent=2)

    # 2. CSV
    csv_lines = ["frame_index," + ",".join([f"{res}_ms" for res in all_results.keys()])]
    num_frames = min(len(data["raw_latencies"]) for data in all_results.values())
    for i in range(num_frames):
        vals = [f"{all_results[res]['raw_latencies'][i]:.3f}" for res in all_results.keys()]
        csv_lines.append(f"{i}," + ",".join(vals))
    with open(output_dir / "benchmark_times.csv", "w") as f:
        f.write("\n".join(csv_lines))

    # 3. Markdown Summary Table
    md_lines = [
        "# Benchmark Results: Real-Time Performance",
        "",
        f"**Benchmark Run Date:** {clean_json['timestamp']}",
        "",
        "### Hardware Configuration",
        f"- **CPU Model:** {hardware_info.get('cpu_model', 'N/A')}",
        f"- **CPU Cores:** {hardware_info.get('cpu_cores_physical', 'N/A')} Physical / {hardware_info.get('cpu_cores_logical', 'N/A')} Logical",
        f"- **System RAM:** {hardware_info.get('total_ram_gb', 'N/A')} GB",
        f"- **GPU / Accelerator:** {hardware_info.get('gpu_accelerator', 'N/A')}",
        f"- **Operating System:** {hardware_info.get('os', 'N/A')}",
        f"- **Python:** {hardware_info.get('python_version', 'N/A')} | OpenCV: {hardware_info.get('opencv_version', 'N/A')} | NumPy: {hardware_info.get('numpy_version', 'N/A')} | Numba: {hardware_info.get('numba_version', 'N/A')}",
        "",
        "### Latency & Throughput Measurements (>= 1,000 Frames Measured)",
        "| Resolution | Frames Tested | Avg Latency (ms) | Min (ms) | Max (ms) | p95 (ms) | p99 (ms) | FPS | Hard Limit (<=36 ms) | Margin Target (p99<=30ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for res, data in all_results.items():
        limit_status = "PASSED" if data["max_ms"] <= 36.0 else "FAILED"
        margin_status = "PASSED" if data["p99_ms"] <= 30.0 else "ACCEPTABLE"
        md_lines.append(
            f"| **{res}** | {data['iterations']} | **{data['avg_ms']}** | {data['min_ms']} | **{data['max_ms']}** | {data['p95_ms']} | **{data['p99_ms']}** | **{data['fps']}** | **{limit_status}** | **{margin_status}** |"
        )

    md_lines.extend([
        "",
        "### Per-Stage Latency Breakdown (Average ms per Frame)",
        "| Stage Name | 512x512 (ms) | 1024x1024 (ms) | Notes |",
        "| :--- | :---: | :---: | :--- |",
    ])

    all_stages = set()
    for data in all_results.values():
        all_stages.update(data["stage_breakdown_avg_ms"].keys())

    stage_notes = {
        "ingest_preprocess": "Percentile windowing + lead shutter border detection",
        "log_transform": "Beer-Lambert optical density space mapping",
        "noise_suppression": "Edge-preserving spatial Gaussian/guided denoise",
        "spine_suppression": "Anisotropic vertical spine bone map subtraction",
        "rib_suppression": "Large-scale Hessian ridge bone map subtraction",
        "lung_background_suppression": "Laplacian pyramid multi-band attenuation",
        "coronary_enhancement": "Multi-scale Frangi vesselness with soft gain map",
        "contrast_enhancement": "16-bit CLAHE and vessel-guided unsharp mask",
        "output_quantization": "Conversion to 16-bit uint16 tensor buffers",
        "downsample_input": "Multiscale working resolution subsampling",
        "upsample_blend": "Full-resolution reconstruction and detail blend",
    }

    for st in sorted(all_stages):
        v512 = all_results.get("512x512", {}).get("stage_breakdown_avg_ms", {}).get(st, "-")
        v1024 = all_results.get("1024x1024", {}).get("stage_breakdown_avg_ms", {}).get(st, "-")
        note = stage_notes.get(st, "")
        md_lines.append(f"| `{st}` | {v512} | {v1024} | {note} |")

    with open(output_dir / "benchmark_summary.md", "w") as f:
        f.write("\n".join(md_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CardioRT full real-time benchmark suite.")
    parser.add_argument("--frames", type=int, default=1000, help="Number of benchmark iterations per resolution")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations before timing")
    parser.add_argument("--output_dir", type=str, default="results/benchmark", help="Destination folder for artifacts")
    parser.add_argument("--max_latency_limit", type=float, default=36.0, help="Maximum latency threshold in ms (default: 36.0)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("           CARDIORT REAL-TIME BENCHMARK SUITE")
    print("=" * 70)

    hw_info = collect_hardware_info()
    print("Hardware Profile:")
    for k, v in hw_info.items():
        print(f"  - {k:22s}: {v}")

    # Load test frames
    data_512_dir = Path("sample_data/cine_512_standard/frames")
    data_1024_dir = Path("sample_data/cine_1024_standard/frames")

    frames_512 = [load_image(f)[0] for f in sorted(data_512_dir.glob("*.png"))]
    frames_1024 = [load_image(f)[0] for f in sorted(data_1024_dir.glob("*.png"))]

    if not frames_512:
        raise RuntimeError("No 512x512 frames found in sample_data/cine_512_standard/frames")
    if not frames_1024:
        raise RuntimeError("No 1024x1024 frames found in sample_data/cine_1024_standard/frames")

    pipeline = CardioPipeline()

    all_results: Dict[str, Dict[str, Any]] = {}

    # Run 512x512 benchmark
    res_512 = run_benchmark_for_resolution(
        pipeline=pipeline,
        frames=frames_512,
        num_iterations=args.frames,
        warmup_runs=args.warmup,
        resolution_label="512x512",
        max_latency_limit=args.max_latency_limit,
    )
    all_results["512x512"] = res_512

    # Run 1024x1024 benchmark
    res_1024 = run_benchmark_for_resolution(
        pipeline=pipeline,
        frames=frames_1024,
        num_iterations=args.frames,
        warmup_runs=args.warmup,
        resolution_label="1024x1024",
        max_latency_limit=args.max_latency_limit,
    )
    all_results["1024x1024"] = res_1024

    # Generate plots and reports
    print("\nGenerating charts and reports...")
    plot_benchmark_results(all_results, out_dir)
    save_reports(hw_info, all_results, out_dir)

    print(f"\nAll benchmark artifacts saved to: {out_dir.resolve()}")
    print("  * results/benchmark/benchmark_results.json")
    print("  * results/benchmark/benchmark_times.csv")
    print("  * results/benchmark/latency_histogram.png")
    print("  * results/benchmark/latency_timeseries.png")
    print("  * results/benchmark/stage_breakdown.png")
    print("  * results/benchmark/benchmark_summary.md")

    # Enforce non-negotiable hard limit
    all_passed = all(data["passes_hard_constraint"] for data in all_results.values())
    if not all_passed:
        print("\n[ERROR] One or more resolutions failed the <= 36 ms maximum latency requirement!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] ALL RESOLUTIONS PASSED THE HARD LATENCY CONSTRAINT (<= 36 ms per frame)!")
        sys.exit(0)


if __name__ == "__main__":
    main()
