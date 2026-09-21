"""
Command-line interface to run CardioRT pipeline on a single image or image directory.
Exports true 16-bit images, 8-bit display previews, and 3-panel comparison panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

from cardio_rt.io import (
    load_image,
    save_image_16bit,
    save_image_8bit_display,
    save_comparison_panel,
)
from cardio_rt.pipeline import CardioPipeline, PipelineConfig


def process_single_file(
    input_path: str | Path,
    output_dir: str | Path,
    pipeline: CardioPipeline,
) -> None:
    p_in = Path(input_path)
    p_out = Path(output_dir)
    stem = p_in.stem

    # Load 16-bit
    img_u16, meta = load_image(p_in)
    print(f"\nProcessing: {p_in.name} [{img_u16.shape[1]}x{img_u16.shape[0]}], dtype: {meta['original_dtype']}")

    # Measure pipeline execution
    t_start = time.perf_counter_ns()
    out = pipeline.process_frame(img_u16)
    t_end = time.perf_counter_ns()
    io_pipeline_latency_ms = (t_end - t_start) / 1e6

    print(f"  -> Pipeline Core Latency: {out.total_latency_ms:.2f} ms ({1000.0 / out.total_latency_ms:.1f} FPS)")
    print(f"  -> Pipeline + Mem Latency: {io_pipeline_latency_ms:.2f} ms")
    print("  -> Stage Latency Breakdown (ms):")
    for stage, lat in out.stage_latencies_ms.items():
        print(f"     * {stage:30s}: {lat:6.2f} ms")

    # Create destination folders
    dir_16 = p_out / "16bit"
    dir_8 = p_out / "8bit_display"
    dir_comp = p_out / "comparisons"

    # Save 16-bit outputs
    save_image_16bit(dir_16 / f"{stem}_original.png", out.original_u16)
    save_image_16bit(dir_16 / f"{stem}_processed.png", out.processed_u16)
    save_image_16bit(dir_16 / f"{stem}_enhanced.png", out.enhanced_u16)

    # Save 8-bit display previews
    save_image_8bit_display(dir_8 / f"{stem}_original_disp.png", out.original_u16)
    save_image_8bit_display(dir_8 / f"{stem}_processed_disp.png", out.processed_u16)
    save_image_8bit_display(dir_8 / f"{stem}_enhanced_disp.png", out.enhanced_u16)

    # Save 3-panel comparison panel (Original | Processed | Enhanced)
    comp_path = dir_comp / f"{stem}_comparison.png"
    save_comparison_panel(
        comp_path,
        out.original_u16,
        out.processed_u16,
        out.enhanced_u16,
        titles=(
            f"Original ({img_u16.shape[1]}x{img_u16.shape[0]})",
            f"Processed (Suppressed) [{out.total_latency_ms:.1f}ms]",
            "Enhanced (Coronary Boost)",
        ),
    )
    print(f"  Saved comparison panel: {comp_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CardioRT pipeline on single frame or folder.")
    parser.add_argument("--input", type=str, default="sample_data/cine_512_standard/frames/frame_0015.png", help="Input image file or folder")
    parser.add_argument("--output_dir", type=str, default="results/single_frames", help="Output directory")
    args = parser.parse_args()

    in_path = Path(args.input)
    pipeline = CardioPipeline()
    pipeline.warmup((512, 512), 10)

    if in_path.is_file():
        process_single_file(in_path, args.output_dir, pipeline)
    elif in_path.is_dir():
        valid_exts = {".png", ".tif", ".tiff", ".npy"}
        files = sorted([f for f in in_path.iterdir() if f.suffix.lower() in valid_exts])
        print(f"Processing {len(files)} files in directory: {in_path}")
        for f in files[:5]:  # process first 5 for sample CLI run
            process_single_file(f, args.output_dir, pipeline)
    else:
        raise FileNotFoundError(f"Path does not exist: {in_path}")


if __name__ == "__main__":
    main()
