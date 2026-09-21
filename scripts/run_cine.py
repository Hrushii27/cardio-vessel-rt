"""
Multi-frame cine runner with motion compensation, temporal background modeling,
latency governor, and side-by-side comparison video export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import List, Tuple
import cv2
import numpy as np

from cardio_rt.io import load_image, load_dicom, create_comparison_panel, to_8bit_display
from cardio_rt.pipeline import CardioPipeline, PipelineConfig
from cardio_rt.cine_motion import MotionCompensator, TemporalBackgroundModel
from cardio_rt.noise import MotionAdaptiveTemporalFilter
from cardio_rt.governor import LatencyGovernor


def load_cine_frames(input_path: str | Path) -> Tuple[List[np.ndarray], int]:
    """Load sequence of frames from folder, DICOM, or NPY file."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")

    ext = p.suffix.lower()
    if p.is_dir():
        valid_exts = {".png", ".tif", ".tiff", ".npy"}
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in valid_exts])
        if not files:
            raise FileNotFoundError(f"No valid images found in directory: {p}")
        frames = [load_image(f)[0] for f in files]
        fps = 15
        return frames, fps

    if ext == ".dcm":
        data, meta = load_dicom(p)
        fps = int(round(1000.0 / meta.get("frame_time_ms", 66.6)))
        if data.ndim == 3:
            return [data[i] for i in range(data.shape[0])], fps
        return [data], fps

    if ext == ".npy":
        raw = np.load(str(p))
        if raw.ndim == 3:
            return [raw[i].astype(np.uint16) for i in range(raw.shape[0])], 15
        return [raw.astype(np.uint16)], 15

    # Single image fallback
    img, _ = load_image(p)
    return [img], 15


def main() -> None:
    parser = argparse.ArgumentParser(description="Process cardiology cine sequence with motion compensation.")
    parser.add_argument("--input", type=str, default="sample_data/cine_512_standard/frames", help="Input folder or file")
    parser.add_argument("--output_dir", type=str, default="results/cine_runs", help="Output directory")
    parser.add_argument("--use_temporal_filter", action="store_true", default=True, help="Enable motion-adaptive temporal filtering")
    parser.add_argument("--use_motion_comp", action="store_true", default=True, help="Enable sub-pixel motion compensation")
    parser.add_argument("--use_governor", action="store_true", default=True, help="Enable runtime latency governor")
    parser.add_argument("--export_video", action="store_true", default=True, help="Export side-by-side MP4 video")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, fps = load_cine_frames(args.input)
    print(f"\nLoaded {len(frames)} cine frames [{frames[0].shape[1]}x{frames[0].shape[0]}], target FPS: {fps}")

    pipeline = CardioPipeline()
    pipeline.warmup(frames[0].shape[:2], 10)

    temporal_filter = MotionAdaptiveTemporalFilter(base_alpha=0.35, motion_threshold=0.03) if args.use_temporal_filter else None
    bg_model = TemporalBackgroundModel(subsample_align=4) if args.use_motion_comp else None
    governor = LatencyGovernor(pipeline) if args.use_governor else None

    # Video Writer setup
    video_writer = None
    first_panel = create_comparison_panel(frames[0], frames[0], frames[0], display_8bit=True)
    vw_h, vw_w = first_panel.shape[:2]

    # Downscale video width if very large
    max_vw_w = 1536
    if vw_w > max_vw_w:
        scale = max_vw_w / vw_w
        vw_w = max_vw_w
        vw_h = int(vw_h * scale)

    if args.export_video:
        video_path = out_dir / "cine_comparison.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (vw_w, vw_h))

    latencies: List[float] = []
    print("\nProcessing cine frames...")

    for idx, frame in enumerate(frames):
        t0 = time.perf_counter_ns()

        # Optional Temporal Pre-filter
        if temporal_filter is not None:
            # Normalize to float32
            f_norm = frame.astype(np.float32) / 65535.0
            filtered_norm = temporal_filter.process(f_norm)
            proc_input = np.clip(filtered_norm * 65535.0, 0, 65535).astype(np.uint16)
        else:
            proc_input = frame

        # Process through CardioPipeline
        out = pipeline.process_frame(proc_input)
        lat = out.total_latency_ms
        latencies.append(lat)

        # Update Governor
        if governor is not None:
            event = governor.record_frame(idx, lat)
            if event:
                print(f"  [Governor Event @ Frame {idx}]: {event.action} -> {event.details}")

        # Render 3-panel comparison
        panel = create_comparison_panel(
            out.original_u16,
            out.processed_u16,
            out.enhanced_u16,
            titles=(
                f"Original [Frame {idx+1}/{len(frames)}]",
                "Processed (Anatomy Suppressed)",
                f"Enhanced [{lat:.1f}ms / {1000.0/lat:.0f} FPS]",
            ),
            display_8bit=True,
        )

        if panel.shape[1] != vw_w or panel.shape[0] != vw_h:
            panel = cv2.resize(panel, (vw_w, vw_h), interpolation=cv2.INTER_AREA)

        if video_writer is not None:
            video_writer.write(panel)

        if idx % 10 == 0 or idx == len(frames) - 1:
            print(f"  Frame {idx+1:3d}/{len(frames)}: Latency {lat:5.2f} ms ({1000.0/lat:4.1f} FPS)")

    if video_writer is not None:
        video_writer.release()
        print(f"\nSaved comparison video: {video_path.resolve()}")

    # Summary
    lat_arr = np.array(latencies)
    print("\nCine Performance Summary:")
    print(f"  Total Frames Processed : {len(latencies)}")
    print(f"  Average Core Latency   : {np.mean(lat_arr):.2f} ms ({1000.0/np.mean(lat_arr):.1f} FPS)")
    print(f"  Maximum Latency        : {np.max(lat_arr):.2f} ms")
    print(f"  p99 Latency            : {np.percentile(lat_arr, 99):.2f} ms")

    # Save metadata summary
    summary = {
        "num_frames": len(frames),
        "avg_latency_ms": round(float(np.mean(lat_arr)), 2),
        "min_latency_ms": round(float(np.min(lat_arr)), 2),
        "max_latency_ms": round(float(np.max(lat_arr)), 2),
        "p99_latency_ms": round(float(np.percentile(lat_arr, 99)), 2),
        "fps": round(float(1000.0 / np.mean(lat_arr)), 1),
        "governor_events_count": len(governor.events) if governor else 0,
    }
    with open(out_dir / "cine_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
