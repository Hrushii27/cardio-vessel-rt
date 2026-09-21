"""
Ablation Study Runner for CardioRT.
Evaluates the quantitative impact of each individual module on:
  - Contrast-to-Noise Ratio (CNR)
  - Background Suppression Ratio (BSR)
  - Execution Latency (ms)
Outputs a comprehensive markdown table and JSON record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Any, List
import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cardio_rt.io import load_image
from cardio_rt.pipeline import CardioPipeline, PipelineConfig
from scripts.evaluate import compute_cnr, compute_background_suppression_ratio


def run_ablation() -> Dict[str, Any]:
    sample_file = Path("sample_data/cine_512_standard/frames/frame_0015.png")
    mask_file = Path("sample_data/cine_512_standard/masks/mask_0015.png")

    raw_img, _ = load_image(sample_file)
    gt_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

    experiments = [
        ("Full Pipeline (All Active)", PipelineConfig()),
        ("w/o Noise Suppression", PipelineConfig(enable_noise_suppression=False)),
        ("w/o Spine Suppression", PipelineConfig(enable_spine_suppression=False)),
        ("w/o Rib Suppression", PipelineConfig(enable_rib_suppression=False)),
        ("w/o Lung Background Suppression", PipelineConfig(enable_lung_background_suppression=False)),
        ("w/o Coronary Enhancement", PipelineConfig(enable_coronary_enhancement=False)),
        ("w/o Contrast Enhancement", PipelineConfig(enable_contrast_enhancement=False)),
        ("Baseline (No Suppression/Enhancement)", PipelineConfig(
            enable_noise_suppression=False,
            enable_spine_suppression=False,
            enable_rib_suppression=False,
            enable_lung_background_suppression=False,
            enable_coronary_enhancement=False,
            enable_contrast_enhancement=False,
        )),
    ]

    results: List[Dict[str, Any]] = []

    print("=" * 70)
    print("                 CARDIORT ABLATION STUDY")
    print("=" * 70)

    for name, cfg in experiments:
        pipeline = CardioPipeline(cfg)
        pipeline.warmup((512, 512), iterations=10)

        # Profile latency across 20 iterations
        lats: List[float] = []
        for _ in range(20):
            out = pipeline.process_frame(raw_img)
            lats.append(out.total_latency_ms)

        avg_lat = float(np.mean(lats))
        cnr_enh = compute_cnr(out.enhanced_u16, gt_mask)
        bsr, energy_red = compute_background_suppression_ratio(out.original_u16, out.processed_u16, gt_mask)

        row = {
            "experiment": name,
            "latency_ms": round(avg_lat, 2),
            "cnr_enhanced": round(cnr_enh, 2),
            "bsr": round(bsr, 2),
            "clutter_reduction_pct": round(energy_red, 1),
        }
        results.append(row)
        print(f"{name:40s} | Latency: {avg_lat:5.2f} ms | CNR: {cnr_enh:4.2f} | BSR: {bsr:4.2f}x")

    return {"ablation_results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CardioRT ablation study.")
    parser.add_argument("--output_dir", type=str, default="results/ablation", help="Destination folder")
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    data = run_ablation()

    # Save JSON
    with open(out_p / "ablation_results.json", "w") as f:
        json.dump(data, f, indent=2)

    # Save Markdown
    md_lines = [
        "# CardioRT Ablation Study: Component Impact Analysis",
        "",
        "| Pipeline Configuration | Avg Latency (ms) | Enhanced CNR | Background Suppression Ratio | Clutter Reduction % | Primary Impact |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]

    notes = {
        "Full Pipeline (All Active)": "Optimal balance of anatomical suppression and vessel visibility",
        "w/o Noise Suppression": "Elevated high-frequency background quantum mottle",
        "w/o Spine Suppression": "Prominent vertebral column shadow retained behind LCA",
        "w/o Rib Suppression": "Dense oblique rib ridges interfering with diagonal branches",
        "w/o Lung Background Suppression": "Strong lateral non-uniformities from thoracic density gradient",
        "w/o Coronary Enhancement": "Loss of distal branch contrast boost (faint vessels remain dark)",
        "w/o Contrast Enhancement": "Reduced dynamic range differentiation between lumen and tissue",
        "Baseline (No Suppression/Enhancement)": "Unfiltered optical density baseline",
    }

    for row in data["ablation_results"]:
        exp = row["experiment"]
        note = notes.get(exp, "")
        md_lines.append(
            f"| **{exp}** | {row['latency_ms']} ms | **{row['cnr_enhanced']}** | **{row['bsr']}x** | {row['clutter_reduction_pct']}% | {note} |"
        )

    with open(out_p / "ablation_table.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"\nAblation table saved to: {(out_p / 'ablation_table.md').resolve()}")


if __name__ == "__main__":
    main()
