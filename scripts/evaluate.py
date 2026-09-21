"""
Quantitative Evaluation Suite for Cardiology Angiography Image Processing.
Computes:
  1. Contrast-to-Noise Ratio (CNR): Vessel vs local anatomical background
  2. Background Suppression Ratio (BSR): Rib, spine, and lung clutter reduction
  3. Vessel Preservation: SSIM & Contrast retention inside ground-truth vessel lumen
  4. Dice Similarity Coefficient: Vessel segmentation overlap vs ground truth
  5. Quantum Noise Standard Deviation in homogeneous background ROIs
Generates multi-case visual comparison grids (Easy, Average, Hard).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from cardio_rt.io import load_image, create_comparison_panel, to_8bit_display
from cardio_rt.pipeline import CardioPipeline, PipelineConfig


def compute_cnr(
    img: np.ndarray,
    vessel_mask: np.ndarray,
    dilation_radius: int = 15,
) -> float:
    """
    Compute Contrast-to-Noise Ratio (CNR):
    CNR = |mean(vessel) - mean(background)| / std(background)
    where background is the immediate perivascular local neighborhood.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius, dilation_radius))
    dilated = cv2.dilate(vessel_mask.astype(np.uint8), kernel)
    local_bg_mask = (dilated > 0) & (vessel_mask == 0)

    vessel_pixels = img[vessel_mask > 0].astype(np.float64)
    bg_pixels = img[local_bg_mask].astype(np.float64)

    if vessel_pixels.size == 0 or bg_pixels.size == 0:
        return 0.0

    mu_v = np.mean(vessel_pixels)
    mu_bg = np.mean(bg_pixels)
    sigma_bg = np.std(bg_pixels)

    if sigma_bg < 1e-6:
        return 0.0

    return float(np.abs(mu_v - mu_bg) / sigma_bg)


def compute_background_suppression_ratio(
    original_img: np.ndarray,
    processed_img: np.ndarray,
    vessel_mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute background clutter energy reduction in non-vessel regions:
    BSR = CV(original_bg) / CV(processed_bg)
    Energy reduction % = (1 - CV(proc)^2 / CV(orig)^2) * 100
    """
    bg_mask = (vessel_mask == 0)
    orig_bg = original_img[bg_mask].astype(np.float64)
    proc_bg = processed_img[bg_mask].astype(np.float64)

    cv_orig = float(np.std(orig_bg) / (np.mean(orig_bg) + 1e-6))
    cv_proc = float(np.std(proc_bg) / (np.mean(proc_bg) + 1e-6))

    bsr = float(cv_orig / (cv_proc + 1e-6))
    var_reduction_pct = float(max(0.0, (1.0 - (cv_proc**2) / (cv_orig**2 + 1e-6)) * 100.0))

    return bsr, var_reduction_pct


def compute_vessel_preservation(
    original_img: np.ndarray,
    enhanced_img: np.ndarray,
    vessel_mask: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute SSIM inside vessel mask and vessel contrast retention ratio.
    """
    o_norm = original_img.astype(np.float32) / 65535.0
    e_norm = enhanced_img.astype(np.float32) / 65535.0

    score, ssim_map = ssim(o_norm, e_norm, full=True, data_range=1.0)
    vessel_ssim = float(np.mean(ssim_map[vessel_mask > 0])) if np.sum(vessel_mask) > 0 else float(score)

    v_pixels_orig = o_norm[vessel_mask > 0]
    v_pixels_enh = e_norm[vessel_mask > 0]
    if v_pixels_orig.size > 0:
        retention = float(np.percentile(v_pixels_enh, 95) / (np.percentile(v_pixels_orig, 95) + 1e-6))
    else:
        retention = 1.0

    return vessel_ssim, retention


def compute_dice(
    vesselness_map: np.ndarray,
    gt_mask: np.ndarray,
    threshold: Optional[float] = None,
) -> float:
    """Compute Dice Similarity Coefficient between detected vessel map and ground truth."""
    if threshold is None:
        pos = vesselness_map[vesselness_map > 1e-4]
        threshold = float(np.percentile(pos, 50)) if pos.size > 0 else 0.01

    pred_bin = vesselness_map > threshold
    gt_bin = gt_mask > 0

    intersection = np.sum(pred_bin & gt_bin)
    total = np.sum(pred_bin) + np.sum(gt_bin)
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def compute_noise_std(
    img: np.ndarray,
    roi_coords: Tuple[int, int, int, int] = (20, 20, 60, 60),
) -> float:
    """Compute standard deviation in a flat homogeneous background patch."""
    y1, x1, y2, x2 = roi_coords
    patch = img[y1:y2, x1:x2].astype(np.float64)
    return float(np.std(patch))


def create_evaluation_figure_grid(
    cases: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]],
    output_path: Path,
) -> None:
    """
    Generate clean comparative figure grid:
    Rows: Easy, Average, Hard
    Columns: Original, Processed, Enhanced, Metrics
    """
    num_cases = len(cases)
    fig, axes = plt.subplots(num_cases, 3, figsize=(13, 4.2 * num_cases), dpi=200)

    if num_cases == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, (case_name, (orig, proc, enh, metrics)) in enumerate(cases.items()):
        # Convert to 8-bit for plotting
        o8 = to_8bit_display(orig)
        p8 = to_8bit_display(proc)
        e8 = to_8bit_display(enh)

        axes[row_idx, 0].imshow(o8, cmap="gray")
        axes[row_idx, 0].set_title(f"{case_name}: Original (Raw 16-bit)", fontsize=11, fontweight="bold")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(p8, cmap="gray")
        axes[row_idx, 1].set_title(
            f"Processed (Suppressed)\nBSR: {metrics['bsr']:.1f}x | Noise: -{metrics['noise_reduction_pct']:.0f}%",
            fontsize=10,
        )
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(e8, cmap="gray")
        axes[row_idx, 2].set_title(
            f"Enhanced (Vessel Boost)\nCNR: {metrics['cnr_enhanced']:.1f} (Raw: {metrics['cnr_raw']:.1f}) | Dice: {metrics['dice']:.2f}",
            fontsize=10,
        )
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def evaluate_frame(
    raw_u16: np.ndarray,
    gt_mask_u8: np.ndarray,
    pipeline: CardioPipeline,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Run pipeline and compute all metrics for a single frame."""
    out = pipeline.process_frame(raw_u16)

    # 1. CNR
    cnr_raw = compute_cnr(out.original_u16, gt_mask_u8)
    cnr_proc = compute_cnr(out.processed_u16, gt_mask_u8)
    cnr_enh = compute_cnr(out.enhanced_u16, gt_mask_u8)

    # 2. Background Suppression
    bsr, energy_red = compute_background_suppression_ratio(out.original_u16, out.processed_u16, gt_mask_u8)

    # 3. Vessel Preservation
    v_ssim, contrast_ret = compute_vessel_preservation(out.original_u16, out.enhanced_u16, gt_mask_u8)

    # 4. Dice Score
    vessel_map = out.vesselness_map if out.vesselness_map is not None else np.zeros_like(raw_u16, dtype=np.float32)
    dice_score = compute_dice(vessel_map, gt_mask_u8)

    # 5. Noise std in homogeneous corner
    noise_raw = compute_noise_std(out.original_u16)
    noise_proc = compute_noise_std(out.processed_u16)
    noise_red = max(0.0, (1.0 - noise_proc / (noise_raw + 1e-6)) * 100.0)

    metrics = {
        "cnr_raw": round(cnr_raw, 2),
        "cnr_processed": round(cnr_proc, 2),
        "cnr_enhanced": round(cnr_enh, 2),
        "cnr_improvement_ratio": round(cnr_enh / (cnr_raw + 1e-6), 2),
        "bsr": round(bsr, 2),
        "background_energy_reduction_pct": round(energy_red, 1),
        "vessel_ssim": round(v_ssim, 3),
        "contrast_retention": round(contrast_ret, 2),
        "dice": round(dice_score, 3),
        "noise_std_raw": round(noise_raw, 1),
        "noise_std_proc": round(noise_proc, 1),
        "noise_reduction_pct": round(noise_red, 1),
        "latency_ms": round(out.total_latency_ms, 2),
    }

    return out.processed_u16, out.enhanced_u16, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cardiology image processing pipeline.")
    parser.add_argument("--output_dir", type=str, default="results/evaluation", help="Evaluation output folder")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = CardioPipeline()
    pipeline.warmup((512, 512), 10)

    print("=" * 70)
    print("        CARDIORT QUANTITATIVE EVALUATION SUITE")
    print("=" * 70)

    cases: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]] = {}
    all_metrics: Dict[str, Dict[str, float]] = {}

    # Define test scenarios: Easy (frame 25 standard), Average (frame 15 standard), Hard (frame 15 hard)
    test_specs = [
        ("Easy Case (High Contrast)", "sample_data/cine_512_standard/frames/frame_0025.png", "sample_data/cine_512_standard/masks/mask_0025.png"),
        ("Average Case (Nominal Cine)", "sample_data/cine_512_standard/frames/frame_0015.png", "sample_data/cine_512_standard/masks/mask_0015.png"),
        ("Hard Case (Low Contrast + High Noise)", "sample_data/cine_512_hard/frames/frame_0015.png", "sample_data/cine_512_hard/masks/mask_0015.png"),
    ]

    for label, frame_p, mask_p in test_specs:
        raw_img, _ = load_image(frame_p)
        gt_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_p}")

        proc_u16, enh_u16, metrics = evaluate_frame(raw_img, gt_mask, pipeline)
        cases[label] = (raw_img, proc_u16, enh_u16, metrics)
        all_metrics[label] = metrics

        print(f"\n[{label}] Evaluation Results:")
        print(f"  - CNR (Raw -> Enhanced)       : {metrics['cnr_raw']} -> {metrics['cnr_enhanced']} ({metrics['cnr_improvement_ratio']}x improvement)")
        print(f"  - Background Suppression Ratio: {metrics['bsr']}x ({metrics['background_energy_reduction_pct']}% clutter energy reduction)")
        print(f"  - Vessel SSIM Preservation    : {metrics['vessel_ssim']} (scale 0..1)")
        print(f"  - Dice Similarity Coefficient : {metrics['dice']}")
        print(f"  - Noise Standard Deviation    : {metrics['noise_std_raw']} -> {metrics['noise_std_proc']} (-{metrics['noise_reduction_pct']}%)")
        print(f"  - Per-frame Latency           : {metrics['latency_ms']} ms")

    # Generate Figure Grid
    fig_path = out_dir / "evaluation_figure_grid.png"
    create_evaluation_figure_grid(cases, fig_path)
    print(f"\nSaved evaluation figure grid: {fig_path.resolve()}")

    # Save JSON metrics
    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save Markdown report
    md_lines = [
        "# Quantitative Evaluation Report: CardioRT Performance",
        "",
        "### Key Quantitative Metrics Across Clinical Scenarios",
        "| Scenario | CNR Raw | CNR Enhanced | CNR Boost | BG Suppression Ratio | Clutter Reduction % | Vessel SSIM | Dice Score | Noise Reduction % | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for label, m in all_metrics.items():
        md_lines.append(
            f"| **{label}** | {m['cnr_raw']} | **{m['cnr_enhanced']}** | **{m['cnr_improvement_ratio']}x** | **{m['bsr']}x** | {m['background_energy_reduction_pct']}% | **{m['vessel_ssim']}** | **{m['dice']}** | **-{m['noise_reduction_pct']}%** | {m['latency_ms']} ms |"
        )

    with open(out_dir / "evaluation_summary.md", "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved evaluation markdown report: {(out_dir / 'evaluation_summary.md').resolve()}")


if __name__ == "__main__":
    main()
