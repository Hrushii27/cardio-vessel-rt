"""
Synthetic True 16-bit Cardiology Cine & Frame Generator.
Procedurally synthesizes coronary artery trees with contrast inflow, ribs, spine,
lung fields, diaphragm, catheter, cardiac/respiratory motion, and Poisson noise.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2


class BranchSegment:
    """A spline/parametric vessel branch segment."""
    def __init__(
        self,
        start_pt: Tuple[float, float],
        end_pt: Tuple[float, float],
        ctrl_pt: Tuple[float, float],
        start_radius: float,
        end_radius: float,
        inflow_start_frame: int,
        inflow_duration: int,
        contrast_density: float = 0.85,
    ):
        self.start_pt = start_pt
        self.end_pt = end_pt
        self.ctrl_pt = ctrl_pt
        self.start_radius = start_radius
        self.end_radius = end_radius
        self.inflow_start_frame = inflow_start_frame
        self.inflow_duration = inflow_duration
        self.contrast_density = contrast_density

    def sample_curve(self, num_pts: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample quadratic Bezier points and interpolated radii."""
        t = np.linspace(0.0, 1.0, num_pts)
        p0 = np.array(self.start_pt)
        p1 = np.array(self.ctrl_pt)
        p2 = np.array(self.end_pt)
        # B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
        b = ((1.0 - t)[:, None] ** 2) * p0 + (2.0 * (1.0 - t)[:, None] * t[:, None]) * p1 + (t[:, None] ** 2) * p2
        radii = (1.0 - t) * self.start_radius + t * self.end_radius
        return b[:, 0], b[:, 1], radii


def build_coronary_tree(scale: float = 1.0) -> List[BranchSegment]:
    """
    Construct realistic Left Coronary Artery tree:
    Main trunk -> LAD (proximal, mid, distal) with diagonals (D1, D2)
    and LCx (proximal, obtuse marginal OM1, distal).
    """
    s = scale
    tree = [
        # Left Main Trunk (LMCA)
        BranchSegment(
            start_pt=(210 * s, 100 * s),
            end_pt=(240 * s, 170 * s),
            ctrl_pt=(220 * s, 135 * s),
            start_radius=5.5 * s,
            end_radius=4.8 * s,
            inflow_start_frame=0,
            inflow_duration=4,
            contrast_density=0.90,
        ),
        # Proximal LAD
        BranchSegment(
            start_pt=(240 * s, 170 * s),
            end_pt=(270 * s, 260 * s),
            ctrl_pt=(250 * s, 215 * s),
            start_radius=4.5 * s,
            end_radius=3.8 * s,
            inflow_start_frame=3,
            inflow_duration=5,
            contrast_density=0.88,
        ),
        # Mid LAD
        BranchSegment(
            start_pt=(270 * s, 260 * s),
            end_pt=(310 * s, 370 * s),
            ctrl_pt=(295 * s, 315 * s),
            start_radius=3.8 * s,
            end_radius=2.8 * s,
            inflow_start_frame=7,
            inflow_duration=6,
            contrast_density=0.85,
        ),
        # Distal LAD (thin distal vessel)
        BranchSegment(
            start_pt=(310 * s, 370 * s),
            end_pt=(335 * s, 450 * s),
            ctrl_pt=(325 * s, 410 * s),
            start_radius=2.6 * s,
            end_radius=1.5 * s,
            inflow_start_frame=12,
            inflow_duration=6,
            contrast_density=0.75,
        ),
        # Diagonal branch 1 (D1)
        BranchSegment(
            start_pt=(260 * s, 225 * s),
            end_pt=(360 * s, 290 * s),
            ctrl_pt=(310 * s, 245 * s),
            start_radius=3.0 * s,
            end_radius=1.8 * s,
            inflow_start_frame=6,
            inflow_duration=5,
            contrast_density=0.80,
        ),
        # Diagonal branch 2 (D2)
        BranchSegment(
            start_pt=(290 * s, 315 * s),
            end_pt=(375 * s, 385 * s),
            ctrl_pt=(340 * s, 345 * s),
            start_radius=2.4 * s,
            end_radius=1.4 * s,
            inflow_start_frame=10,
            inflow_duration=5,
            contrast_density=0.75,
        ),
        # Left Circumflex (LCx) proximal
        BranchSegment(
            start_pt=(240 * s, 170 * s),
            end_pt=(175 * s, 245 * s),
            ctrl_pt=(195 * s, 195 * s),
            start_radius=4.2 * s,
            end_radius=3.2 * s,
            inflow_start_frame=4,
            inflow_duration=5,
            contrast_density=0.85,
        ),
        # Obtuse Marginal 1 (OM1)
        BranchSegment(
            start_pt=(175 * s, 245 * s),
            end_pt=(130 * s, 350 * s),
            ctrl_pt=(145 * s, 290 * s),
            start_radius=3.0 * s,
            end_radius=1.8 * s,
            inflow_start_frame=8,
            inflow_duration=6,
            contrast_density=0.80,
        ),
        # Distal LCx
        BranchSegment(
            start_pt=(175 * s, 245 * s),
            end_pt=(200 * s, 340 * s),
            ctrl_pt=(180 * s, 295 * s),
            start_radius=2.8 * s,
            end_radius=1.6 * s,
            inflow_start_frame=8,
            inflow_duration=6,
            contrast_density=0.78,
        ),
    ]
    return tree


def render_vessels(
    shape: Tuple[int, int],
    tree: List[BranchSegment],
    frame_idx: int,
    cardiac_phase: float,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render coronary vessels with cardiac pulsation / motion deformation and contrast inflow.
    Returns:
        (optical_density_map, ground_truth_binary_mask)
    """
    h, w = shape
    od_map = np.zeros((h, w), dtype=np.float32)
    gt_mask = np.zeros((h, w), dtype=np.uint8)

    # Cardiac contraction deformation field (radial toward center of heart + slight twist)
    heart_cx, heart_cy = 260 * scale, 280 * scale
    motion_amp = 8.0 * scale * math.sin(cardiac_phase)

    for seg in tree:
        # Inflow progress [0.0, 1.0]
        if frame_idx < seg.inflow_start_frame:
            progress = 0.0
        elif frame_idx >= seg.inflow_start_frame + seg.inflow_duration:
            progress = 1.0
        else:
            progress = (frame_idx - seg.inflow_start_frame) / float(seg.inflow_duration)

        if progress <= 0.01:
            continue

        xs, ys, radii = seg.sample_curve(num_pts=80)
        num_active = max(2, int(round(len(xs) * progress)))

        for k in range(num_active - 1):
            x1, y1, r1 = xs[k], ys[k], radii[k]
            x2, y2, r2 = xs[k + 1], ys[k + 1], radii[k + 1]

            # Apply cardiac motion offset
            dx1 = (x1 - heart_cx) / (w + 1e-4) * motion_amp
            dy1 = (y1 - heart_cy) / (h + 1e-4) * motion_amp
            dx2 = (x2 - heart_cx) / (w + 1e-4) * motion_amp
            dy2 = (y2 - heart_cy) / (h + 1e-4) * motion_amp

            pt1 = (int(round(x1 + dx1)), int(round(y1 + dy1)))
            pt2 = (int(round(x2 + dx2)), int(round(y2 + dy2)))
            thickness = max(1, int(round((r1 + r2))))

            # Optical density: parabolic cross-section (Beer-Lambert attenuation of cylinder)
            cv2.line(od_map, pt1, pt2, float(seg.contrast_density), thickness)
            cv2.line(gt_mask, pt1, pt2, 255, thickness)

    # Slight blur to create natural continuous vessel cross-section
    if scale >= 1.5:
        cv2.GaussianBlur(od_map, (5, 5), 1.0 * scale, dst=od_map)
    else:
        cv2.GaussianBlur(od_map, (3, 3), 0.7, dst=od_map)

    return od_map, (gt_mask > 127).astype(np.uint8)


def render_anatomy(
    shape: Tuple[int, int],
    respiration_phase: float,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Render anatomical structures in optical density space:
      - Spine (vertical column)
      - Ribs (oblique curved bands)
      - Lung fields (bilateral low density)
      - Diaphragm edge (dome moving with breathing)
    Returns:
        (spine_od, rib_od, lung_od, total_bg_od)
    """
    h, w = shape
    spine_od = np.zeros((h, w), dtype=np.float32)
    rib_od = np.zeros((h, w), dtype=np.float32)
    lung_od = np.zeros((h, w), dtype=np.float32)

    # Respiratory vertical drift: 6 pixels
    resp_shift = 6.0 * scale * math.sin(respiration_phase)

    # 1. Spine: vertical column around x = 200..260 with intervertebral bodies
    spine_cx = 220.0 * scale
    spine_half_w = 40.0 * scale
    x_grid = np.arange(w, dtype=np.float32)[np.newaxis, :]
    y_grid = np.arange(h, dtype=np.float32)[:, np.newaxis]

    # Transverse spine profile
    spine_profile = np.exp(-0.5 * ((x_grid - spine_cx) / spine_half_w) ** 2)
    # Vertebral body segmentations (periodic modulation along Y)
    vert_mod = 0.85 + 0.15 * np.cos(2.0 * np.pi * (y_grid - resp_shift) / (45.0 * scale))
    spine_od = (0.75 * spine_profile * vert_mod).astype(np.float32)

    # 2. Ribs: 4 oblique curved bands crossing thoracic cage
    rib_angles = [-18.0, -15.0, -12.0, -10.0]
    rib_y_bases = [90.0, 190.0, 290.0, 390.0]
    for angle, y_base in zip(rib_angles, rib_y_bases):
        rad = math.radians(angle)
        # Rib centerline: y - y0 = tan(angle) * (x - x0) + curvature * (x - x0)^2
        y0 = (y_base * scale) + resp_shift
        rib_line = y0 + math.tan(rad) * (x_grid - w * 0.5) + (0.0006 / scale) * (x_grid - w * 0.5) ** 2
        rib_dist = np.abs(y_grid - rib_line)
        rib_width = 16.0 * scale
        rib_band = np.exp(-0.5 * (rib_dist / (rib_width * 0.5)) ** 2)
        rib_od += (0.45 * rib_band).astype(np.float32)

    # 3. Lung fields: smooth lateral gradients (lungs are radiolucent -> low X-ray density)
    lung_row = (0.35 + 0.30 * (1.0 - np.exp(-0.5 * ((x_grid - w * 0.5) / (w * 0.35)) ** 2))).astype(np.float32)
    lung_od = np.repeat(lung_row, h, axis=0)

    # 4. Diaphragm: dome at lower right
    diaph_y = (h * 0.78) + resp_shift - (w * 0.12) * np.sin((x_grid / w) * np.pi)
    diaph_mask = 1.0 / (1.0 + np.exp(-(y_grid - diaph_y) / (12.0 * scale)))
    lung_od += (0.50 * diaph_mask).astype(np.float32)

    total_bg_od = spine_od + rib_od + lung_od
    return spine_od, rib_od, lung_od, total_bg_od


def render_catheter(
    shape: Tuple[int, int],
    tip_pt: Tuple[int, int],
    scale: float = 1.0,
) -> np.ndarray:
    """Render radio-opaque guide catheter entering coronary ostium."""
    h, w = shape
    catheter_od = np.zeros((h, w), dtype=np.float32)
    # Catheter enters from top border
    pts = np.array([
        [int(w * 0.46), 0],
        [int(w * 0.44), int(h * 0.10)],
        [tip_pt[0], tip_pt[1]],
    ], dtype=np.int32)
    cv2.polylines(catheter_od, [pts], isClosed=False, color=1.4, thickness=int(round(4 * scale)))
    cv2.GaussianBlur(catheter_od, (3, 3), 0.8, dst=catheter_od)
    return catheter_od


def generate_cine_sequence(
    num_frames: int = 30,
    shape: Tuple[int, int] = (512, 512),
    difficulty: str = "standard",
    fps: int = 15,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, Any]]:
    """
    Generate an entire cine angiography sequence in true 16-bit uint16.

    Args:
        num_frames: Total number of frames in cine run.
        shape: (Height, Width) e.g. (512, 512) or (1024, 1024).
        difficulty: "standard" or "hard" (low contrast, higher noise, large motion).
        fps: Cine frame rate (default: 15 fps).

    Returns:
        (frames_u16, gt_masks_u8, metadata_dict)
    """
    h, w = shape
    scale = h / 512.0

    # Difficulty tuning
    if difficulty == "hard":
        contrast_multiplier = 0.55   # Faint contrast injection
        noise_level = 0.045          # Heavy quantum noise
        motion_factor = 1.6          # High tachycardia / patient motion
    else:
        contrast_multiplier = 1.0    # Nominal contrast
        noise_level = 0.018          # Typical clinical quantum mottle
        motion_factor = 1.0

    tree = build_coronary_tree(scale=scale)

    frames_u16: List[np.ndarray] = []
    gt_masks: List[np.ndarray] = []

    for i in range(num_frames):
        # 75 bpm cardiac cycle at 15 fps -> 1.25 Hz -> ~12 frames per beat
        cardiac_phase = 2.0 * np.pi * (i / 12.0) * motion_factor
        # 15 bpm respiration cycle -> ~60 frames per breath
        respiration_phase = 2.0 * np.pi * (i / 60.0)

        # 1. Vessels
        vessel_od, gt_mask = render_vessels(shape, tree, i, cardiac_phase, scale=scale)
        vessel_od *= contrast_multiplier

        # 2. Anatomy
        spine_od, rib_od, lung_od, bg_od = render_anatomy(shape, respiration_phase, scale=scale)

        # 3. Catheter
        catheter_od = render_catheter(shape, (int(210 * scale), int(100 * scale)), scale=scale)

        # Total optical density in Beer-Lambert domain
        total_od = bg_od + vessel_od + catheter_od

        # Convert back to transmission intensity: I = I_0 * exp(-total_od)
        # Scale baseline dynamic range so tissues span ~12000 to ~55000 in 16-bit
        i_norm = np.exp(-total_od * 0.85)
        # Normalize to [0.15, 0.90]
        i_norm = 0.12 + 0.82 * (i_norm - np.min(i_norm)) / (np.max(i_norm) - np.min(i_norm) + 1e-6)

        # 4. Realistic Poisson + Gaussian noise simulation
        # Photon count N ~ Poisson(I_norm * N_max)
        n_max = 50000.0 / (1.0 + noise_level * 20.0)
        noisy = np.random.poisson(np.clip(i_norm * n_max, 10, None)).astype(np.float32) / n_max

        # Electronic readout Gaussian noise
        read_noise = np.random.normal(0.0, noise_level * 0.25, size=shape).astype(np.float32)
        noisy += read_noise

        # Circular collimator shutter border:
        cy, cx = h // 2, w // 2
        radius = int(min(h, w) * 0.48)
        y_idx, x_idx = np.ogrid[:h, :w]
        shutter = ((x_idx - cx)**2 + (y_idx - cy)**2) <= (radius**2)
        noisy[~shutter] *= 0.01  # Unexposed dark collimator area

        # Map to true 16-bit [0, 65535]
        u16_frame = np.clip(noisy * 65535.0, 0, 65535).astype(np.uint16)

        frames_u16.append(u16_frame)
        gt_masks.append(gt_mask)

    meta = {
        "num_frames": num_frames,
        "width": w,
        "height": h,
        "fps": fps,
        "difficulty": difficulty,
        "bit_depth": 16,
        "vessel_anatomy": "LCA (LMCA, LAD, LCx, D1, D2, OM1)",
        "simulated_effects": [
            "contrast_inflow",
            "cardiac_pulsation",
            "respiratory_drift",
            "vertebral_spine",
            "oblique_ribs",
            "bilateral_lungs",
            "diaphragm",
            "catheter",
            "poisson_quantum_noise",
            "electronic_read_noise",
            "circular_collimator_shutter",
        ],
    }

    return frames_u16, gt_masks, meta


def save_synthetic_dataset(
    output_dir: str | Path,
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    meta: Dict[str, Any],
) -> None:
    """Save generated frames, masks, and metadata to disk."""
    out_p = Path(output_dir)
    frames_dir = out_p / "frames"
    masks_dir = out_p / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for idx, (frame, mask) in enumerate(zip(frames, masks)):
        # 16-bit PNG
        cv2.imwrite(str(frames_dir / f"frame_{idx:04d}.png"), frame)
        # Ground truth mask (uint8 0/255)
        cv2.imwrite(str(masks_dir / f"mask_{idx:04d}.png"), mask * 255)

    with open(out_p / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic 16-bit cardiology cine datasets.")
    parser.add_argument("--output_dir", type=str, default="sample_data", help="Output directory for sample data")
    parser.add_argument("--num_frames", type=int, default=30, help="Frames per sequence")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate")
    args = parser.parse_args()

    out_base = Path(args.output_dir)

    print("Generating standard 512x512 cine sequence...")
    f_512, m_512, meta_512 = generate_cine_sequence(
        num_frames=args.num_frames, shape=(512, 512), difficulty="standard", fps=args.fps
    )
    save_synthetic_dataset(out_base / "cine_512_standard", f_512, m_512, meta_512)

    print("Generating hard 512x512 cine sequence (low contrast, high noise)...")
    f_hard, m_hard, meta_hard = generate_cine_sequence(
        num_frames=args.num_frames, shape=(512, 512), difficulty="hard", fps=args.fps
    )
    save_synthetic_dataset(out_base / "cine_512_hard", f_hard, m_hard, meta_hard)

    print("Generating standard 1024x1024 sequence (10 frames for sample_data)...")
    f_1024, m_1024, meta_1024 = generate_cine_sequence(
        num_frames=10, shape=(1024, 1024), difficulty="standard", fps=args.fps
    )
    save_synthetic_dataset(out_base / "cine_1024_standard", f_1024, m_1024, meta_1024)

    print("Synthetic datasets successfully generated in:", out_base.resolve())


if __name__ == "__main__":
    main()
