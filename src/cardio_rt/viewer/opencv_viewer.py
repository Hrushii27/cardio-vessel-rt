"""
Real-time OpenCV interactive side-by-side viewer for cardiology fluoroscopy frames and cines.
Displays [Original | Processed | Enhanced] panels with real-time latency HUD and toggleable modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import cv2

from cardio_rt.io import load_image, create_comparison_panel, to_8bit_display
from cardio_rt.pipeline import CardioPipeline, PipelineConfig

logger = logging.getLogger(__name__)


class CardioViewer:
    """
    Interactive OpenCV viewer for real-time cardiology inspection.
    """

    def __init__(self, pipeline: Optional[CardioPipeline] = None, window_name: str = "CardioRT Live Inspection"):
        self.pipeline = pipeline or CardioPipeline()
        self.window_name = window_name
        self.paused = False
        self.frame_delay_ms = 33  # ~30 fps base playback rate

    def render_hud(self, panel_bgr: np.ndarray, frame_idx: int, total_frames: int, latency_ms: float) -> np.ndarray:
        """Draw HUD banner with latency, FPS, and toggleable module statuses."""
        h, w = panel_bgr.shape[:2]
        hud_h = 36
        hud = panel_bgr.copy()

        # Semi-transparent bottom banner
        overlay = hud.copy()
        cv2.rectangle(overlay, (0, h - hud_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, hud, 0.15, 0, dst=hud)

        # Build module status string
        cfg = self.pipeline.config
        mods = (
            f"Noise[n]:{'ON' if cfg.enable_noise_suppression else 'OFF'} | "
            f"Ribs[r]:{'ON' if cfg.enable_rib_suppression else 'OFF'} | "
            f"Spine[s]:{'ON' if cfg.enable_spine_suppression else 'OFF'} | "
            f"Lung[l]:{'ON' if cfg.enable_lung_background_suppression else 'OFF'} | "
            f"Vessel[v]:{'ON' if cfg.enable_coronary_enhancement else 'OFF'} | "
            f"Contrast[c]:{'ON' if cfg.enable_contrast_enhancement else 'OFF'}"
        )

        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        stats = f"Frame: {frame_idx + 1}/{total_frames} | Latency: {latency_ms:.1f}ms ({fps:.1f} FPS) | {mods}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42 if w >= 1000 else 0.35
        cv2.putText(hud, stats, (15, h - 12), font, scale, (0, 255, 200), 1, cv2.LINE_AA)
        return hud

    def run_cine(
        self,
        source: Union[str, Path, List[np.ndarray]],
        loop: bool = True,
        max_display_width: int = 1536,
    ) -> None:
        """
        Play cine loop with live pipeline processing and comparison display.

        Args:
            source: Directory containing frames, or list of uint16 frames.
            loop: Whether to loop playback infinitely until 'q' is pressed.
            max_display_width: Resize composite panel if it exceeds screen width.
        """
        frames: List[np.ndarray] = []
        if isinstance(source, (str, Path)):
            src_p = Path(source)
            if src_p.is_dir():
                valid_exts = {".png", ".tif", ".tiff", ".npy"}
                files = sorted([f for f in src_p.iterdir() if f.suffix.lower() in valid_exts])
                if not files:
                    raise FileNotFoundError(f"No valid image files found in {src_p}")
                for f in files:
                    img, _ = load_image(f)
                    frames.append(img)
            elif src_p.is_file():
                img, _ = load_image(src_p)
                frames.append(img)
            else:
                raise FileNotFoundError(f"Source not found: {src_p}")
        else:
            frames = list(source)

        if not frames:
            print("No frames to display.")
            return

        print("\n" + "=" * 60)
        print("CardioRT OpenCV Viewer Started")
        print("Controls:")
        print("  [q] / ESC : Exit viewer")
        print("  [Space]   : Pause / Resume cine")
        print("  [n]       : Toggle Noise Suppression")
        print("  [r]       : Toggle Rib Suppression")
        print("  [s]       : Toggle Spine Suppression")
        print("  [l]       : Toggle Lung Background Suppression")
        print("  [v]       : Toggle Coronary Artery Enhancement")
        print("  [c]       : Toggle Contrast Enhancement (CLAHE)")
        print("=" * 60 + "\n")

        # Warmup pipeline on the first frame
        self.pipeline.warmup(frames[0].shape[:2], iterations=5)

        idx = 0
        total = len(frames)

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            logger.warning(f"Could not open graphical window (headless environment?): {e}")
            return

        while True:
            frame_u16 = frames[idx]

            # Process frame through live pipeline
            out = self.pipeline.process_frame(frame_u16)

            # Generate 3-panel comparison
            panel_bgr = create_comparison_panel(
                out.original_u16,
                out.processed_u16,
                out.enhanced_u16,
                titles=(
                    f"Original ({frame_u16.shape[1]}x{frame_u16.shape[0]})",
                    "Processed (Background Suppressed)",
                    f"Enhanced ({out.total_latency_ms:.1f}ms)",
                ),
                display_8bit=True,
            )

            # Draw HUD
            display_img = self.render_hud(panel_bgr, idx, total, out.total_latency_ms)

            # Downscale for display if panel exceeds screen width
            if display_img.shape[1] > max_display_width:
                aspect = display_img.shape[0] / display_img.shape[1]
                dh = int(max_display_width * aspect)
                display_img = cv2.resize(display_img, (max_display_width, dh), interpolation=cv2.INTER_AREA)

            cv2.imshow(self.window_name, display_img)

            # Key handling
            key = cv2.waitKey(self.frame_delay_ms if not self.paused else 50) & 0xFF
            if key in (27, ord("q")):  # ESC or 'q'
                break
            elif key == ord(" "):     # Space: pause
                self.paused = not self.paused
            elif key == ord("n"):
                self.pipeline.config.enable_noise_suppression = not self.pipeline.config.enable_noise_suppression
                print(f"Noise suppression: {'ON' if self.pipeline.config.enable_noise_suppression else 'OFF'}")
            elif key == ord("r"):
                self.pipeline.config.enable_rib_suppression = not self.pipeline.config.enable_rib_suppression
                print(f"Rib suppression: {'ON' if self.pipeline.config.enable_rib_suppression else 'OFF'}")
            elif key == ord("s"):
                self.pipeline.config.enable_spine_suppression = not self.pipeline.config.enable_spine_suppression
                print(f"Spine suppression: {'ON' if self.pipeline.config.enable_spine_suppression else 'OFF'}")
            elif key == ord("l"):
                self.pipeline.config.enable_lung_background_suppression = not self.pipeline.config.enable_lung_background_suppression
                print(f"Lung suppression: {'ON' if self.pipeline.config.enable_lung_background_suppression else 'OFF'}")
            elif key == ord("v"):
                self.pipeline.config.enable_coronary_enhancement = not self.pipeline.config.enable_coronary_enhancement
                print(f"Coronary enhancement: {'ON' if self.pipeline.config.enable_coronary_enhancement else 'OFF'}")
            elif key == ord("c"):
                self.pipeline.config.enable_contrast_enhancement = not self.pipeline.config.enable_contrast_enhancement
                print(f"Contrast enhancement: {'ON' if self.pipeline.config.enable_contrast_enhancement else 'OFF'}")

            if not self.paused:
                idx = (idx + 1) % total
                if idx == 0 and not loop:
                    break

        cv2.destroyAllWindows()
