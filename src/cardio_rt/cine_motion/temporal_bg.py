"""
Temporal background estimation and motion-compensated DSA-like subtraction for cine mode.
Tracks running background optical density with sub-pixel phase correlation alignment.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2

from cardio_rt.cine_motion.phase_corr import MotionCompensator


class TemporalBackgroundModel:
    """
    Maintains a running anatomical background model across cine frames.
    Uses motion compensation to align the reference background before subtraction,
    isolating contrast-filled coronary arteries from static/moving ribs and spine.
    """

    def __init__(
        self,
        history_size: int = 5,
        subsample_align: int = 4,
        background_decay: float = 0.05,
    ):
        self.history_size = history_size
        self.background_decay = background_decay
        self.compensator = MotionCompensator(subsample=subsample_align)

        self._bg_od: Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._frame_count: int = 0

    def reset(self) -> None:
        """Reset temporal background history."""
        self._bg_od = None
        self._prev_frame = None
        self._frame_count = 0

    def process(
        self,
        current_od: np.ndarray,
        out_od: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """
        Estimate background and subtract from current optical density map.

        Args:
            current_od: Current optical density float32 image.
            out_od: Optional preallocated output buffer.

        Returns:
            (subtracted_od, aligned_bg_od, (shift_x, shift_y))
        """
        if out_od is None:
            out_od = np.empty_like(current_od)

        # First frame: initialize background
        if self._bg_od is None or self._bg_od.shape != current_od.shape:
            self._bg_od = current_od.copy()
            self._prev_frame = current_od.copy()
            self._frame_count = 1
            np.copyto(out_od, current_od)
            return out_od, self._bg_od, (0.0, 0.0)

        # 1. Motion compensation: estimate translational shift from previous frame to current
        dx, dy, response = self.compensator.estimate_shift(current_od, self._prev_frame)

        # 2. Warp running background to align with current frame
        aligned_bg = self.compensator.warp_reference(self._bg_od, dx, dy)

        # 3. Subtraction: Vessels are positive peaks in OD.
        # subtracted = max(0, current_od - aligned_bg)
        np.subtract(current_od, aligned_bg * 0.85, out=out_od)
        np.clip(out_od, 0.0, None, out=out_od)

        # 4. Update running background (min-hold / running median approximation):
        # Background is where optical density is lowest (before contrast bolus enters)
        # bg = min(aligned_bg, current_od) with slight temporal relaxation
        updated_bg = np.minimum(aligned_bg, current_od)
        if self.background_decay > 0:
            # Slow relaxation towards current to adapt to respiratory drift
            aligned_bg = (1.0 - self.background_decay) * updated_bg + self.background_decay * current_od
        else:
            aligned_bg = updated_bg

        np.copyto(self._bg_od, aligned_bg)
        np.copyto(self._prev_frame, current_od)
        self._frame_count += 1

        return out_od, aligned_bg, (dx, dy)
