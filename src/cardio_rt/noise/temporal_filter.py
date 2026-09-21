"""
Motion-adaptive recursive temporal filter for cardiology cine angiography.
Suppresses quantum noise in static background regions while dynamically disabling
averaging where cardiac motion or contrast inflow is detected, preventing vessel smearing.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import cv2


class MotionAdaptiveTemporalFilter:
    """
    Recursive IIR temporal filter:
    I_t = alpha(x,y) * Frame_t + (1 - alpha(x,y)) * I_{t-1}
    where alpha is high (0.9-1.0) on moving vessels and low (0.3-0.4) on static tissue.
    """

    def __init__(
        self,
        base_alpha: float = 0.40,
        motion_threshold: float = 0.03,
        motion_smooth_ksize: int = 5,
    ):
        self.base_alpha = base_alpha
        self.motion_threshold = motion_threshold
        self.motion_smooth_ksize = motion_smooth_ksize
        self._prev_frame: Optional[np.ndarray] = None
        self._diff_buf: Optional[np.ndarray] = None
        self._alpha_map: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clear temporal history (e.g. at the start of a new cine run)."""
        self._prev_frame = None

    def process(
        self,
        current_frame: np.ndarray,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Filter current frame using motion-adaptive temporal recursion.

        Args:
            current_frame: float32 normalized image [0.0, 1.0].
            out: Optional preallocated output buffer.

        Returns:
            Temporally denoised float32 image.
        """
        if out is None:
            out = np.empty_like(current_frame)

        if self._prev_frame is None or self._prev_frame.shape != current_frame.shape:
            self._prev_frame = current_frame.copy()
            self._diff_buf = np.empty_like(current_frame)
            self._alpha_map = np.empty_like(current_frame)
            np.copyto(out, current_frame)
            return out

        # 1. Detect frame-to-frame absolute difference
        np.subtract(current_frame, self._prev_frame, out=self._diff_buf)
        np.abs(self._diff_buf, out=self._diff_buf)

        # Smooth difference map to create continuous motion envelopes
        k = self.motion_smooth_ksize
        cv2.boxFilter(self._diff_buf, -1, (k, k), dst=self._diff_buf, borderType=cv2.BORDER_REFLECT)

        # 2. Compute motion-adaptive alpha map:
        # alpha = base_alpha + (1.0 - base_alpha) * (diff / (diff + motion_thresh))
        # High motion -> alpha -> 1.0 (pass through without smearing)
        # Low motion -> alpha -> base_alpha (noise reduction)
        np.divide(self._diff_buf, self._diff_buf + self.motion_threshold, out=self._alpha_map)
        np.multiply(self._alpha_map, 1.0 - self.base_alpha, out=self._alpha_map)
        np.add(self._alpha_map, self.base_alpha, out=self._alpha_map)

        # 3. Recursive blend: out = alpha * current + (1 - alpha) * prev
        # out = prev + alpha * (current - prev)
        np.subtract(current_frame, self._prev_frame, out=self._diff_buf)
        np.multiply(self._alpha_map, self._diff_buf, out=self._diff_buf)
        np.add(self._prev_frame, self._diff_buf, out=out)

        # Update history
        np.copyto(self._prev_frame, out)
        return out
