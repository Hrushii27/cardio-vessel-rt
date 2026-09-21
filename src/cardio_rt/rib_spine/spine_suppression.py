"""
Spine suppression module for cardiology X-ray angiography.
Suppresses wide, vertically oriented vertebral column structures while preserving
tortuous, branching coronary arteries.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2


class SpineSuppression:
    """
    Isolated, toggleable module for spine suppression.
    Uses anisotropic directional filtering and vertical column profiling
    to estimate the vertebral bone attenuation map and subtract it in the density domain.
    """

    def __init__(
        self,
        enabled: bool = True,
        sigma_vertical: float = 35.0,
        sigma_horizontal: float = 7.0,
        suppression_strength: float = 0.75,
    ):
        self.enabled = enabled
        self.sigma_vertical = sigma_vertical
        self.sigma_horizontal = sigma_horizontal
        self.suppression_strength = suppression_strength
        self._buffers: dict[str, np.ndarray] = {}

    def allocate_buffers(self, shape: tuple[int, int]) -> None:
        """Preallocate memory buffers for zero allocations during processing."""
        h, w = shape
        # Subsampled 1/2 resolution for low-frequency spine estimation
        ds_h, ds_w = h // 2, w // 2
        self._buffers = {
            "ds_input": np.empty((ds_h, ds_w), dtype=np.float32),
            "ds_spine": np.empty((ds_h, ds_w), dtype=np.float32),
            "up_spine": np.empty((h, w), dtype=np.float32),
            "bone_map": np.empty((h, w), dtype=np.float32),
        }

    def compute_spine_map(self, od_map: np.ndarray) -> np.ndarray:
        """
        Estimate the 2D spine density contribution.

        Args:
            od_map: Optical density float32 image.

        Returns:
            Estimated spine density map [H, W].
        """
        h, w = od_map.shape
        if not self._buffers or self._buffers["up_spine"].shape != (h, w):
            self.allocate_buffers((h, w))

        ds_in = self._buffers["ds_input"]
        ds_spine = self._buffers["ds_spine"]
        up_spine = self._buffers["up_spine"]

        # Downsample 2x to isolate macro structures and maximize speed (< 0.5 ms)
        cv2.resize(od_map, (ds_in.shape[1], ds_in.shape[0]), dst=ds_in, interpolation=cv2.INTER_AREA)

        # Anisotropic separable filtering: large sigma along Y (vertical column), moderate along X
        # Scaled down to half-res coordinates
        sy = max(3.0, self.sigma_vertical / 2.0)
        sx = max(2.0, self.sigma_horizontal / 2.0)

        # Separable Gaussian filters
        cv2.GaussianBlur(ds_in, (0, 0), sigmaX=sx, sigmaY=sy, dst=ds_spine, borderType=cv2.BORDER_REFLECT)

        # Upsample back to full resolution
        cv2.resize(ds_spine, (w, h), dst=up_spine, interpolation=cv2.INTER_LINEAR)

        # Spine is primarily located in the central vertical corridor.
        # Construct soft horizontal window weighting to isolate central column:
        x_coords = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        # Vertebral column profile: Gaussian centered horizontally
        spine_weight = np.exp(-0.5 * (x_coords / 0.5) ** 2)[np.newaxis, :]

        bone_map = self._buffers["bone_map"]
        np.multiply(up_spine, spine_weight, out=bone_map)

        return bone_map

    def process(
        self,
        od_map: np.ndarray,
        out: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subtract the estimated spine density from the optical density map.

        Args:
            od_map: Optical density float32 image.
            out: Optional preallocated output buffer.

        Returns:
            (spine_suppressed_od, estimated_spine_map)
        """
        if out is None:
            out = np.empty_like(od_map)

        if not self.enabled:
            np.copyto(out, od_map)
            return out, np.zeros_like(od_map)

        spine_map = self.compute_spine_map(od_map)

        # Subtraction in optical density space with attenuation strength:
        # out = od_map - suppression_strength * spine_map
        # We ensure vessel signals are not driven negative by clipping to 0 baseline
        np.multiply(spine_map, self.suppression_strength, out=out)
        np.subtract(od_map, out, out=out)
        np.clip(out, 0.0, None, out=out)

        return out, spine_map
