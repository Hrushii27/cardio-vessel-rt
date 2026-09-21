"""
Rib suppression module for cardiology X-ray angiography.
Suppresses wide, oblique rib bone structures via large-sigma Hessian ridge maps
subtracted as an estimated bone map.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2


class RibSuppression:
    """
    Isolated, toggleable module for rib suppression.
    Uses large-scale Hessian ridge filtering at downsampled resolution to isolate
    wide rib bones while leaving thin coronary arteries (small scale) untouched.
    """

    def __init__(
        self,
        enabled: bool = True,
        bone_scale_sigma: float = 14.0,
        suppression_strength: float = 0.65,
        subsample_factor: int = 4,
    ):
        self.enabled = enabled
        self.bone_scale_sigma = bone_scale_sigma
        self.suppression_strength = suppression_strength
        self.subsample_factor = subsample_factor
        self._buffers: dict[str, np.ndarray] = {}

    def allocate_buffers(self, shape: tuple[int, int]) -> None:
        """Preallocate buffers for zero runtime allocation."""
        h, w = shape
        sf = self.subsample_factor
        ds_h, ds_w = h // sf, w // sf
        self._buffers = {
            "ds_in": np.empty((ds_h, ds_w), dtype=np.float32),
            "ds_smooth": np.empty((ds_h, ds_w), dtype=np.float32),
            "dxx": np.empty((ds_h, ds_w), dtype=np.float32),
            "dyy": np.empty((ds_h, ds_w), dtype=np.float32),
            "dxy": np.empty((ds_h, ds_w), dtype=np.float32),
            "ds_rib": np.empty((ds_h, ds_w), dtype=np.float32),
            "rib_map": np.empty((h, w), dtype=np.float32),
        }

    def compute_rib_map(self, od_map: np.ndarray) -> np.ndarray:
        """
        Compute the large-sigma Hessian bone ridge response for ribs.

        Args:
            od_map: Optical density float32 image.

        Returns:
            Estimated rib bone map [H, W].
        """
        h, w = od_map.shape
        if not self._buffers or self._buffers["rib_map"].shape != (h, w):
            self.allocate_buffers((h, w))

        ds_in = self._buffers["ds_in"]
        ds_smooth = self._buffers["ds_smooth"]
        dxx = self._buffers["dxx"]
        dyy = self._buffers["dyy"]
        dxy = self._buffers["dxy"]
        ds_rib = self._buffers["ds_rib"]
        rib_map = self._buffers["rib_map"]

        # Subsample to working scale
        cv2.resize(od_map, (ds_in.shape[1], ds_in.shape[0]), dst=ds_in, interpolation=cv2.INTER_AREA)

        # Scale down sigma to the subsampled coordinate system
        ds_sigma = max(1.5, self.bone_scale_sigma / self.subsample_factor)
        ksize = int(round(ds_sigma * 6)) | 1  # ensure odd

        # Gaussian smoothing
        cv2.GaussianBlur(ds_in, (ksize, ksize), ds_sigma, dst=ds_smooth, borderType=cv2.BORDER_REFLECT)

        # Compute 2nd derivatives using Sobel
        cv2.Sobel(ds_smooth, cv2.CV_32F, 2, 0, dst=dxx, ksize=3)
        cv2.Sobel(ds_smooth, cv2.CV_32F, 0, 2, dst=dyy, ksize=3)
        cv2.Sobel(ds_smooth, cv2.CV_32F, 1, 1, dst=dxy, ksize=3)

        # Closed-form 2x2 Hessian eigenvalues:
        # Trace = dxx + dyy
        # Det = dxx * dyy - dxy^2
        # lambda1 = (Trace + sqrt((dxx - dyy)^2 + 4 * dxy^2)) / 2
        # lambda2 = (Trace - sqrt((dxx - dyy)^2 + 4 * dxy^2)) / 2
        # In optical density space, ribs are ridge-like structures with negative second derivative
        # in the direction normal to the rib ridge.
        diff = dxx - dyy
        term = np.sqrt(diff * diff + 4.0 * dxy * dxy)
        # lambda_min has maximum negative curvature (highest ridge response for positive features)
        lambda_ridge = 0.5 * (dxx + dyy - term)

        # Invert sign so positive values correspond to ridges
        # and select oblique/curved ribs (where curvature in rib cross-section is significant)
        np.negative(lambda_ridge, out=ds_rib)
        np.clip(ds_rib, 0.0, None, out=ds_rib)

        # Normalize rib response scale
        max_val = float(np.percentile(ds_rib, 99.0))
        if max_val > 1e-6:
            ds_rib /= max_val

        # Upscale back to full resolution
        cv2.resize(ds_rib, (w, h), dst=rib_map, interpolation=cv2.INTER_LINEAR)

        # Multiply by local optical density to preserve natural bone attenuation proportion
        np.multiply(rib_map, od_map, out=rib_map)

        return rib_map

    def process(
        self,
        od_map: np.ndarray,
        out: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subtract the estimated rib bone map from the optical density map.

        Args:
            od_map: Optical density float32 image.
            out: Optional preallocated output buffer.

        Returns:
            (rib_suppressed_od, estimated_rib_map)
        """
        if out is None:
            out = np.empty_like(od_map)

        if not self.enabled:
            np.copyto(out, od_map)
            return out, np.zeros_like(od_map)

        rib_map = self.compute_rib_map(od_map)

        # Subtract bone map
        np.multiply(rib_map, self.suppression_strength, out=out)
        np.subtract(od_map, out, out=out)
        np.clip(out, 0.0, None, out=out)

        return out, rib_map
