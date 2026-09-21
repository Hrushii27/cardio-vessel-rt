"""
Multi-scale Laplacian pyramid and local flat-field normalization for lung background suppression.
Attenuates macro lung density variations and coarse gradients while keeping vessel-scale bands.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np
import cv2


class LungBackgroundSuppression:
    """
    Suppresses low-frequency lung fields, diaphragm gradients, and soft-tissue background
    using multi-band Laplacian pyramid decomposition with band-wise attenuation and local flat-fielding.
    """

    def __init__(
        self,
        enabled: bool = True,
        num_levels: int = 4,
        coarse_band_weight: float = 0.12,
        mid_coarse_weight: float = 0.35,
        vessel_band_weight: float = 1.0,
        fine_noise_weight: float = 0.70,
        enable_flat_field: bool = True,
    ):
        self.enabled = enabled
        self.num_levels = num_levels
        self.coarse_band_weight = coarse_band_weight
        self.mid_coarse_weight = mid_coarse_weight
        self.vessel_band_weight = vessel_band_weight
        self.fine_noise_weight = fine_noise_weight
        self.enable_flat_field = enable_flat_field
        self._buffers: dict[str, Any] = {}

    def _build_gaussian_pyramid(self, img: np.ndarray) -> List[np.ndarray]:
        """Construct Gaussian pyramid down to num_levels."""
        gp = [img]
        current = img
        for _ in range(self.num_levels):
            down = cv2.pyrDown(current)
            gp.append(down)
            current = down
        return gp

    def _build_laplacian_pyramid(self, gp: List[np.ndarray]) -> List[np.ndarray]:
        """Construct Laplacian pyramid from Gaussian pyramid."""
        lp = []
        for i in range(self.num_levels):
            h, w = gp[i].shape[:2]
            up = cv2.pyrUp(gp[i + 1], dstsize=(w, h))
            lap = cv2.subtract(gp[i], up)
            lp.append(lap)
        lp.append(gp[self.num_levels])  # Coarsest base residual
        return lp

    def _reconstruct_laplacian_pyramid(self, lp: List[np.ndarray]) -> np.ndarray:
        """Reconstruct full image from modulated Laplacian pyramid levels."""
        current = lp[-1]
        for i in range(self.num_levels - 1, -1, -1):
            h, w = lp[i].shape[:2]
            up = cv2.pyrUp(current, dstsize=(w, h))
            current = cv2.add(up, lp[i])
        return current

    def process(
        self,
        od_map: np.ndarray,
        out: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose into Laplacian pyramid, modulate band weights, and reconstruct.

        Args:
            od_map: Optical density float32 image.
            out: Optional preallocated output buffer.

        Returns:
            (suppressed_od, background_map)
        """
        if out is None:
            out = np.empty_like(od_map)

        if not self.enabled:
            np.copyto(out, od_map)
            return out, np.zeros_like(od_map)

        h, w = od_map.shape

        # Optional flat-field normalization for extreme lung-to-cardiac intensity ramps
        working_od = od_map
        if self.enable_flat_field:
            # Subsample 8x, apply wide blur, upsample to estimate global illumination field
            small_w, small_h = max(16, w // 8), max(16, h // 8)
            low_res = cv2.resize(od_map, (small_w, small_h), interpolation=cv2.INTER_AREA)
            low_res_blur = cv2.GaussianBlur(low_res, (21, 21), 7.0)
            flat_bg = cv2.resize(low_res_blur, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            flat_bg = np.zeros_like(od_map)

        # Pad to multiple of 2^num_levels if necessary
        divisor = 2**self.num_levels
        pad_h = (divisor - (h % divisor)) % divisor
        pad_w = (divisor - (w % divisor)) % divisor

        if pad_h > 0 or pad_w > 0:
            padded = cv2.copyMakeBorder(
                working_od, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
            )
        else:
            padded = working_od

        # Multi-scale Laplacian pyramid decomposition
        gp = self._build_gaussian_pyramid(padded)
        lp = self._build_laplacian_pyramid(gp)

        # lp[0]: Finest band (1-2 px details + noise)
        # lp[1]: Thin vessels (2-4 px)
        # lp[2]: Main vessels (4-8 px)
        # lp[3]: Coarse rib/spine residuals (8-16 px)
        # lp[4]: Base residual (macro lung fields and DC background)

        # Modulate bands dynamically:
        # Band 0: finest band
        if len(lp) > 1:
            lp[0] *= self.fine_noise_weight
        # Intermediate vessel bands
        for lvl in range(1, len(lp) - 2):
            lp[lvl] *= self.vessel_band_weight
        # Coarse structure band
        if len(lp) > 2:
            lp[-2] *= self.mid_coarse_weight
        # Base residual: severe attenuation of lung fields
        lp[-1] *= self.coarse_band_weight

        reconstructed = self._reconstruct_laplacian_pyramid(lp)

        # Crop back if padded
        if pad_h > 0 or pad_w > 0:
            reconstructed = reconstructed[:h, :w]

        # Ensure non-negativity
        np.clip(reconstructed, 0.0, None, out=out)

        # Estimate background map that was removed
        bg_removed = np.maximum(0.0, od_map - out)

        return out, bg_removed
