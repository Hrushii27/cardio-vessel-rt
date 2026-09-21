"""
Contrast enhancement module for cardiology angiography.
Includes native 16-bit CLAHE, precomputed 16-bit LUT gamma/tone curve,
and vessel-guided unsharp masking to avoid background noise amplification.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import cv2


class ContrastEnhancement:
    """
    Isolated, toggleable contrast enhancement module.
    Applies 16-bit CLAHE, tone mapping, and vessel-gated high-frequency sharpening.
    """

    def __init__(
        self,
        enabled: bool = True,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        gamma: float = 0.90,
        unsharp_strength: float = 0.50,
        unsharp_radius: int = 2,
    ):
        self.enabled = enabled
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.gamma = gamma
        self.unsharp_strength = unsharp_strength
        self.unsharp_radius = unsharp_radius
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Precompute 16-bit LUT for instantaneous gamma/tone mapping (< 0.8 ms)
        x = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
        if abs(self.gamma - 1.0) > 1e-3:
            y = np.power(x, self.gamma)
        else:
            y = x
        self._lut = np.clip(y * 65535.0, 0, 65535).astype(np.uint16)
        self._buffers: dict[str, np.ndarray] = {}

    def allocate_buffers(self, shape: tuple[int, int]) -> None:
        """Preallocate buffers for zero runtime allocation."""
        self._buffers = {
            "u16_tmp": np.empty(shape, dtype=np.uint16),
            "clahe_out": np.empty(shape, dtype=np.uint16),
            "blur_tmp": np.empty(shape, dtype=np.float32),
            "diff_tmp": np.empty(shape, dtype=np.float32),
        }

    def process(
        self,
        img_norm: np.ndarray,
        apply_clahe: bool = True,
        vessel_gain_map: Optional[np.ndarray] = None,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Enhance image contrast using 16-bit CLAHE and vessel-guided sharpening.

        Args:
            img_norm: [0.0, 1.0] normalized float32 image.
            apply_clahe: Whether to apply tile-based CLAHE.
            vessel_gain_map: Optional [H, W] float32 vesselness/gain weights to gate sharpening.
            out: Optional preallocated float32 output buffer.

        Returns:
            Contrast-enhanced [0.0, 1.0] float32 image.
        """
        if out is None:
            out = np.empty_like(img_norm)

        if not self.enabled:
            np.copyto(out, img_norm)
            return out

        h, w = img_norm.shape
        if not self._buffers or self._buffers["u16_tmp"].shape != (h, w):
            self.allocate_buffers((h, w))

        # 1. Map float32 [0.0, 1.0] to uint16 [0, 65535]
        u16_tmp = self._buffers["u16_tmp"]
        np.multiply(img_norm, 65535.0, out=out)
        np.clip(out, 0, 65535, out=out)
        u16_tmp[...] = out.astype(np.uint16)

        # 2. Fast 16-bit LUT Tone mapping
        mapped_u16 = self._lut[u16_tmp]

        # 3. 16-bit CLAHE (optional for secondary panels)
        if apply_clahe:
            clahe_out = self._buffers["clahe_out"]
            self.clahe.apply(mapped_u16, dst=clahe_out)
            np.multiply(clahe_out, 1.0 / 65535.0, out=out, dtype=np.float32)
        else:
            np.multiply(mapped_u16, 1.0 / 65535.0, out=out, dtype=np.float32)

        # 4. Mild vessel-gated unsharp mask:
        if self.unsharp_strength > 0.0 and vessel_gain_map is not None:
            blur_tmp = self._buffers["blur_tmp"]
            diff_tmp = self._buffers["diff_tmp"]
            k = 2 * self.unsharp_radius + 1

            cv2.GaussianBlur(out, (k, k), 1.0, dst=blur_tmp, borderType=cv2.BORDER_REFLECT)
            np.subtract(out, blur_tmp, out=diff_tmp)

            # Gate sharpening to vessel structures only
            gate = np.clip(vessel_gain_map - 1.0, 0.0, 1.0)
            np.multiply(diff_tmp, gate, out=diff_tmp)
            np.multiply(diff_tmp, self.unsharp_strength, out=diff_tmp)
            np.add(out, diff_tmp, out=out)

        np.clip(out, 0.0, 1.0, out=out)
        return out
