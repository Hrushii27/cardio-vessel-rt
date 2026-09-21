"""
Fast multi-scale 2D vesselness filter (Frangi, Sato) using closed-form analytical
2x2 Hessian eigenvalues and parallel Numba JIT acceleration.
Generates continuous soft vessel gain maps for real-time coronary enhancement.
"""

from __future__ import annotations

from typing import Optional, Tuple, Sequence
import numpy as np
import cv2

try:
    import numba

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def _numba_frangi_kernel(
        dxx: np.ndarray,
        dyy: np.ndarray,
        dxy: np.ndarray,
        beta_sq2: float,
        c_sq2: float,
        out: np.ndarray,
    ) -> None:
        """
        Numba-accelerated parallel kernel computing closed-form eigenvalues
        and Frangi vesselness in a single cache-friendly pass.
        """
        h, w = dxx.shape
        for i in numba.prange(h):
            for j in range(w):
                a = dxx[i, j]
                b = dyy[i, j]
                c_val = dxy[i, j]

                diff = a - b
                disc = np.sqrt(diff * diff + 4.0 * c_val * c_val)
                mu1 = 0.5 * (a + b + disc)
                mu2 = 0.5 * (a + b - disc)

                # Order so |l1| <= |l2|
                if abs(mu1) <= abs(mu2):
                    l1 = mu1
                    l2 = mu2
                else:
                    l1 = mu2
                    l2 = mu1

                # In optical density space, vessels are ridges (cross-section l2 < 0)
                if l2 < 0.0:
                    denom = l2 * l2
                    rb_sq = (l1 * l1) / (denom + 1e-8)
                    s_sq = l1 * l1 + denom
                    v = np.exp(-rb_sq / beta_sq2) * (1.0 - np.exp(-s_sq / c_sq2))
                    if v > out[i, j]:
                        out[i, j] = v

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False


def _numpy_frangi_kernel(
    dxx: np.ndarray,
    dyy: np.ndarray,
    dxy: np.ndarray,
    beta_sq2: float,
    c_sq2: float,
    out: np.ndarray,
) -> None:
    """Vectorized NumPy fallback if Numba is not present."""
    diff = dxx - dyy
    disc = np.sqrt(diff * diff + 4.0 * dxy * dxy)
    trace = dxx + dyy

    mu1 = 0.5 * (trace + disc)
    mu2 = 0.5 * (trace - disc)

    cond = np.abs(mu1) <= np.abs(mu2)
    l1 = np.where(cond, mu1, mu2)
    l2 = np.where(cond, mu2, mu1)

    vessel_mask = l2 < 0.0
    denom = l2 * l2 + 1e-8
    rb_sq = (l1 * l1) / denom
    s_sq = l1 * l1 + l2 * l2

    v = np.exp(-rb_sq / beta_sq2) * (1.0 - np.exp(-s_sq / c_sq2))
    v = np.where(vessel_mask, v, 0.0)
    np.maximum(out, v, out=out)


class FastVesselnessEnhancer:
    """
    Multi-scale Hessian vesselness engine. Produces a continuous soft gain map
    for coronary artery contrast boosting without hard binary thresholding.
    """

    def __init__(
        self,
        enabled: bool = True,
        scales: Sequence[float] = (1.5, 3.0),
        method: str = "frangi",
        beta: float = 0.5,
        c: float = 12.0,
        gain_strength: float = 1.5,
        subsample_factor: int = 2,
    ):
        self.enabled = enabled
        self.scales = tuple(scales)
        self.method = method
        self.beta = beta
        self.c = c
        self.gain_strength = gain_strength
        self.subsample_factor = subsample_factor
        self._buffers: dict[str, np.ndarray] = {}

    def allocate_buffers(self, shape: tuple[int, int]) -> None:
        """Preallocate buffers for zero runtime allocation."""
        h, w = shape
        sf = self.subsample_factor
        wh, ww = max(1, h // sf), max(1, w // sf)
        self._buffers = {
            "work_in": np.empty((wh, ww), dtype=np.float32),
            "smoothed": np.empty((wh, ww), dtype=np.float32),
            "dxx": np.empty((wh, ww), dtype=np.float32),
            "dyy": np.empty((wh, ww), dtype=np.float32),
            "dxy": np.empty((wh, ww), dtype=np.float32),
            "v_small": np.empty((wh, ww), dtype=np.float32),
            "full_vesselness": np.empty((h, w), dtype=np.float32),
            "gain_map": np.empty((h, w), dtype=np.float32),
        }

    def compute_vesselness(self, od_map: np.ndarray) -> np.ndarray:
        """
        Compute multi-scale vesselness probability map [0.0, 1.0].
        """
        h, w = od_map.shape
        if not self._buffers or self._buffers["full_vesselness"].shape != (h, w):
            self.allocate_buffers((h, w))

        sf = self.subsample_factor
        work_in = self._buffers["work_in"]
        v_small = self._buffers["v_small"]
        v_small.fill(0.0)

        if sf > 1:
            cv2.resize(od_map, (work_in.shape[1], work_in.shape[0]), dst=work_in, interpolation=cv2.INTER_AREA)
        else:
            np.copyto(work_in, od_map)

        beta_sq2 = 2.0 * self.beta * self.beta
        c_sq2 = 2.0 * self.c * self.c

        dxx = self._buffers["dxx"]
        dyy = self._buffers["dyy"]
        dxy = self._buffers["dxy"]
        smoothed = self._buffers["smoothed"]

        for s in self.scales:
            scale_adj = s / sf
            if scale_adj < 0.6:
                continue

            ksize = int(round(scale_adj * 4)) | 1
            if ksize > 1:
                cv2.GaussianBlur(work_in, (ksize, ksize), scale_adj, dst=smoothed, borderType=cv2.BORDER_REFLECT)
            else:
                np.copyto(smoothed, work_in)

            # 2nd derivative Sobel filters
            cv2.Sobel(smoothed, cv2.CV_32F, 2, 0, dst=dxx, ksize=3)
            cv2.Sobel(smoothed, cv2.CV_32F, 0, 2, dst=dyy, ksize=3)
            cv2.Sobel(smoothed, cv2.CV_32F, 1, 1, dst=dxy, ksize=3)

            # Lindeberg gamma normalization (gamma = 2.0)
            scale_norm = scale_adj * scale_adj
            dxx *= scale_norm
            dyy *= scale_norm
            dxy *= scale_norm

            if HAS_NUMBA:
                _numba_frangi_kernel(dxx, dyy, dxy, beta_sq2, c_sq2, v_small)
            else:
                _numpy_frangi_kernel(dxx, dyy, dxy, beta_sq2, c_sq2, v_small)

        # Upscale back to full resolution
        full_v = self._buffers["full_vesselness"]
        if sf > 1:
            cv2.resize(v_small, (w, h), dst=full_v, interpolation=cv2.INTER_LINEAR)
        else:
            np.copyto(full_v, v_small)

        # Percentile scale normalization (strided for 0.1 ms execution)
        v_max = float(np.percentile(full_v[::4, ::4], 99.8))
        if v_max > 1e-6:
            full_v /= v_max
            np.clip(full_v, 0.0, 1.0, out=full_v)

        return full_v

    def get_soft_gain_map(self, vesselness: np.ndarray) -> np.ndarray:
        """
        Compute soft gain map: G = 1.0 + gain_strength * vesselness.
        """
        gain_map = self._buffers["gain_map"]
        np.multiply(vesselness, self.gain_strength, out=gain_map)
        np.add(gain_map, 1.0, out=gain_map)
        return gain_map

    def process(
        self,
        od_map: np.ndarray,
        out: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute vesselness and apply soft vessel gain map.

        Args:
            od_map: Optical density float32 image.
            out: Optional preallocated output buffer.

        Returns:
            (enhanced_od, vesselness_map, soft_gain_map)
        """
        if out is None:
            out = np.empty_like(od_map)

        if not self.enabled:
            np.copyto(out, od_map)
            zeros = np.zeros_like(od_map)
            ones = np.ones_like(od_map)
            return out, zeros, ones

        vesselness = self.compute_vesselness(od_map)
        gain_map = self.get_soft_gain_map(vesselness)

        np.multiply(od_map, gain_map, out=out)

        return out, vesselness, gain_map
