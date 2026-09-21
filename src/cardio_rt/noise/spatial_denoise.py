"""
Fast edge-preserving spatial denoising for cardiology fluoroscopy.
Includes Anscombe Variance-Stabilizing Transform (VST) for Poisson photon counting noise
and an O(N) Guided Filter / fast bilateral approximation.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import cv2


def anscombe_transform(img: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Anscombe Variance-Stabilizing Transform: transforms Poisson noise to approximately
    additive Gaussian noise with unit variance: f(x) = 2 * sqrt(x + 3/8).
    """
    if out is None:
        out = np.empty_like(img, dtype=np.float32)
    np.add(img, 3.0 / 8.0, out=out)
    np.sqrt(out, out=out)
    np.multiply(out, 2.0, out=out)
    return out


def inverse_anscombe_transform(ans: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Asymptotically unbiased inverse Anscombe transform:
    x = (ans / 2)^2 - 3/8.
    """
    if out is None:
        out = np.empty_like(ans, dtype=np.float32)
    np.multiply(ans, 0.5, out=out)
    np.square(out, out=out)
    np.subtract(out, 3.0 / 8.0, out=out)
    np.clip(out, 0.0, None, out=out)
    return out


def fast_guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 3,
    eps: float = 1e-3,
    out: Optional[np.ndarray] = None,
    buffer_dict: Optional[dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Fast edge-preserving Guided Filter (He et al.).
    Runs in O(N) time using SIMD OpenCV boxFilter operations.
    Preserves thin coronary vessels while smoothing quantum mottle/noise.
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    if buffer_dict is not None and "mean_I" in buffer_dict:
        mean_I = buffer_dict["mean_I"]
        mean_p = buffer_dict["mean_p"]
        mean_Ip = buffer_dict["mean_Ip"]
        mean_II = buffer_dict["mean_II"]
        a = buffer_dict["a"]
        b = buffer_dict["b"]
    else:
        mean_I = np.empty_like(guide)
        mean_p = np.empty_like(src)
        mean_Ip = np.empty_like(guide)
        mean_II = np.empty_like(guide)
        a = np.empty_like(guide)
        b = np.empty_like(guide)

    cv2.boxFilter(guide, -1, ksize, dst=mean_I, borderType=cv2.BORDER_REFLECT)
    cv2.boxFilter(src, -1, ksize, dst=mean_p, borderType=cv2.BORDER_REFLECT)

    # mean_Ip = boxFilter(I * p)
    np.multiply(guide, src, out=mean_Ip)
    cv2.boxFilter(mean_Ip, -1, ksize, dst=mean_Ip, borderType=cv2.BORDER_REFLECT)

    # mean_II = boxFilter(I * I)
    np.multiply(guide, guide, out=mean_II)
    cv2.boxFilter(mean_II, -1, ksize, dst=mean_II, borderType=cv2.BORDER_REFLECT)

    # cov_Ip = mean_Ip - mean_I * mean_p
    # var_I = mean_II - mean_I * mean_I
    # a = cov_Ip / (var_I + eps)
    # b = mean_p - a * mean_I
    # We reuse arrays to minimize allocations:
    np.multiply(mean_I, mean_p, out=b)
    np.subtract(mean_Ip, b, out=mean_Ip)  # now cov_Ip

    np.multiply(mean_I, mean_I, out=b)
    np.subtract(mean_II, b, out=mean_II)  # now var_I
    np.add(mean_II, eps, out=mean_II)

    np.divide(mean_Ip, mean_II, out=a)
    np.multiply(a, mean_I, out=b)
    np.subtract(mean_p, b, out=b)

    # mean_a = boxFilter(a), mean_b = boxFilter(b)
    cv2.boxFilter(a, -1, ksize, dst=a, borderType=cv2.BORDER_REFLECT)
    cv2.boxFilter(b, -1, ksize, dst=b, borderType=cv2.BORDER_REFLECT)

    # out = mean_a * guide + mean_b
    if out is None:
        out = np.empty_like(src)
    np.multiply(a, guide, out=out)
    np.add(out, b, out=out)
    return out


class FastSpatialDenoise:
    """
    Configurable, toggleable edge-preserving spatial denoising block.
    Supports guided filtering, bilateral approximation, and Anscombe VST.
    """

    def __init__(
        self,
        method: str = "guided",
        radius: int = 3,
        eps: float = 0.002,
        use_vst: bool = False,
    ):
        self.method = method
        self.radius = radius
        self.eps = eps
        self.use_vst = use_vst
        self._buffers: dict[str, np.ndarray] = {}

    def allocate_buffers(self, shape: tuple[int, int]) -> None:
        """Preallocate scratch buffers for zero-allocation hot loops."""
        self._buffers = {
            "mean_I": np.empty(shape, dtype=np.float32),
            "mean_p": np.empty(shape, dtype=np.float32),
            "mean_Ip": np.empty(shape, dtype=np.float32),
            "mean_II": np.empty(shape, dtype=np.float32),
            "a": np.empty(shape, dtype=np.float32),
            "b": np.empty(shape, dtype=np.float32),
            "vst": np.empty(shape, dtype=np.float32),
        }

    def process(
        self,
        img: np.ndarray,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Denoise image while strictly preserving vessel edges.

        Args:
            img: float32 normalized image [0.0, 1.0] or optical density.
            out: Optional preallocated output buffer.

        Returns:
            Denoised float32 image.
        """
        if out is None:
            out = np.empty_like(img)

        # Allocate on shape change if needed
        if not self._buffers or self._buffers["mean_I"].shape != img.shape:
            self.allocate_buffers(img.shape)

        src = img
        if self.use_vst:
            src = anscombe_transform(img, out=self._buffers["vst"])

        if self.method == "guided":
            fast_guided_filter(
                guide=src,
                src=src,
                radius=self.radius,
                eps=self.eps,
                out=out,
                buffer_dict=self._buffers,
            )
        elif self.method == "bilateral":
            # Fast bilateral with small diameter
            cv2.bilateralFilter(src, d=5, sigmaColor=0.08, sigmaSpace=3.0, dst=out)
        else:
            # Separable Gaussian fallback
            cv2.GaussianBlur(src, (2 * self.radius + 1, 2 * self.radius + 1), 1.2, dst=out)

        if self.use_vst:
            inverse_anscombe_transform(out, out=out)

        return out
