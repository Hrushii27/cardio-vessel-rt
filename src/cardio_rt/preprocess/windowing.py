"""
Percentile windowing and collimator/shutter mask detection for 16-bit cardiology frames.
Designed for real-time execution with preallocated buffer support.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import cv2


def percentile_normalize(
    img: np.ndarray,
    p_low: float = 0.5,
    p_high: float = 99.5,
    out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize uint16 or float32 image to [0.0, 1.0] range using robust percentiles.
    To ensure real-time speed, percentiles are computed on a 4x subsampled grid.

    Args:
        img: Input image (uint16 or float32), 2D array.
        p_low: Lower percentile (default: 0.5).
        p_high: Upper percentile (default: 99.5).
        out: Optional preallocated float32 buffer [H, W].

    Returns:
        (normalized_float32, vmin, vmax)
    """
    if out is None:
        out = np.empty(img.shape, dtype=np.float32)

    # Subsample for ultra-fast percentile computation (avoids sorting full array)
    sub = img[::8, ::8]
    vmin = float(np.percentile(sub, p_low))
    vmax = float(np.percentile(sub, p_high))

    if vmax <= vmin:
        vmax = vmin + 1.0

    inv_range = 1.0 / (vmax - vmin)

    # Vectorized fast scaling and clipping into output buffer
    np.subtract(img, vmin, out=out, dtype=np.float32)
    np.multiply(out, inv_range, out=out)
    np.clip(out, 0.0, 1.0, out=out)

    return out, vmin, vmax


def detect_collimator_mask(
    img_norm: np.ndarray,
    dark_thresh: float = 0.03,
    out_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Detect the circular or rectangular radiation collimator / lead shutter border.
    Collimator borders are unexposed (very dark / zero intensity) and should not
    corrupt background statistics or cause edge ringing.

    Args:
        img_norm: [0.0, 1.0] normalized float32 image.
        dark_thresh: Threshold below which pixels are treated as lead shutter.
        out_mask: Optional preallocated uint8 buffer [H, W].

    Returns:
        Binary mask (uint8, 1 = valid active anatomy field, 0 = collimator border).
    """
    if out_mask is None:
        out_mask = np.ones(img_norm.shape, dtype=np.uint8)
    else:
        out_mask.fill(1)

    # Downscale for ultra-fast morphological cleanup
    h, w = img_norm.shape
    ds_w = min(128, w)
    ds_h = min(128, h)
    small = cv2.resize(img_norm, (ds_w, ds_h), interpolation=cv2.INTER_AREA)

    # Threshold dark shutter
    small_bin = (small > dark_thresh).astype(np.uint8)

    # Morphological closing then opening to keep the central active cone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    small_clean = cv2.morphologyEx(small_bin, cv2.MORPH_CLOSE, kernel)
    small_clean = cv2.morphologyEx(small_clean, cv2.MORPH_OPEN, kernel)

    # Find the largest connected component (the active field of view)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(small_clean)
    if num_labels > 1:
        # Background is 0; find largest non-zero component
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        small_mask = (labels == largest_idx).astype(np.uint8)
    else:
        small_mask = small_clean

    # Upscale back to full resolution
    cv2.resize(small_mask, (w, h), dst=out_mask, interpolation=cv2.INTER_NEAREST)
    return out_mask
