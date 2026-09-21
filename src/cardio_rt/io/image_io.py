"""
Image I/O supporting 16-bit grayscale (PNG, TIFF, NPY) with automatic 8-bit detection,
flagging, and display conversion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple, Optional
import numpy as np
import cv2
import tifffile

logger = logging.getLogger(__name__)


def to_8bit_display(
    img: np.ndarray,
    p_low: float = 0.5,
    p_high: float = 99.5,
) -> np.ndarray:
    """
    Convert a 16-bit or float32 image to an 8-bit display image using robust percentile windowing.
    Never used in the internal computation pipeline - purely for display and GUI rendering.
    """
    arr = img.astype(np.float32)
    vmin = np.percentile(arr, p_low)
    vmax = np.percentile(arr, p_high)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def load_image(path: str | Path) -> Tuple[np.ndarray, dict[str, Any]]:
    """
    Load an image from disk as uint16 grayscale.

    Supports:
      - 16-bit PNG (via cv2.IMREAD_UNCHANGED)
      - 16-bit TIFF / TIF (via tifffile or cv2)
      - NumPy .npy files
      - 8-bit formats (PNG, JPG, BMP) - upscaled to 16-bit and flagged in metadata.

    Returns:
      (image_uint16, metadata_dict)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found at: {p.resolve()}")

    ext = p.suffix.lower()
    meta: dict[str, Any] = {
        "file_path": str(p),
        "extension": ext,
        "original_dtype": "unknown",
        "bit_depth": 16,
        "was_upscaled_from_8bit": False,
        "shape": (),
    }

    if ext == ".npy":
        raw = np.load(str(p))
        meta["original_dtype"] = str(raw.dtype)
        if raw.ndim == 3 and raw.shape[2] in (3, 4):
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY if raw.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
        elif raw.ndim != 2:
            raise ValueError(f"NPY array must be 2D, got shape {raw.shape}")

        if raw.dtype == np.uint8:
            meta["was_upscaled_from_8bit"] = True
            logger.warning(f"File {p.name} is 8-bit NPY; upscaling to 16-bit.")
            img_16 = (raw.astype(np.uint16) * 257)
        elif raw.dtype == np.uint16:
            img_16 = raw
        elif np.issubdtype(raw.dtype, np.floating):
            # If normalized 0..1 or 0..65535
            max_val = np.max(raw) if raw.size > 0 else 1.0
            if max_val <= 1.05:
                img_16 = np.clip(raw * 65535.0, 0, 65535).astype(np.uint16)
            else:
                img_16 = np.clip(raw, 0, 65535).astype(np.uint16)
        else:
            img_16 = np.clip(raw, 0, 65535).astype(np.uint16)

    elif ext in (".tif", ".tiff"):
        try:
            raw = tifffile.imread(str(p))
        except Exception:
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

        if raw is None:
            raise IOError(f"Failed to read TIFF image: {p}")

        if raw.ndim == 3 and raw.shape[-1] in (3, 4):
            raw = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        elif raw.ndim > 2:
            raw = raw[0]  # Take first slice if multi-page

        meta["original_dtype"] = str(raw.dtype)
        if raw.dtype == np.uint8:
            meta["was_upscaled_from_8bit"] = True
            logger.warning(f"File {p.name} is 8-bit TIFF; upscaling to 16-bit.")
            img_16 = (raw.astype(np.uint16) * 257)
        elif raw.dtype == np.uint16:
            img_16 = raw
        elif np.issubdtype(raw.dtype, np.floating):
            max_val = np.max(raw) if raw.size > 0 else 1.0
            if max_val <= 1.05:
                img_16 = np.clip(raw * 65535.0, 0, 65535).astype(np.uint16)
            else:
                img_16 = np.clip(raw, 0, 65535).astype(np.uint16)
        else:
            img_16 = np.clip(raw, 0, 65535).astype(np.uint16)

    else:
        # Standard image reading via OpenCV (supports 16-bit PNG if saved with uint16)
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise IOError(f"Failed to read image with OpenCV: {p}")

        if raw.ndim == 3 and raw.shape[2] in (3, 4):
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY if raw.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)

        meta["original_dtype"] = str(raw.dtype)
        if raw.dtype == np.uint8:
            meta["was_upscaled_from_8bit"] = True
            logger.warning(f"File {p.name} is 8-bit image; upscaling to 16-bit.")
            img_16 = (raw.astype(np.uint16) * 257)
        elif raw.dtype == np.uint16:
            img_16 = raw
        else:
            img_16 = np.clip(raw, 0, 65535).astype(np.uint16)

    meta["shape"] = img_16.shape
    return img_16, meta


def save_image_16bit(path: str | Path, img: np.ndarray) -> None:
    """Save an image as a true 16-bit grayscale PNG or TIFF."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr16 = np.clip(img, 0, 65535).astype(np.uint16)
    ext = p.suffix.lower()

    if ext in (".tif", ".tiff"):
        tifffile.imwrite(str(p), arr16)
    elif ext == ".npy":
        np.save(str(p), arr16)
    else:
        # Defaults to cv2 imwrite (e.g. .png)
        success = cv2.imwrite(str(p), arr16)
        if not success:
            raise IOError(f"cv2.imwrite failed for 16-bit output: {p}")


def save_image_8bit_display(
    path: str | Path,
    img: np.ndarray,
    p_low: float = 0.5,
    p_high: float = 99.5,
) -> None:
    """Save an 8-bit PNG display copy using robust percentile windowing."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr8 = to_8bit_display(img, p_low=p_low, p_high=p_high)
    success = cv2.imwrite(str(p), arr8)
    if not success:
        raise IOError(f"cv2.imwrite failed for 8-bit output: {p}")


def create_comparison_panel(
    original: np.ndarray,
    processed: np.ndarray,
    enhanced: np.ndarray,
    titles: tuple[str, str, str] = (
        "Original / Raw (16-bit)",
        "Processed (Background Suppressed)",
        "Enhanced (Coronary Boosted)",
    ),
    display_8bit: bool = True,
) -> np.ndarray:
    """
    Create a side-by-side composite panel:
    [Original | Processed | Enhanced]
    With clean top title banners for clear clinical review.
    """
    if display_8bit:
        o8 = to_8bit_display(original)
        p8 = to_8bit_display(processed)
        e8 = to_8bit_display(enhanced)
        panels = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in (o8, p8, e8)]
    else:
        # Keep 16-bit 3-channel
        panels = [cv2.cvtColor(np.clip(img, 0, 65535).astype(np.uint16), cv2.COLOR_GRAY2BGR) for img in (original, processed, enhanced)]

    h, w = panels[0].shape[:2]
    header_h = 42

    composite_w = w * 3
    composite_h = h + header_h

    if display_8bit:
        composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
        header_color = (35, 35, 35)
        text_color = (240, 240, 240)
    else:
        composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint16)
        header_color = (9000, 9000, 9000)
        text_color = (62000, 62000, 62000)

    # Fill header
    composite[:header_h, :] = header_color

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65 if w >= 512 else 0.45
    thickness = 2 if w >= 512 else 1

    for i, (panel, title) in enumerate(zip(panels, titles)):
        x_start = i * w
        composite[header_h:, x_start : x_start + w] = panel

        # Render text centered in header section
        text_size, _ = cv2.getTextSize(title, font, font_scale, thickness)
        tx = x_start + (w - text_size[0]) // 2
        ty = (header_h + text_size[1]) // 2
        cv2.putText(composite, title, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Draw vertical separator line
        if i > 0:
            sep_color = (80, 80, 80) if display_8bit else (20000, 20000, 20000)
            cv2.line(composite, (x_start, 0), (x_start, composite_h), sep_color, 2)

    return composite


def save_comparison_panel(
    path: str | Path,
    original: np.ndarray,
    processed: np.ndarray,
    enhanced: np.ndarray,
    titles: tuple[str, str, str] = (
        "Original / Raw (16-bit)",
        "Processed (Background Suppressed)",
        "Enhanced (Coronary Boosted)",
    ),
) -> None:
    """Save the comparison panel as an 8-bit PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    panel_8 = create_comparison_panel(original, processed, enhanced, titles, display_8bit=True)
    success = cv2.imwrite(str(p), panel_8)
    if not success:
        raise IOError(f"Failed to save comparison panel: {p}")
