"""
Raw headerless binary image and cine loader for cardiology fluoroscopy.
Supports arbitrary width, height, data types, byte offsets, and endianness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np


def load_raw(
    file_path: str | Path,
    width: int = 512,
    height: int = 512,
    dtype: str = "uint16",
    offset: int = 0,
    byte_order: str = "little",  # "little" (<) or "big" (>)
    num_frames: int = 1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load raw binary image/cine data from disk.

    Args:
        file_path: Path to raw binary file.
        width: Image width in pixels.
        height: Image height in pixels.
        dtype: Data type ('uint16', 'uint8', 'float32', 'int16').
        offset: Header byte offset to skip.
        byte_order: 'little' or 'big' endian.
        num_frames: Number of cine frames (default: 1).

    Returns:
        (image_u16, metadata_dict)
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"RAW file not found: {p.resolve()}")

    prefix = "<" if byte_order.lower().startswith("l") else ">"
    np_dtype = np.dtype(dtype).newbyteorder(prefix)
    pixels_per_frame = width * height
    total_pixels = pixels_per_frame * num_frames

    with open(p, "rb") as f:
        f.seek(offset)
        raw_bytes = f.read(total_pixels * np_dtype.itemsize)

    arr = np.frombuffer(raw_bytes, dtype=np_dtype)

    if arr.size != total_pixels:
        raise ValueError(
            f"Expected {total_pixels} pixels ({total_pixels * np_dtype.itemsize} bytes), "
            f"got {arr.size} pixels ({len(raw_bytes)} bytes) from {p.name}"
        )

    if num_frames > 1:
        arr = arr.reshape((num_frames, height, width))
    else:
        arr = arr.reshape((height, width))

    meta = {
        "file_path": str(p),
        "width": width,
        "height": height,
        "dtype": dtype,
        "offset": offset,
        "byte_order": byte_order,
        "num_frames": num_frames,
    }

    # Normalize to 16-bit uint16 if needed
    if arr.dtype == np.uint8:
        arr_u16 = (arr.astype(np.uint16) * 257)
        meta["was_upscaled_from_8bit"] = True
    elif arr.dtype == np.uint16:
        arr_u16 = arr
    else:
        arr_f32 = arr.astype(np.float32)
        vmin, vmax = np.min(arr_f32), np.max(arr_f32)
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = np.clip((arr_f32 - vmin) / (vmax - vmin), 0.0, 1.0)
        arr_u16 = (norm * 65535.0).astype(np.uint16)

    return arr_u16, meta


def save_raw(
    file_path: str | Path,
    data: np.ndarray,
) -> None:
    """Save raw binary dump to disk."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(data.tobytes())
