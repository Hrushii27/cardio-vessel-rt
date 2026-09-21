"""
Beer-Lambert optical density log transform for cardiology X-ray angiography.
Converts multiplicative tissue attenuation into an additive density domain.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def log_transform(
    img_norm: np.ndarray,
    polarity: str = "dark_vessels",
    epsilon: float = 1e-4,
    i0: float = 1.0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply Beer-Lambert logarithmic transformation to normalize input to optical density space.

    In physical X-ray transmission:
        I(x,y) = I_0 * exp(-integral mu(x,y,z) dz)

    In the optical density domain:
        OD(x,y) = -ln((I(x,y) + epsilon) / I_0) = ln(I_0) - ln(I(x,y) + epsilon)

    When polarity == "dark_vessels" (standard angiogram fluoroscopy), radio-opaque
    contrast agent creates higher optical density. In OD space, vessels become
    positive/prominent features on top of anatomical background structures (additive domain).

    Args:
        img_norm: Normalized [0.0, 1.0] float32 image.
        polarity: "dark_vessels" (standard) or "bright_vessels".
        epsilon: Small positive constant to prevent log(0).
        i0: Incident X-ray intensity normalization reference (default 1.0).
        out: Optional preallocated float32 buffer [H, W].

    Returns:
        Optical density float32 map [H, W].
    """
    if out is None:
        out = np.empty_like(img_norm, dtype=np.float32)

    # out = img_norm + epsilon
    np.add(img_norm, epsilon, out=out)

    # out = ln(img_norm + epsilon)
    np.log(out, out=out)

    log_i0 = float(np.log(i0 + epsilon))

    if polarity == "dark_vessels":
        # OD = log(I_0) - log(I) -> vessels with low I have high OD
        np.subtract(log_i0, out, out=out)
    else:
        # OD = log(I) - log(I_0)
        np.subtract(out, log_i0, out=out)

    return out


def inverse_log_transform(
    od_map: np.ndarray,
    polarity: str = "dark_vessels",
    i0: float = 1.0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Invert optical density map back to normalized transmission intensity space [0.0, 1.0].

    Args:
        od_map: Optical density float32 image.
        polarity: "dark_vessels" or "bright_vessels".
        i0: Incident intensity reference.
        out: Optional preallocated float32 buffer.

    Returns:
        Normalized intensity [0.0, 1.0] float32 image.
    """
    if out is None:
        out = np.empty_like(od_map, dtype=np.float32)

    if polarity == "dark_vessels":
        # I = I_0 * exp(-OD)
        np.negative(od_map, out=out)
        np.exp(out, out=out)
        np.multiply(out, i0, out=out)
    else:
        # I = I_0 * exp(OD)
        np.exp(od_map, out=out)
        np.multiply(out, i0, out=out)

    np.clip(out, 0.0, 1.0, out=out)
    return out
