"""
Noise suppression modules for cardiology X-ray image processing.
"""

from cardio_rt.noise.spatial_denoise import (
    FastSpatialDenoise,
    anscombe_transform,
    inverse_anscombe_transform,
)

__all__ = [
    "FastSpatialDenoise",
    "anscombe_transform",
    "inverse_anscombe_transform",
]
