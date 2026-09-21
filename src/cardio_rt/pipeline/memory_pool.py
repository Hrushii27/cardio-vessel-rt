"""
Zero-allocation memory pool for real-time angiography pipeline hot loop.
Preallocates and reuses internal float32 and uint16 working buffers.
"""

from __future__ import annotations

from typing import Dict
import numpy as np


class PipelineMemoryPool:
    """
    Manages preallocated numpy array buffers for a given frame resolution.
    Eliminates heap allocation and garbage collection pauses in the processing loop.
    """

    def __init__(self, shape: tuple[int, int] = (512, 512)):
        self.shape = shape
        self.buffers: Dict[str, np.ndarray] = {}
        self.allocate(shape)

    def allocate(self, shape: tuple[int, int]) -> None:
        """Allocate all working buffers for the requested image shape."""
        self.shape = shape
        h, w = shape

        self.buffers = {
            "norm_in": np.empty((h, w), dtype=np.float32),
            "collimator_mask": np.empty((h, w), dtype=np.uint8),
            "log_od": np.empty((h, w), dtype=np.float32),
            "denoised_od": np.empty((h, w), dtype=np.float32),
            "spine_od": np.empty((h, w), dtype=np.float32),
            "rib_od": np.empty((h, w), dtype=np.float32),
            "lung_od": np.empty((h, w), dtype=np.float32),
            "vessel_od": np.empty((h, w), dtype=np.float32),
            "processed_linear": np.empty((h, w), dtype=np.float32),
            "enhanced_linear": np.empty((h, w), dtype=np.float32),
            "processed_u16": np.empty((h, w), dtype=np.uint16),
            "enhanced_u16": np.empty((h, w), dtype=np.uint16),
            "scratch_f32_1": np.empty((h, w), dtype=np.float32),
            "scratch_f32_2": np.empty((h, w), dtype=np.float32),
        }

    def get(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        """Get preallocated buffer, reallocating if the resolution changed."""
        if shape != self.shape or name not in self.buffers:
            self.allocate(shape)
        return self.buffers[name]
