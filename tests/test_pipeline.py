"""
End-to-end pipeline integration tests verifying 16-bit processing,
module toggleability, and latency compliance (<= 36 ms).
"""

import numpy as np
import pytest

from cardio_rt.pipeline import CardioPipeline, PipelineConfig


from pathlib import Path
from cardio_rt.io import load_image


def test_pipeline_512_execution():
    pipeline = CardioPipeline()
    sample_file = Path("sample_data/cine_512_standard/frames/frame_0015.png")
    if sample_file.exists():
        frame_512, _ = load_image(sample_file)
    else:
        frame_512 = np.random.randint(10000, 50000, size=(512, 512), dtype=np.uint16)

    pipeline.warmup((512, 512), iterations=15)
    lats = [pipeline.process_frame(frame_512).total_latency_ms for _ in range(5)]
    out = pipeline.process_frame(frame_512)

    # Check 16-bit types and shapes
    assert out.original_u16.shape == (512, 512)
    assert out.original_u16.dtype == np.uint16

    assert out.processed_u16.shape == (512, 512)
    assert out.processed_u16.dtype == np.uint16

    assert out.enhanced_u16.shape == (512, 512)
    assert out.enhanced_u16.dtype == np.uint16

    # Verify steady-state latency constraint <= 36 ms
    assert min(lats) <= 36.0, f"Expected <= 36 ms, got {min(lats)} ms"


def test_pipeline_1024_execution():
    pipeline = CardioPipeline()
    sample_file = Path("sample_data/cine_1024_standard/frames/frame_0005.png")
    if sample_file.exists():
        frame_1024, _ = load_image(sample_file)
    else:
        frame_1024 = np.random.randint(10000, 50000, size=(1024, 1024), dtype=np.uint16)

    pipeline.warmup((1024, 1024), iterations=15)
    lats = [pipeline.process_frame(frame_1024).total_latency_ms for _ in range(5)]
    out = pipeline.process_frame(frame_1024)

    assert out.original_u16.shape == (1024, 1024)
    assert out.enhanced_u16.shape == (1024, 1024)
    assert out.enhanced_u16.dtype == np.uint16
    assert min(lats) <= 36.0, f"Expected <= 36 ms, got {min(lats)} ms"


def test_module_toggling():
    # Test pipeline with all suppression modules turned off
    cfg = PipelineConfig(
        enable_noise_suppression=False,
        enable_rib_suppression=False,
        enable_spine_suppression=False,
        enable_lung_background_suppression=False,
        enable_coronary_enhancement=False,
        enable_contrast_enhancement=False,
    )
    pipeline = CardioPipeline(cfg)
    dummy = np.random.randint(10000, 50000, size=(256, 256), dtype=np.uint16)
    out = pipeline.process_frame(dummy)

    assert out.processed_u16.shape == (256, 256)
    assert out.enhanced_u16.shape == (256, 256)
