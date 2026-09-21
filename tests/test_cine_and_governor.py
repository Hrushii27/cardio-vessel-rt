"""
Unit tests for motion compensation, temporal filtering, and latency governor.
"""

import numpy as np
import pytest

from cardio_rt.cine_motion import MotionCompensator, TemporalBackgroundModel
from cardio_rt.noise import MotionAdaptiveTemporalFilter
from cardio_rt.governor import LatencyGovernor
from cardio_rt.pipeline import CardioPipeline


def test_motion_compensator():
    comp = MotionCompensator(subsample=2)
    # Create base image with a bright feature
    ref = np.zeros((128, 128), dtype=np.float32)
    ref[40:80, 40:80] = 1.0

    # Shifted image by (dx=+5, dy=+3)
    cur = np.zeros((128, 128), dtype=np.float32)
    cur[43:83, 45:85] = 1.0

    dx, dy, response = comp.estimate_shift(cur, ref)
    assert abs(dx - 5.0) < 1.5
    assert abs(dy - 3.0) < 1.5

    # Test warp
    warped = comp.warp_reference(ref, dx, dy)
    assert warped.shape == ref.shape


def test_temporal_filter():
    tf = MotionAdaptiveTemporalFilter(base_alpha=0.4)
    f1 = np.ones((64, 64), dtype=np.float32) * 0.5
    f2 = np.ones((64, 64), dtype=np.float32) * 0.5

    out1 = tf.process(f1)
    out2 = tf.process(f2)
    assert out1.shape == (64, 64)
    assert out2.shape == (64, 64)


def test_latency_governor():
    pipe = CardioPipeline()
    gov = LatencyGovernor(pipe, budget_limit_ms=36.0, downgrade_thresh_ms=30.0, window_size=4, cooldown_frames=2)

    # Simulate spike: 4 frames at 34 ms
    events = []
    for i, lat in enumerate([34.0, 35.0, 34.0, 35.0]):
        ev = gov.record_frame(i, lat)
        if ev:
            events.append(ev)

    # Governor should have triggered degradation to protect 36 ms budget
    assert len(events) > 0
    assert events[0].action == "DEGRADE"
    assert pipe.config.max_working_res <= 320
