"""
Unit tests for individual pipeline processing modules.
"""

import numpy as np
import pytest

from cardio_rt.preprocess.windowing import percentile_normalize, detect_collimator_mask
from cardio_rt.preprocess.log_transform import log_transform, inverse_log_transform
from cardio_rt.noise.spatial_denoise import FastSpatialDenoise
from cardio_rt.rib_spine.spine_suppression import SpineSuppression
from cardio_rt.rib_spine.rib_suppression import RibSuppression
from cardio_rt.lung_bg.lung_suppression import LungBackgroundSuppression
from cardio_rt.vesselness.frangi_fast import FastVesselnessEnhancer
from cardio_rt.enhance.contrast_enhance import ContrastEnhancement


def test_percentile_normalize():
    arr = np.linspace(1000, 60000, 10000, dtype=np.uint16).reshape(100, 100)
    norm, vmin, vmax = percentile_normalize(arr, 1.0, 99.0)
    assert norm.shape == (100, 100)
    assert norm.dtype == np.float32
    assert 0.0 <= np.min(norm) <= 0.05
    assert 0.95 <= np.max(norm) <= 1.0


def test_log_transform_invertibility():
    img = np.random.uniform(0.1, 0.9, size=(128, 128)).astype(np.float32)
    od = log_transform(img, polarity="dark_vessels")
    reconstructed = inverse_log_transform(od, polarity="dark_vessels")
    np.testing.assert_allclose(img, reconstructed, atol=1e-3)


def test_spine_suppression_toggle():
    s_on = SpineSuppression(enabled=True)
    s_off = SpineSuppression(enabled=False)

    od = np.random.uniform(0.1, 0.8, size=(128, 128)).astype(np.float32)
    out_on, spine_map = s_on.process(od)
    out_off, _ = s_off.process(od)

    assert out_on.shape == (128, 128)
    np.testing.assert_array_equal(out_off, od)


def test_rib_suppression_toggle():
    r_on = RibSuppression(enabled=True)
    r_off = RibSuppression(enabled=False)

    od = np.random.uniform(0.1, 0.8, size=(128, 128)).astype(np.float32)
    out_on, rib_map = r_on.process(od)
    out_off, _ = r_off.process(od)

    assert out_on.shape == (128, 128)
    np.testing.assert_array_equal(out_off, od)


def test_lung_suppression_reconstruction():
    lung = LungBackgroundSuppression(enabled=True, num_levels=3)
    od = np.random.uniform(0.2, 0.7, size=(128, 128)).astype(np.float32)
    out, bg = lung.process(od)
    assert out.shape == (128, 128)
    assert np.all(out >= 0.0)


def test_fast_vesselness_soft_gain():
    enh = FastVesselnessEnhancer(enabled=True, scales=(1.5, 3.0), subsample_factor=1)
    od = np.random.uniform(0.1, 0.5, size=(128, 128)).astype(np.float32)
    out_od, vesselness, gain = enh.process(od)

    assert out_od.shape == (128, 128)
    assert np.all(gain >= 1.0)  # Continuous soft gain >= 1.0
    assert np.all(vesselness >= 0.0) and np.all(vesselness <= 1.0)


def test_contrast_enhancement():
    ce = ContrastEnhancement(enabled=True)
    img = np.random.uniform(0.1, 0.8, size=(128, 128)).astype(np.float32)
    out = ce.process(img, apply_clahe=True)
    assert out.shape == (128, 128)
    assert 0.0 <= np.min(out) <= np.max(out) <= 1.0
