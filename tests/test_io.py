"""
Unit tests for CardioRT image, DICOM, and RAW I/O modules.
"""

from pathlib import Path
import numpy as np
import pytest

from cardio_rt.io import (
    load_image,
    save_image_16bit,
    save_image_8bit_display,
    create_comparison_panel,
    load_dicom,
    save_synthetic_dicom,
    load_raw,
    save_raw,
)


def test_png_16bit_io(tmp_path: Path):
    dummy_16 = np.random.randint(5000, 60000, size=(256, 256), dtype=np.uint16)
    out_file = tmp_path / "test_frame.png"
    save_image_16bit(out_file, dummy_16)

    loaded, meta = load_image(out_file)
    assert loaded.dtype == np.uint16
    assert loaded.shape == (256, 256)
    assert np.array_equal(dummy_16, loaded)
    assert meta["bit_depth"] == 16
    assert not meta["was_upscaled_from_8bit"]


def test_8bit_upscaling_and_flagging(tmp_path: Path):
    dummy_8 = np.random.randint(20, 240, size=(128, 128), dtype=np.uint8)
    npy_file = tmp_path / "test_8bit.npy"
    np.save(str(npy_file), dummy_8)

    loaded, meta = load_image(npy_file)
    assert loaded.dtype == np.uint16
    assert loaded.shape == (128, 128)
    assert meta["was_upscaled_from_8bit"] is True


def test_dicom_multiframe_io(tmp_path: Path):
    dummy_cine = np.random.randint(10000, 50000, size=(4, 128, 128), dtype=np.uint16)
    dcm_file = tmp_path / "cine.dcm"
    save_synthetic_dicom(dcm_file, dummy_cine, fps=15)

    loaded, meta = load_dicom(dcm_file)
    assert loaded.dtype == np.uint16
    assert loaded.shape == (4, 128, 128)
    assert meta["num_frames"] == 4


def test_raw_binary_io(tmp_path: Path):
    dummy_raw = np.random.randint(1000, 60000, size=(64, 64), dtype=np.uint16)
    raw_file = tmp_path / "frame.raw"
    save_raw(raw_file, dummy_raw)

    loaded, meta = load_raw(raw_file, width=64, height=64, dtype="uint16")
    assert loaded.dtype == np.uint16
    assert np.array_equal(dummy_raw, loaded)


def test_comparison_panel():
    o = np.zeros((100, 100), dtype=np.uint16)
    p = np.ones((100, 100), dtype=np.uint16) * 30000
    e = np.ones((100, 100), dtype=np.uint16) * 60000

    panel = create_comparison_panel(o, p, e, display_8bit=True)
    assert panel.ndim == 3
    assert panel.shape[1] == 300  # 3 panels side-by-side
    assert panel.dtype == np.uint8
