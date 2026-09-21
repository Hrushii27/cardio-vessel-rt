"""
DICOM loader for cardiology angiography frames and multi-frame cine series.
Handles MONOCHROME1/2 photometric interpretations, RescaleSlope/Intercept,
and metadata extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

try:
    import pydicom
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

logger = logging.getLogger(__name__)


def load_dicom(
    file_path: str | Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load DICOM cardiology angiography file.
    Supports single-frame images and multi-frame cine series (N, H, W).

    Args:
        file_path: Path to DICOM file.

    Returns:
        (frames_u16, metadata_dict)
        where frames_u16 is either [H, W] or [N, H, W] uint16.
    """
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required for loading DICOM files. Install via `pip install pydicom`.")

    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"DICOM file not found at: {p.resolve()}")

    ds = pydicom.dcmread(str(p))

    # Apply modality LUT (RescaleSlope, RescaleIntercept) if available
    try:
        pixel_array = apply_modality_lut(ds.pixel_array, ds)
    except Exception:
        pixel_array = ds.pixel_array

    # Extract clinical metadata
    meta: Dict[str, Any] = {
        "file_path": str(p),
        "modality": getattr(ds, "Modality", "XA"),
        "patient_id": getattr(ds, "PatientID", "ANONYMOUS"),
        "study_date": getattr(ds, "StudyDate", ""),
        "series_description": getattr(ds, "SeriesDescription", ""),
        "photometric_interpretation": getattr(ds, "PhotometricInterpretation", "MONOCHROME2"),
        "rows": getattr(ds, "Rows", pixel_array.shape[-2]),
        "columns": getattr(ds, "Columns", pixel_array.shape[-1]),
        "bits_allocated": getattr(ds, "BitsAllocated", 16),
        "bits_stored": getattr(ds, "BitsStored", 12),
        "kvp": getattr(ds, "KVP", None),
        "frame_time_ms": getattr(ds, "FrameTime", 66.6),  # Default 15 fps
        "num_frames": 1,
    }

    if pixel_array.ndim == 3 and pixel_array.shape[0] > 1:
        meta["num_frames"] = pixel_array.shape[0]

    # Handle Photometric Interpretation:
    # MONOCHROME1: 0 is white (bright), max is black (dark) -> invert to MONOCHROME2
    if meta["photometric_interpretation"] == "MONOCHROME1":
        max_possible = (2 ** meta["bits_stored"]) - 1
        pixel_array = max_possible - pixel_array
        meta["inverted_from_monochrome1"] = True
    else:
        meta["inverted_from_monochrome1"] = False

    # Normalize to 16-bit uint16 [0, 65535]
    arr_f32 = pixel_array.astype(np.float32)
    p_low = np.percentile(arr_f32, 0.1)
    p_high = np.percentile(arr_f32, 99.9)
    if p_high <= p_low:
        p_high = p_low + 1.0

    norm = np.clip((arr_f32 - p_low) / (p_high - p_low), 0.0, 1.0)
    arr_u16 = (norm * 65535.0).astype(np.uint16)

    return arr_u16, meta


def save_synthetic_dicom(
    file_path: str | Path,
    frames_u16: np.ndarray,
    fps: int = 15,
) -> None:
    """
    Export synthetic frames as a compliant multi-frame DICOM file for testing.
    """
    if not HAS_PYDICOM:
        return

    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(p), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "XA"
    ds.PatientID = "SYNTH_PATIENT_01"
    ds.StudyDescription = "Cine Coronary Angiography"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.FrameTime = round(1000.0 / fps, 1)

    if frames_u16.ndim == 3:
        ds.NumberOfFrames = frames_u16.shape[0]
        ds.Rows = frames_u16.shape[1]
        ds.Columns = frames_u16.shape[2]
    else:
        ds.NumberOfFrames = 1
        ds.Rows = frames_u16.shape[0]
        ds.Columns = frames_u16.shape[1]

    ds.PixelData = frames_u16.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(p))
