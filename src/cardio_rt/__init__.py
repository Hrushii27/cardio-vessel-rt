"""
CardioRT: Real-Time Cardiology X-ray & Cine Coronary Angiography Processing System
Designed for <= 36 ms per frame low-latency clinical imaging pipelines.
"""

__version__ = "1.0.0"
__author__ = "Cardiology Real-Time Imaging Team"

from cardio_rt.pipeline.cardio_pipeline import CardioPipeline, PipelineConfig

__all__ = ["CardioPipeline", "PipelineConfig"]
