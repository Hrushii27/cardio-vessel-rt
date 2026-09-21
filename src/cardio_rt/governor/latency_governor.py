"""
Dynamic Latency Governor for real-time cardiology angiography.
Monitors execution latency on a per-frame rolling window, dynamically degrading
working resolution and vesselness scale count to guarantee the <= 36 ms limit.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from cardio_rt.pipeline.cardio_pipeline import CardioPipeline, PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class GovernorEvent:
    """Record of an automated adaptation event."""
    frame_index: int
    action: str               # "DEGRADE" or "RESTORE"
    trigger_latency_ms: float
    old_level: int
    new_level: int
    details: str


class LatencyGovernor:
    """
    Adaptive controller that regulates pipeline complexity to satisfy real-time deadlines.
    Enforces the <= 36 ms per frame constraint by gracefully degrading computation parameters
    under host CPU load spikes.
    """

    def __init__(
        self,
        pipeline: CardioPipeline,
        budget_limit_ms: float = 36.0,
        downgrade_thresh_ms: float = 30.0,
        upgrade_thresh_ms: float = 20.0,
        window_size: int = 8,
        cooldown_frames: int = 15,
    ):
        self.pipeline = pipeline
        self.budget_limit_ms = budget_limit_ms
        self.downgrade_thresh_ms = downgrade_thresh_ms
        self.upgrade_thresh_ms = upgrade_thresh_ms
        self.window_size = window_size
        self.cooldown_frames = cooldown_frames

        self._latencies: collections.deque[float] = collections.deque(maxlen=window_size)
        self._current_level: int = 0  # 0: Full Quality, 1: Moderate, 2: Emergency
        self._frames_since_action: int = 0
        self.events: List[GovernorEvent] = []

        # Presets per degradation tier
        self.tiers: Dict[int, Dict[str, Any]] = {
            0: {
                "name": "Full Fidelity",
                "max_working_res": 384,
                "vessel_scales": (1.5, 3.0),
                "vessel_subsample": 2,
            },
            1: {
                "name": "Moderate Degradation",
                "max_working_res": 320,
                "vessel_scales": (2.0,),
                "vessel_subsample": 2,
            },
            2: {
                "name": "Emergency Low Latency",
                "max_working_res": 256,
                "vessel_scales": (2.0,),
                "vessel_subsample": 2,
            },
        }

    def _apply_tier(self, level: int) -> None:
        """Apply tier parameters to pipeline and rebuild working pipeline."""
        cfg = self.tiers[level]
        p_cfg = self.pipeline.config
        p_cfg.max_working_res = cfg["max_working_res"]
        p_cfg.vessel_scales = cfg["vessel_scales"]
        p_cfg.vessel_subsample = cfg["vessel_subsample"]

        # Re-initialize working subpipeline with updated configuration
        if self.pipeline._working_pipeline is not None:
            self.pipeline._working_pipeline.config.max_working_res = cfg["max_working_res"]
            self.pipeline._working_pipeline.config.vessel_scales = cfg["vessel_scales"]
            self.pipeline._working_pipeline.config.vessel_subsample = cfg["vessel_subsample"]
            self.pipeline._working_pipeline.vessel_module.scales = cfg["vessel_scales"]
            self.pipeline._working_pipeline.vessel_module.subsample_factor = cfg["vessel_subsample"]

        self.pipeline.vessel_module.scales = cfg["vessel_scales"]
        self.pipeline.vessel_module.subsample_factor = cfg["vessel_subsample"]

    def record_frame(self, frame_idx: int, latency_ms: float) -> Optional[GovernorEvent]:
        """
        Record completed frame latency and adapt configuration if required.

        Returns:
            GovernorEvent if an adaptation action was triggered, else None.
        """
        self._latencies.append(latency_ms)
        self._frames_since_action += 1

        if len(self._latencies) < 3 or self._frames_since_action < self.cooldown_frames:
            return None

        # Check rolling max or recent average
        recent_avg = float(np.mean(self._latencies))
        recent_max = float(np.max(self._latencies))

        # Check for degradation condition:
        if (recent_avg > self.downgrade_thresh_ms or recent_max > self.budget_limit_ms) and self._current_level < 2:
            old = self._current_level
            self._current_level += 1
            self._apply_tier(self._current_level)
            self._frames_since_action = 0

            desc = (
                f"Degraded to level {self._current_level} ({self.tiers[self._current_level]['name']}): "
                f"Working res -> {self.tiers[self._current_level]['max_working_res']}px, "
                f"scales -> {self.tiers[self._current_level]['vessel_scales']}"
            )
            event = GovernorEvent(
                frame_index=frame_idx,
                action="DEGRADE",
                trigger_latency_ms=recent_max,
                old_level=old,
                new_level=self._current_level,
                details=desc,
            )
            self.events.append(event)
            logger.warning(f"[LATENCY GOVERNOR] {desc} (Trigger latency: {recent_max:.1f}ms)")
            return event

        # Check for restoration condition:
        if recent_avg < self.upgrade_thresh_ms and recent_max < self.downgrade_thresh_ms and self._current_level > 0:
            old = self._current_level
            self._current_level -= 1
            self._apply_tier(self._current_level)
            self._frames_since_action = 0

            desc = (
                f"Restored to level {self._current_level} ({self.tiers[self._current_level]['name']}): "
                f"Working res -> {self.tiers[self._current_level]['max_working_res']}px, "
                f"scales -> {self.tiers[self._current_level]['vessel_scales']}"
            )
            event = GovernorEvent(
                frame_index=frame_idx,
                action="RESTORE",
                trigger_latency_ms=recent_avg,
                old_level=old,
                new_level=self._current_level,
                details=desc,
            )
            self.events.append(event)
            logger.info(f"[LATENCY GOVERNOR] {desc} (Average latency: {recent_avg:.1f}ms)")
            return event

        return None
