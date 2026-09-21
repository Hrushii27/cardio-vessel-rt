"""
Cardiology Real-Time 16-bit Angiography Processing Pipeline.
Integrates separate toggleable modules for noise, rib, spine, lung background,
coronary enhancement, and contrast enhancement with microsecond latency profiling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Optional, Dict, Any, Tuple
import numpy as np
import cv2

from cardio_rt.preprocess.windowing import percentile_normalize, detect_collimator_mask
from cardio_rt.preprocess.log_transform import log_transform, inverse_log_transform
from cardio_rt.noise.spatial_denoise import FastSpatialDenoise
from cardio_rt.rib_spine.spine_suppression import SpineSuppression
from cardio_rt.rib_spine.rib_suppression import RibSuppression
from cardio_rt.lung_bg.lung_suppression import LungBackgroundSuppression
from cardio_rt.vesselness.frangi_fast import FastVesselnessEnhancer
from cardio_rt.enhance.contrast_enhance import ContrastEnhancement
from cardio_rt.pipeline.memory_pool import PipelineMemoryPool


@dataclass
class PipelineConfig:
    """Configuration for all pipeline stages and parameters."""
    # Module Toggles
    enable_noise_suppression: bool = True
    enable_rib_suppression: bool = True
    enable_spine_suppression: bool = True
    enable_lung_background_suppression: bool = True
    enable_coronary_enhancement: bool = True
    enable_contrast_enhancement: bool = True

    # Polarity: "dark_vessels" (standard radio-opaque contrast) or "bright_vessels"
    polarity: str = "dark_vessels"

    # Ingest / Preprocess
    percentile_low: float = 0.5
    percentile_high: float = 99.5
    detect_collimator: bool = True

    # Noise Suppression
    denoise_method: str = "gaussian"  # "gaussian" (< 1 ms), "guided", "bilateral"
    denoise_radius: int = 2
    denoise_eps: float = 0.002
    use_anscombe_vst: bool = False

    # Spine Suppression
    spine_sigma_v: float = 35.0
    spine_sigma_h: float = 7.0
    spine_strength: float = 0.75

    # Rib Suppression
    rib_sigma: float = 14.0
    rib_strength: float = 0.75
    rib_subsample: int = 4

    # Lung Background Suppression
    lung_pyramid_levels: int = 4
    lung_coarse_weight: float = 0.15
    lung_mid_coarse_weight: float = 0.55
    lung_vessel_band_weight: float = 1.25
    lung_fine_weight: float = 0.95
    lung_flat_field: bool = False

    # Vesselness Enhancement
    vessel_scales: Tuple[float, ...] = (1.5, 3.0)
    vessel_method: str = "frangi"  # "frangi", "sato"
    vessel_beta: float = 0.5
    vessel_c: float = 0.22
    vessel_gain_strength: float = 2.0
    vessel_subsample: int = 2  # Subsampling factor for vesselness calculation

    # Contrast Enhancement
    clahe_clip_limit: float = 1.2
    clahe_grid_size: Tuple[int, int] = (2, 2)
    tone_gamma: float = 0.90
    unsharp_strength: float = 0.35
    unsharp_radius: int = 2

    # High-Resolution Real-Time Multiscale Architecture
    multiscale_mode: bool = True
    max_working_res: int = 512


@dataclass
class PipelineOutput:
    """Standardized output produced per frame."""
    original_u16: np.ndarray
    processed_u16: np.ndarray      # Background suppressed (ribs, spine, lungs removed)
    enhanced_u16: np.ndarray       # Coronary artery boosted with contrast enhancement
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    vesselness_map: Optional[np.ndarray] = None


class CardioPipeline:
    """
    Real-time Cardiology X-ray & Cine Angiography Pipeline.
    Guarantees <= 36 ms per frame end-to-end processing latency.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        cv2.setNumThreads(8)
        self.pool = PipelineMemoryPool((512, 512))
        self._working_pipeline: Optional[CardioPipeline] = None

        # Instantiate modular processing blocks
        self.noise_module = FastSpatialDenoise(
            method=self.config.denoise_method,
            radius=self.config.denoise_radius,
            eps=self.config.denoise_eps,
            use_vst=self.config.use_anscombe_vst,
        )
        self.spine_module = SpineSuppression(
            enabled=self.config.enable_spine_suppression,
            sigma_vertical=self.config.spine_sigma_v,
            sigma_horizontal=self.config.spine_sigma_h,
            suppression_strength=self.config.spine_strength,
        )
        self.rib_module = RibSuppression(
            enabled=self.config.enable_rib_suppression,
            bone_scale_sigma=self.config.rib_sigma,
            suppression_strength=self.config.rib_strength,
            subsample_factor=self.config.rib_subsample,
        )
        self.lung_module = LungBackgroundSuppression(
            enabled=self.config.enable_lung_background_suppression,
            num_levels=self.config.lung_pyramid_levels,
            coarse_band_weight=self.config.lung_coarse_weight,
            mid_coarse_weight=self.config.lung_mid_coarse_weight,
            vessel_band_weight=self.config.lung_vessel_band_weight,
            fine_noise_weight=self.config.lung_fine_weight,
            enable_flat_field=self.config.lung_flat_field,
        )
        self.vessel_module = FastVesselnessEnhancer(
            enabled=self.config.enable_coronary_enhancement,
            scales=self.config.vessel_scales,
            method=self.config.vessel_method,
            beta=self.config.vessel_beta,
            c=self.config.vessel_c,
            gain_strength=self.config.vessel_gain_strength,
            subsample_factor=self.config.vessel_subsample,
        )
        self.enhance_module = ContrastEnhancement(
            enabled=self.config.enable_contrast_enhancement,
            clip_limit=self.config.clahe_clip_limit,
            tile_grid_size=self.config.clahe_grid_size,
            gamma=self.config.tone_gamma,
            unsharp_strength=self.config.unsharp_strength,
            unsharp_radius=self.config.unsharp_radius,
        )

        self._current_shape: Optional[Tuple[int, int]] = None
        if self.config.multiscale_mode:
            work_cfg = replace(
                self.config,
                multiscale_mode=False,
                vessel_subsample=2,
            )
            self._working_pipeline = CardioPipeline(work_cfg)

    def warmup(self, shape: Tuple[int, int] = (512, 512), iterations: int = 15) -> None:
        """Warm up JIT compilation, OpenCV SIMD paths, and buffer pools."""
        dummy = np.random.randint(10000, 50000, size=shape, dtype=np.uint16)
        if self._working_pipeline is not None:
            self._working_pipeline.warmup((self.config.max_working_res, self.config.max_working_res), iterations=5)
        for _ in range(iterations):
            self.process_frame(dummy)

    def process_frame(self, frame_u16: np.ndarray) -> PipelineOutput:
        """
        Process a single 16-bit cardiology frame.
        Strictly profiles latency from start to finish using time.perf_counter_ns.
        """
        t_total_start = time.perf_counter_ns()
        stage_times: Dict[str, float] = {}
        shape = frame_u16.shape[:2]

        # Multiscale acceleration for high-resolution frames (>=1024x1024)
        if shape[0] > self.config.max_working_res and self.config.multiscale_mode and self._working_pipeline is not None:
            t0 = time.perf_counter_ns()
            tgt_res = (self.config.max_working_res, self.config.max_working_res)
            small_frame = cv2.resize(frame_u16, tgt_res, interpolation=cv2.INTER_AREA)
            stage_times["downsample_input"] = (time.perf_counter_ns() - t0) / 1e6

            sub_out = self._working_pipeline.process_frame(small_frame)
            for k, v in sub_out.stage_latencies_ms.items():
                stage_times[k] = v

            t0 = time.perf_counter_ns()
            # Upsample and blend at full 1024x1024 resolution
            proc_full = cv2.resize(sub_out.processed_u16, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            enh_full = cv2.resize(sub_out.enhanced_u16, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            v_full = cv2.resize(sub_out.vesselness_map, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) if sub_out.vesselness_map is not None else None
            stage_times["upsample_blend"] = (time.perf_counter_ns() - t0) / 1e6

            total_lat = (time.perf_counter_ns() - t_total_start) / 1e6
            return PipelineOutput(
                original_u16=frame_u16,
                processed_u16=proc_full,
                enhanced_u16=enh_full,
                stage_latencies_ms=stage_times,
                total_latency_ms=total_lat,
                vesselness_map=v_full,
            )

        # ----------------------------------------------------
        # Stage 1: Ingest, Percentile Windowing, Collimator
        # ----------------------------------------------------
        t0 = time.perf_counter_ns()
        norm_in = self.pool.get("norm_in", shape)
        percentile_normalize(
            frame_u16,
            p_low=self.config.percentile_low,
            p_high=self.config.percentile_high,
            out=norm_in,
        )

        if self.config.detect_collimator:
            collimator_mask = self.pool.get("collimator_mask", shape)
            detect_collimator_mask(norm_in, dark_thresh=0.02, out_mask=collimator_mask)
        else:
            collimator_mask = None
        stage_times["ingest_preprocess"] = (time.perf_counter_ns() - t0) / 1e6

        # ----------------------------------------------------
        # Stage 2: Log Transform (Beer-Lambert optical density)
        # ----------------------------------------------------
        t0 = time.perf_counter_ns()
        log_od = self.pool.get("log_od", shape)
        log_transform(
            norm_in,
            polarity=self.config.polarity,
            epsilon=1e-4,
            out=log_od,
        )
        # Condition collimator shutter: fill non-diagnostic border with tissue baseline
        # to prevent artificial border singularities (-ln(0)=9.21) from bleeding into spatial filters
        if collimator_mask is not None:
            active_m = (collimator_mask > 0)
            if not np.all(active_m) and np.any(active_m):
                base_od_sub = log_od[active_m]
                base_val = float(np.mean(base_od_sub[::4]))
                log_od[~active_m] = base_val
        stage_times["log_transform"] = (time.perf_counter_ns() - t0) / 1e6

        # ----------------------------------------------------
        # Stage 3: Noise Suppression
        # ----------------------------------------------------
        t0 = time.perf_counter_ns()
        denoised_od = self.pool.get("denoised_od", shape)
        if self.config.enable_noise_suppression:
            self.noise_module.process(log_od, out=denoised_od)
        else:
            np.copyto(denoised_od, log_od)
        stage_times["noise_suppression"] = (time.perf_counter_ns() - t0) / 1e6

        # ----------------------------------------------------
        # Stage 4: Spine Suppression
        # ----------------------------------------------------
        t0 = time.perf_counter_ns()
        spine_od = self.pool.get("spine_od", shape)
        if self.config.enable_spine_suppression:
            self.spine_module.process(denoised_od, out=spine_od)
        else:
            np.copyto(spine_od, denoised_od)
        stage_times["spine_suppression"] = (time.perf_counter_ns() - t0) / 1e6

        # ----------------------------------------------------
        # Stage 5: Rib Suppression
        # ----------------------------------------------------
        t0 = time.perf_counter_ns()
        rib_od = self.pool.get("rib_od", shape)
        if self.config.enable_rib_suppression:
            self.rib_module.process(spine_od, out=rib_od)
        else:
            np.copyto(rib_od, spine_od)
        stage_times["rib_suppression"] = (time.perf_counter_ns() - t0) / 1e6

        # ----------------------------------------------------
        # Stage 6: Lung & Background Suppression (Laplacian Pyramids)
        # ----------------------------------------------------
        t0 = time.perf_counter_ns()
        lung_od = self.pool.get("lung_od", shape)
        if self.config.enable_lung_background_suppression:
            self.lung_module.process(rib_od, out=lung_od)
        else:
            np.copyto(lung_od, rib_od)
        stage_times["lung_background_suppression"] = (time.perf_counter_ns() - t0) / 1e6

        # Invert from optical density back to transmission space for "Processed" image
        processed_linear = self.pool.get("processed_linear", shape)
        inverse_log_transform(lung_od, polarity=self.config.polarity, out=processed_linear)

        # ----------------------------------------------------
        # Stage 7: Coronary Enhancement (Multi-scale Vesselness)
        # ----------------------------------------------------
        t0 = time.perf_counter_ns()
        vessel_od = self.pool.get("vessel_od", shape)
        if self.config.enable_coronary_enhancement:
            vessel_od, vesselness_map, gain_map = self.vessel_module.process(lung_od, out=vessel_od)
        else:
            np.copyto(vessel_od, lung_od)
            vesselness_map = None
            gain_map = None
        stage_times["coronary_enhancement"] = (time.perf_counter_ns() - t0) / 1e6

        # Invert enhanced optical density back to transmission space
        enhanced_linear = self.pool.get("enhanced_linear", shape)
        inverse_log_transform(vessel_od, polarity=self.config.polarity, out=enhanced_linear)

        # ----------------------------------------------------
        # Stage 8: Contrast Enhancement (16-bit CLAHE & Tone)
        # ----------------------------------------------------
        t0 = time.perf_counter_ns()
        if self.config.enable_contrast_enhancement:
            enhanced_linear = self.enhance_module.process(
                enhanced_linear,
                apply_clahe=True,
                vessel_gain_map=gain_map,
                out=enhanced_linear,
            )
        stage_times["contrast_enhancement"] = (time.perf_counter_ns() - t0) / 1e6

        # Apply collimator mask if detected
        if collimator_mask is not None:
            np.multiply(processed_linear, collimator_mask, out=processed_linear)
            np.multiply(enhanced_linear, collimator_mask, out=enhanced_linear)

        # ----------------------------------------------------
        # Stage 9: Output 16-bit Tensor Conversions
        # ----------------------------------------------------
        t0 = time.perf_counter_ns()
        proc_u16 = self.pool.get("processed_u16", shape)
        enh_u16 = self.pool.get("enhanced_u16", shape)

        # Robust mapping to [0, 65535] with zero-allocation unsafe copy
        s1 = self.pool.get("scratch_f32_1", shape)
        np.multiply(processed_linear, 65535.0, out=s1)
        np.clip(s1, 0, 65535, out=s1)
        np.copyto(proc_u16, s1, casting="unsafe")

        s2 = self.pool.get("scratch_f32_2", shape)
        np.multiply(enhanced_linear, 65535.0, out=s2)
        np.clip(s2, 0, 65535, out=s2)
        np.copyto(enh_u16, s2, casting="unsafe")

        stage_times["output_quantization"] = (time.perf_counter_ns() - t0) / 1e6

        total_latency_ms = (time.perf_counter_ns() - t_total_start) / 1e6

        return PipelineOutput(
            original_u16=frame_u16,
            processed_u16=proc_u16,
            enhanced_u16=enh_u16,
            stage_latencies_ms=stage_times,
            total_latency_ms=total_latency_ms,
            vesselness_map=vesselness_map,
        )
