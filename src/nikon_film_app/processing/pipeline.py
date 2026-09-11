from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .accelerator import Accelerator, BackendMode
from .film_presets import get_presets, GrainParams, GrainType


@dataclass
class ProcessOptions:
    preset_name: str
    strength_percent: int = 100  # 0..100
    enable_grain: bool = True
    grain_type: GrainType = GrainType.SILVER_HALIDE
    grain_size: int = 30          # 0..100
    grain_density: int = 30       # 0..100
    grain_roughness: int = 40     # 0..100
    grain_chroma_mix: int = 0     # 0..100
    grain_seed: Optional[int] = None
    # Vignette: off/auto/manual
    vignette_mode: str = "off"    # "off" | "auto" | "manual"
    vignette_amount: int = 0      # 0..100, used when mode=manual
    enable_auto_baseline: bool = True
    # Manual exposure and color temperature
    exposure_ev_x100: int = 0     # -200..200 represents -2..+2 EV
    temp_bias: int = 0            # -100..100 (cooler..warmer)


class ImageProcessor:
    def __init__(self, accelerator: Optional[Accelerator] = None) -> None:
        self.accel = accelerator or Accelerator(BackendMode.AUTO)
        self.presets = get_presets()

    def list_presets(self) -> list[str]:
        return list(self.presets.keys())

    def process_bgr01(self, img_bgr01: np.ndarray, options: ProcessOptions) -> np.ndarray:
        # Ensure float32 in 0..1 and 3 channels
        if img_bgr01.dtype != np.float32:
            img = img_bgr01.astype(np.float32)
        else:
            img = img_bgr01.copy()
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        strength01 = max(0.0, min(1.0, options.strength_percent / 100.0))
        preset = self.presets.get(options.preset_name)
        if preset is None:
            return img
        grain = GrainParams(
            enabled=options.enable_grain,
            grain_type=options.grain_type,
            size01=max(0.0, min(1.0, options.grain_size / 100.0)),
            density01=max(0.0, min(1.0, options.grain_density / 100.0)),
            roughness01=max(0.0, min(1.0, options.grain_roughness / 100.0)),
            chroma_mix01=max(0.0, min(1.0, options.grain_chroma_mix / 100.0)),
            seed=options.grain_seed,
        )
        # Vignette selection
        if options.vignette_mode == "manual":
            enable_vignette = options.vignette_amount > 0
            vignette_override = max(0.0, min(1.0, options.vignette_amount / 100.0))
        elif options.vignette_mode == "auto":
            enable_vignette = True
            vignette_override = None
        else:
            enable_vignette = False
            vignette_override = None
        # Manual exposure and color temperature
        exposure_ev = max(-2.0, min(2.0, options.exposure_ev_x100 / 100.0))
        temp01 = max(-1.0, min(1.0, options.temp_bias / 100.0))
        out = preset.process(
            img,
            self.accel,
            strength01,
            grain,
            enable_vignette,
            vignette_override,
            options.enable_auto_baseline,
            options.grain_seed,
            exposure_ev,
            temp01,
        )
        return np.clip(out, 0.0, 1.0).astype(np.float32)

