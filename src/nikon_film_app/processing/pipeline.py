from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .accelerator import Accelerator, BackendMode
from .film_presets import get_presets


@dataclass
class ProcessOptions:
    preset_name: str
    strength_percent: int = 100  # 0..100
    enable_grain: bool = True
    enable_vignette: bool = False
    enable_auto_baseline: bool = True


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
        out = preset.process(
            img,
            self.accel,
            strength01,
            options.enable_grain,
            options.enable_vignette,
            options.enable_auto_baseline,
        )
        return np.clip(out, 0.0, 1.0).astype(np.float32)

