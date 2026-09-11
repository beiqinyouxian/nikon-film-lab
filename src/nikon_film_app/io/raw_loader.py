from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import rawpy  # type: ignore


@dataclass
class RawLoadResult:
    image_bgr8: np.ndarray  # uint8 0..255 BGR
    size: Tuple[int, int]  # (h, w)


def _lift_exposure_bgr8(bgr8: np.ndarray, target_median: float = 0.42) -> np.ndarray:
    """If the demosaic result is still dark, gently lift midtones toward a natural median."""
    img = bgr8.astype(np.float32) / 255.0
    # Rec.709-ish luminance on BGR
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    y = 0.0722 * b + 0.7152 * g + 0.2126 * r
    med = float(np.median(y)) + 1e-6
    if med >= target_median:
        return bgr8
    # Cap gain so we don't blow highlights too hard
    gain = min(target_median / med, 3.0)
    out = np.clip(img * gain, 0.0, 1.0)
    # Soft highlight rolloff after gain
    hi = np.maximum(0.0, out - 0.85) / 0.15
    out = out - hi * hi * 0.08
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_nef_to_bgr8(path: str, auto_bright: bool = True, bright: float = 1.35) -> RawLoadResult:
    """Demosaic Nikon NEF to displayable BGR8 with natural exposure.

    Defaults:
    - camera white balance
    - rawpy auto-bright (more aggressive threshold)
    - slightly elevated ``bright``
    - native 8-bit sRGB
    - extra midtone lift if the frame is still dark
    """
    with rawpy.imread(path) as raw:
        kwargs = dict(
            output_bps=8,
            user_flip=0,
            no_auto_bright=not auto_bright,
            use_camera_wb=True,
            # Allow a bit more highlight clipping so midtones lift (rawpy default is 0.01)
            auto_bright_thr=0.001 if auto_bright else 0.01,
            gamma=(2.222, 4.5),
            bright=float(bright),
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            output_color=rawpy.ColorSpace.sRGB,
            highlight_mode=2,  # Blend
        )
        try:
            rgb8 = raw.postprocess(**kwargs)
        except TypeError:
            kwargs.pop("highlight_mode", None)
            kwargs.pop("output_color", None)
            kwargs.pop("auto_bright_thr", None)
            rgb8 = raw.postprocess(**kwargs)
    h, w, _ = rgb8.shape
    bgr8 = rgb8[..., ::-1].copy()
    bgr8 = _lift_exposure_bgr8(bgr8, target_median=0.42)
    return RawLoadResult(image_bgr8=bgr8, size=(h, w))
