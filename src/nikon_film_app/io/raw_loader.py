from __future__ import annotations

"""Nikon NEF demosaic via LibRaw/rawpy.

Correct-ish baseline for camera-like sRGB preview/export:
- camera white balance from NEF
- LibRaw auto brightness (default threshold — do NOT lower it)
- sRGB output color with LibRaw's default gamma (do not override)
- native 8-bit output (no 16-bit >> 8)
- respect file orientation (user_flip=None)
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import rawpy  # type: ignore


@dataclass
class RawLoadResult:
    image_bgr8: np.ndarray  # uint8 0..255 BGR, contiguous
    size: Tuple[int, int]  # (h, w)


def load_nef_to_bgr8(path: str, auto_bright: bool = True) -> RawLoadResult:
    """Demosaic a Nikon NEF to BGR8 sRGB for display and further processing."""
    with rawpy.imread(path) as raw:
        # Prefer camera WB; fall back to daylight multipliers if missing/zero.
        use_cam_wb = True
        try:
            wb = getattr(raw, "camera_whitebalance", None)
            if wb is not None and hasattr(wb, "__len__") and len(wb) >= 3:
                if not (wb[0] > 0 and wb[1] > 0 and wb[2] > 0):
                    use_cam_wb = False
        except Exception:
            use_cam_wb = True

        kwargs = dict(
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            half_size=False,
            four_color_rgb=False,
            use_camera_wb=use_cam_wb,
            use_auto_wb=not use_cam_wb,
            user_wb=None,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=8,
            # None = use orientation from the RAW file (Nikon often sets this)
            user_flip=None,
            no_auto_bright=not auto_bright,
            # Keep LibRaw default threshold (~0.01). Smaller values brighten LESS.
            auto_bright_thr=0.01,
            adjust_maximum_thr=0.75,
            bright=1.0,
            # Do NOT pass custom gamma — LibRaw applies the correct sRGB transfer
            # when output_color=sRGB. Custom gamma was a common cause of dark looks.
            highlight_mode=getattr(rawpy.HighlightMode, "Blend", 2),
            exp_shift=None,
            no_auto_scale=False,
        )
        try:
            rgb8 = raw.postprocess(**kwargs)
        except TypeError:
            # Older rawpy builds may not accept every kwarg
            for k in ("highlight_mode", "adjust_maximum_thr", "exp_shift", "four_color_rgb"):
                kwargs.pop(k, None)
            try:
                rgb8 = raw.postprocess(**kwargs)
            except TypeError:
                kwargs.pop("output_color", None)
                kwargs.pop("auto_bright_thr", None)
                rgb8 = raw.postprocess(**kwargs)

    if rgb8.ndim != 3 or rgb8.shape[2] != 3:
        raise ValueError(f"Unexpected demosaic shape: {getattr(rgb8, 'shape', None)}")
    if rgb8.dtype != np.uint8:
        # Safety: some builds may return 16-bit even if asked for 8
        if rgb8.dtype == np.uint16:
            rgb8 = (rgb8.astype(np.float32) / 257.0 + 0.5).astype(np.uint8)
        else:
            rgb8 = np.clip(rgb8, 0, 255).astype(np.uint8)

    h, w, _ = rgb8.shape
    # Contiguous BGR for OpenCV / Qt conversion paths
    bgr8 = np.ascontiguousarray(rgb8[..., ::-1])
    return RawLoadResult(image_bgr8=bgr8, size=(h, w))
