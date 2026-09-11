from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rawpy  # type: ignore


@dataclass
class RawLoadResult:
    image_bgr8: np.ndarray  # uint8 0..255 BGR
    size: Tuple[int, int]  # (h, w)


def load_nef_to_bgr8(path: str, auto_bright: bool = False) -> RawLoadResult:
    # Use rawpy to demosaic to 16-bit RGB, convert to BGR uint8
    with rawpy.imread(path) as raw:
        # Postprocess: full size, camera white balance, sRGB gamma
        rgb16 = raw.postprocess(
            output_bps=16,
            user_flip=0,
            no_auto_bright=not auto_bright,
            use_camera_wb=True,
            gamma=(2.2, 4.5),  # sRGB-ish
            bright=1.0,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    h, w, _ = rgb16.shape
    # Convert to 8-bit with simple scaling
    rgb8 = np.right_shift(rgb16, 8).astype(np.uint8)
    bgr8 = rgb8[..., ::-1].copy()
    return RawLoadResult(image_bgr8=bgr8, size=(h, w))

