from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import rawpy  # type: ignore


@dataclass
class RawLoadResult:
    image_bgr8: np.ndarray  # uint8 0..255 BGR
    size: Tuple[int, int]  # (h, w)


def load_nef_to_bgr8(path: str, auto_bright: bool = True) -> RawLoadResult:
    """Demosaic Nikon NEF to displayable BGR8.

    Defaults favor natural exposure: camera WB + rawpy auto-bright,
    native 8-bit sRGB (avoids the dark look from blind 16-bit >> 8).
    """
    with rawpy.imread(path) as raw:
        kwargs = dict(
            output_bps=8,
            user_flip=0,
            no_auto_bright=not auto_bright,
            use_camera_wb=True,
            gamma=(2.222, 4.5),
            bright=1.0,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            output_color=rawpy.ColorSpace.sRGB,
            # 2 = Blend highlights (widely supported as int)
            highlight_mode=2,
        )
        try:
            rgb8 = raw.postprocess(**kwargs)
        except TypeError:
            # Older rawpy without some kwargs
            kwargs.pop("highlight_mode", None)
            kwargs.pop("output_color", None)
            rgb8 = raw.postprocess(**kwargs)
    h, w, _ = rgb8.shape
    bgr8 = rgb8[..., ::-1].copy()
    return RawLoadResult(image_bgr8=bgr8, size=(h, w))
