from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import piexif
from PIL import Image, ImageOps
import numpy as np


@dataclass
class JpegLoadResult:
    image_bgr8: np.ndarray
    size: Tuple[int, int]
    exif_bytes: Optional[bytes]


def load_jpeg_bgr8(path: str) -> JpegLoadResult:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        exif_bytes = im.info.get("exif", None)
        rgb = im.convert("RGB")
        w, h = rgb.size
        arr = np.array(rgb, dtype=np.uint8)
    bgr = arr[..., ::-1].copy()
    return JpegLoadResult(image_bgr8=bgr, size=(h, w), exif_bytes=exif_bytes)


def save_jpeg_bgr8(path: str, bgr8: np.ndarray, quality: int = 95, exif_bytes: Optional[bytes] = None) -> None:
    rgb = bgr8[..., ::-1]
    im = Image.fromarray(rgb, mode="RGB")
    if exif_bytes:
        try:
            im.save(path, "JPEG", quality=quality, subsampling=0, exif=exif_bytes)
            return
        except Exception:
            pass
    im.save(path, "JPEG", quality=quality, subsampling=0)

