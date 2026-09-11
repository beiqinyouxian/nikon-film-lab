from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import cv2
import numpy as np

from .accelerator import Accelerator


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _apply_tone_curve(img: np.ndarray, curve: np.ndarray) -> np.ndarray:
    # img: float32 0..1, curve: 256 LUT per channel or 256x3
    lut = (curve * 255.0).astype(np.uint8)
    img8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
    if lut.ndim == 1:
        lut = np.stack([lut, lut, lut], axis=-1)
    b, g, r = cv2.split(img8)
    b = cv2.LUT(b, lut[:, 0])
    g = cv2.LUT(g, lut[:, 1])
    r = cv2.LUT(r, lut[:, 2])
    out8 = cv2.merge([b, g, r])
    return out8.astype(np.float32) / 255.0


def _apply_color_matrix(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # matrix 3x3 applied in RGB
    rgb = img[..., ::-1]  # BGR to RGB
    h, w, _ = rgb.shape
    reshaped = rgb.reshape((-1, 3))
    transformed = reshaped @ matrix.T
    transformed = transformed.reshape((h, w, 3))
    bgr = transformed[..., ::-1]
    return _clip01(bgr)


def _adjust_contrast(img: np.ndarray, contrast: float) -> np.ndarray:
    # contrast >0 increases contrast around mid-gray 0.5
    return _clip01((img - 0.5) * (1.0 + contrast) + 0.5)


def _adjust_saturation(img: np.ndarray, saturation: float) -> np.ndarray:
    # convert to HSV, scale S
    img8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * (1.0 + saturation), 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out.astype(np.float32) / 255.0


def _vignette(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return img
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    ry, rx = h / 2.0, w / 2.0
    dist = np.sqrt(((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2)
    mask = 1.0 - np.clip(dist, 0.0, 1.0)
    mask = mask ** (1.0 + 3.0 * strength)
    return _clip01(img * mask[..., None] + (1 - mask[..., None]) * img * (1 - 0.15 * strength))


def _grain(img: np.ndarray, amount: float, seed: int = 123) -> np.ndarray:
    if amount <= 0.0:
        return img
    h, w = img.shape[:2]
    # Resolution-aware: normalize to a 12MP reference (4000x3000)
    ref = 4000 * 3000
    scale = np.sqrt((w * h) / ref)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, size=(h, w, 1)).astype(np.float32)
    sigma = 0.02 * amount * (0.75 + 0.25 * scale)
    noisy = img + sigma * noise
    return _clip01(noisy)


def _make_curve(shadows: float, mids: float, highlights: float) -> np.ndarray:
    # Simple parametric curve: Bezier-like between (0,shadows), (0.5,mids), (1,highlights)
    x = np.linspace(0, 1, 256, dtype=np.float32)
    p0 = np.array([0.0, shadows], dtype=np.float32)
    p1 = np.array([0.5, mids], dtype=np.float32)
    p2 = np.array([1.0, highlights], dtype=np.float32)
    # Quadratic Bezier interpolation of y as a function of x index
    # For simplicity map index to t in [0,1]
    t = x
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return np.clip(y, 0.0, 1.0)


def _blend(original: np.ndarray, processed: np.ndarray, strength01: float) -> np.ndarray:
    return _clip01(original * (1.0 - strength01) + processed * strength01)


def _auto_baseline(img: np.ndarray) -> np.ndarray:
    # gray-world WB + simple exposure normalization to target median luminance
    eps = 1e-6
    b, g, r = cv2.split(img)
    mb, mg, mr = float(b.mean()), float(g.mean()), float(r.mean())
    mean = (mb + mg + mr) / 3.0 + eps
    b = np.clip(b * (mean / (mb + eps)), 0, 1)
    g = np.clip(g * (mean / (mg + eps)), 0, 1)
    r = np.clip(r * (mean / (mr + eps)), 0, 1)
    img = cv2.merge([b, g, r])
    # exposure: bring median of V to ~0.5
    img8 = (img * 255.0).astype(np.uint8)
    v = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32) / 255.0
    med = float(np.median(v)) + eps
    gain = np.clip(0.5 / med, 0.5, 2.0)
    return _clip01(img * gain)


ProcessFunc = Callable[[np.ndarray, Accelerator, float, bool, bool, bool], np.ndarray]


@dataclass
class FilmPreset:
    name: str
    process: ProcessFunc


def _build_preset(name: str, curve_params: Tuple[float, float, float], color_matrix: np.ndarray,
                  contrast: float, saturation: float, default_grain: float, vignette_strength: float) -> FilmPreset:
    curve = _make_curve(*curve_params)

    def proc(img: np.ndarray, accel: Accelerator, strength01: float, enable_grain: bool, enable_vignette: bool, auto_base: bool) -> np.ndarray:
        work = img.copy()
        if auto_base:
            work = _auto_baseline(work)
        work = _apply_tone_curve(work, curve)
        work = _apply_color_matrix(work, color_matrix)
        work = _adjust_contrast(work, contrast)
        work = _adjust_saturation(work, saturation)
        if enable_vignette:
            work = _vignette(work, vignette_strength)
        if enable_grain:
            work = _grain(work, default_grain)
        return _blend(img, work, strength01)

    return FilmPreset(name=name, process=proc)


def get_presets() -> Dict[str, FilmPreset]:
    # color matrices approximate look biases
    def mat(rg: Tuple[float, float, float], gg: Tuple[float, float, float], bg: Tuple[float, float, float]) -> np.ndarray:
        return np.array([rg, gg, bg], dtype=np.float32)

    presets = [
        _build_preset("Kodak Portra 400", (0.02, 0.55, 0.98), mat((1.05, 0.02, -0.02), (0.00, 1.02, -0.01), (-0.01, 0.02, 0.98)), 0.08, -0.05, 0.3, 0.25),
        _build_preset("Kodak Gold 200", (0.01, 0.52, 0.99), mat((1.10, -0.02, -0.02), (0.00, 1.00, 0.00), (-0.02, 0.02, 0.95)), 0.10, 0.05, 0.35, 0.20),
        _build_preset("Fuji Velvia 50", (0.00, 0.50, 1.00), mat((1.08, 0.02, -0.05), (-0.02, 1.05, -0.02), (-0.02, -0.02, 1.05)), 0.12, 0.25, 0.2, 0.15),
        _build_preset("Fuji Pro 400H", (0.02, 0.54, 0.98), mat((1.02, 0.00, -0.01), (0.00, 1.02, -0.02), (-0.02, 0.02, 1.00)), 0.06, -0.02, 0.25, 0.20),
        _build_preset("Ilford HP5 (B&W)", (0.03, 0.52, 0.97), mat((0.33, 0.33, 0.33), (0.33, 0.33, 0.33), (0.33, 0.33, 0.33)), 0.15, -1.0, 0.30, 0.25),
        _build_preset("Cinestill 800T", (0.00, 0.48, 0.98), mat((0.95, 0.05, 0.10), (0.00, 1.00, 0.05), (-0.05, 0.00, 1.05)), 0.10, 0.05, 0.25, 0.25),
        _build_preset("Agfa Vista", (0.02, 0.53, 0.99), mat((1.06, -0.01, -0.02), (0.00, 1.01, -0.01), (-0.01, 0.00, 0.98)), 0.09, 0.08, 0.25, 0.18),
        _build_preset("Kodak Tri-X", (0.02, 0.50, 0.96), mat((0.33, 0.33, 0.33), (0.33, 0.33, 0.33), (0.33, 0.33, 0.33)), 0.20, -1.0, 0.35, 0.30),
    ]
    return {p.name: p for p in presets}

