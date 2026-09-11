from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Tuple, Optional

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


class GrainType(str, Enum):
    SILVER_HALIDE = "Silver halide"
    MODERN_FINE = "Modern fine"
    COARSE_PUSH = "Coarse push"


@dataclass
class GrainParams:
    enabled: bool = True
    grain_type: GrainType = GrainType.SILVER_HALIDE
    size01: float = 0.3          # 0..1, relative size
    density01: float = 0.3       # 0..1, amplitude
    roughness01: float = 0.4     # 0..1, clumping
    chroma_mix01: float = 0.0    # 0..1, 0 mono only, 1 full color
    seed: Optional[int] = None


def _resize_like(noise: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    h, w = noise.shape[:2]
    if (h, w) == (th, tw):
        return noise
    return cv2.resize(noise, (tw, th), interpolation=cv2.INTER_CUBIC)


def _apply_grain(img: np.ndarray, params: GrainParams) -> np.ndarray:
    if not params.enabled or params.density01 <= 0.0:
        return img
    h, w = img.shape[:2]
    rng = np.random.default_rng(params.seed if params.seed is not None else 123)

    # Determine scale relative to reference width ~4000px
    ref_w = 4000.0
    scale = max(h, w) / ref_w

    # Base frequency via downsample-upsample to control "grain size"
    # Map size01 in [0,1] to factor in [1, 24]
    factor = max(1, int(1 + round(24.0 * params.size01)))
    low_h = max(1, h // factor)
    low_w = max(1, w // factor)

    # Type-dependent characteristics
    if params.grain_type == GrainType.SILVER_HALIDE:
        # Mono luminance, noticeable clumps
        base = rng.normal(0.0, 1.0, size=(low_h, low_w, 1)).astype(np.float32)
        # Clumping: blend coarse and fine
        coarse = cv2.GaussianBlur(base, (0, 0), sigmaX=1.0 + 3.0 * params.roughness01)
        fine = rng.normal(0.0, 1.0, size=(low_h, low_w, 1)).astype(np.float32) * 0.5
        mixed = (1.0 - params.roughness01) * fine + params.roughness01 * coarse
        noise = _resize_like(mixed, (h, w))
        # Apply to luminance (HSV V)
        img8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[..., 2] / 255.0
        v = np.clip(v + params.density01 * 0.08 * (0.75 + 0.5 * scale) * noise[..., 0], 0, 1)
        hsv[..., 2] = (v * 255.0).astype(np.float32)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        # Optional chroma noise mix
        if params.chroma_mix01 > 0.0:
            cnoise = rng.normal(0.0, 1.0, size=(low_h, low_w, 3)).astype(np.float32)
            cnoise = _resize_like(cnoise, (h, w))
            out = _clip01(out + params.chroma_mix01 * params.density01 * 0.02 * cnoise)
        return out

    elif params.grain_type == GrainType.MODERN_FINE:
        # Fine high-frequency, low chroma
        base = rng.normal(0.0, 1.0, size=(h, w, 1)).astype(np.float32)
        noise = cv2.GaussianBlur(base, (0, 0), sigmaX=0.6)
        out = _clip01(img + params.density01 * 0.015 * (0.7 + 0.3 * scale) * noise)
        if params.chroma_mix01 > 0.0:
            c = rng.normal(0.0, 1.0, size=(h, w, 3)).astype(np.float32)
            out = _clip01(out + params.chroma_mix01 * params.density01 * 0.01 * c)
        return out

    else:  # COARSE_PUSH
        # Coarse, stronger, some chroma
        base = rng.normal(0.0, 1.0, size=(low_h, low_w, 1)).astype(np.float32)
        noise = _resize_like(base, (h, w))
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=0.8 + 2.0 * params.roughness01)
        out = _clip01(img + params.density01 * 0.05 * (0.8 + 0.4 * scale) * noise)
        if params.chroma_mix01 > 0.0:
            c = rng.normal(0.0, 1.0, size=(low_h, low_w, 3)).astype(np.float32)
            c = _resize_like(c, (h, w))
            out = _clip01(out + params.chroma_mix01 * params.density01 * 0.02 * c)
        return out


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


ProcessFunc = Callable[[np.ndarray, Accelerator, float, GrainParams, bool, bool, Optional[int]], np.ndarray]


@dataclass
class FilmPreset:
    name: str
    process: ProcessFunc


def _build_preset(name: str, curve_params: Tuple[float, float, float], color_matrix: np.ndarray,
                  contrast: float, saturation: float, default_vignette: float) -> FilmPreset:
    curve = _make_curve(*curve_params)

    def proc(img: np.ndarray, accel: Accelerator, strength01: float, grain: GrainParams, enable_vignette: bool, auto_base: bool, grain_seed: Optional[int]) -> np.ndarray:
        work = img.copy()
        if auto_base:
            work = _auto_baseline(work)
        work = _apply_tone_curve(work, curve)
        work = _apply_color_matrix(work, color_matrix)
        work = _adjust_contrast(work, contrast)
        work = _adjust_saturation(work, saturation)
        if enable_vignette:
            work = _vignette(work, default_vignette)
        if grain.seed is None:
            grain.seed = grain_seed
        if grain.enabled:
            work = _apply_grain(work, grain)
        return _blend(img, work, strength01)

    return FilmPreset(name=name, process=proc)


def get_presets() -> Dict[str, FilmPreset]:
    # color matrices approximate look biases
    def mat(rg: Tuple[float, float, float], gg: Tuple[float, float, float], bg: Tuple[float, float, float]) -> np.ndarray:
        return np.array([rg, gg, bg], dtype=np.float32)

    presets = [
        _build_preset("Kodak Portra 400", (0.02, 0.55, 0.98), mat((1.05, 0.02, -0.02), (0.00, 1.02, -0.01), (-0.01, 0.02, 0.98)), 0.08, -0.05, 0.25),
        _build_preset("Kodak Gold 200", (0.01, 0.52, 0.99), mat((1.10, -0.02, -0.02), (0.00, 1.00, 0.00), (-0.02, 0.02, 0.95)), 0.10, 0.05, 0.20),
        _build_preset("Fuji Velvia 50", (0.00, 0.50, 1.00), mat((1.08, 0.02, -0.05), (-0.02, 1.05, -0.02), (-0.02, -0.02, 1.05)), 0.12, 0.25, 0.15),
        _build_preset("Fuji Pro 400H", (0.02, 0.54, 0.98), mat((1.02, 0.00, -0.01), (0.00, 1.02, -0.02), (-0.02, 0.02, 1.00)), 0.06, -0.02, 0.20),
        _build_preset("Ilford HP5 (B&W)", (0.03, 0.52, 0.97), mat((0.33, 0.33, 0.33), (0.33, 0.33, 0.33), (0.33, 0.33, 0.33)), 0.15, -1.0, 0.25),
        _build_preset("Cinestill 800T", (0.00, 0.48, 0.98), mat((0.95, 0.05, 0.10), (0.00, 1.00, 0.05), (-0.05, 0.00, 1.05)), 0.10, 0.05, 0.25),
        _build_preset("Agfa Vista", (0.02, 0.53, 0.99), mat((1.06, -0.01, -0.02), (0.00, 1.01, -0.01), (-0.01, 0.00, 0.98)), 0.09, 0.08, 0.18),
        _build_preset("Kodak Tri-X", (0.02, 0.50, 0.96), mat((0.33, 0.33, 0.33), (0.33, 0.33, 0.33), (0.33, 0.33, 0.33)), 0.20, -1.0, 0.30),
        # Leica-inspired
        _build_preset("Leica Color Modern", (0.02, 0.53, 0.99), mat((1.04, 0.02, -0.01), (0.00, 1.02, -0.01), (-0.01, 0.01, 0.99)), 0.10, -0.02, 0.12),
        _build_preset("Leica Classic Mono", (0.01, 0.50, 0.95), mat((0.33, 0.33, 0.33), (0.33, 0.33, 0.33), (0.33, 0.33, 0.33)), 0.22, -1.0, 0.20),
        _build_preset("Leica Chrome Vivid", (0.00, 0.48, 0.98), mat((1.06, 0.00, -0.04), (-0.01, 1.04, -0.01), (-0.01, -0.01, 1.03)), 0.12, 0.18, 0.15),
    ]
    return {p.name: p for p in presets}

