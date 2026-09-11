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


def _luminance_bgr(img: np.ndarray) -> np.ndarray:
    # Return luminance (0..1) from BGR
    b, g, r = cv2.split(img)
    return 0.0722 * r + 0.7152 * g + 0.2126 * b  # using swapped due to BGR vs RGB; still fine as relative mix


def _highlight_rolloff(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return img
    x = img
    t = 0.7
    low = np.minimum(x, t)
    high = np.maximum(0.0, x - t) / (1.0 - t)
    # compress top with a smooth shoulder
    compressed = 1.0 - np.power(1.0 - high, 1.0 + 2.5 * amount)
    y = low + (1.0 - t) * compressed
    return _clip01(y)


def _shadow_lift(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return img
    l = _luminance_bgr(img)
    weight = 1.0 - l  # stronger in shadows
    lift = amount * 0.15
    return _clip01(img + lift * weight[..., None])


def _split_tone(img: np.ndarray, shadow_tint_bgr: Tuple[float, float, float], highlight_tint_bgr: Tuple[float, float, float],
                shadow_amt: float, highlight_amt: float) -> np.ndarray:
    if shadow_amt <= 0.0 and highlight_amt <= 0.0:
        return img
    l = _luminance_bgr(img)
    st = np.array(shadow_tint_bgr, dtype=np.float32)[None, None, :]
    ht = np.array(highlight_tint_bgr, dtype=np.float32)[None, None, :]
    shadow_w = (1.0 - l)[..., None]
    highlight_w = l[..., None]
    out = img + shadow_amt * st * shadow_w + highlight_amt * ht * highlight_w
    return _clip01(out)


def _microcontrast(img: np.ndarray, amount: float, radius: float = 1.6) -> np.ndarray:
    if amount <= 0.0:
        return img
    # Unsharp mask style
    sigma = max(0.1, radius)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    detail = img - blur
    out = img + amount * detail
    return _clip01(out)


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
    # Normalize input to 2D or 3D with channels last
    if noise.ndim == 3 and noise.shape[2] in (1, 3):
        resized = cv2.resize(noise, (tw, th), interpolation=cv2.INTER_CUBIC)
        # OpenCV may drop the channel dim when it's 1 in some environments; ensure it exists
        if resized.ndim == 2:
            resized = resized[..., None]
        return resized
    else:
        # Treat as single-channel and re-add channel axis
        resized = cv2.resize(noise, (tw, th), interpolation=cv2.INTER_CUBIC)
        if resized.ndim == 2:
            resized = resized[..., None]
        return resized


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


def _build_preset(
    name: str,
    curve_params: Tuple[float, float, float],
    color_matrix: np.ndarray,
    contrast: float,
    saturation: float,
    default_vignette: float,
    *,
    rolloff: float = 0.0,
    shadow_lift_amt: float = 0.0,
    split_shadow_bgr: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    split_high_bgr: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    split_shadow_amt: float = 0.0,
    split_high_amt: float = 0.0,
    microcontrast_amt: float = 0.0,
    monochrome: bool = False,
    halation: float = 0.0,
) -> FilmPreset:
    curve = _make_curve(*curve_params)

    def proc(img: np.ndarray, accel: Accelerator, strength01: float, grain: GrainParams, enable_vignette: bool, auto_base: bool, grain_seed: Optional[int]) -> np.ndarray:
        work = img.copy()
        if auto_base:
            work = _auto_baseline(work)
        work = _apply_tone_curve(work, curve)
        work = _highlight_rolloff(work, rolloff)
        work = _apply_color_matrix(work, color_matrix)
        work = _adjust_contrast(work, contrast)
        work = _adjust_saturation(work, saturation)
        if shadow_lift_amt > 0.0:
            work = _shadow_lift(work, shadow_lift_amt)
        if split_shadow_amt > 0.0 or split_high_amt > 0.0:
            work = _split_tone(work, split_shadow_bgr, split_high_bgr, split_shadow_amt, split_high_amt)
        if microcontrast_amt > 0.0:
            work = _microcontrast(work, microcontrast_amt)
        if monochrome:
            l = _luminance_bgr(work)
            work = np.dstack([l, l, l]).astype(np.float32)
        if halation > 0.0:
            # simple bloom on highlights with warm tint
            v = _luminance_bgr(work)
            mask = np.clip((v - 0.7) / 0.3, 0.0, 1.0)
            glow_src = cv2.GaussianBlur(work, (0, 0), sigmaX=3.0)
            warm = np.array([0.02, 0.03, 0.0], dtype=np.float32)  # slight warm glow
            glow = glow_src + warm
            work = _clip01(work + halation * mask[..., None] * (glow - work))
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
        # Kodak family
        _build_preset(
            "Kodak Portra 400",
            (0.03, 0.57, 0.98),
            mat((1.07, 0.02, -0.02), (0.00, 1.03, -0.02), (-0.02, 0.03, 0.98)),
            contrast=0.10,
            saturation=-0.03,
            default_vignette=0.20,
            rolloff=0.25,
            split_shadow_bgr=(0.02, 0.03, 0.00),  # slight cyan/green in shadows (BGR)
            split_high_bgr=(0.00, 0.01, 0.02),    # warm highlights
            split_shadow_amt=0.05,
            split_high_amt=0.04,
            microcontrast_amt=0.06,
        ),
        _build_preset(
            "Kodak Gold 200",
            (0.02, 0.54, 0.99),
            mat((1.12, -0.02, -0.02), (0.00, 1.00, 0.00), (-0.02, 0.02, 0.94)),
            contrast=0.12,
            saturation=0.10,
            default_vignette=0.18,
            rolloff=0.18,
            split_shadow_bgr=(0.00, 0.01, 0.00),
            split_high_bgr=(0.00, 0.02, 0.04),
            split_high_amt=0.05,
            microcontrast_amt=0.04,
        ),
        # Fuji
        _build_preset(
            "Fuji Velvia 50",
            (0.00, 0.48, 1.00),
            mat((1.10, 0.02, -0.06), (-0.02, 1.07, -0.02), (-0.02, -0.02, 1.08)),
            contrast=0.18,
            saturation=0.35,
            default_vignette=0.15,
            rolloff=0.10,
            split_shadow_bgr=(0.02, 0.01, 0.00),
            split_high_bgr=(0.00, 0.01, 0.02),
            split_shadow_amt=0.03,
            split_high_amt=0.03,
            microcontrast_amt=0.08,
        ),
        _build_preset(
            "Fuji Pro 400H",
            (0.02, 0.56, 0.99),
            mat((1.02, -0.01, -0.01), (-0.01, 1.03, -0.02), (-0.02, 0.02, 1.00)),
            contrast=0.02,
            saturation=-0.08,
            default_vignette=0.18,
            rolloff=0.22,
            shadow_lift_amt=0.10,
            split_shadow_bgr=(0.03, 0.05, 0.00),
            split_high_bgr=(0.00, 0.01, 0.01),
            split_shadow_amt=0.06,
            split_high_amt=0.02,
            microcontrast_amt=0.03,
        ),
        # B&W
        _build_preset(
            "Ilford HP5 (B&W)",
            (0.04, 0.54, 0.97),
            mat((0.33, 0.33, 0.33), (0.33, 0.33, 0.33), (0.33, 0.33, 0.33)),
            contrast=0.22,
            saturation=-1.0,
            default_vignette=0.22,
            rolloff=0.12,
            microcontrast_amt=0.10,
            monochrome=True,
        ),
        _build_preset(
            "Kodak Tri-X",
            (0.03, 0.50, 0.95),
            mat((0.33, 0.33, 0.33), (0.33, 0.33, 0.33), (0.33, 0.33, 0.33)),
            contrast=0.30,
            saturation=-1.0,
            default_vignette=0.28,
            rolloff=0.10,
            microcontrast_amt=0.14,
            monochrome=True,
        ),
        # Cinestill
        _build_preset(
            "Cinestill 800T",
            (0.00, 0.50, 0.98),
            mat((0.95, 0.05, 0.10), (0.00, 1.00, 0.06), (-0.04, 0.00, 1.06)),
            contrast=0.12,
            saturation=0.08,
            default_vignette=0.25,
            rolloff=0.20,
            split_shadow_bgr=(0.08, 0.04, 0.00),   # teal shadows (B channel up)
            split_high_bgr=(0.00, 0.02, 0.06),     # warm highlights
            split_shadow_amt=0.08,
            split_high_amt=0.06,
            microcontrast_amt=0.06,
            halation=0.12,
        ),
        # Agfa
        _build_preset(
            "Agfa Vista",
            (0.02, 0.54, 0.99),
            mat((1.06, -0.01, -0.03), (-0.01, 1.02, -0.01), (-0.01, -0.01, 0.99)),
            contrast=0.10,
            saturation=0.06,
            default_vignette=0.18,
            rolloff=0.14,
            split_shadow_bgr=(0.02, 0.01, 0.00),
            split_high_bgr=(0.00, 0.01, 0.01),
            split_shadow_amt=0.03,
            split_high_amt=0.02,
            microcontrast_amt=0.05,
        ),
        # Leica-inspired
        _build_preset(
            "Leica Color Modern",
            (0.02, 0.54, 0.99),
            mat((1.05, 0.02, -0.01), (0.00, 1.03, -0.01), (-0.01, 0.01, 0.99)),
            contrast=0.14,
            saturation=0.02,
            default_vignette=0.12,
            rolloff=0.12,
            microcontrast_amt=0.16,
        ),
        _build_preset(
            "Leica Classic Mono",
            (0.02, 0.50, 0.95),
            mat((0.33, 0.33, 0.33), (0.33, 0.33, 0.33), (0.33, 0.33, 0.33)),
            contrast=0.28,
            saturation=-1.0,
            default_vignette=0.20,
            rolloff=0.10,
            microcontrast_amt=0.18,
            monochrome=True,
        ),
        _build_preset(
            "Leica Chrome Vivid",
            (0.00, 0.50, 0.99),
            mat((1.08, 0.00, -0.04), (-0.01, 1.05, -0.01), (-0.01, -0.01, 1.04)),
            contrast=0.16,
            saturation=0.16,
            default_vignette=0.15,
            rolloff=0.10,
            microcontrast_amt=0.20,
        ),
    ]
    return {p.name: p for p in presets}

