from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .accelerator import Accelerator, BackendMode
from .film_presets import (
    get_presets,
    GrainParams,
    GrainType,
    _apply_exposure,
    _apply_color_temp,
    _apply_clarity,
    _adjust_contrast,
    _apply_highlights_shadows,
    _apply_vibrance,
    _adjust_saturation,
)


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
    # Specialty FX (independent of film presets)
    enable_lens_aging: bool = False
    lens_aging: int = 0                 # 0..100
    enable_scratches: bool = False
    scratches: int = 0                  # 0..100
    enable_film_defects: bool = False
    film_defects: int = 0               # 0..100
    enable_partial_exposure: bool = False
    partial_exposure: int = 0           # 0..100
    fx_seed: Optional[int] = None
    # Color splitter (HSL per-color bands; independent of presets)
    enable_color_splitter: bool = True
    hsl_sat8: list[int] | None = None  # [-100..100] x8 for 红/橙/黄/绿/青/蓝/紫/品红
    hsl_lum8: list[int] | None = None  # [-100..100] x8
    # Vignette: off/auto/manual
    vignette_mode: str = "off"    # "off" | "auto" | "manual"
    vignette_amount: int = 0      # 0..100, used when mode=manual
    enable_auto_baseline: bool = True
    # Manual exposure and color temperature
    exposure_ev_x100: int = 0     # -200..200 represents -2..+2 EV
    temp_bias: int = 0            # -100..100 (cooler..warmer)
    clarity: int = 0              # -100..100 (negative softens), default 0 center
    contrast: int = 0             # -100..100, default 0 center (left softer, right harder)
    highlights: int = 0           # -100..100
    shadows: int = 0              # -100..100
    vibrance: int = 0             # -100..100
    saturation: int = 0           # -100..100
 


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
        clarity_amount = max(-1.0, min(1.0, options.clarity / 100.0))
        contrast_amount = max(-1.0, min(1.0, options.contrast / 100.0))
        highlights = max(-1.0, min(1.0, options.highlights / 100.0))
        shadows = max(-1.0, min(1.0, options.shadows / 100.0))
        vibrance = max(-1.0, min(1.0, options.vibrance / 100.0))
        saturation = max(-1.0, min(1.0, options.saturation / 100.0))
        # Film preset look only (manual params zeroed), blended by strength
        out = preset.process(
            img,
            self.accel,
            strength01,
            grain,
            enable_vignette,
            vignette_override,
            options.enable_auto_baseline,
            options.grain_seed,
            0.0,  # exposure inside preset
            0.0,  # temp
            0.0,  # clarity
            0.0,  # user_contrast
            0.0,  # highlights
            0.0,  # shadows
            0.0,  # vibrance
            0.0,  # user_saturation
        )
        # Apply manual tone/color controls AFTER preset blend (decoupled from strength)
        if abs(exposure_ev) > 1e-6:
            out = _apply_exposure(out, exposure_ev)
        if abs(temp01) > 1e-6:
            out = _apply_color_temp(out, temp01)
        if abs(clarity_amount) > 1e-6:
            out = _apply_clarity(out, clarity_amount)
        if abs(contrast_amount) > 1e-6:
            out = _adjust_contrast(out, 0.8 * contrast_amount)
        if abs(highlights) > 1e-6 or abs(shadows) > 1e-6:
            out = _apply_highlights_shadows(out, highlights, shadows)
        if abs(vibrance) > 1e-6:
            out = _apply_vibrance(out, vibrance)
        if abs(saturation) > 1e-6:
            out = _adjust_saturation(out, 0.7 * saturation)
        # Apply color splitter after manual controls, before defects
        if getattr(options, "hsl_sat8", None) is None:
            options.hsl_sat8 = [0] * 8
        if getattr(options, "hsl_lum8", None) is None:
            options.hsl_lum8 = [0] * 8
        if options.enable_color_splitter and (any(v != 0 for v in options.hsl_sat8) or any(v != 0 for v in options.hsl_lum8)):
            out = self._apply_color_splitter(out, options.hsl_sat8, options.hsl_lum8)
        # Apply specialty FX independently from preset
        out = self._apply_special_fx(out, options)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    # ---- Specialty FX helpers (BGR float32 0..1) ----
    @staticmethod
    def _clip01(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _ensure_3c(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr
        if arr.ndim == 2:
            return np.repeat(arr[..., None], 3, axis=2)
        if arr.ndim == 3 and arr.shape[2] == 1:
            return np.repeat(arr, 3, axis=2)
        raise ValueError("Unsupported array shape")

    @staticmethod
    def _radial_mask(h: int, w: int) -> np.ndarray:
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        ry, rx = h / 2.0, w / 2.0
        dist = np.sqrt(((y - cy) / max(ry, 1e-6)) ** 2 + ((x - cx) / max(rx, 1e-6)) ** 2)
        m = 1.0 - np.clip(dist, 0.0, 1.0)
        return m.astype(np.float32)  # HxW

    def _lens_aging(self, img: np.ndarray, t: float) -> np.ndarray:
        if t <= 1e-6:
            return img
        h, w = img.shape[:2]
        # Mild yellow/amber cast: R up a bit, B down a bit, G slightly up
        b, g, r = cv2.split(img.astype(np.float32))
        r = np.clip(r * (1.0 + 0.05 * t), 0, 1)
        g = np.clip(g * (1.0 + 0.02 * t), 0, 1)
        b = np.clip(b * (1.0 - 0.05 * t), 0, 1)
        colored = cv2.merge([b, g, r]).astype(np.float32)
        # Haze / contrast loss: blend towards a soft base and mid-gray slightly
        base = cv2.GaussianBlur(colored, (0, 0), sigmaX=0.8 + 1.2 * t)
        softened = colored * (1.0 - 0.25 * t) + base * (0.25 * t)
        # Edge softness via radial mask
        mask = 1.0 - self._radial_mask(h, w)  # 0 center, 1 edges
        mask3 = self._ensure_3c(mask)
        blur_edges = cv2.GaussianBlur(softened, (0, 0), sigmaX=1.5 + 2.0 * t)
        mixed = softened * (1.0 - 0.25 * t * mask3) + blur_edges * (0.25 * t * mask3)
        # Very slight vignette (falloff)
        vig = 0.08 * t
        if vig > 0.0:
            vmask = (self._radial_mask(h, w) ** (1.2 + 2.0 * t)).astype(np.float32)
            mixed = self._clip01(mixed * (0.96 + 0.04 * vmask[..., None]))
        return self._clip01(mixed)

    def _scratches(self, img: np.ndarray, t: float, seed: Optional[int]) -> np.ndarray:
        if t <= 1e-6:
            return img
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed if seed is not None else 0)
        overlay = img.copy()
        alpha = np.zeros((h, w, 1), dtype=np.float32)
        # Number of scratches scales with area and t but capped
        base_count = int(2 + 6 * t)
        area_scale = max(1.0, np.sqrt(h * w) / 1000.0)
        count = int(min(60, base_count * area_scale))
        for i in range(count):
            # Random line endpoints slightly beyond frame to span across
            x0 = int(rng.integers(-w//4, w + w//4))
            y0 = int(rng.integers(-h//4, h + h//4))
            angle = float(rng.uniform(0, np.pi))
            length = int(rng.uniform(0.3, 1.2) * max(h, w))
            x1 = int(x0 + length * np.cos(angle))
            y1 = int(y0 - length * np.sin(angle))
            thickness = int(rng.integers(1, 3))
            bright = rng.random() < 0.5
            color = (1.0, 1.0, 1.0) if bright else (0.0, 0.0, 0.0)
            # Draw on overlay with small opacity
            cv2.line(overlay, (x0, y0), (x1, y1), color, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.line(alpha, (x0, y0), (x1, y1), (0.15 + 0.25 * t,), thickness=thickness, lineType=cv2.LINE_AA)
        # Occasional hairline arcs
        for i in range(max(0, int(1 * t))):
            center = (int(rng.integers(-w//2, w + w//2)), int(rng.integers(-h//2, h + h//2)))
            axes = (int(rng.uniform(0.6, 1.2) * w), int(rng.uniform(0.6, 1.2) * h))
            start_angle = int(rng.uniform(0, 360))
            end_angle = start_angle + int(rng.uniform(20, 120))
            thickness = 1
            bright = rng.random() < 0.5
            color = (1.0, 1.0, 1.0) if bright else (0.0, 0.0, 0.0)
            cv2.ellipse(overlay, center, axes, 0, start_angle, end_angle, color, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.ellipse(alpha, center, axes, 0, start_angle, end_angle, (0.10 + 0.20 * t,), thickness=thickness, lineType=cv2.LINE_AA)
        alpha3 = self._ensure_3c(alpha)
        out = self._clip01(img * (1.0 - alpha3) + overlay * alpha3)
        return out

    def _film_defects(self, img: np.ndarray, t: float, seed: Optional[int]) -> np.ndarray:
        if t <= 1e-6:
            return img
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed if seed is not None else 1)
        out = img.copy()
        # Dust specks: small dark or bright soft dots
        speck_count = int(min(200, (3 + 12 * t) * max(1.0, (h * w) / (1200 * 1200))))
        dust = out.copy()
        alpha = np.zeros((h, w, 1), dtype=np.float32)
        for _ in range(speck_count):
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))
            radius = int(max(1, int(rng.uniform(0.3, 1.8) * (min(h, w) / 300.0))))
            color = (0.0, 0.0, 0.0) if rng.random() < 0.7 else (1.0, 1.0, 1.0)
            cv2.circle(dust, (cx, cy), radius, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(alpha, (cx, cy), int(radius * 1.5), (0.05 + 0.20 * t,), thickness=-1, lineType=cv2.LINE_AA)
        out = self._clip01(out * (1.0 - self._ensure_3c(alpha)) + dust * self._ensure_3c(alpha))
        # Light leak edges: pick a random edge and add warm gradient
        if rng.random() < 0.9:  # often present when enabled
            leak_color = np.array([0.06, 0.04, 0.0], dtype=np.float32)  # warm in BGR
            side = int(rng.integers(0, 4))  # 0=L,1=T,2=R,3=B
            # distance map from chosen edge
            if side == 0:
                dist = np.tile(np.linspace(0, 1, w, dtype=np.float32)[None, :, None], (h, 1, 1))
            elif side == 2:
                dist = np.tile(np.linspace(1, 0, w, dtype=np.float32)[None, :, None], (h, 1, 1))
            elif side == 1:
                dist = np.tile(np.linspace(0, 1, h, dtype=np.float32)[:, None, None], (1, w, 1))
            else:
                dist = np.tile(np.linspace(1, 0, h, dtype=np.float32)[:, None, None], (1, w, 1))
            grad = np.exp(-4.0 * dist)  # stronger near edge
            strength = (0.08 + 0.25 * t) * float(rng.uniform(0.6, 1.2))
            out = self._clip01(out + strength * grad * leak_color[None, None, :])
        # Subtle frame edge unevenness: very low amplitude random mask
        noise = rng.normal(0.0, 1.0, size=(h, w, 1)).astype(np.float32)
        lowfreq = cv2.GaussianBlur(noise, (0, 0), sigmaX=40.0)
        edge_mask = 1.0 - self._radial_mask(h, w)
        edge_mask = cv2.GaussianBlur(edge_mask.astype(np.float32), (0, 0), sigmaX=6.0)[..., None]
        out = self._clip01(out * (1.0 - 0.03 * t * edge_mask * lowfreq))
        return out

    def _partial_exposure(self, img: np.ndarray, t: float, seed: Optional[int]) -> np.ndarray:
        if t <= 1e-6:
            return img
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed if seed is not None else 2)
        out = img.copy()
        # Choose an edge and an angular wedge
        side = int(rng.integers(0, 4))
        over = rng.random() < 0.7  # mostly overexposed light strike
        color = np.array([0.05, 0.03, 0.0], dtype=np.float32) if over else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Build a mask that decays from the edge inward and varies by angle
        yy, xx = np.mgrid[0:h, 0:w]
        if side == 0:  # left
            d = xx.astype(np.float32) / max(1, w)
        elif side == 2:  # right
            d = (w - 1 - xx).astype(np.float32) / max(1, w)
        elif side == 1:  # top
            d = yy.astype(np.float32) / max(1, h)
        else:  # bottom
            d = (h - 1 - yy).astype(np.float32) / max(1, h)
        d = np.clip(d, 0.0, 1.0)
        ang = np.arctan2(yy - h / 2.0, xx - w / 2.0)  # -pi..pi
        center_angle = float(rng.uniform(-np.pi, np.pi))
        spread = float(rng.uniform(0.6, 1.6))  # radians
        ang_weight = np.exp(-((ang - center_angle) ** 2) / (2 * (spread ** 2)))
        mask = (np.exp(-5.0 * d) * ang_weight).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8.0)
        mask3 = self._ensure_3c(mask)
        # Apply as localized exposure shift with slight warm color if overexposed
        amt = (0.15 + 0.35 * t) * float(rng.uniform(0.7, 1.3))
        if over:
            out = self._clip01(out * (1.0 - amt * mask3) + (out + color) * (amt * mask3))
        else:
            out = self._clip01(out * (1.0 - 0.5 * amt * mask3))  # localized underexposure
        return out

    def _apply_special_fx(self, img: np.ndarray, options: ProcessOptions) -> np.ndarray:
        out = img
        # Normalize intensities
        lens_t = max(0.0, min(1.0, options.lens_aging / 100.0)) if options.enable_lens_aging else 0.0
        scratch_t = max(0.0, min(1.0, options.scratches / 100.0)) if options.enable_scratches else 0.0
        defects_t = max(0.0, min(1.0, options.film_defects / 100.0)) if options.enable_film_defects else 0.0
        partial_t = max(0.0, min(1.0, options.partial_exposure / 100.0)) if options.enable_partial_exposure else 0.0
        seed = options.fx_seed if options.fx_seed is not None else options.grain_seed
        # Order: lens aging (base look) -> partial exposure -> film defects -> scratches (top-most)
        if lens_t > 0.0:
            out = self._lens_aging(out, lens_t)
        if partial_t > 0.0:
            out = self._partial_exposure(out, partial_t, None if seed is None else seed + 17)
        if defects_t > 0.0:
            out = self._film_defects(out, defects_t, None if seed is None else seed + 29)
        if scratch_t > 0.0:
            out = self._scratches(out, scratch_t, None if seed is None else seed + 41)
        return out
    
    # ---- Color Splitter (HSL-style 8-band) ----
    @staticmethod
    def _apply_color_splitter(img: np.ndarray, sat8: list[int], lum8: list[int]) -> np.ndarray:
        img8 = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV).astype(np.float32)
        h = hsv[..., 0] * 2.0  # degrees 0..360
        s = hsv[..., 1] / 255.0
        v = hsv[..., 2] / 255.0
        centers = np.array([0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0], dtype=np.float32)
        width = 40.0
        sat_adj = np.array([np.clip(x, -100, 100) / 100.0 for x in (sat8 or [0]*8)], dtype=np.float32)
        lum_adj = np.array([np.clip(x, -100, 100) / 100.0 for x in (lum8 or [0]*8)], dtype=np.float32)
        total_sat = np.zeros_like(s, dtype=np.float32)
        total_lum = np.zeros_like(v, dtype=np.float32)
        for i in range(8):
            c = centers[i]
            d = np.abs(h - c)
            d = np.minimum(d, 360.0 - d)
            wmask = np.clip(1.0 - d / width, 0.0, 1.0)
            wmask = wmask * wmask
            if sat_adj[i] != 0.0:
                total_sat += wmask * sat_adj[i]
            if lum_adj[i] != 0.0:
                total_lum += wmask * lum_adj[i]
        s2 = np.clip(s * (1.0 + total_sat), 0.0, 1.0)
        v2 = np.clip(v * (1.0 + total_lum), 0.0, 1.0)
        hsv[..., 1] = (s2 * 255.0).astype(np.float32)
        hsv[..., 2] = (v2 * 255.0).astype(np.float32)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        return out

