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
    # Renamed FX
    enable_expired_film: bool = False
    expired_film: int = 0               # 0..100
    enable_light_leak: bool = False
    light_leak: int = 0                 # 0..100
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
    
    def recommended_strength(self, preset_name: str) -> int:
        p = self.presets.get(preset_name)
        if p is None:
            return 0
        return int(getattr(p, "recommended_strength", 0))


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
        """Front-element hairline scratches: thin, anti-aliased, mostly diagonal, subtle opacity."""
        if t <= 1e-6:
            return img
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed if seed is not None else 0)
        overlay = img.copy()
        alpha2d = np.zeros((h, w), dtype=np.float32)
        # Density scales with area and intensity; prefer hairline (thickness 1)
        area_scale = max(1.0, np.sqrt(h * w) / 900.0)
        base = 12 + int(36 * t * area_scale)
        count = int(min(220, base))
        # Edge bias mask (scratches更容易出现在边缘)
        edge = 1.0 - self._radial_mask(h, w)  # 0 center, 1 edge
        edge = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), sigmaX=6.0)
        # Draw hairlines
        for _ in range(count):
            # Start near edge more often
            if rng.random() < 0.6:
                # pick an edge and position
                side = int(rng.integers(0, 4))
                if side == 0:
                    x0, y0 = 0, int(rng.integers(0, h))
                elif side == 1:
                    x0, y0 = w - 1, int(rng.integers(0, h))
                elif side == 2:
                    x0, y0 = int(rng.integers(0, w)), 0
                else:
                    x0, y0 = int(rng.integers(0, w)), h - 1
            else:
                x0 = int(rng.integers(0, w))
                y0 = int(rng.integers(0, h))
            # Diagonal bias angles around 30°/150°
            if rng.random() < 0.5:
                angle = float(rng.normal(np.deg2rad(30), np.deg2rad(12)))
            else:
                angle = float(rng.normal(np.deg2rad(150), np.deg2rad(12)))
            length = int(rng.uniform(0.2, 0.9) * max(h, w))
            x1 = int(x0 + length * np.cos(angle))
            y1 = int(y0 - length * np.sin(angle))
            thickness = 1  # hairline
            # Bright specular vs faint dark groove
            bright = rng.random() < 0.55
            color = (1.0, 1.0, 1.0) if bright else (0.0, 0.0, 0.0)
            # Opacity scales with t and edge weight around the midpoint
            midx = int((x0 + x1) / 2)
            midy = int((y0 + y1) / 2)
            e_w = float(edge[np.clip(midy, 0, h - 1), np.clip(midx, 0, w - 1)])
            opa = (0.06 + 0.12 * t) * (0.6 + 0.6 * e_w)
            cv2.line(overlay, (x0, y0), (x1, y1), color, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.line(alpha2d, (x0, y0), (x1, y1), opa, thickness=thickness, lineType=cv2.LINE_AA)
        # Occasional arcs (very subtle)
        arc_n = int(1 + 2 * t)
        for _ in range(arc_n):
            if rng.random() < 0.4:
                center = (int(rng.integers(-w // 2, w + w // 2)), int(rng.integers(-h // 2, h + h // 2)))
                axes = (int(rng.uniform(0.5, 1.2) * w), int(rng.uniform(0.5, 1.2) * h))
                start_angle = int(rng.uniform(0, 360))
                end_angle = start_angle + int(rng.uniform(12, 60))
                bright = rng.random() < 0.5
                color = (1.0, 1.0, 1.0) if bright else (0.0, 0.0, 0.0)
                opa = 0.04 + 0.08 * t
                cv2.ellipse(overlay, center, axes, 0, start_angle, end_angle, color, thickness=1, lineType=cv2.LINE_AA)
                cv2.ellipse(alpha2d, center, axes, 0, start_angle, end_angle, opa, thickness=1, lineType=cv2.LINE_AA)
        # Feather overall alpha a bit to avoid harshness
        alpha2d = cv2.GaussianBlur(alpha2d, (0, 0), sigmaX=0.8)
        alpha3 = self._ensure_3c(alpha2d)
        out = self._clip01(img * (1.0 - alpha3) + overlay * alpha3)
        return out

    def _expired_film(self, img: np.ndarray, t: float, seed: Optional[int]) -> np.ndarray:
        """Expired film look: dye fade, muddy shadows, lifted blacks, color cast, slight mottling."""
        if t <= 1e-6:
            return img
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed if seed is not None else 1)
        out = img.astype(np.float32).copy()
        # Base fog (lift blacks)
        fog = 0.04 + 0.10 * t
        out = self._clip01(out * (1.0 - 0.5 * t) + fog)
        # Mild overall saturation loss
        out = _adjust_saturation(out, -0.25 * t)
        # Cross-process style color shift (random green-magenta or warm)
        if rng.random() < 0.5:
            cast = np.array([0.00, 0.03 + 0.10 * t, 0.06 * t], dtype=np.float32)[None, None, :]  # BGR ~ cyan/green
        else:
            cast = np.array([0.02 + 0.08 * t, 0.02 * t, 0.00], dtype=np.float32)[None, None, :]  # warm
        out = self._clip01(out + cast)
        # Lower contrast a bit and muddy shadows (lift)
        out = _adjust_contrast(out, -0.20 * t)
        out = self._clip01(out + (0.08 * t) * (1.0 - self._radial_mask(h, w))[..., None] * 0.5)
        # Uneven dye fade via low-frequency color noise
        noise = rng.normal(0.0, 1.0, size=(h, w, 3)).astype(np.float32)
        low = cv2.GaussianBlur(noise, (0, 0), sigmaX=18.0)
        out = self._clip01(out + 0.04 * t * low)
        # Sparse fine dust at very low opacity
        specks = np.zeros((h, w), dtype=np.float32)
        n = int(20 * t)
        for _ in range(n):
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))
            r = int(max(1, rng.integers(1, 2 + int(2 * t))))
            cv2.circle(specks, (cx, cy), r, color=1.0, thickness=-1, lineType=cv2.LINE_AA)
        specks = cv2.GaussianBlur(specks, (0, 0), sigmaX=0.6)
        out = self._clip01(out * (1.0 - 0.06 * t * self._ensure_3c(specks)))
        return out

    def _light_leak(self, img: np.ndarray, t: float, seed: Optional[int]) -> np.ndarray:
        """Analog-like light leak from edges/corners with organic shapes and warm/magenta casts."""
        if t <= 1e-6:
            return img
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed if seed is not None else 2)
        out = img.copy()
        # Base edge distance
        y, x = np.ogrid[:h, :w]
        dist_left = x / max(1, w - 1)
        dist_right = (w - 1 - x) / max(1, w - 1)
        dist_top = y / max(1, h - 1)
        dist_bottom = (h - 1 - y) / max(1, h - 1)
        dists = [dist_left, dist_right, dist_top, dist_bottom]
        # Pick 1-2 edges
        edges = rng.choice(4, size=int(rng.integers(1, 3)), replace=False)
        mask2d = np.zeros((h, w), dtype=np.float32)
        for e in edges:
            d = dists[e].astype(np.float32)
            # Organic shape via thresholded blurred noise multiplied by distance falloff
            shape = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
            shape = cv2.GaussianBlur(shape, (0, 0), sigmaX=8.0 + 12.0 * t)
            shape = (shape - shape.min()) / max(1e-6, shape.max() - shape.min())
            fall = np.exp(-4.0 * d)
            mask2d = np.maximum(mask2d, (shape * fall).astype(np.float32))
        mask2d = cv2.GaussianBlur(mask2d, (0, 0), sigmaX=6.0)
        mask2d = np.clip(mask2d, 0.0, 1.0)
        # Warm or magenta cast
        if rng.random() < 0.75:
            color = np.array([0.06, 0.10 + 0.25 * t, 0.01], dtype=np.float32)  # warm/orange (BGR)
        else:
            color = np.array([0.12, 0.02, 0.10 + 0.20 * t], dtype=np.float32)  # cool magenta
        mask3 = self._ensure_3c(mask2d ** (1.2))
        strength = 0.20 + 0.60 * t
        out = self._clip01(out + strength * mask3 * color[None, None, :])
        # Subtle bloom on strong mask areas
        glow = cv2.GaussianBlur(out, (0, 0), sigmaX=3.0)
        out = self._clip01(out * (1.0 - 0.25 * t * mask3) + glow * (0.25 * t * mask3))
        return out

    def _apply_special_fx(self, img: np.ndarray, options: ProcessOptions) -> np.ndarray:
        out = img
        # Normalize intensities
        lens_t = max(0.0, min(1.0, options.lens_aging / 100.0)) if options.enable_lens_aging else 0.0
        scratch_t = max(0.0, min(1.0, options.scratches / 100.0)) if options.enable_scratches else 0.0
        # Backward-compat attribute names
        enable_expired = getattr(options, "enable_expired_film", getattr(options, "enable_film_defects", False))
        expired_val = getattr(options, "expired_film", getattr(options, "film_defects", 0))
        enable_leak = getattr(options, "enable_light_leak", getattr(options, "enable_partial_exposure", False))
        leak_val = getattr(options, "light_leak", getattr(options, "partial_exposure", 0))
        defects_t = max(0.0, min(1.0, expired_val / 100.0)) if enable_expired else 0.0
        partial_t = max(0.0, min(1.0, leak_val / 100.0)) if enable_leak else 0.0
        seed = options.fx_seed if options.fx_seed is not None else options.grain_seed
        # Order: lens aging (base look) -> partial exposure -> film defects -> scratches (top-most)
        if lens_t > 0.0:
            out = self._lens_aging(out, lens_t)
        if partial_t > 0.0:
            out = self._light_leak(out, partial_t, None if seed is None else seed + 17)
        if defects_t > 0.0:
            out = self._expired_film(out, defects_t, None if seed is None else seed + 29)
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

