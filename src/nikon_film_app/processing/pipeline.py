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
        # 数值健壮性：清理 NaN/Inf，避免后续 UI 转换崩溃
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
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

    @staticmethod
    def _luminance_bgr(img: np.ndarray) -> np.ndarray:
        """Rec.709-ish luminance for BGR float image; returns HxW float32."""
        b = img[..., 0]
        g = img[..., 1]
        r = img[..., 2]
        return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)

    @staticmethod
    def _adjust_contrast(img: np.ndarray, amount: float) -> np.ndarray:
        # amount in roughly [-1, 1]; 0 = no change
        mid = 0.5
        return np.clip((img - mid) * (1.0 + amount) + mid, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _adjust_saturation(img: np.ndarray, amount: float) -> np.ndarray:
        # amount in roughly [-1, 1]
        y = ImageProcessor._luminance_bgr(img)[..., None]
        return np.clip(y + (img - y) * (1.0 + amount), 0.0, 1.0).astype(np.float32)

    def _lens_aging(self, img: np.ndarray, t: float) -> np.ndarray:
        """Aged lens: warm cast, veiling glare, contrast loss, edge softness, mild CA."""
        if t <= 1e-6:
            return img
        h, w = img.shape[:2]
        vis = float(np.power(max(t, 0.0), 0.75))
        b, g, r = cv2.split(img.astype(np.float32))
        r = np.clip(r * (1.0 + 0.08 * vis), 0.0, 1.0)
        g = np.clip(g * (1.0 + 0.03 * vis), 0.0, 1.0)
        b = np.clip(b * (1.0 - 0.07 * vis), 0.0, 1.0)
        colored = cv2.merge([b, g, r]).astype(np.float32)
        # Veiling glare: luminance-gated bloom
        y = self._luminance_bgr(colored)
        hi = np.clip((y - 0.55) / 0.45, 0.0, 1.0).astype(np.float32)
        bloom = cv2.GaussianBlur(colored, (0, 0), sigmaX=2.0 + 4.0 * vis)
        hi3 = self._ensure_3c(hi)
        mixed = colored * (1.0 - 0.35 * vis * hi3) + bloom * (0.35 * vis * hi3)
        # Global haze / contrast loss
        haze = cv2.GaussianBlur(mixed, (0, 0), sigmaX=0.8 + 1.5 * vis)
        mixed = mixed * (1.0 - 0.28 * vis) + haze * (0.28 * vis)
        mixed = self._adjust_contrast(mixed, -0.22 * vis)
        # Edge softness
        edge = 1.0 - self._radial_mask(h, w)
        edge3 = self._ensure_3c(edge.astype(np.float32))
        soft = cv2.GaussianBlur(mixed, (0, 0), sigmaX=1.2 + 2.5 * vis)
        mixed = mixed * (1.0 - 0.30 * vis * edge3) + soft * (0.30 * vis * edge3)
        # Mild CA toward corners: scale R/B slightly
        # Use mgrid so both remap maps are full HxW (ogrid leaves map_y as Hx1 and breaks OpenCV).
        yy, xx = np.mgrid[:h, :w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        rad = np.sqrt(((yy - cy) / max(cy, 1e-6)) ** 2 + ((xx - cx) / max(cx, 1e-6)) ** 2)
        rad = np.clip(rad, 0.0, 1.0).astype(np.float32)
        shift = (0.004 + 0.010 * vis) * rad
        map_x = (xx + shift * (xx - cx)).astype(np.float32)
        map_y = yy.astype(np.float32)
        map_x2 = (xx - shift * (xx - cx)).astype(np.float32)
        bb, gg, rr = cv2.split(mixed)
        rr2 = cv2.remap(rr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        bb2 = cv2.remap(bb, map_x2, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mixed = cv2.merge([bb2, gg, rr2]).astype(np.float32)
        # Slight vignette
        vmask = (self._radial_mask(h, w) ** (1.2 + 1.5 * vis)).astype(np.float32)
        mixed = mixed * (0.94 + 0.06 * vmask[..., None])
        return self._clip01(mixed)

    def _scratches(self, img: np.ndarray, t: float, seed: Optional[int]) -> np.ndarray:
        # Realistic thin anti-aliased hairline scratches:
        # - predominantly faint dark grooves
        # - specular highlights only as thin companions gated by image highlights
        # - micro-wobble and clustered parallels
        # - edge-biased occurrence
        # - soft local haze along the scratch path only
        if t <= 1e-6:
            return img
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed if seed is not None else 0)
        img_f = img.astype(np.float32)
        # Nonlinear visibility so 30–50% is clearly visible but natural
        vis = float(np.clip(0.08 + (t ** 0.62) * 0.92, 0.0, 1.0))
        # Highlight gate (strong specular only where base image is bright)
        b, g, r = cv2.split(img_f)
        luma = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)
        gate = np.clip((luma - 0.58) / 0.25, 0.0, 1.0)  # 0 below ~0.58, 1 above ~0.83
        gate = gate * gate  # tighten
        # Masks (HxW)
        dark_mask = np.zeros((h, w), dtype=np.float32)
        bright_mask = np.zeros((h, w), dtype=np.float32)
        haze_mask = np.zeros((h, w), dtype=np.float32)
        # Edge bias via inverse radial mask
        edge_bias = (1.0 - self._radial_mask(h, w))  # 0 center, 1 edges
        area_scale = max(1.0, np.sqrt(h * w) / 900.0)
        n_clusters = int(min(90, (6 + 16 * vis) * area_scale))
        # Helper to draw a polyline path with micro wobble
        def draw_cluster(start_xy: tuple[int, int], direction: np.ndarray, length_px: float) -> None:
            dir_vec = direction / (np.linalg.norm(direction) + 1e-6)
            # Perpendicular
            nrm = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float32)
            # Steps along the path
            step_px = 8.0
            steps = max(4, int(length_px / step_px))
            pts = []
            wob_amp = 0.6 + 1.4 * vis  # micro wobble amplitude in px
            wob = 0.0
            x, y = float(start_xy[0]), float(start_xy[1])
            for i in range(steps):
                # Cumulative small wobble for smoothness
                wob += float(rng.normal(0.0, 0.35))
                off = np.clip(wob, -wob_amp, wob_amp)
                px = x + dir_vec[0] * (i * step_px) + nrm[0] * off
                py = y + dir_vec[1] * (i * step_px) + nrm[1] * off
                pts.append((int(round(px)), int(round(py))))
            if len(pts) < 2:
                return
            # Central dark groove
            thickness = 1 if rng.random() < 0.85 else 2
            cv2.polylines(dark_mask, [np.array(pts, dtype=np.int32)], False, color=1.0, thickness=thickness, lineType=cv2.LINE_AA)
            # Parallel companions (very close)
            companions = 1 if rng.random() < 0.7 else 2 if rng.random() < 0.3 else 0
            for k in range(companions):
                off_sign = -1.0 if (k % 2 == 0) else 1.0
                off_amt = (0.7 + 0.6 * rng.random()) * off_sign
                pts2 = [(int(round(px + nrm[0] * off_amt)), int(round(py + nrm[1] * off_amt))) for (px, py) in pts]
                cv2.polylines(dark_mask, [np.array(pts2, dtype=np.int32)], False, color=0.9, thickness=1, lineType=cv2.LINE_AA)
            # Specular companion exactly on the groove line (ultra thin)
            cv2.polylines(bright_mask, [np.array(pts, dtype=np.int32)], False, color=1.0, thickness=1, lineType=cv2.LINE_AA)
            # Haze: slightly wider mark for local softening
            cv2.polylines(haze_mask, [np.array(pts, dtype=np.int32)], False, color=1.0, thickness=2 + (1 if thickness > 1 else 0), lineType=cv2.LINE_AA)

        # Spawn clusters, biased to edges and spanning inward
        for _ in range(n_clusters):
            # Pick a side (0=L,1=R,2=T,3=B)
            side = int(rng.integers(0, 4))
            if side == 0:  # left
                y0 = int(rng.integers(-h // 8, h + h // 8))
                start = (-8, y0)
                direction = np.array([1.0, rng.uniform(-0.35, 0.35)], dtype=np.float32)
                inward = np.array([1.0, 0.0], dtype=np.float32)
                d_edge = 1.0
            elif side == 1:  # right
                y0 = int(rng.integers(-h // 8, h + h // 8))
                start = (w + 8, y0)
                direction = np.array([-1.0, rng.uniform(-0.35, 0.35)], dtype=np.float32)
                inward = np.array([-1.0, 0.0], dtype=np.float32)
                d_edge = 1.0
            elif side == 2:  # top
                x0 = int(rng.integers(-w // 8, w + w // 8))
                start = (x0, -8)
                direction = np.array([rng.uniform(-0.35, 0.35), 1.0], dtype=np.float32)
                inward = np.array([0.0, 1.0], dtype=np.float32)
                d_edge = 1.0
            else:  # bottom
                x0 = int(rng.integers(-w // 8, w + w // 8))
                start = (x0, h + 8)
                direction = np.array([rng.uniform(-0.35, 0.35), -1.0], dtype=np.float32)
                inward = np.array([0.0, -1.0], dtype=np.float32)
                d_edge = 1.0
            # Length scaled by vis and size, slight inward bias
            base_len = (0.45 + 0.75 * rng.random()) * (0.65 + 0.7 * vis) * float(max(h, w))
            # Occasional short cluster (micro scuffs)
            if rng.random() < (0.25 + 0.35 * (1.0 - vis)):
                base_len *= 0.45
            # Small inward "pull" to avoid exiting immediately
            direction = (direction * 0.85 + inward * 0.15).astype(np.float32)
            draw_cluster(start, direction, base_len)

        # Post-process masks
        if dark_mask.max() > 0:
            dark_mask = cv2.GaussianBlur(dark_mask, (0, 0), sigmaX=0.6)
        if bright_mask.max() > 0:
            # Gate by highlights; keep ultra thin and sparse
            bright_mask = bright_mask * (gate ** 1.6)
            bright_mask = cv2.GaussianBlur(bright_mask, (0, 0), sigmaX=0.5)
        if haze_mask.max() > 0:
            haze_mask = cv2.GaussianBlur(haze_mask, (0, 0), sigmaX=1.6)
            # Only haze where there is a dark groove
            haze_mask = np.minimum(haze_mask, cv2.GaussianBlur(dark_mask, (0, 0), sigmaX=1.2))
        dark_mask = np.clip(dark_mask, 0.0, 1.0).astype(np.float32)
        bright_mask = np.clip(bright_mask, 0.0, 1.0).astype(np.float32)
        haze_mask = np.clip(haze_mask, 0.0, 1.0).astype(np.float32)
        # Strengths
        dark_k = np.float32(0.18 + 0.36 * vis)    # multiplicative dimming
        bright_k = np.float32(0.05 + 0.22 * vis)  # additive highlight (gated)
        haze_k = np.float32(0.06 + 0.18 * vis)    # local soft haze
        # Apply dark grooves (edge-biased slightly stronger)
        edge_boost = (0.85 + 0.30 * edge_bias).astype(np.float32)
        dark3 = self._ensure_3c(dark_mask * edge_boost)
        out = self._clip01(img_f * (1.0 - dark_k * dark3))
        # Apply specular companions (white, highlight-gated)
        if bright_k > 0:
            spec3 = self._ensure_3c(bright_mask * gate)
            out = self._clip01(out + bright_k * spec3)
        # Local haze along scratches (blend with a slightly blurred base)
        if haze_k > 0:
            base_blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.35)
            haze3 = self._ensure_3c(haze_mask)
            out = self._clip01(out * (1.0 - 0.5 * haze_k * haze3) + base_blur * (0.5 * haze_k * haze3))
        return out

    def _expired_film(self, img: np.ndarray, t: float, seed: Optional[int]) -> np.ndarray:
        """Expired film: base fog, DR crush, cast, mottled dye fade."""
        if t <= 1e-6:
            return img
        # Soften peak: UI 0..100 maps into ~0..0.55 of the previous max,
        # so mid/high slider steps are finer and less extreme.
        t = float(np.clip(t, 0.0, 1.0)) * 0.55
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed if seed is not None else 1)
        vis = float(np.power(max(t, 0.0), 0.9))
        out = img.astype(np.float32).copy()
        fog = 0.05 + 0.14 * vis
        out = self._clip01(out * (1.0 - 0.55 * vis) + fog)
        out = self._adjust_contrast(out, -0.25 * vis)
        out = self._clip01(1.0 - (1.0 - out) * (1.0 - 0.08 * vis))
        out = self._adjust_saturation(out, -0.30 * vis)
        if rng.random() < 0.5:
            cast = np.array([0.00, 0.05 + 0.12 * vis, 0.08 * vis], dtype=np.float32)[None, None, :]
        else:
            cast = np.array([0.04 + 0.10 * vis, 0.03 * vis, 0.00], dtype=np.float32)[None, None, :]
        out = self._clip01(out + cast)
        y = self._luminance_bgr(out)
        sh = np.clip((0.45 - y) / 0.45, 0.0, 1.0)[..., None]
        out = self._clip01(out + sh * np.array([0.02, 0.00, 0.03], dtype=np.float32)[None, None, :] * (0.6 * vis))
        noise = rng.normal(0.0, 1.0, size=(h, w, 3)).astype(np.float32)
        low = cv2.GaussianBlur(noise, (0, 0), sigmaX=14.0)
        out = self._clip01(out + 0.06 * vis * low)
        specks = np.zeros((h, w), dtype=np.float32)
        n = int(24 * vis)
        for _ in range(max(0, n)):
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))
            rr = int(max(1, rng.integers(1, 2 + int(2 * vis))))
            cv2.circle(specks, (cx, cy), rr, color=1.0, thickness=-1, lineType=cv2.LINE_AA)
        specks = cv2.GaussianBlur(specks, (0, 0), sigmaX=0.6)
        out = self._clip01(out * (1.0 - 0.05 * vis * self._ensure_3c(specks)))
        return out

    def _light_leak(self, img: np.ndarray, t: float, seed: Optional[int]) -> np.ndarray:
        """Organic asymmetric 1–2 corner cubic pools + soft irregular edge bleed (noise-warp).
        Warm orange→amber→magenta/red cast. Screen/additive + gentle bloom. Mid intensity visible."""
        if t <= 1e-6:
            return img
        h, w = img.shape[:2]
        rng = np.random.default_rng(seed if seed is not None else 2)
        img_f = img.astype(np.float32)
        # Nonlinear visibility so 40–60% reads obviously analog
        vis = float(np.clip(0.10 + (t ** 0.65) * 0.90, 0.0, 1.0))
        # Coordinates
        y, x = np.ogrid[:h, :w]
        nx = x.astype(np.float32) / max(1, w - 1)
        ny = y.astype(np.float32) / max(1, h - 1)
        # Corner pools (choose 1–2 corners)
        corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        pick = rng.choice(4, size=int(rng.integers(1, 3)), replace=False)
        pool = np.zeros((h, w), dtype=np.float32)
        for idx in pick:
            cx, cy = corners[idx]
            dx = np.abs(nx - cx)
            dy = np.abs(ny - cy)
            d = np.sqrt(dx * dx + dy * dy)
            # Cubic pool with slight random size/power
            pwr = 2.6 + 0.9 * rng.random()
            rad = 1.0
            v = np.clip(1.0 - (d / rad) ** pwr, 0.0, 1.0) ** 3.0
            # Slight asym wobble via smoothed noise
            n = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=10.0 + 18.0 * vis)
            n = (n - n.min()) / max(1e-6, n.max() - n.min())
            v *= (0.80 + 0.35 * n)
            pool = np.maximum(pool, v.astype(np.float32))
        # Irregular edge bleed (distance from nearest edge with noise warp)
        dist_l = nx
        dist_r = 1.0 - nx
        dist_t = ny
        dist_b = 1.0 - ny
        dmin = np.minimum(np.minimum(dist_l, dist_r), np.minimum(dist_t, dist_b))
        bleed = np.exp(-5.0 * (dmin ** (0.75)))  # strong at edges, falls inwards
        # Noise warp (low-frequency)
        wn = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        wn = cv2.GaussianBlur(wn, (0, 0), sigmaX=12.0 + 15.0 * vis)
        wn = (wn - wn.min()) / max(1e-6, wn.max() - wn.min())
        bleed *= (0.65 + 0.45 * wn)
        bleed = cv2.GaussianBlur(bleed, (0, 0), sigmaX=3.0)
        # Combine masks
        leak_mask = np.clip(np.maximum(pool, 0.55 * bleed), 0.0, 1.0)
        # Color ramp: orange -> amber -> magenta/red (BGR)
        orange = np.array([0.03, 0.22, 0.56], dtype=np.float32)
        amber = np.array([0.05, 0.30, 0.50], dtype=np.float32)
        magenta_red = np.array([0.14, 0.06, 0.62], dtype=np.float32)
        r1 = float(rng.uniform(0.25, 0.85))
        r2 = float(rng.uniform(0.25, 0.95))
        warm = orange * (1.0 - r1) + amber * r1
        color = warm * (1.0 - r2) + magenta_red * r2
        # Strength and shaping
        mask_shaped = leak_mask ** (0.95)  # preserve core intensity
        k = 0.28 + 0.60 * vis
        # Screen blend: out = 1 - (1-img)*(1 - k * mask * color)
        mask3 = self._ensure_3c(mask_shaped) * color[None, None, :]
        add = np.clip(k * mask3, 0.0, 1.0).astype(np.float32)
        out = 1.0 - (1.0 - img_f) * (1.0 - add)
        out = self._clip01(out)
        # Gentle bloom on high mask regions
        if leak_mask.max() > 1e-6:
            glow_mask = (cv2.GaussianBlur(leak_mask, (0, 0), sigmaX=2.8)) ** 1.1
            glow3 = self._ensure_3c(np.clip(glow_mask, 0.0, 1.0))
            blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=2.2 + 1.8 * vis)
            bloom_k = 0.15 + 0.25 * vis
            out = self._clip01(out * (1.0 - bloom_k * glow3) + blurred * (bloom_k * glow3))
        return out

    def _apply_special_fx(self, img: np.ndarray, options: ProcessOptions) -> np.ndarray:
        out = img
        lens_t = max(0.0, min(1.0, options.lens_aging / 100.0)) if options.enable_lens_aging else 0.0
        scratch_t = max(0.0, min(1.0, options.scratches / 100.0)) if options.enable_scratches else 0.0
        enable_expired = getattr(options, "enable_expired_film", getattr(options, "enable_film_defects", False))
        expired_val = getattr(options, "expired_film", getattr(options, "film_defects", 0))
        enable_leak = getattr(options, "enable_light_leak", getattr(options, "enable_partial_exposure", False))
        leak_val = getattr(options, "light_leak", getattr(options, "partial_exposure", 0))
        defects_t = max(0.0, min(1.0, expired_val / 100.0)) if enable_expired else 0.0
        partial_t = max(0.0, min(1.0, leak_val / 100.0)) if enable_leak else 0.0
        seed = options.fx_seed if options.fx_seed is not None else options.grain_seed
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

