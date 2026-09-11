import numpy as np

from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions


def _luminance(bgr01: np.ndarray) -> float:
    b, g, r = bgr01[..., 0], bgr01[..., 1], bgr01[..., 2]
    return float((0.0722 * r + 0.7152 * g + 0.2126 * b).mean())


def test_exposure_effect_independent_of_strength():
    h, w = 240, 320
    img = np.clip(np.linspace(0.1, 0.9, h, dtype=np.float32)[:, None] * np.ones((1, w, 3), dtype=np.float32), 0.0, 1.0)
    proc = ImageProcessor()
    base_opts = dict(
        preset_name="不处理",
        enable_grain=False,
        vignette_mode="off",
        enable_auto_baseline=False,
        temp_bias=0,
        clarity=0,
        contrast=0,
        highlights=0,
        shadows=0,
        vibrance=0,
        saturation=0,
    )
    # strength=0, exposure=+1 EV
    o1 = ProcessOptions(strength_percent=0, exposure_ev_x100=100, **base_opts)
    out1 = proc.process_bgr01(img, o1)
    # strength=100, exposure=+1 EV (manual should apply equally)
    o2 = ProcessOptions(strength_percent=100, exposure_ev_x100=100, **base_opts)
    out2 = proc.process_bgr01(img, o2)
    # Strength shouldn't significantly change exposure effect
    l1 = _luminance(out1) - _luminance(img)
    l2 = _luminance(out2) - _luminance(img)
    assert abs(l1 - l2) / max(1e-6, abs(l1)) < 0.05

