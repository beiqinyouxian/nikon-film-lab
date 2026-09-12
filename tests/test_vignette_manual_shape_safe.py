import numpy as np

from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions


def test_vignette_manual_shape_safe_and_finite():
    h, w = 360, 540
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (0.2 + 0.6 * (0.5 * xx / max(1, w - 1) + 0.5 * yy / max(1, h - 1)))[..., None]
    img = np.clip(np.concatenate([base, base * 0.9, base * 0.8], axis=2), 0.0, 1.0).astype(np.float32)
    proc = ImageProcessor()
    opts = ProcessOptions(
        preset_name="不处理",
        strength_percent=0,
        enable_grain=False,
        vignette_mode="manual",
        vignette_amount=60,
        enable_auto_baseline=False,
        exposure_ev_x100=0,
        temp_bias=0,
        clarity=0,
        contrast=0,
        highlights=0,
        shadows=0,
        vibrance=0,
        saturation=0,
    )
    out = proc.process_bgr01(img, opts)
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape
    assert np.isfinite(out).all()
