import numpy as np

from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions


def test_expired_and_lightleak_no_broadcast_shapes():
    # Synthetic HxWx3 float image in 0..1
    h, w = 1065, 1600
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    grad = np.clip(0.3 + 0.7 * (0.5 * x + 0.5 * y), 0.0, 1.0)
    img = np.stack([grad, grad * 0.9, grad * 0.8], axis=-1).astype(np.float32)

    proc = ImageProcessor()
    opts = ProcessOptions(
        preset_name="不处理",
        strength_percent=0,
        enable_grain=False,
        vignette_mode="off",
        enable_auto_baseline=False,
        exposure_ev_x100=0,
        temp_bias=0,
        clarity=0,
        contrast=0,
        enable_expired_film=True,
        expired_film=40,
        enable_light_leak=True,
        light_leak=40,
        defects_seed=123,
    )

    out = proc.process_bgr01(img, opts)
    assert isinstance(out, np.ndarray)
    assert out.shape == (h, w, 3)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 - 1e-6 and out.max() <= 1.0 + 1e-6

