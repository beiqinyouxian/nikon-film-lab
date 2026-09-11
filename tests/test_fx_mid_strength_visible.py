import numpy as np
from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions


def _make_img(h=256, w=384):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    b = (0.2 + 0.6 * (xx / max(1, w - 1)))[..., None]
    g = (0.2 + 0.6 * (yy / max(1, h - 1)))[..., None]
    r = (0.2 + 0.6 * (0.5 * xx / max(1, w - 1) + 0.5 * yy / max(1, h - 1)))[..., None]
    return np.clip(np.concatenate([b, g, r], axis=2), 0.0, 1.0).astype(np.float32)


def _proc_with(opts: dict):
    img = _make_img()
    proc = ImageProcessor()
    o = ProcessOptions(
        preset_name="不处理",
        strength_percent=0,
        enable_grain=False,
        vignette_mode="off",
        enable_auto_baseline=False,
        exposure_ev_x100=0,
        temp_bias=0,
        clarity=0,
        contrast=0,
        highlights=0,
        shadows=0,
        vibrance=0,
        saturation=0,
        **opts,
    )
    out = proc.process_bgr01(img, o)
    return img, out


def test_fx_mid_strength_have_visible_effects():
    # lens aging
    img, out1 = _proc_with(dict(enable_lens_aging=True, lens_aging=50, fx_seed=1))
    # light leak
    img, out2 = _proc_with(dict(enable_light_leak=True, light_leak=50, fx_seed=2))
    # expired film
    img, out3 = _proc_with(dict(enable_expired_film=True, expired_film=50, fx_seed=3))
    diffs = [
        float(np.abs(out1 - img).mean()),
        float(np.abs(out2 - img).mean()),
        float(np.abs(out3 - img).mean()),
    ]
    # Mid-level effects should be clearly visible
    for d in diffs:
        assert d > 0.005

