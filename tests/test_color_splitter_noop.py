import numpy as np

from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions


def test_color_splitter_noop_returns_same_image():
    h, w = 240, 320
    # simple gradient color chart
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (xx / max(1, w - 1))[..., None]
    img = np.clip(
        np.concatenate(
            [
                base,                # B
                0.5 * np.ones_like(base),  # G
                (yy / max(1, h - 1))[..., None],  # R
            ],
            axis=2,
        ),
        0.0,
        1.0,
    ).astype(np.float32)
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
        highlights=0,
        shadows=0,
        vibrance=0,
        saturation=0,
        enable_color_splitter=True,
        hsl_sat8=[0] * 8,
        hsl_lum8=[0] * 8,
    )
    out = proc.process_bgr01(img, opts)
    assert out.shape == img.shape
    # allow tiny rounding differences due to conversions
    diff = np.abs(out - img).max()
    assert diff < 1e-3

