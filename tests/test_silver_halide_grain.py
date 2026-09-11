import numpy as np

from nikon_film_app.processing.film_presets import GrainParams, GrainType, _apply_grain  # type: ignore
from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions


def test_apply_grain_silver_halide_no_broadcast():
    h, w = 133, 200
    img = np.random.RandomState(123).rand(h, w, 3).astype(np.float32)
    params = GrainParams(
        enabled=True,
        grain_type=GrainType.SILVER_HALIDE,
        size01=0.4,
        density01=0.5,
        roughness01=0.6,
        chroma_mix01=0.0,
        seed=1234,
    )
    out = _apply_grain(img, params)
    assert out.shape == img.shape


def test_pipeline_with_silver_halide_no_broadcast():
    proc = ImageProcessor()
    h, w = 133, 200
    img = np.random.RandomState(321).rand(h, w, 3).astype(np.float32)
    opts = ProcessOptions(
        preset_name="Kodak Portra 400",
        strength_percent=100,
        enable_grain=True,
        grain_type=GrainType.SILVER_HALIDE,
        grain_size=40,
        grain_density=60,
        grain_roughness=60,
        grain_chroma_mix=0,
        vignette_mode="auto",
        enable_auto_baseline=True,
    )
    out = proc.process_bgr01(img, opts)
    assert out.shape == img.shape

