import numpy as np
import pytest

from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions
from nikon_film_app.processing.film_presets import GrainType


@pytest.mark.parametrize("preset", [
    "Cinestill 800T",
    "Leica Chrome Vivid",
    "Kodak Portra 400",
    "Ilford HP5 (B&W)",
])
def test_pipeline_no_broadcast_errors_float(preset: str):
    proc = ImageProcessor()
    h, w = 133, 200  # small preview-like size
    img = np.random.RandomState(42).rand(h, w, 3).astype(np.float32)
    opts = ProcessOptions(
        preset_name=preset,
        strength_percent=100,
        enable_grain=True,
        grain_type=GrainType.SILVER_HALIDE,
        grain_size=35,
        grain_density=40,
        grain_roughness=30,
        grain_chroma_mix=10,
        enable_vignette=True,
        enable_auto_baseline=True,
    )
    out = proc.process_bgr01(img, opts)
    assert out.shape == img.shape


def test_pipeline_uint8_input_and_shape():
    proc = ImageProcessor()
    h, w = 133, 200
    img8 = (np.random.RandomState(0).rand(h, w, 3) * 255).astype(np.uint8)
    opts = ProcessOptions(
        preset_name="Cinestill 800T",
        strength_percent=100,
        enable_grain=True,
        grain_type=GrainType.COARSE_PUSH,
        grain_size=50,
        grain_density=60,
        grain_roughness=60,
        grain_chroma_mix=30,
        enable_vignette=True,
        enable_auto_baseline=True,
    )
    out = proc.process_bgr01(img8.astype(np.float32) / 255.0, opts)
    assert out.shape == img8.shape
