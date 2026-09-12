import numpy as np
from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions
from nikon_film_app.processing.film_presets import GrainType


def test_manual_controls_pipeline_shapes():
    proc = ImageProcessor()
    h, w = 133, 200
    img = np.random.RandomState(7).rand(h, w, 3).astype(np.float32)
    opts = ProcessOptions(
        preset_name="Cinestill 800T",
        strength_percent=100,
        enable_grain=True,
        grain_type=GrainType.MODERN_FINE,
        grain_size=20,
        grain_density=30,
        grain_roughness=20,
        grain_chroma_mix=0,
        vignette_mode="manual",
        vignette_amount=35,
        enable_auto_baseline=False,
        exposure_ev_x100=50,  # +0.5 EV
        temp_bias=30,         # warmer
    )
    out = proc.process_bgr01(img, opts)
    assert out.shape == img.shape
