import numpy as np

from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions


def test_resolution_unchanged():
    proc = ImageProcessor()
    h, w = 480, 640
    # synthetic gradient image
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    img = np.dstack([
        x.repeat(h, axis=0),
        y.repeat(w, axis=1),
        0.5 * np.ones((h, w), dtype=np.float32),
    ])
    out = proc.process_bgr01(img, ProcessOptions(preset_name="Fuji Velvia 50", strength_percent=70))
    assert out.shape[:2] == (h, w)
