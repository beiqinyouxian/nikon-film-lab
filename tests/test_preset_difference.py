import numpy as np

from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions


def test_preset_not_identity():
    proc = ImageProcessor()
    h, w = 64, 64
    img = np.random.RandomState(0).rand(h, w, 3).astype(np.float32)
    out = proc.process_bgr01(img, ProcessOptions(preset_name=\"Fuji Velvia 50\", strength_percent=100))
    diff = np.mean(np.abs(out - img))
    assert diff > 1e-3
