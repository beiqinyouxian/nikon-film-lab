from nikon_film_app.processing.accelerator import Accelerator, BackendMode


def test_backend_opencl_fallback_no_crash():
    acc = Accelerator(BackendMode.OPENCL)
    # Regardless of availability, the object should be created and usable
    # If OpenCL is unavailable, it should fall back to CPU gracefully
    assert acc is not None
    assert acc.mode in (BackendMode.OPENCL, BackendMode.AUTO, BackendMode.CPU)
