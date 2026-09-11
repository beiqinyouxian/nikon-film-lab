from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class BackendMode(str, Enum):
    AUTO = "Auto"
    CPU = "CPU"
    OPENCL = "OpenCL"


@dataclass
class Accelerator:
    mode: BackendMode = BackendMode.AUTO
    opencl_enabled: bool = False
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        self._configure_backend(self.mode)

    def _configure_backend(self, mode: BackendMode) -> None:
        self.mode = mode
        self.opencl_enabled = False
        self.reason = None
        try:
            have_opencl = bool(cv2.ocl.haveOpenCL())
        except Exception:
            have_opencl = False
        if mode == BackendMode.CPU:
            cv2.ocl.setUseOpenCL(False)
            self.opencl_enabled = False
            self.reason = "CPU forced"
            return
        if mode in (BackendMode.OPENCL, BackendMode.AUTO):
            if have_opencl:
                try:
                    cv2.ocl.setUseOpenCL(True)
                    if bool(cv2.ocl.useOpenCL()):
                        self.opencl_enabled = True
                        self.reason = "OpenCL enabled"
                        return
                except Exception as e:
                    self.reason = f"OpenCL request failed: {e}"
            # fallback
            cv2.ocl.setUseOpenCL(False)
            self.opencl_enabled = False
            if mode == BackendMode.OPENCL:
                self.reason = "OpenCL requested but unavailable; fell back to CPU"
            else:
                self.reason = "Auto selected CPU (no OpenCL)"

    def set_mode(self, mode: BackendMode) -> None:
        self._configure_backend(mode)

    def to_mat(self, array: np.ndarray):
        # If OpenCL is active, convert to UMat for operations that support it
        if self.opencl_enabled:
            try:
                return cv2.UMat(array)
            except Exception:
                return array
        return array

    def from_mat(self, mat) -> np.ndarray:
        # Convert back to numpy when needed
        if isinstance(mat, cv2.UMat):
            return mat.get()
        return mat

