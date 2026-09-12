from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import zlib

from nikon_film_app.processing.accelerator import Accelerator, BackendMode
from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions
from nikon_film_app.processing.film_presets import GrainType
from nikon_film_app.io.jpeg_utils import load_jpeg_bgr8, save_jpeg_bgr8
from nikon_film_app.io.raw_loader import load_nef_to_bgr8


SUPPORTED_EXT = {".jpg", ".jpeg", ".nef"}


def is_supported(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_EXT


def bgr_to_qpixmap(bgr: np.ndarray, max_side: int = 800) -> QtGui.QPixmap:
    # Guard bad arrays so QImage never hard-crashes the process
    try:
        if not isinstance(bgr, np.ndarray) or bgr.ndim < 2:
            raise ValueError("invalid array")
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        if bgr.shape[2] != 3 or bgr.size == 0:
            raise ValueError("expect non-empty HxWx3")
        if bgr.dtype != np.uint8:
            bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    except Exception:
        pm = QtGui.QPixmap(max_side, max_side)
        pm.fill(QtGui.QColor("#555555"))
        return pm
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    display = bgr
    if scale < 1.0:
        display = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(display[..., ::-1])
    h2, w2 = rgb.shape[:2]
    qimg = QtGui.QImage(rgb.data, w2, h2, 3 * w2, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


@dataclass
class QueueItem:
    path: str
    is_raw: bool
    exif_bytes: Optional[bytes]
    src_bgr8: np.ndarray


class ProcessorThread(QtCore.QThread):
    progress_changed = QtCore.Signal(int, int)  # current, total
    file_processed = QtCore.Signal(str, bool, str)  # path, success, message
    preview_ready = QtCore.Signal(np.ndarray, np.ndarray)  # before, after

    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue: List[QueueItem] = []
        self.cancelled = False
        self.processor = ImageProcessor()
        self.options = ProcessOptions(
            preset_name="不处理",
            strength_percent=0,
            enable_grain=False,
            vignette_mode="off",
            vignette_amount=0,
            enable_auto_baseline=False,
            exposure_ev_x100=0,
            temp_bias=0,
            clarity=0,
            contrast=0,
            highlights=0,
            shadows=0,
            vibrance=0,
            saturation=0,
        )
        self.export_dir = os.getcwd()
        self.backend = BackendMode.AUTO

    def set_backend(self, mode: BackendMode) -> None:
        self.processor.accel.set_mode(mode)
        self.backend = mode

    def run(self) -> None:
        total = len(self.queue)
        for idx, item in enumerate(self.queue, start=1):
            if self.cancelled:
                break
            try:
                self.progress_changed.emit(idx, total)
                # Deterministic grain per file
                self.options.grain_seed = zlib.adler32(item.path.encode("utf-8")) & 0xFFFFFFFF
                # Deterministic FX per file (separate namespace)
                self.options.fx_seed = zlib.adler32((item.path + "#fx").encode("utf-8")) & 0xFFFFFFFF
                out = self._process_bgr(item.src_bgr8)
                # Save
                base = os.path.basename(item.path)
                name, _ = os.path.splitext(base)
                out_path = os.path.join(self.export_dir, f"{name}_film.jpg")
                exif = item.exif_bytes if not item.is_raw else None
                save_jpeg_bgr8(out_path, out, quality=95, exif_bytes=exif)
                self.file_processed.emit(item.path, True, out_path)
                if idx == 1:
                    self.preview_ready.emit(item.src_bgr8, out)
            except Exception as e:
                self.file_processed.emit(item.path, False, str(e))
        self.progress_changed.emit(total, total)

    def _process_bgr(self, bgr8: np.ndarray) -> np.ndarray:
        img = bgr8.astype(np.float32) / 255.0
        out = self.processor.process_bgr01(img, self.options)
        out8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return out8


class DropListWidget(QtWidgets.QListWidget):
    files_dropped = QtCore.Signal(list)
    resized = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.resized.emit()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in files:
                        fp = os.path.join(root, f)
                        if is_supported(fp):
                            paths.append(fp)
            elif is_supported(p):
                paths.append(p)
        if paths:
            self.files_dropped.emit(paths)



class ThumbWorker(QtCore.QThread):
    """Background thumbnail loader for the queue list."""

    thumb_ready = QtCore.Signal(str, object)  # path, bgr8 uint8 HxWx3

    def __init__(self, parent=None, max_side: int = 512) -> None:
        super().__init__(parent)
        self.max_side = int(max_side)
        self._q: deque[str] = deque()
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._stop = False
        self._wake = threading.Event()

    def enqueue(self, path: str) -> None:
        if not path:
            return
        with self._lock:
            if path in self._pending:
                return
            self._pending.add(path)
            self._q.append(path)
        self._wake.set()

    def clear_pending(self) -> None:
        with self._lock:
            self._q.clear()
            self._pending.clear()
        self._wake.set()

    def stop(self) -> None:
        self._stop = True
        self._wake.set()

    def _pop(self) -> Optional[str]:
        with self._lock:
            if not self._q:
                return None
            path = self._q.popleft()
            self._pending.discard(path)
            return path

    def _make_thumb(self, path: str) -> Optional[np.ndarray]:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".jpg", ".jpeg"):
                # Fast path: decode then downscale
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    r = load_jpeg_bgr8(path)
                    img = r.image_bgr8
            elif ext == ".nef":
                # Prefer embedded JPEG preview when available (much faster)
                img = None
                try:
                    import rawpy  # type: ignore

                    with rawpy.imread(path) as raw:
                        try:
                            thumb = raw.extract_thumb()
                            if getattr(thumb, "format", None) == rawpy.ThumbFormat.JPEG:
                                buf = np.frombuffer(thumb.data, dtype=np.uint8)
                                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        except Exception:
                            img = None
                except Exception:
                    img = None
                if img is None:
                    r2 = load_nef_to_bgr8(path)
                    img = r2.image_bgr8
            else:
                return None
            if img is None or getattr(img, "size", 0) == 0:
                return None
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.ndim != 3 or img.shape[2] != 3:
                return None
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            h, w = img.shape[:2]
            scale = min(1.0, float(self.max_side) / float(max(h, w)))
            if scale < 1.0:
                img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            return np.ascontiguousarray(img)
        except Exception:
            return None

    def run(self) -> None:
        while not self._stop:
            path = self._pop()
            if path is None:
                self._wake.wait(0.25)
                self._wake.clear()
                continue
            thumb = self._make_thumb(path)
            if thumb is not None and not self._stop:
                self.thumb_ready.emit(path, thumb)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nikon Film Lab")
        self.thread = ProcessorThread()
        self.preview_max_side = 1600
        self.preview_cache: Dict[str, np.ndarray] = {}
        self._current_before_bgr: Optional[np.ndarray] = None
        self._current_after_bgr: Optional[np.ndarray] = None
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(200)  # debounce ~200ms
        self.preview_timer.timeout.connect(self._render_preview)
        self._preview_seq = 0

        # UI - single preview
        self.preview_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(200, 200)
        self.preview_label.setScaledContents(False)

        self.list_widget = DropListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        # One-column thumbs that track the queue panel width
        self.list_widget.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.list_widget.setFlow(QtWidgets.QListView.Flow.TopToBottom)
        self.list_widget.setWrapping(False)
        self.list_widget.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list_widget.setMovement(QtWidgets.QListView.Movement.Static)
        self.list_widget.setSpacing(6)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setWordWrap(True)
        # Thumbnails (decode larger, display scaled to panel width)
        self._thumb_worker = ThumbWorker(self, max_side=512)
        self._thumb_worker.thumb_ready.connect(self._on_thumb_ready)
        self._thumb_worker.start()
        self._thumb_bgr: Dict[str, np.ndarray] = {}
        self._path_to_item: Dict[str, QtWidgets.QListWidgetItem] = {}
        self._queue_thumb_w = 160
        self._placeholder_icon = self._make_placeholder_icon(self._queue_thumb_w)
        self.list_widget.setIconSize(QtCore.QSize(self._queue_thumb_w, self._queue_thumb_w))

        self.add_btn = QtWidgets.QPushButton("添加文件")
        self.add_dir_btn = QtWidgets.QPushButton("添加文件夹")
        self.clear_btn = QtWidgets.QPushButton("清空")
        self.process_btn = QtWidgets.QPushButton("开始处理")
        self.cancel_btn = QtWidgets.QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.reset_btn = QtWidgets.QPushButton("复位")

        self.export_edit = QtWidgets.QLineEdit(os.getcwd())
        self.export_btn = QtWidgets.QPushButton("选择导出文件夹")

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(self.thread.processor.list_presets())
        # default to "不处理"
        idx_id = self.preset_combo.findText("不处理")
        if idx_id >= 0:
            self.preset_combo.setCurrentIndex(idx_id)
        self.strength_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setValue(0)
        self.strength_preset_btn = QtWidgets.QPushButton("推荐强度")
        self.strength_preset_btn.setToolTip("应用当前预设的推荐强度")
        self.grain_check = QtWidgets.QCheckBox("颗粒")
        self.grain_check.setChecked(False)
        self.grain_preset_btn = QtWidgets.QPushButton("推荐参数")
        self.grain_preset_btn.setToolTip("应用当前颗粒类型的推荐滑条参数")
        self.grain_type = QtWidgets.QComboBox()
        self.grain_type.addItems([GrainType.SILVER_HALIDE.value, GrainType.MODERN_FINE.value, GrainType.COARSE_PUSH.value])
        self.grain_size = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.grain_size.setRange(0, 100)
        self.grain_size.setValue(0)
        self.grain_density = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.grain_density.setRange(0, 100)
        self.grain_density.setValue(0)
        self.grain_rough = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.grain_rough.setRange(0, 100)
        self.grain_rough.setValue(0)
        self.grain_chroma = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.grain_chroma.setRange(0, 100)
        self.grain_chroma.setValue(0)
        self.vignette_mode = QtWidgets.QComboBox()
        self.vignette_mode.addItems(["不处理", "自动", "手动"])
        self.vignette_mode.setCurrentIndex(0)
        self.vignette_amount = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.vignette_amount.setRange(0, 100)
        self.vignette_amount.setValue(0)
        self.vignette_amount.setEnabled(False)
        # 独立特色效果控件
        self.fx_lens_check = QtWidgets.QCheckBox("镜头老化")
        self.fx_lens_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fx_lens_slider.setRange(0, 100)
        self.fx_lens_slider.setValue(0)
        self.fx_scratches_check = QtWidgets.QCheckBox("镜片划伤")
        self.fx_scratches_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fx_scratches_slider.setRange(0, 100)
        self.fx_scratches_slider.setValue(0)
        self.fx_defects_check = QtWidgets.QCheckBox("胶片过期")
        self.fx_defects_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fx_defects_slider.setRange(0, 100)
        self.fx_defects_slider.setValue(0)
        self.fx_partial_check = QtWidgets.QCheckBox("胶片漏光")
        self.fx_partial_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fx_partial_slider.setRange(0, 100)
        self.fx_partial_slider.setValue(0)
        def _bipolar(slider: QtWidgets.QSlider, lo: int, hi: int) -> None:
            slider.setRange(lo, hi)
            slider.setValue(0)
            slider.setTickPosition(QtWidgets.QSlider.TickPosition.NoTicks)

        def _short_slider(slider: QtWidgets.QSlider) -> QtWidgets.QSlider:
            slider.setMinimumWidth(72)
            slider.setMaximumWidth(140)
            slider.setFixedHeight(22)
            return slider

        self.exposure_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        _bipolar(self.exposure_slider, -200, 200)  # -2.00 .. +2.00 EV (x100), center 0
        self.temp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        _bipolar(self.temp_slider, -100, 100)  # cooler .. warmer, center 0
        self.clarity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        _bipolar(self.clarity_slider, -100, 100)  # softer .. clearer, center 0
        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        _bipolar(self.contrast_slider, -100, 100)  # flatter .. punchier, center 0
        self.highlights_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        _bipolar(self.highlights_slider, -100, 100)
        self.shadows_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        _bipolar(self.shadows_slider, -100, 100)
        self.vibrance_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        _bipolar(self.vibrance_slider, -100, 100)
        self.saturation_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        _bipolar(self.saturation_slider, -100, 100)
        self.auto_check = QtWidgets.QCheckBox("自动基线（曝光/色温）")
        self.auto_check.setChecked(False)

        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems([BackendMode.AUTO.value, BackendMode.CPU.value, BackendMode.OPENCL.value])

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)  # avoid Windows style overlapping % text
        self.progress_label = QtWidgets.QLabel("0%")
        self.progress_label.setMinimumWidth(64)
        self.progress_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._progress_style_busy = "QProgressBar::chunk { background-color: #3b82f6; }"
        self._progress_style_done = "QProgressBar::chunk { background-color: #22c55e; }"
        self.progress.setStyleSheet(self._progress_style_busy)
        self._progress_anim: Optional[QtCore.QVariantAnimation] = None
        self._slider_anims: Dict[QtWidgets.QSlider, QtCore.QVariantAnimation] = {}

        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("队列（拖放 .nef/.jpg）："))
        left.addWidget(self.list_widget, 1)
        left_buttons = QtWidgets.QHBoxLayout()
        for b in (self.add_btn, self.add_dir_btn, self.clear_btn):
            left_buttons.addWidget(b)
        left.addLayout(left_buttons)
        left.addWidget(QtWidgets.QLabel("导出文件夹："))
        exp = QtWidgets.QHBoxLayout()
        exp.addWidget(self.export_edit, 1)
        exp.addWidget(self.export_btn)
        left.addLayout(exp)
        left.addWidget(QtWidgets.QLabel("进度："))
        prog_row = QtWidgets.QHBoxLayout()
        prog_row.addWidget(self.progress, 1)
        prog_row.addWidget(self.progress_label)
        left.addLayout(prog_row)

        # Compact params: short sliders, multi-column grid to reduce vertical scroll
        for _s in (
            self.strength_slider,
            self.vignette_amount,
            self.exposure_slider,
            self.temp_slider,
            self.clarity_slider,
            self.contrast_slider,
            self.highlights_slider,
            self.shadows_slider,
            self.vibrance_slider,
            self.saturation_slider,
            self.fx_lens_slider,
            self.fx_scratches_slider,
            self.fx_defects_slider,
            self.fx_partial_slider,
            self.grain_size,
            self.grain_density,
            self.grain_rough,
            self.grain_chroma,
        ):
            _short_slider(_s)

        self.preset_combo.setMaximumWidth(180)
        self.vignette_mode.setMaximumWidth(100)
        self.backend_combo.setMaximumWidth(120)
        self.grain_type.setMaximumWidth(140)
        self.grain_preset_btn.setMaximumWidth(88)

        def _pair_row(label: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(w)
            row.setContentsMargins(0, 0, 4, 0)
            row.setSpacing(4)
            lab = QtWidgets.QLabel(label)
            lab.setMinimumWidth(48)
            lab.setMaximumWidth(56)
            lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(lab)
            row.addWidget(widget, 1)
            return w

        # —— 分组与辅助线 —— #
        def _hline() -> QtWidgets.QFrame:
            ln = QtWidgets.QFrame()
            ln.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            ln.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            ln.setStyleSheet("QFrame{color:#dcdcdc; background:#dcdcdc; max-height:1px;}")
            return ln

        # 基础 / 预设
        basics_box = QtWidgets.QGroupBox("基础 / 预设")
        bgrid = QtWidgets.QGridLayout(basics_box)
        bgrid.setContentsMargins(6, 8, 6, 6)
        bgrid.setHorizontalSpacing(8)
        bgrid.setVerticalSpacing(4)
        bgrid.addWidget(_pair_row("预设", self.preset_combo), 0, 0)
        # 强度行：滑条 + 推荐按钮
        strength_row = QtWidgets.QWidget()
        strength_layout = QtWidgets.QHBoxLayout(strength_row)
        strength_layout.setContentsMargins(0, 0, 0, 0)
        strength_layout.setSpacing(4)
        strength_layout.addWidget(self.strength_slider, 1)
        strength_layout.addWidget(self.strength_preset_btn)
        bgrid.addWidget(_pair_row("强度", strength_row), 0, 1)
        bgrid.addWidget(_pair_row("暗角", self.vignette_mode), 0, 2)
        bgrid.addWidget(_pair_row("暗角量", self.vignette_amount), 0, 3)
        bgrid.addWidget(self.auto_check, 1, 0)
        bgrid.addWidget(_pair_row("后端", self.backend_combo), 1, 1)
        for c in range(4):
            bgrid.setColumnStretch(c, 1)

        # 影调 / 色彩
        tone_box = QtWidgets.QGroupBox("影调 / 色彩")
        tgrid = QtWidgets.QGridLayout(tone_box)
        tgrid.setContentsMargins(6, 8, 6, 6)
        tgrid.setHorizontalSpacing(8)
        tgrid.setVerticalSpacing(4)
        tgrid.addWidget(_pair_row("曝光", self.exposure_slider), 0, 0)
        tgrid.addWidget(_pair_row("色温", self.temp_slider), 0, 1)
        tgrid.addWidget(_pair_row("对比", self.contrast_slider), 0, 2)
        tgrid.addWidget(_pair_row("清晰", self.clarity_slider), 0, 3)
        tgrid.addWidget(_pair_row("高光", self.highlights_slider), 1, 0)
        tgrid.addWidget(_pair_row("阴影", self.shadows_slider), 1, 1)
        tgrid.addWidget(_pair_row("鲜艳", self.vibrance_slider), 1, 2)
        tgrid.addWidget(_pair_row("饱和", self.saturation_slider), 1, 3)
        for c in range(4):
            tgrid.setColumnStretch(c, 1)

        # 颗粒
        grain_box = QtWidgets.QGroupBox("颗粒")
        ggrid = QtWidgets.QGridLayout(grain_box)
        ggrid.setContentsMargins(6, 8, 6, 6)
        ggrid.setHorizontalSpacing(8)
        ggrid.setVerticalSpacing(4)
        grain_head = QtWidgets.QWidget()
        gh = QtWidgets.QHBoxLayout(grain_head)
        gh.setContentsMargins(0, 0, 4, 0)
        gh.setSpacing(4)
        gh.addWidget(self.grain_check)
        gh.addWidget(self.grain_preset_btn)
        gh.addStretch(1)
        ggrid.addWidget(grain_head, 0, 0)
        ggrid.addWidget(_pair_row("类型", self.grain_type), 0, 1)
        ggrid.addWidget(_pair_row("大小", self.grain_size), 0, 2)
        ggrid.addWidget(_pair_row("密度", self.grain_density), 0, 3)
        ggrid.addWidget(_pair_row("粗糙", self.grain_rough), 1, 0)
        ggrid.addWidget(_pair_row("彩混", self.grain_chroma), 1, 1)
        for c in range(4):
            ggrid.setColumnStretch(c, 1)

        # 镜头与胶片缺陷
        fx_box = QtWidgets.QGroupBox("镜头与胶片缺陷")
        fgrid = QtWidgets.QGridLayout(fx_box)
        fgrid.setContentsMargins(6, 8, 6, 6)
        fgrid.setHorizontalSpacing(8)
        fgrid.setVerticalSpacing(4)
        def _fx_cell(check: QtWidgets.QCheckBox, slider: QtWidgets.QSlider) -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            h.addWidget(check)
            h.addWidget(slider, 1)
            return w
        fgrid.addWidget(_fx_cell(self.fx_lens_check, self.fx_lens_slider), 0, 0)
        fgrid.addWidget(_fx_cell(self.fx_scratches_check, self.fx_scratches_slider), 0, 1)
        fgrid.addWidget(_fx_cell(self.fx_defects_check, self.fx_defects_slider), 0, 2)
        fgrid.addWidget(_fx_cell(self.fx_partial_check, self.fx_partial_slider), 0, 3)
        for c in range(4):
            fgrid.setColumnStretch(c, 1)
        # 分色器：两行（饱和/明度）x 8 列，紧凑滑条（独立分组）
        splitter_box = QtWidgets.QGroupBox("分色器")
        split_grid = QtWidgets.QGridLayout(splitter_box)
        split_grid.setContentsMargins(6, 6, 6, 6)
        split_grid.setHorizontalSpacing(6)
        split_grid.setVerticalSpacing(2)
        bands = ["红", "橙", "黄", "绿", "青", "蓝", "紫", "品红"]
        self.hsl_sat_sliders: List[QtWidgets.QSlider] = []
        self.hsl_lum_sliders: List[QtWidgets.QSlider] = []
        # Header labels
        for col, name in enumerate(bands):
            lab = QtWidgets.QLabel(name)
            lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter)
            split_grid.addWidget(lab, 0, col)
        for col in range(8):
            s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            s.setRange(-100, 100)
            s.setValue(0)
            _short_slider(s)
            self.hsl_sat_sliders.append(s)
            split_grid.addWidget(s, 1, col)
        for col in range(8):
            l = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            l.setRange(-100, 100)
            l.setValue(0)
            _short_slider(l)
            self.hsl_lum_sliders.append(l)
            split_grid.addWidget(l, 2, col)
        # 行说明文本放在第 9 列，避免覆盖第 8 列（品红）
        leg_col = 8
        leg_sat = QtWidgets.QLabel("饱和")
        leg_sat.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        leg_lum = QtWidgets.QLabel("明度")
        leg_lum.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        split_grid.addWidget(leg_sat, 1, leg_col, 1, 1)
        split_grid.addWidget(leg_lum, 2, leg_col, 1, 1)
        # 列伸展：0..7 为色带列，均匀分布；第 8 列为小标签列
        for c in range(8):
            split_grid.setColumnStretch(c, 1)
        split_grid.setColumnStretch(leg_col, 0)
        split_grid.setColumnMinimumWidth(leg_col, 36)

        # 组装外层布局（带分割线）
        params_outer = QtWidgets.QVBoxLayout()
        params_outer.setContentsMargins(4, 4, 4, 4)
        params_outer.setSpacing(6)
        params_outer.addWidget(basics_box)
        params_outer.addWidget(_hline())
        params_outer.addWidget(tone_box)
        params_outer.addWidget(_hline())
        params_outer.addWidget(grain_box)
        params_outer.addWidget(_hline())
        params_outer.addWidget(fx_box)
        params_outer.addWidget(_hline())
        params_outer.addWidget(splitter_box)

        btns = QtWidgets.QHBoxLayout()
        btns.setSpacing(6)
        btns.addStretch(1)
        btns.addWidget(self.process_btn)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.reset_btn)
        params_outer.addLayout(btns)
        tip = QtWidgets.QLabel("提示：拖动粗分隔条调节预览/参数高度；导出始终全分辨率。")
        tip.setWordWrap(True)
        params_outer.addWidget(tip)
        params_widget = QtWidgets.QWidget()
        params_widget.setLayout(params_outer)
        params_scroll = QtWidgets.QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(params_widget)
        params_scroll.setMinimumHeight(120)
        params_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        params_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        preview_wrap = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_wrap)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.preview_label)

        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.right_splitter.addWidget(preview_wrap)
        self.right_splitter.addWidget(params_scroll)
        self.right_splitter.setChildrenCollapsible(False)
        self.right_splitter.setStretchFactor(0, 3)
        self.right_splitter.setStretchFactor(1, 1)
        self.right_splitter.setHandleWidth(10)  # thicker vertical drag handle
        self.right_splitter.setSizes([480, 300])

        lw = QtWidgets.QWidget()
        lw.setLayout(left)
        lw.setMinimumWidth(160)
        lw.setMaximumWidth(360)
        self.root_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.root_splitter.addWidget(lw)
        self.root_splitter.addWidget(self.right_splitter)
        self.root_splitter.setChildrenCollapsible(False)
        self.root_splitter.setStretchFactor(0, 0)
        self.root_splitter.setStretchFactor(1, 1)
        self.root_splitter.setHandleWidth(6)
        self.root_splitter.setSizes([200, 1000])  # narrower queue by default
        self.setCentralWidget(self.root_splitter)
        self._apply_splitter_style()
        self._restore_splitters()
        self._queue_resize_timer = QtCore.QTimer(self)
        self._queue_resize_timer.setSingleShot(True)
        self._queue_resize_timer.setInterval(50)
        self._queue_resize_timer.timeout.connect(self._sync_queue_thumb_size)
        self.list_widget.resized.connect(lambda: self._queue_resize_timer.start())
        self.root_splitter.splitterMoved.connect(lambda *_: self._queue_resize_timer.start())
        QtCore.QTimer.singleShot(0, self._sync_queue_thumb_size)

        # Signals
        self.add_btn.clicked.connect(self.on_add_files)
        self.add_dir_btn.clicked.connect(self.on_add_dir)
        self.clear_btn.clicked.connect(self.on_clear)
        self.export_btn.clicked.connect(self.on_pick_export)
        self.list_widget.files_dropped.connect(self.on_files_dropped)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_widget.itemClicked.connect(self.on_selection_changed)
        self.process_btn.clicked.connect(self.on_process)
        self.cancel_btn.clicked.connect(self.on_cancel)
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        self.strength_slider.valueChanged.connect(self.on_strength_changed)
        self.strength_preset_btn.clicked.connect(self.on_strength_recommended_clicked)
        self.grain_check.toggled.connect(self.on_grain_enabled_changed)
        self.grain_preset_btn.clicked.connect(self.on_grain_preset_clicked)
        self.grain_type.currentTextChanged.connect(self.on_grain_type_changed)
        self.grain_size.valueChanged.connect(self.on_grain_params_changed)
        self.grain_density.valueChanged.connect(self.on_grain_params_changed)
        self.grain_rough.valueChanged.connect(self.on_grain_params_changed)
        self.grain_chroma.valueChanged.connect(self.on_grain_params_changed)
        self.vignette_mode.currentIndexChanged.connect(self.on_vignette_mode_changed)
        self.vignette_amount.valueChanged.connect(self.on_manual_adjust_changed)
        self.exposure_slider.valueChanged.connect(self.on_manual_adjust_changed)
        self.temp_slider.valueChanged.connect(self.on_manual_adjust_changed)
        self.clarity_slider.valueChanged.connect(self.on_manual_adjust_changed)
        self.contrast_slider.valueChanged.connect(self.on_manual_adjust_changed)
        self.highlights_slider.valueChanged.connect(self.on_manual_adjust_changed)
        self.shadows_slider.valueChanged.connect(self.on_manual_adjust_changed)
        self.vibrance_slider.valueChanged.connect(self.on_manual_adjust_changed)
        self.saturation_slider.valueChanged.connect(self.on_manual_adjust_changed)
        self.auto_check.toggled.connect(self.on_flags_changed)
        # 特色效果信号
        self.fx_lens_check.toggled.connect(self.on_special_fx_changed)
        self.fx_lens_slider.valueChanged.connect(self.on_special_fx_changed)
        self.fx_scratches_check.toggled.connect(self.on_special_fx_changed)
        self.fx_scratches_slider.valueChanged.connect(self.on_special_fx_changed)
        self.fx_defects_check.toggled.connect(self.on_special_fx_changed)
        self.fx_defects_slider.valueChanged.connect(self.on_special_fx_changed)
        self.fx_partial_check.toggled.connect(self.on_special_fx_changed)
        self.fx_partial_slider.valueChanged.connect(self.on_special_fx_changed)
        # 拖动即自动启用对应特效（不自动取消；0 为 no-op）
        self.fx_lens_slider.sliderPressed.connect(lambda: self._auto_check_fx(self.fx_lens_check))
        self.fx_scratches_slider.sliderPressed.connect(lambda: self._auto_check_fx(self.fx_scratches_check))
        self.fx_defects_slider.sliderPressed.connect(lambda: self._auto_check_fx(self.fx_defects_check))
        self.fx_partial_slider.sliderPressed.connect(lambda: self._auto_check_fx(self.fx_partial_check))
        # 分色器信号
        for s in self.hsl_sat_sliders + self.hsl_lum_sliders:
            s.valueChanged.connect(self.on_hsl_changed)
        self.backend_combo.currentTextChanged.connect(self.on_backend_changed)
        self.reset_btn.clicked.connect(self.on_reset)

        self.thread.progress_changed.connect(self.on_progress)
        self.thread.file_processed.connect(self.on_file_processed)
        self.thread.preview_ready.connect(self.on_preview_ready)


    def _apply_splitter_style(self) -> None:
        # Make the vertical (preview/params) handle easier to grab
        self.right_splitter.setStyleSheet(
            "QSplitter::handle:vertical {"
            "  height: 10px;"
            "  background: #c5c5c5;"
            "  margin: 1px 8px;"
            "  border-radius: 3px;"
            "}"
            "QSplitter::handle:vertical:hover { background: #9aa0a6; }"
            "QSplitter::handle:horizontal {"
            "  width: 6px;"
            "  background: #d0d0d0;"
            "}"
        )

    def _restore_splitters(self) -> None:
        try:
            settings = QtCore.QSettings("nikon-film-lab", "nikon-film-lab")
            rs = settings.value("right_splitter_state", None)
            if isinstance(rs, QtCore.QByteArray):
                self.right_splitter.restoreState(rs)
            elif isinstance(rs, bytes):
                self.right_splitter.restoreState(QtCore.QByteArray(rs))
            rts = settings.value("root_splitter_state", None)
            if isinstance(rts, QtCore.QByteArray):
                self.root_splitter.restoreState(rts)
            elif isinstance(rts, bytes):
                self.root_splitter.restoreState(QtCore.QByteArray(rts))
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            settings = QtCore.QSettings("nikon-film-lab", "nikon-film-lab")
            settings.setValue("right_splitter_state", self.right_splitter.saveState())
            settings.setValue("root_splitter_state", self.root_splitter.saveState())
            # 停止缩略图工作线程
            try:
                if hasattr(self, "_thumb_worker") and self._thumb_worker.isRunning():
                    self._thumb_worker.stop()
                    self._thumb_worker.wait(800)
            except Exception:
                pass
            # 停止处理线程，避免退出时 QThread 仍在运行导致闪退
            try:
                if hasattr(self, "thread") and isinstance(self.thread, QtCore.QThread) and self.thread.isRunning():
                    if hasattr(self.thread, "cancelled"):
                        self.thread.cancelled = True
                    try:
                        self.thread.quit()
                    except Exception:
                        pass
                    self.thread.wait(1200)
            except Exception:
                pass
        except Exception:
            pass
        super().closeEvent(event)

    def _settings(self) -> QtCore.QSettings:
        return QtCore.QSettings("nikon-film-lab", "nikon-film-lab")

    def _last_import_dir(self) -> str:
        try:
            d = self._settings().value("last_import_dir", "", type=str)
        except Exception:
            d = ""
        if isinstance(d, str) and d and os.path.isdir(d):
            return d
        return os.getcwd()

    def _remember_import_dir(self, path: str) -> None:
        if not path:
            return
        d = path if os.path.isdir(path) else os.path.dirname(path)
        if d and os.path.isdir(d):
            try:
                self._settings().setValue("last_import_dir", d)
            except Exception:
                pass

    def on_files_dropped(self, paths: List[str]) -> None:
        self._add_paths(paths)
        self._request_preview_update()

    def on_add_files(self) -> None:
        start = self._last_import_dir()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "选择文件", start, "Images (*.nef *.NEF *.jpg *.jpeg)"
        )
        if files:
            self._remember_import_dir(files[0])
        self._add_paths(files)
        self._request_preview_update()

    def on_add_dir(self) -> None:
        start = self._last_import_dir()
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹", start)
        if not d:
            return
        self._remember_import_dir(d)
        paths = []
        for root, _, files in os.walk(d):
            for f in files:
                fp = os.path.join(root, f)
                if is_supported(fp):
                    paths.append(fp)
        self._add_paths(paths)
        self._request_preview_update()

    def _make_placeholder_icon(self, side: int = 160) -> QtGui.QIcon:
        side = max(48, int(side))
        pix = QtGui.QPixmap(side, side)
        pix.fill(QtGui.QColor("#555555"))
        painter = QtGui.QPainter(pix)
        painter.setPen(QtGui.QPen(QtGui.QColor("#888888")))
        m = max(4, side // 16)
        painter.drawRect(m, m, side - 2 * m - 1, side - 2 * m - 1)
        painter.end()
        return QtGui.QIcon(pix)

    def _queue_content_width(self) -> int:
        vw = int(self.list_widget.viewport().width())
        # leave a little room for scrollbar / spacing so icon fits panel width
        return max(64, vw - 8)

    def _bgr_to_icon(self, bgr: np.ndarray, width: int) -> QtGui.QIcon:
        width = max(32, int(width))
        try:
            if not isinstance(bgr, np.ndarray) or bgr.size == 0:
                return self._make_placeholder_icon(width)
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            if bgr.ndim != 3 or bgr.shape[2] != 3:
                return self._make_placeholder_icon(width)
            if bgr.dtype != np.uint8:
                bgr = np.clip(bgr, 0, 255).astype(np.uint8)
            h, w = bgr.shape[:2]
            if w <= 0 or h <= 0:
                return self._make_placeholder_icon(width)
            scale = width / float(w)
            nh = max(1, int(round(h * scale)))
            resized = cv2.resize(bgr, (width, nh), interpolation=cv2.INTER_AREA)
            rgb = np.ascontiguousarray(resized[..., ::-1])
            qimg = QtGui.QImage(rgb.data, width, nh, 3 * width, QtGui.QImage.Format.Format_RGB888).copy()
            return QtGui.QIcon(QtGui.QPixmap.fromImage(qimg))
        except Exception:
            return self._make_placeholder_icon(width)

    def _sync_queue_thumb_size(self) -> None:
        w = self._queue_content_width()
        if abs(w - getattr(self, "_queue_thumb_w", 0)) < 2 and self.list_widget.iconSize().width() == w:
            return
        self._queue_thumb_w = w
        # Keep square icon box; image itself is width-fitted and may be shorter
        self.list_widget.setIconSize(QtCore.QSize(w, w))
        self.list_widget.setGridSize(QtCore.QSize(w + 8, w + 28))
        self._placeholder_icon = self._make_placeholder_icon(w)
        for path, item in list(self._path_to_item.items()):
            item.setSizeHint(QtCore.QSize(w + 8, w + 28))
            bgr = self._thumb_bgr.get(path)
            if bgr is not None:
                item.setIcon(self._bgr_to_icon(bgr, w))
            else:
                item.setIcon(self._placeholder_icon)

    def _add_paths(self, paths: List[str]) -> None:
        added = False
        w = self._queue_content_width()
        self._queue_thumb_w = w
        self.list_widget.setIconSize(QtCore.QSize(w, w))
        self.list_widget.setGridSize(QtCore.QSize(w + 8, w + 28))
        for p in paths:
            if not is_supported(p):
                continue
            if p in self._path_to_item:
                continue
            name = os.path.basename(p)
            item = QtWidgets.QListWidgetItem(self._placeholder_icon, name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            item.setSizeHint(QtCore.QSize(w + 8, w + 28))
            self.list_widget.addItem(item)
            self._path_to_item[p] = item
            if p in self._thumb_bgr:
                item.setIcon(self._bgr_to_icon(self._thumb_bgr[p], w))
            else:
                self._thumb_worker.enqueue(p)
            if not added:
                self._remember_import_dir(p)
                added = True
        # Select the last added item to show immediate preview
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def on_clear(self) -> None:
        self.list_widget.clear()
        self._path_to_item.clear()
        self._thumb_bgr.clear()
        try:
            self._thumb_worker.clear_pending()
        except Exception:
            pass
        self.preview_cache.clear()
        self.preview_label.clear()
        self._current_before_bgr = None
        self._current_after_bgr = None

    def on_pick_export(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出文件夹", self.export_dir())
        if d:
            self.export_edit.setText(d)

    def export_dir(self) -> str:
        d = self.export_edit.text().strip()
        return d or os.getcwd()

    def on_process(self) -> None:
        items = []
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            p = it.data(QtCore.Qt.ItemDataRole.UserRole) or it.toolTip() or it.text()
            items.append(str(p))
        if not items:
            QtWidgets.QMessageBox.information(self, "提示", "请先添加文件。")
            return
        q: List[QueueItem] = []
        for p in items:
            ext = os.path.splitext(p)[1].lower()
            if ext in (".jpg", ".jpeg"):
                r = load_jpeg_bgr8(p)
                q.append(QueueItem(path=p, is_raw=False, exif_bytes=r.exif_bytes, src_bgr8=r.image_bgr8))
            elif ext == ".nef":
                r2 = load_nef_to_bgr8(p)
                q.append(QueueItem(path=p, is_raw=True, exif_bytes=None, src_bgr8=r2.image_bgr8))
        self.thread.queue = q
        self.thread.export_dir = self.export_dir()
        self.thread.cancelled = False
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setStyleSheet(self._progress_style_busy)
        self.progress.setValue(0)
        self.progress_label.setText("0%")
        self.thread.start()

    def on_cancel(self) -> None:
        self.thread.cancelled = True
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def on_preset_changed(self, name: str) -> None:
        self.thread.options.preset_name = name
        # 自动应用预设推荐强度（不影响手动曝光等）
        rec = self.thread.processor.recommended_strength(name)
        self._animate_slider_to(self.strength_slider, int(rec))
        self.thread.options.strength_percent = int(rec)
        self._request_preview_update()

    def on_strength_changed(self, value: int) -> None:
        self.thread.options.strength_percent = value
        self._request_preview_update()
    
    def on_strength_recommended_clicked(self) -> None:
        name = self.preset_combo.currentText()
        rec = self.thread.processor.recommended_strength(name)
        self._animate_slider_to(self.strength_slider, int(rec))
    
    def _safe_stop_anim(self, anim: Optional[QtCore.QVariantAnimation]) -> None:
        if anim is None:
            return
        try:
            from shiboken6 import isValid
            if not isValid(anim):
                return
            anim.stop()
        except RuntimeError:
            pass

    def _animate_slider_to(self, slider: QtWidgets.QSlider, target: int, duration_ms: int = 180) -> None:
        # 仅用于程序触发的跳变；用户拖动不干预
        # KeepWhenStopped：避免 DeleteWhenStopped 后 Python 引用变成悬空 C++ 对象
        self._safe_stop_anim(self._slider_anims.get(slider))
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(int(slider.value()))
        anim.setEndValue(int(target))
        anim.setDuration(duration_ms)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        anim.valueChanged.connect(lambda v, s=slider: s.setValue(int(v)))
        self._slider_anims[slider] = anim
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.KeepWhenStopped)

    def on_flags_changed(self) -> None:
        self.thread.options.enable_grain = self.grain_check.isChecked()
        self.thread.options.enable_auto_baseline = self.auto_check.isChecked()
        self._request_preview_update()

    def on_vignette_mode_changed(self, _idx: int = 0) -> None:
        mode_text = self.vignette_mode.currentText()
        mapping = {"不处理": "off", "自动": "auto", "手动": "manual"}
        mode = mapping.get(mode_text, "off")
        self.thread.options.vignette_mode = mode
        self.vignette_amount.setEnabled(mode == "manual")
        if mode == "manual":
            self.thread.options.vignette_amount = self.vignette_amount.value()
        self._request_preview_update()

    def on_manual_adjust_changed(self, _value: int = 0) -> None:
        self.thread.options.vignette_amount = self.vignette_amount.value()
        self.thread.options.exposure_ev_x100 = self.exposure_slider.value()
        self.thread.options.temp_bias = self.temp_slider.value()
        self.thread.options.clarity = self.clarity_slider.value()
        self.thread.options.contrast = self.contrast_slider.value()
        self.thread.options.highlights = self.highlights_slider.value()
        self.thread.options.shadows = self.shadows_slider.value()
        self.thread.options.vibrance = self.vibrance_slider.value()
        self.thread.options.saturation = self.saturation_slider.value()
        self._request_preview_update()

    def on_backend_changed(self, text: str) -> None:
        mode = BackendMode(text)
        self.thread.set_backend(mode)
        self._request_preview_update()

    def on_selection_changed(self) -> None:
        # Show original immediately, then schedule processed preview
        if self.list_widget.count() == 0:
            return
        idx = self.list_widget.currentRow()
        if idx < 0:
            idx = 0
        it = self.list_widget.item(idx)
        path = it.data(QtCore.Qt.ItemDataRole.UserRole) or it.toolTip() or it.text()
        try:
            before_bgr = self._load_preview_source(path)
            self._current_before_bgr = before_bgr
            self._set_label_image_fit(self.preview_label, before_bgr)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "预览错误", f"加载失败：{os.path.basename(path)}\n{e}")
            self.preview_label.setText("加载失败")
        self._request_preview_update()

    # Recommended "best look" defaults per grain type (user can still tweak)
    GRAIN_PRESETS = {
        "Silver halide": {"size": 40, "density": 48, "rough": 58, "chroma": 8},
        "Modern fine": {"size": 18, "density": 32, "rough": 22, "chroma": 4},
        "Coarse push": {"size": 62, "density": 60, "rough": 72, "chroma": 18},
    }

    def _apply_grain_preset(self, grain_name: str | None = None) -> None:
        name = grain_name or self.grain_type.currentText()
        preset = self.GRAIN_PRESETS.get(name)
        if not preset:
            return
        for w, key in (
            (self.grain_size, "size"),
            (self.grain_density, "density"),
            (self.grain_rough, "rough"),
            (self.grain_chroma, "chroma"),
        ):
            w.blockSignals(True)
            w.setValue(int(preset[key]))
            w.blockSignals(False)
        self.thread.options.grain_type = GrainType(name)
        self.thread.options.grain_size = int(preset["size"])
        self.thread.options.grain_density = int(preset["density"])
        self.thread.options.grain_roughness = int(preset["rough"])
        self.thread.options.grain_chroma_mix = int(preset["chroma"])

    def on_grain_preset_clicked(self) -> None:
        self._apply_grain_preset()
        if not self.grain_check.isChecked():
            self.grain_check.setChecked(True)
        else:
            self._request_preview_update()

    def on_grain_enabled_changed(self, checked: bool) -> None:
        self.thread.options.enable_grain = bool(checked)
        if checked:
            self._apply_grain_preset()
        self._request_preview_update()

    def on_grain_type_changed(self, name: str) -> None:
        self.thread.options.grain_type = GrainType(name)
        self._apply_grain_preset(name)
        self._request_preview_update()

    def on_grain_params_changed(self) -> None:
        self.thread.options.grain_type = GrainType(self.grain_type.currentText())
        self.thread.options.grain_size = self.grain_size.value()
        self.thread.options.grain_density = self.grain_density.value()
        self.thread.options.grain_roughness = self.grain_rough.value()
        self.thread.options.grain_chroma_mix = self.grain_chroma.value()
        self._request_preview_update()

    def _request_preview_update(self) -> None:
        # debounce preview updates
        self._preview_seq += 1
        self.preview_timer.start()

    def _update_preview_first(self) -> None:
        # Update preview on current selection or first item
        if self.list_widget.count() == 0:
            return
        idx = self.list_widget.currentRow()
        if idx < 0:
            idx = 0
        it = self.list_widget.item(idx)
        path = it.data(QtCore.Qt.ItemDataRole.UserRole) or it.toolTip() or it.text()
        try:
            before_bgr = self._load_preview_source(path)
            after_bgr = self._process_preview(before_bgr)
            self._show_previews(before_bgr, after_bgr)
        except Exception:
            pass

    def _render_preview(self) -> None:
        local_seq = self._preview_seq
        if self.list_widget.count() == 0:
            return
        idx = self.list_widget.currentRow()
        if idx < 0:
            idx = 0
        it = self.list_widget.item(idx)
        path = it.data(QtCore.Qt.ItemDataRole.UserRole) or it.toolTip() or it.text()
        try:
            before_bgr = self._load_preview_source(path)
            if not self._should_process():
                self._current_preview_bgr = before_bgr
            else:
                after_bgr = self._process_preview(before_bgr)
                self._current_preview_bgr = after_bgr
            # If a newer request arrived, discard this result
            if local_seq != self._preview_seq:
                return
            self._refit_previews()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "预览错误", f"处理失败：{os.path.basename(path)}\n{e}")
            self.preview_label.setText("处理失败")

    def _downscale_max_side(self, bgr: np.ndarray, max_side: int) -> np.ndarray:
        h, w = bgr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale >= 1.0:
            return bgr
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _load_preview_source(self, path: str) -> np.ndarray:
        cached = self.preview_cache.get(path)
        if cached is not None:
            return cached
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            r = load_jpeg_bgr8(path)
            src = r.image_bgr8
        else:
            r2 = load_nef_to_bgr8(path)
            src = r2.image_bgr8
        preview = self._downscale_max_side(src, self.preview_max_side)
        self.preview_cache[path] = preview
        # Set deterministic grain seed per image path
        self._set_grain_seed_for_path(path)
        self._set_fx_seed_for_path(path)
        return preview
    
    def _on_thumb_ready(self, path: str, bgr8: object) -> None:
        # Keep BGR source; scale icon to current queue width on UI thread
        try:
            if not isinstance(bgr8, np.ndarray):
                return
            if bgr8.ndim == 2:
                bgr8 = cv2.cvtColor(bgr8, cv2.COLOR_GRAY2BGR)
            if bgr8.ndim != 3 or bgr8.shape[2] != 3 or bgr8.size == 0:
                return
            if bgr8.dtype != np.uint8:
                bgr8 = np.clip(bgr8, 0, 255).astype(np.uint8)
            self._thumb_bgr[path] = bgr8
            it = self._path_to_item.get(path)
            if it is not None:
                it.setIcon(self._bgr_to_icon(bgr8, self._queue_content_width()))
        except Exception:
            pass

    def _process_preview(self, bgr8: np.ndarray) -> np.ndarray:
        img = bgr8.astype(np.float32) / 255.0
        out = self.thread.processor.process_bgr01(img, self.thread.options)
        out8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return out8

    def _should_process(self) -> bool:
        # If preset is '不处理' and strength==0 and all toggles off, show original
        no_preset = (self.thread.options.preset_name == "不处理") or (self.preset_combo.currentText() == "不处理")
        no_strength = self.thread.options.strength_percent <= 0
        no_vignette = (getattr(self.thread.options, "vignette_mode", "off") == "off")
        no_exposure = (getattr(self.thread.options, "exposure_ev_x100", 0) == 0)
        no_temp = (getattr(self.thread.options, "temp_bias", 0) == 0)
        no_clarity = (getattr(self.thread.options, "clarity", 0) == 0)
        no_contrast = (getattr(self.thread.options, "contrast", 0) == 0)
        no_special = (not getattr(self.thread.options, "enable_lens_aging", False)) and \
                     (not getattr(self.thread.options, "enable_scratches", False)) and \
                     (not getattr(self.thread.options, "enable_expired_film", getattr(self.thread.options, "enable_film_defects", False))) and \
                     (not getattr(self.thread.options, "enable_light_leak", getattr(self.thread.options, "enable_partial_exposure", False)))
        no_hi = (getattr(self.thread.options, "highlights", 0) == 0)
        no_sh = (getattr(self.thread.options, "shadows", 0) == 0)
        no_vib = (getattr(self.thread.options, "vibrance", 0) == 0)
        no_sat = (getattr(self.thread.options, "saturation", 0) == 0)
        hsl_sat = getattr(self.thread.options, "hsl_sat8", None) or [0] * 8
        hsl_lum = getattr(self.thread.options, "hsl_lum8", None) or [0] * 8
        no_splitter = all(v == 0 for v in hsl_sat) and all(v == 0 for v in hsl_lum)
        no_fx = (not self.thread.options.enable_grain) and no_vignette and (not self.thread.options.enable_auto_baseline) and no_exposure and no_temp and no_clarity and no_contrast and no_hi and no_sh and no_vib and no_sat and no_special and no_splitter
        return not (no_preset and no_strength and no_fx)

    def _show_preview(self, bgr: np.ndarray) -> None:
        self._current_preview_bgr = bgr
        self._refit_previews()

    def _refit_previews(self) -> None:
        if getattr(self, "_current_preview_bgr", None) is not None:
            self._set_label_image_fit(self.preview_label, self._current_preview_bgr)

    def _set_label_image_fit(self, label: QtWidgets.QLabel, bgr: np.ndarray) -> None:
        try:
            if not isinstance(bgr, np.ndarray) or bgr.ndim < 2 or bgr.size == 0:
                label.clear()
                return
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            if bgr.shape[2] != 3:
                label.clear()
                return
            if bgr.dtype != np.uint8:
                bgr = np.clip(bgr, 0, 255).astype(np.uint8)
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            h2, w2 = rgb.shape[:2]
            if h2 <= 0 or w2 <= 0:
                label.clear()
                return
            bytes_per_line = 3 * w2
            qimg = QtGui.QImage(rgb.data, w2, h2, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()
            pix = QtGui.QPixmap.fromImage(qimg)
            scaled = pix.scaled(label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
            label.setPixmap(scaled)
        except Exception:
            try:
                label.clear()
            except Exception:
                pass

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._sync_queue_thumb_size()
        self._refit_previews()

    def _set_grain_seed_for_path(self, path: str) -> None:
        self.thread.options.grain_seed = zlib.adler32(path.encode("utf-8")) & 0xFFFFFFFF
    def _set_fx_seed_for_path(self, path: str) -> None:
        # Use a slightly different namespace to decorrelate from grain
        self.thread.options.fx_seed = zlib.adler32((path + "#fx").encode("utf-8")) & 0xFFFFFFFF

    def on_progress(self, cur: int, total: int) -> None:
        total = max(total, 1)
        self.progress.setMaximum(total)
        # 非线性动画到目标进度
        start_val = self.progress.value()
        end_val = cur
        self._safe_stop_anim(self._progress_anim)
        self._progress_anim = QtCore.QVariantAnimation(self)
        self._progress_anim.setStartValue(start_val)
        self._progress_anim.setEndValue(end_val)
        delta = abs(end_val - start_val)
        dur = int(min(450, max(120, 12 * delta)))
        self._progress_anim.setDuration(dur)
        self._progress_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._progress_anim.valueChanged.connect(lambda v: self.progress.setValue(int(v)))
        self._progress_anim.start(QtCore.QAbstractAnimation.DeletionPolicy.KeepWhenStopped)
        pct = int(round(100.0 * cur / total)) if total else 0
        if cur >= total:
            self.progress.setStyleSheet(self._progress_style_done)
            self.progress_label.setText(f"{pct}%  ✓")
            self.process_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
        else:
            self.progress.setStyleSheet(self._progress_style_busy)
            self.progress_label.setText(f"{pct}%")


    def on_file_processed(self, path: str, success: bool, message: str) -> None:
        if not success:
            QtWidgets.QMessageBox.warning(self, "错误", f"{os.path.basename(path)} 处理失败：{message}")

    def on_preview_ready(self, before: np.ndarray, after: np.ndarray) -> None:
        # For batch thread preview: show processed result if applicable, else original
        self._current_preview_bgr = after if self._should_process() else before
        self._refit_previews()

    # Reset controls to show original image
    def on_reset(self) -> None:
        self.strength_slider.setValue(0)
        self.grain_check.setChecked(False)
        self.vignette_mode.setCurrentIndex(0)
        self.vignette_amount.setValue(0)
        self.vignette_amount.setEnabled(False)
        # 特色效果复位
        self.fx_lens_check.setChecked(False)
        self.fx_lens_slider.setValue(0)
        self.fx_scratches_check.setChecked(False)
        self.fx_scratches_slider.setValue(0)
        self.fx_defects_check.setChecked(False)
        self.fx_defects_slider.setValue(0)
        self.fx_partial_check.setChecked(False)
        self.fx_partial_slider.setValue(0)
        self.exposure_slider.setValue(0)
        self.temp_slider.setValue(0)
        self.clarity_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.highlights_slider.setValue(0)
        self.shadows_slider.setValue(0)
        self.vibrance_slider.setValue(0)
        self.saturation_slider.setValue(0)
        self.auto_check.setChecked(False)
        self.grain_size.setValue(0)
        self.grain_density.setValue(0)
        self.grain_rough.setValue(0)
        self.grain_chroma.setValue(0)
        # 分色器复位
        for s in self.hsl_sat_sliders:
            s.setValue(0)
        for l in self.hsl_lum_sliders:
            l.setValue(0)
        idx_id = self.preset_combo.findText("不处理")
        if idx_id >= 0:
            self.preset_combo.setCurrentIndex(idx_id)
        self.thread.options.strength_percent = 0
        self.thread.options.enable_grain = False
        self.thread.options.vignette_mode = "off"
        self.thread.options.vignette_amount = 0
        self.thread.options.enable_auto_baseline = False
        self.thread.options.grain_size = 0
        self.thread.options.grain_density = 0
        self.thread.options.grain_roughness = 0
        self.thread.options.grain_chroma_mix = 0
        self.thread.options.exposure_ev_x100 = 0
        self.thread.options.temp_bias = 0
        self.thread.options.clarity = 0
        self.thread.options.contrast = 0
        self.thread.options.highlights = 0
        self.thread.options.shadows = 0
        self.thread.options.vibrance = 0
        self.thread.options.saturation = 0
        # Color splitter options
        self.thread.options.enable_color_splitter = True
        self.thread.options.hsl_sat8 = [0] * 8
        self.thread.options.hsl_lum8 = [0] * 8
        # FX options
        self.thread.options.enable_lens_aging = False
        self.thread.options.lens_aging = 0
        self.thread.options.enable_scratches = False
        self.thread.options.scratches = 0
        # 新命名字段
        self.thread.options.enable_expired_film = False
        self.thread.options.expired_film = 0
        self.thread.options.enable_light_leak = False
        self.thread.options.light_leak = 0
        self.thread.options.preset_name = "不处理"
        self._request_preview_update()

    def on_special_fx_changed(self) -> None:
        # 更新独立特效参数并触发预览
        self.thread.options.enable_lens_aging = self.fx_lens_check.isChecked()
        self.thread.options.lens_aging = self.fx_lens_slider.value()
        self.thread.options.enable_scratches = self.fx_scratches_check.isChecked()
        self.thread.options.scratches = self.fx_scratches_slider.value()
        self.thread.options.enable_expired_film = self.fx_defects_check.isChecked()
        self.thread.options.expired_film = self.fx_defects_slider.value()
        self.thread.options.enable_light_leak = self.fx_partial_check.isChecked()
        self.thread.options.light_leak = self.fx_partial_slider.value()
        self._request_preview_update()

    def _auto_check_fx(self, checkbox: QtWidgets.QCheckBox) -> None:
        if not checkbox.isChecked():
            checkbox.setChecked(True)
            # on_special_fx_changed 会在 valueChanged 触发时同步参数；此处先行 request 以获得更及时的反馈
            self._request_preview_update()

    def on_hsl_changed(self, _value: int = 0) -> None:
        # Collect current HSL per-band values and push to options
        sat = [int(s.value()) for s in self.hsl_sat_sliders]
        lum = [int(l.value()) for l in self.hsl_lum_sliders]
        self.thread.options.hsl_sat8 = sat
        self.thread.options.hsl_lum8 = lum
        self.thread.options.enable_color_splitter = True
        self._request_preview_update()
