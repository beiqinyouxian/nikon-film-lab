from __future__ import annotations

import os
from dataclasses import dataclass
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
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    display = bgr
    if scale < 1.0:
        display = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rgb = display[..., ::-1].copy()
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

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
        self.grain_check = QtWidgets.QCheckBox("颗粒")
        self.grain_check.setChecked(False)
        self.grain_type = QtWidgets.QComboBox()
        self.grain_type.addItems([GrainType.SILVER_HALIDE.value, GrainType.MODERN_FINE.value, GrainType.COARSE_PUSH.value])
        self.grain_size = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.grain_size.setRange(0, 100)
        self.grain_size.setValue(30)
        self.grain_density = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.grain_density.setRange(0, 100)
        self.grain_density.setValue(40)
        self.grain_rough = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.grain_rough.setRange(0, 100)
        self.grain_rough.setValue(40)
        self.grain_chroma = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.grain_chroma.setRange(0, 100)
        self.grain_chroma.setValue(0)
        # Auto baseline + new vignette/exposure/temp controls
        self.auto_check = QtWidgets.QCheckBox("自动基线（曝光/色温）")
        self.auto_check.setChecked(False)
        self.vignette_mode = QtWidgets.QComboBox()
        self.vignette_mode.addItems(["不处理", "自动", "手动"])
        self.vignette_amount = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.vignette_amount.setRange(0, 100)
        self.vignette_amount.setValue(0)
        self.exposure_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(-200, 200)  # -2..+2 EV
        self.exposure_slider.setValue(0)
        self.temp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.temp_slider.setRange(-100, 100)  # cooler..warmer
        self.temp_slider.setValue(0)

        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems([BackendMode.AUTO.value, BackendMode.CPU.value, BackendMode.OPENCL.value])

        self.progress = QtWidgets.QProgressBar()

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
        left.addWidget(self.progress)

        # Build bottom controls widget
        form = QtWidgets.QFormLayout()
        form.addRow("预设：", self.preset_combo)
        form.addRow("强度：", self.strength_slider)
        form.addRow("", self.grain_check)
        grain_box = QtWidgets.QGroupBox("颗粒参数")
        grain_form = QtWidgets.QFormLayout(grain_box)
        grain_form.addRow("类型：", self.grain_type)
        grain_form.addRow("大小：", self.grain_size)
        grain_form.addRow("密度：", self.grain_density)
        grain_form.addRow("粗糙：", self.grain_rough)
        grain_form.addRow("彩色混合：", self.grain_chroma)
        vig_box = QtWidgets.QGroupBox("暗角")
        vig_form = QtWidgets.QFormLayout(vig_box)
        vig_form.addRow("模式：", self.vignette_mode)
        vig_form.addRow("强度：", self.vignette_amount)
        exp_box = QtWidgets.QGroupBox("曝光 / 色温")
        exp_form = QtWidgets.QFormLayout(exp_box)
        exp_form.addRow("曝光(EV)：", self.exposure_slider)
        exp_form.addRow("色温：", self.temp_slider)
        # Clarity control
        clarity_box = QtWidgets.QGroupBox("清晰度（Clarity）")
        self.clarity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.clarity_slider.setRange(-50, 100)
        self.clarity_slider.setValue(0)
        clarity_layout = QtWidgets.QVBoxLayout(clarity_box)
        clarity_layout.addWidget(self.clarity_slider)
        bottom_controls = QtWidgets.QWidget()
        bc_layout = QtWidgets.QVBoxLayout(bottom_controls)
        bc_layout.setContentsMargins(0, 0, 0, 0)
        bc_layout.addLayout(form)
        bc_layout.addWidget(grain_box)
        bc_layout.addWidget(vig_box)
        bc_layout.addWidget(exp_box)
        bc_layout.addWidget(clarity_box)
        bc_layout.addWidget(self.auto_check)
        backend_row = QtWidgets.QHBoxLayout()
        backend_row.addWidget(QtWidgets.QLabel("后端："))
        backend_row.addWidget(self.backend_combo, 1)
        bc_layout.addLayout(backend_row)
        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.process_btn)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.reset_btn)
        bc_layout.addLayout(btns)
        bc_layout.addWidget(QtWidgets.QLabel("提示：预览为单图等比适配；导出始终为原始全分辨率。"))
        bottom_controls.setMinimumHeight(200)

        # Right vertical splitter
        self.right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.right_splitter.setObjectName("right_splitter")
        preview_wrap = QtWidgets.QWidget()
        pw_layout = QtWidgets.QVBoxLayout(preview_wrap)
        pw_layout.setContentsMargins(0, 0, 0, 0)
        pw_layout.addWidget(self.preview_label, 1)
        preview_wrap.setMinimumHeight(240)
        self.right_splitter.addWidget(preview_wrap)
        self.right_splitter.addWidget(bottom_controls)
        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 0)
        self.right_splitter.splitterMoved.connect(lambda *_: self._refit_previews())

        root = QtWidgets.QSplitter()
        lw = QtWidgets.QWidget()
        rw = QtWidgets.QWidget()
        lw.setLayout(left)
        rw_layout = QtWidgets.QVBoxLayout(rw)
        rw_layout.setContentsMargins(0, 0, 0, 0)
        rw_layout.addWidget(self.right_splitter, 1)
        root.addWidget(lw)
        root.addWidget(rw)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setObjectName("root_splitter")
        self.root_splitter = root
        # Minimum sizes
        lw.setMinimumWidth(220)
        preview_wrap.setMinimumHeight(240)
        self.setCentralWidget(root)
        # Restore saved splitter states if any
        self._restore_splitters()
        QtCore.QTimer.singleShot(0, self._refit_previews)

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
        self.grain_check.toggled.connect(self.on_flags_changed)
        self.grain_type.currentTextChanged.connect(self.on_grain_params_changed)
        self.grain_size.valueChanged.connect(self.on_grain_params_changed)
        self.grain_density.valueChanged.connect(self.on_grain_params_changed)
        self.grain_rough.valueChanged.connect(self.on_grain_params_changed)
        self.grain_chroma.valueChanged.connect(self.on_grain_params_changed)
        self.vignette_mode.currentTextChanged.connect(self.on_vignette_changed)
        self.vignette_amount.valueChanged.connect(self.on_vignette_changed)
        self.exposure_slider.valueChanged.connect(self.on_exposure_temp_changed)
        self.temp_slider.valueChanged.connect(self.on_exposure_temp_changed)
        self.clarity_slider.valueChanged.connect(self.on_clarity_changed)
        self.auto_check.toggled.connect(self.on_flags_changed)
        self.backend_combo.currentTextChanged.connect(self.on_backend_changed)
        self.reset_btn.clicked.connect(self.on_reset)

        self.thread.progress_changed.connect(self.on_progress)
        self.thread.file_processed.connect(self.on_file_processed)
        self.thread.preview_ready.connect(self.on_preview_ready)

    def on_files_dropped(self, paths: List[str]) -> None:
        self._add_paths(paths)
        self._request_preview_update()

    def on_add_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "选择文件", "", "Images (*.nef *.NEF *.jpg *.jpeg)")
        self._add_paths(files)
        self._request_preview_update()

    def on_add_dir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹", "")
        if not d:
            return
        paths = []
        for root, _, files in os.walk(d):
            for f in files:
                fp = os.path.join(root, f)
                if is_supported(fp):
                    paths.append(fp)
        self._add_paths(paths)
        self._request_preview_update()

    def _add_paths(self, paths: List[str]) -> None:
        for p in paths:
            if not is_supported(p):
                continue
            item = QtWidgets.QListWidgetItem(p)
            self.list_widget.addItem(item)
        # Select the last added item to show immediate preview
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(self.list_widget.count() - 1)

    def on_clear(self) -> None:
        self.list_widget.clear()
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
        items = [self.list_widget.item(i).text() for i in range(self.list_widget.count())]
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
        self.thread.start()

    def on_cancel(self) -> None:
        self.thread.cancelled = True
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def on_preset_changed(self, name: str) -> None:
        self.thread.options.preset_name = name
        self._request_preview_update()

    def on_strength_changed(self, value: int) -> None:
        self.thread.options.strength_percent = value
        self._request_preview_update()

    def on_flags_changed(self) -> None:
        self.thread.options.enable_grain = self.grain_check.isChecked()
        self.thread.options.enable_auto_baseline = self.auto_check.isChecked()
        self._request_preview_update()

    def on_vignette_changed(self) -> None:
        mode_text = self.vignette_mode.currentText()
        if mode_text == "不处理":
            self.thread.options.vignette_mode = "off"
        elif mode_text == "自动":
            self.thread.options.vignette_mode = "auto"
        else:
            self.thread.options.vignette_mode = "manual"
        self.thread.options.vignette_amount = self.vignette_amount.value()
        self.vignette_amount.setEnabled(self.thread.options.vignette_mode == "manual")
        self._request_preview_update()

    def on_exposure_temp_changed(self) -> None:
        self.thread.options.exposure_ev_x100 = self.exposure_slider.value()
        self.thread.options.temp_bias = self.temp_slider.value()
        self._request_preview_update()

    def on_clarity_changed(self) -> None:
        self.thread.options.clarity = self.clarity_slider.value()
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
        path = self.list_widget.item(idx).text()
        try:
            before_bgr = self._load_preview_source(path)
            self._current_before_bgr = before_bgr
            self._set_label_image_fit(self.preview_label, before_bgr)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "预览错误", f"加载失败：{os.path.basename(path)}\n{e}")
            self.preview_label.setText("加载失败")
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
        path = self.list_widget.item(idx).text()
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
        path = self.list_widget.item(idx).text()
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
        return preview

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
        no_fx = (not self.thread.options.enable_grain) and no_vignette and (not self.thread.options.enable_auto_baseline) and no_exposure and no_temp and no_clarity
        return not (no_preset and no_strength and no_fx)

    def _show_preview(self, bgr: np.ndarray) -> None:
        self._current_preview_bgr = bgr
        self._refit_previews()

    def _refit_previews(self) -> None:
        if getattr(self, "_current_preview_bgr", None) is not None:
            self._set_label_image_fit(self.preview_label, self._current_preview_bgr)

    def _set_label_image_fit(self, label: QtWidgets.QLabel, bgr: np.ndarray) -> None:
        rgb = bgr[..., ::-1].copy()
        h2, w2 = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w2, h2, 3 * w2, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refit_previews()

    def _set_grain_seed_for_path(self, path: str) -> None:
        self.thread.options.grain_seed = zlib.adler32(path.encode("utf-8")) & 0xFFFFFFFF

    def on_progress(self, cur: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(cur)
        if cur == total:
            self.process_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

    def on_file_processed(self, path: str, success: bool, message: str) -> None:
        if not success:
            QtWidgets.QMessageBox.warning(self, "错误", f"{os.path.basename(path)} 处理失败：{message}")

    def on_preview_ready(self, before: np.ndarray, after: np.ndarray) -> None:
        # For batch thread preview: show processed result if applicable, else original
        self._current_preview_bgr = after if self._should_process() else before
        self._refit_previews()

    # Reset controls to show original image
    def on_reset(self) -> None:
        # Reset UI controls to neutral
        self.strength_slider.setValue(0)
        self.grain_check.setChecked(False)
        self.vignette_mode.setCurrentIndex(0)  # 不处理
        self.vignette_amount.setValue(0)
        self.auto_check.setChecked(False)
        self.grain_size.setValue(0)
        self.grain_density.setValue(0)
        self.grain_rough.setValue(0)
        self.grain_chroma.setValue(0)
        self.exposure_slider.setValue(0)
        self.temp_slider.setValue(0)
        self.clarity_slider.setValue(0)
        # preset back to 不处理
        idx_id = self.preset_combo.findText("不处理")
        if idx_id >= 0:
            self.preset_combo.setCurrentIndex(idx_id)
        # Update options directly
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
        self.thread.options.preset_name = "不处理"
        self._request_preview_update()

