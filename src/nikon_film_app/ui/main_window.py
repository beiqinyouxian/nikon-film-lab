from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from nikon_film_app.processing.accelerator import Accelerator, BackendMode
from nikon_film_app.processing.pipeline import ImageProcessor, ProcessOptions
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
        self.options = ProcessOptions(preset_name="Kodak Portra 400")
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
                out = self._process_bgr(item.src_bgr8)
                # Save
                base = os.path.basename(item.path)
                name, _ = os.path.splitext(base)
                out_path = os.path.join(self.export_dir, f\"{name}_film.jpg\")
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

        # UI
        self.before_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.after_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.before_label.setMinimumSize(200, 200)
        self.after_label.setMinimumSize(200, 200)

        self.list_widget = DropListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        self.add_btn = QtWidgets.QPushButton("添加文件")
        self.add_dir_btn = QtWidgets.QPushButton("添加文件夹")
        self.clear_btn = QtWidgets.QPushButton("清空")
        self.process_btn = QtWidgets.QPushButton("开始处理")
        self.cancel_btn = QtWidgets.QPushButton("取消")
        self.cancel_btn.setEnabled(False)

        self.export_edit = QtWidgets.QLineEdit(os.getcwd())
        self.export_btn = QtWidgets.QPushButton("选择导出文件夹")

        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(self.thread.processor.list_presets())
        self.strength_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setValue(80)
        self.grain_check = QtWidgets.QCheckBox("颗粒")
        self.grain_check.setChecked(True)
        self.vignette_check = QtWidgets.QCheckBox("暗角")
        self.auto_check = QtWidgets.QCheckBox("自动基线（曝光/色温）")
        self.auto_check.setChecked(True)

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

        right = QtWidgets.QVBoxLayout()
        imgs = QtWidgets.QHBoxLayout()
        imgs.addWidget(self.before_label, 1)
        imgs.addWidget(self.after_label, 1)
        right.addLayout(imgs, 1)
        form = QtWidgets.QFormLayout()
        form.addRow("预设：", self.preset_combo)
        form.addRow("强度：", self.strength_slider)
        form.addRow("", self.grain_check)
        form.addRow("", self.vignette_check)
        form.addRow("", self.auto_check)
        form.addRow("后端：", self.backend_combo)
        right.addLayout(form)
        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.process_btn)
        btns.addWidget(self.cancel_btn)
        right.addLayout(btns)
        right.addWidget(QtWidgets.QLabel("提示：导出始终为原始全分辨率，预览仅为缩放显示。"))

        root = QtWidgets.QSplitter()
        lw = QtWidgets.QWidget()
        rw = QtWidgets.QWidget()
        lw.setLayout(left)
        rw.setLayout(right)
        root.addWidget(lw)
        root.addWidget(rw)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        self.setCentralWidget(root)

        # Signals
        self.add_btn.clicked.connect(self.on_add_files)
        self.add_dir_btn.clicked.connect(self.on_add_dir)
        self.clear_btn.clicked.connect(self.on_clear)
        self.export_btn.clicked.connect(self.on_pick_export)
        self.list_widget.files_dropped.connect(self.on_files_dropped)
        self.process_btn.clicked.connect(self.on_process)
        self.cancel_btn.clicked.connect(self.on_cancel)
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        self.strength_slider.valueChanged.connect(self.on_strength_changed)
        self.grain_check.toggled.connect(self.on_flags_changed)
        self.vignette_check.toggled.connect(self.on_flags_changed)
        self.auto_check.toggled.connect(self.on_flags_changed)
        self.backend_combo.currentTextChanged.connect(self.on_backend_changed)

        self.thread.progress_changed.connect(self.on_progress)
        self.thread.file_processed.connect(self.on_file_processed)
        self.thread.preview_ready.connect(self.on_preview_ready)

    def on_add_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "选择文件", "", "Images (*.nef *.NEF *.jpg *.jpeg)")
        self._add_paths(files)

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

    def _add_paths(self, paths: List[str]) -> None:
        for p in paths:
            if not is_supported(p):
                continue
            item = QtWidgets.QListWidgetItem(p)
            self.list_widget.addItem(item)

    def on_clear(self) -> None:
        self.list_widget.clear()
        self.before_label.clear()
        self.after_label.clear()

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
        self._update_preview_first()

    def on_strength_changed(self, value: int) -> None:
        self.thread.options.strength_percent = value
        self._update_preview_first()

    def on_flags_changed(self) -> None:
        self.thread.options.enable_grain = self.grain_check.isChecked()
        self.thread.options.enable_vignette = self.vignette_check.isChecked()
        self.thread.options.enable_auto_baseline = self.auto_check.isChecked()
        self._update_preview_first()

    def on_backend_changed(self, text: str) -> None:
        mode = BackendMode(text)
        self.thread.set_backend(mode)

    def _update_preview_first(self) -> None:
        # Update preview on current selection or first item
        if self.list_widget.count() == 0:
            return
        idx = self.list_widget.currentRow()
        if idx < 0:
            idx = 0
        path = self.list_widget.item(idx).text()
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                r = load_jpeg_bgr8(path)
                src = r.image_bgr8
            else:
                r2 = load_nef_to_bgr8(path)
                src = r2.image_bgr8
            img = src.astype(np.float32) / 255.0
            out = self.thread.processor.process_bgr01(img, self.thread.options)
            out8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            self.before_label.setPixmap(bgr_to_qpixmap(src))
            self.after_label.setPixmap(bgr_to_qpixmap(out8))
        except Exception:
            pass

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
        self.before_label.setPixmap(bgr_to_qpixmap(before))
        self.after_label.setPixmap(bgr_to_qpixmap(after))

