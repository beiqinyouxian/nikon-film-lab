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
        self.fx_defects_check = QtWidgets.QCheckBox("胶片缺陷")
        self.fx_defects_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fx_defects_slider.setRange(0, 100)
        self.fx_defects_slider.setValue(0)
        self.fx_partial_check = QtWidgets.QCheckBox("部分曝光")
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
        bgrid.addWidget(_pair_row("强度", self.strength_slider), 0, 1)
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
        # 行说明文本
        leg_sat = QtWidgets.QLabel("饱和")
        leg_sat.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        leg_lum = QtWidgets.QLabel("明度")
        leg_lum.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        split_grid.addWidget(leg_sat, 1, 7)
        split_grid.addWidget(leg_lum, 2, 7)

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
        btns.addWidget(self.process_btn)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.reset_btn)
        btns.addStretch(1)
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
        except Exception:
            pass
        super().closeEvent(event)

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
        self._request_preview_update()

    def on_strength_changed(self, value: int) -> None:
        self.thread.options.strength_percent = value
        self._request_preview_update()

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
        path = self.list_widget.item(idx).text()
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
        self._set_fx_seed_for_path(path)
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
        no_contrast = (getattr(self.thread.options, "contrast", 0) == 0)
        no_special = (not getattr(self.thread.options, "enable_lens_aging", False)) and \
                     (not getattr(self.thread.options, "enable_scratches", False)) and \
                     (not getattr(self.thread.options, "enable_film_defects", False)) and \
                     (not getattr(self.thread.options, "enable_partial_exposure", False))
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
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        h2, w2 = rgb.shape[:2]
        bytes_per_line = 3 * w2
        qimg = QtGui.QImage(rgb.data, w2, h2, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refit_previews()

    def _set_grain_seed_for_path(self, path: str) -> None:
        self.thread.options.grain_seed = zlib.adler32(path.encode("utf-8")) & 0xFFFFFFFF
    def _set_fx_seed_for_path(self, path: str) -> None:
        # Use a slightly different namespace to decorrelate from grain
        self.thread.options.fx_seed = zlib.adler32((path + "#fx").encode("utf-8")) & 0xFFFFFFFF

    def on_progress(self, cur: int, total: int) -> None:
        total = max(total, 1)
        self.progress.setMaximum(total)
        self.progress.setValue(cur)
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
        self.thread.options.enable_film_defects = False
        self.thread.options.film_defects = 0
        self.thread.options.enable_partial_exposure = False
        self.thread.options.partial_exposure = 0
        self.thread.options.preset_name = "不处理"
        self._request_preview_update()

    def on_special_fx_changed(self) -> None:
        # 更新独立特效参数并触发预览
        self.thread.options.enable_lens_aging = self.fx_lens_check.isChecked()
        self.thread.options.lens_aging = self.fx_lens_slider.value()
        self.thread.options.enable_scratches = self.fx_scratches_check.isChecked()
        self.thread.options.scratches = self.fx_scratches_slider.value()
        self.thread.options.enable_film_defects = self.fx_defects_check.isChecked()
        self.thread.options.film_defects = self.fx_defects_slider.value()
        self.thread.options.enable_partial_exposure = self.fx_partial_check.isChecked()
        self.thread.options.partial_exposure = self.fx_partial_slider.value()
        self._request_preview_update()

    def on_hsl_changed(self, _value: int = 0) -> None:
        # Collect current HSL per-band values and push to options
        sat = [int(s.value()) for s in self.hsl_sat_sliders]
        lum = [int(l.value()) for l in self.hsl_lum_sliders]
        self.thread.options.hsl_sat8 = sat
        self.thread.options.hsl_lum8 = lum
        self.thread.options.enable_color_splitter = True
        self._request_preview_update()
