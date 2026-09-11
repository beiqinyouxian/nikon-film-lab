# nikon-film-lab
Desktop app: Nikon RAW/JPG film-look stylization with CPU/OpenCL acceleration

### 中文使用说明

**功能概述**
- **打开/批量处理**: Nikon NEF（RAW）与 JPG/JPEG
- **多款胶片风格**: 至少 8 种预设，强度可调；可选颗粒、暗角
- **可选自动基线**: 轻微曝光/白平衡矫正
- **高质量导出**: JPG 质量 95，尽可能保留 EXIF（JPG→JPG 时可保留）
- **分辨率不变**: 导出始终为原始解码后的全分辨率（不改变像素尺寸）
- **加速后端**: CPU 默认；如可用，尝试 OpenCL（iGPU）。界面可切换 Auto / CPU / OpenCL

**安装与运行**
1) 准备 Python 3.11+
2) 创建虚拟环境并安装依赖
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
   或使用可编辑安装：
   ```bash
   pip install -e .
   ```
3) 启动图形界面
   ```bash
   python -m nikon_film_app
   ```
   也可使用命令：
   ```bash
   nikon-film-app
   ```

**基本用法**
- 拖拽 .nef/.jpg 到左侧队列，或用“添加文件/文件夹”
- 右侧选择“预设”与“强度”，可勾选“颗粒/暗角/自动基线”
- 选择“后端”：Auto / CPU / OpenCL（若系统不支持 OpenCL，会自动回退至 CPU）
- 选择“导出文件夹”，点击“开始处理”；可批量处理并显示进度；支持“取消”
- 导出文件名规则：原名 + “_film.jpg”
- 非破坏式：不会覆盖原文件（除非你手动选择同名覆盖）

**NEF（RAW）加载说明**
- 使用 rawpy（LibRaw）解码 NEF 为全分辨率 RGB，再进行风格化处理与导出
- 如需测试，可从 `https://raw.pixls.us/` 搜索并下载公开 NEF 样张
- 注意：NEF → JPG 时完整 EXIF 复制较复杂，当前版本优先保留 JPG→JPG 的 EXIF。RAW→JPG 的 EXIF 未来版本将逐步增强（当前可能不完整或不保留）。

**胶片风格（示例）**
- Kodak Portra 400
- Kodak Gold 200
- Fuji Velvia 50
- Fuji Pro 400H
- Ilford HP5（黑白）
- Cinestill 800T
- Agfa Vista
- Kodak Tri-X（黑白）
- Leica Color Modern（类徕卡现代色：微暖、中等对比、克制饱和）
- Leica Classic Mono（类徕卡 Monochrom：深邃黑白、银盐质感）
- Leica Chrome Vivid（类徕卡街头浓郁色）

**测试**
```bash
pytest -q
```
覆盖内容：
- 处理前后分辨率保持不变
- 至少一个预设的输出不是恒等映射
- OpenCL 不可用时后端选择不崩溃（自动回退）

### English notes (install)
1) Python 3.11+
2) Create venv and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```
   or:
   ```bash
   pip install -e .
   ```
3) Launch GUI:
   ```bash
   python -m nikon_film_app
   ```

**Acceleration**
- Auto tries OpenCL via OpenCV (UMat). If unavailable, falls back to CPU.
- Preview uses downscaled display only; export keeps source full resolution.

**预览性能与显示**
- 左侧选择任意队列条目，会立即在右侧显示该照片的全幅（按窗口自适应比例缩放，非裁剪）
- 预览采用降采样副本（最长边约 1600px）进行实时渲染，提升交互流畅度
- 滑动强度/切换预设/勾选选项时，使用约 200ms 防抖并丢弃过期请求，避免卡顿
- 导出始终基于原始全分辨率图像（预览仅用于加速显示）
- 提供“复位”按钮：一键恢复为原片预览（等同于设置 强度=0、关闭颗粒/暗角/自动基线）；导出也将尊重当前参数

**颗粒（Grain）控制**
- 类型：银盐（单色、结团感）、Modern fine（现代细腻）、Coarse push（粗颗粒，似推片）
- 大小：控制颗粒的空间尺度（分辨率自适应；预览与导出一致感）
- 密度：控制颗粒强度/覆盖度
- 粗糙：控制结团/粗糙程度
- 彩色混合：控制彩色颗粒参与度（0 为纯亮度颗粒）

### Windows 可执行文件（.exe）

**从 GitHub Actions 下载**
- 打开仓库的 “Actions” 页面，选择工作流“Windows Build (PyInstaller)”
- 在最新一次运行中，进入 “Artifacts” 下载 `NikonFilmLab-windows-onedir.zip`
- 解压后，进入 `NikonFilmLab/` 文件夹，双击运行 `NikonFilmLab.exe`
- 如遇 Windows SmartScreen 拦截，请选择“仍要运行”
- 如系统缺少运行时导致无法启动，可安装微软“Visual C++ 2015-2022 可再发行组件”（通常无需手动安装）
- OpenCL/iGPU 加速取决于显卡驱动与 OpenCL 运行时是否正确安装（无则自动回退 CPU）

**本地在 Windows 打包（可选）**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --noconfirm --clean packaging/NikonFilmLab.spec
```
生成目录：`dist/NikonFilmLab/`，其中包含 `NikonFilmLab.exe` 与依赖 DLL/资源（onedir 结构更适合 Qt+OpenCV+rawpy 组合）。你也可以将该文件夹压缩分发。

说明
- 入口：`src/nikon_film_app/main.py`（GUI，无控制台窗口）
- 已在 spec 中收集 PySide6、cv2、rawpy 的二进制与资源
- rawpy/LibRaw、OpenCV 与 Qt 插件均已打包到 onedir 结果中

