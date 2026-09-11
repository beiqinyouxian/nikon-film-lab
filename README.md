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

