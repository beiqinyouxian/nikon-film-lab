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

**预览与交互（单图预览）**
- 预览区域为“单图等比适配”，不再并排前后对比。点击左侧队列条目，立即显示该照片“原图”全幅
- 预设新增“【不处理】”（默认），默认强度=0、颗粒=关、暗角=关、自动基线=关 → 启动即看到原图
- 当选择真实预设或开启颗粒/调整参数时，预览显示“处理后”单图（200ms 防抖、降采样预览、导出全分辨率）
- “复位”按钮：一键恢复为原片（预设=不处理、强度=0、颗粒/暗角/自动基线关闭）；导出同样遵循当前参数

**颗粒（Grain）控制**
- 类型：银盐（单色、结团感）、Modern fine（现代细腻）、Coarse push（粗颗粒，似推片）
- 大小：控制颗粒的空间尺度（分辨率自适应；预览与导出一致感）
- 密度：控制颗粒强度/覆盖度
- 粗糙：控制结团/粗糙程度
- 彩色混合：控制彩色颗粒参与度（0 为纯亮度颗粒）

**手动控制**
- 暗角：模式 不处理 / 自动（预设默认）/ 手动（强度滑条）
- 曝光：-2..+2 EV（默认 0 不处理）
- 色温：冷暖调节（默认 0 不处理）
- 清晰度（Clarity）：中灰区域的局部对比（默认 0 不处理；负值轻柔，正值增强），分辨率自适应，预览与导出观感一致

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

---

## Rust 重写（进行中）

为获得更好的运行性能与 Windows 原生发行版，我们在不破坏现有 Python 应用的前提下，旁挂一个 Rust 重写版本（逐步对齐功能）。Rust 项目位于 `rust/`，采用 `cargo` 工作区：

- `rust/crates/film_core`：核心图像处理库（预设/手动/颗粒/暗角/特色 FX/分色器 API）
- `rust/crates/film_gui`：基于 egui/eframe 的桌面 GUI（中文 UI）

目前交付策略（分阶段）：
1. 工作区与 README（本次已完成）
2. 核心处理库：提供预设与手动参数 API + 单元测试（已提供 MVP）
3. GUI 外壳：队列/单图预览/参数连通（已提供 MVP；JPG 优先）
4. Windows 发布工作流：GitHub Actions 产出 `.exe/.zip`（本次已添加工作流文件）
5. 特性清单：已移植 vs 仍由 Python 提供（见下）

### 架构与构建
- UI：egui + eframe（后续如需，也可评估 iced/Slint 替代）
- 图像：image/ndarray + 自写快速路径；并行 rayon；MVP 支持 JPG，NEF 解码计划使用 `rawloader`（纯 Rust）或后续切到 LibRaw 绑定
- 打包：`cargo build --release`，Windows 由 CI 生成 `.zip` 工件；后续可加入安装器

本地构建（Rust）：
```bash
cd rust
cargo build --release -p film_gui
# 运行
cargo run -p film_gui
```
生成可执行文件（Windows/macOS/Linux）：`rust/target/release/nikon-film-lab-rs{.exe}`

### Rust 版功能清单（进展）
- 已实现（Rust）：
  - 队列（添加文件/文件夹，拖拽导入，清空），JPG/JPEG 优先
  - 左侧缩略图，随队列面板宽度自适应缩放（JPG）
  - 单图预览（200ms 防抖）
  - 预设全集（与 Python 对齐的命名与推荐强度）：不处理 / Leica Color Modern / Leica Classic Mono / Leica Chrome Vivid / Chrome 浓彩 / Chrome 经典 / Chrome 鲜艳 / Kodak Portra 400 / Kodak Gold 200 / Fuji Velvia 50 / Fuji Pro 400H / Ilford HP5 (B&W) / Kodak Tri-X / Cinestill 800T / Agfa Vista
  - 手动：曝光、色温、清晰、对比、高光、阴影、鲜艳、饱和
  - 分色器：8 段饱和/明度（GUI 已接线，中文标签）
  - 颗粒：三种类型参数（分辨率自适应）
  - 暗角：不处理/自动/手动（手动强度）
  - 特色 FX：镜头老化、镜片划伤、过期胶片、胶片漏光（拖动滑条自动勾选）
  - 导出：全分辨率 JPEG（质量 95），JPG→JPG 场景尝试 EXIF 透传（尽力而为）
  - 单元测试：形状保持、基础参数有效性
- 待办（仍由 Python 版提供或下一阶段迁移）：
  - RAW→JPG 的 EXIF 更完整的复制（当前仅 JPG→JPG 尝试透传；RAW→JPG 暂无 EXIF）
  - OpenCL/GPU（非阻塞项；先专注 CPU 热路径）

### RAW / NEF 支持（本次新增）
- 解码：优先使用 `rawloader` + `demosaic`（Bayer 去马赛克）实现纯 Rust 流程；如遇质量/兼容性问题，将在后续评估 LibRaw 绑定，但不阻塞当前进度
- 队列：接受 `.nef/.NEF`；缩略与预览来自解码后的缩小图；大图导出使用全分辨率 demosaic → sRGB → JPEG（质量 95）
- EXIF：RAW→JPG 暂不复制 EXIF；README 明确限制；JPG→JPG 维持 APP1 Exif 透传尝试
- 交互：错误信息与提示为中文；保持 UI 响应（预览有防抖；重处理在计算完成后更新）

### 预览 vs 导出（质量与速度）
- 预览/缩略：优先速度，使用“早期降采样 + 快速去马赛克（MHC）”路径，快速响应调参
- 导出：优先质量，使用更高质量的 AHD 去马赛克与全分辨率处理

测试建议
- 可从 `https://raw.pixls.us/` 下载公开 NEF 样张进行验证；比较 Python 版（rawpy/LibRaw）与 Rust 版在外观上的差异

### CI：Windows 可执行文件（Rust）
- 工作流：`.github/workflows/windows-rust.yml`
- 触发：推送到 `cursor/rust-rewrite-0eab` 或手动
- 产物：`nikon-film-lab-rs-windows-x86_64.zip`（包含 `nikon-film-lab-rs.exe`）

如需更多细节，请查看 `rust/crates/film_core/src/lib.rs` 与 `rust/crates/film_gui/src/main.rs`。

