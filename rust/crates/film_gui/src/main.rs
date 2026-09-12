use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
use std::thread;

use eframe::{egui, egui::{ColorImage, TextureHandle, FontDefinitions, FontData, FontFamily}};
use egui_extras::RetainedImage;
use film_core::{self as core, ProcessOptions, GrainType, VignetteMode};
use image::RgbImage;
use std::fs;

fn main() -> eframe::Result<()> {
    env_logger::init();
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("Nikon Film Lab (Rust)"),
        ..Default::default()
    };
    eframe::run_native(
        "Nikon Film Lab (Rust)",
        native_options,
        Box::new(|cc| {
            // Windows CJK 字体注入，避免中文方块（tofu）
            #[cfg(target_os = "windows")]
            {
                if let Some(bytes) = load_windows_cjk_font_bytes() {
                    let mut fonts = FontDefinitions::default();
                    fonts.font_data.insert("cjk".to_owned(), FontData::from_owned(bytes));
                    fonts.families.entry(FontFamily::Proportional).or_default().insert(0, "cjk".to_owned());
                    fonts.families.entry(FontFamily::Monospace).or_default().insert(0, "cjk".to_owned());
                    cc.egui_ctx.set_fonts(fonts);
                }
            }
            Box::new(AppState::default())
        }),
    )
}

#[cfg(target_os = "windows")]
fn load_windows_cjk_font_bytes() -> Option<Vec<u8>> {
    // 优先顺序：Microsoft YaHei UI / Microsoft YaHei / SimHei / Noto Sans CJK / Source Han Sans
    let candidates = [
        r"C:\Windows\Fonts\msyh.ttc",      // Microsoft YaHei
        r"C:\Windows\Fonts\msyhbd.ttc",    // Microsoft YaHei Bold
        r"C:\Windows\Fonts\msyh.ttf",      // Some systems may have ttf
        r"C:\Windows\Fonts\simhei.ttf",    // SimHei
        r"C:\Windows\Fonts\NotoSansCJK-Regular.ttc",
        r"C:\Windows\Fonts\NotoSansSC-Regular.otf",
        r"C:\Windows\Fonts\SourceHanSansSC-Regular.otf",
    ];
    for p in candidates {
        if let Ok(data) = std::fs::read(p) {
            return Some(data);
        }
    }
    None
}

#[derive(Default)]
struct QueueItem {
    path: PathBuf,
    thumb: Option<RetainedImage>,
}

struct AppState {
    // 队列与预览
    items: Vec<QueueItem>,
    current: Option<usize>,
    src_rgb8: Option<RgbImage>,
    preview_tex: Option<TextureHandle>,
    // 参数
    opts: ProcessOptions,
    // 控件状态
    vignette_mode_idx: usize, // 0 不处理, 1 自动, 2 手动
    // 防抖
    last_change: Instant,
    need_render: bool,
    export_dir: PathBuf,
    last_queue_width: f32,
    // 批量导出进度
    exporting: bool,
    export_cur: Arc<AtomicUsize>,
    export_total: Arc<AtomicUsize>,
    export_cancel: Arc<AtomicBool>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            items: vec![],
            current: None,
            src_rgb8: None,
            preview_tex: None,
            opts: ProcessOptions::default(),
            vignette_mode_idx: 0,
            last_change: Instant::now(),
            need_render: false,
            export_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            last_queue_width: 0.0,
            exporting: false,
            export_cur: Arc::new(AtomicUsize::new(0)),
            export_total: Arc::new(AtomicUsize::new(0)),
            export_cancel: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl eframe::App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut queue_width = 260.0_f32;
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("队列（支持 .jpg / .jpeg；NEF 后续开启）");
                if ui.button("添加文件").clicked() {
                    let mut dlg = rfd::FileDialog::new().add_filter("Images", &["jpg", "jpeg", "nef"]);
                    if let Some(dir) = read_last_dir() {
                        dlg = dlg.set_directory(dir);
                    }
                    if let Some(files) = dlg.pick_files() {
                        self.add_files(files);
                    }
                }
                if ui.button("添加文件夹").clicked() {
                    let mut dlg = rfd::FileDialog::new();
                    if let Some(dir) = read_last_dir() {
                        dlg = dlg.set_directory(dir);
                    }
                    if let Some(dir) = dlg.pick_folder() {
                        write_last_dir(&dir);
                        let mut files = vec![];
                        if let Ok(rd) = std::fs::read_dir(dir) {
                            for e in rd.flatten() {
                                let p = e.path();
                                if is_supported(&p) {
                                    files.push(p);
                                }
                            }
                        }
                        self.add_files(files);
                    }
                }
                if ui.button("清空").clicked() {
                    self.items.clear();
                    self.current = None;
                    self.src_rgb8 = None;
                    self.preview_tex = None;
                }
                ui.separator();
                ui.label("导出文件夹：");
                let path_str = self.export_dir.to_string_lossy().to_string();
                ui.monospace(ellipsize_path(&path_str, 48));
                if ui.button("选择…").clicked() {
                    let mut dlg = rfd::FileDialog::new();
                    if let Some(dir) = read_last_dir() {
                        dlg = dlg.set_directory(dir);
                    }
                    if let Some(d) = dlg.pick_folder() {
                        self.export_dir = d;
                        write_last_dir(&self.export_dir);
                    }
                }
                if ui.button("导出当前").clicked() {
                    self.export_current();
                }
                if ui.button("导出全部").clicked() {
                    if !self.exporting {
                        self.start_export_all();
                    }
                }
                if self.exporting {
                    let cur = self.export_cur.load(Ordering::Relaxed);
                    let tot = self.export_total.load(Ordering::Relaxed).max(1);
                    let pct = (cur as f32 / tot as f32 * 100.0).round() as i32;
                    ui.separator();
                    ui.label(format!("批量导出进度：{cur}/{tot}（{pct}%）"));
                    if ui.button("取消导出").clicked() {
                        self.export_cancel.store(true, Ordering::Relaxed);
                    }
                }
            });
        });

        egui::SidePanel::left("queue_panel").resizable(true).default_width(240.0).show(ctx, |ui| {
            queue_width = ui.available_width();
            let mut select_idx: Option<usize> = None;
            egui::ScrollArea::vertical().show(ui, |ui| {
                for (idx, it) in self.items.iter_mut().enumerate() {
                    // 生成或更新缩略
                    if (self.last_queue_width - queue_width).abs() > 8.0 || it.thumb.is_none() {
                        if let Some(img) = load_thumb_rgb8(&it.path, queue_width as u32) {
                            it.thumb = Some(image_to_retained(img));
                        }
                        self.last_queue_width = queue_width;
                    }
                    let name = it.path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
                    ui.horizontal(|ui| {
                        if let Some(t) = &it.thumb {
                            let s = egui::vec2(queue_width.min(220.0), (queue_width.min(220.0) * 0.6).max(60.0));
                            t.show_max_size(ui, s);
                        }
                        if ui.selectable_label(self.current == Some(idx), name).clicked() {
                            select_idx = Some(idx);
                        }
                    });
                    ui.separator();
                }
            });
            if let Some(i) = select_idx {
                self.set_current(i, ctx);
            }
            // 拖放导入
            if let Some(files) = dropped_files(ctx) {
                let files: Vec<PathBuf> = files.into_iter().filter(|p| is_supported(p)).collect();
                if !files.is_empty() {
                    self.add_files(files);
                }
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                // 预设与强度
                ui.label("预设");
                let presets = core::list_presets();
                let mut preset_idx = presets.iter().position(|&n| n == self.opts.preset_name).unwrap_or(0);
                if egui::ComboBox::from_id_source("preset_combo").selected_text(presets[preset_idx]).show_ui(ui, |ui| {
                    for (i, name) in presets.iter().enumerate() {
                        ui.selectable_value(&mut preset_idx, i, *name);
                    }
                }).response.changed() {
                    self.opts.preset_name = presets[preset_idx].to_string();
                    self.opts.strength_percent = core::recommended_strength(&self.opts.preset_name);
                    self.touch();
                }
                ui.label("强度");
                let mut s = self.opts.strength_percent as i32;
                if ui.add(egui::Slider::new(&mut s, 0..=100).clamp_to_range(true).show_value(true)).changed() {
                    self.opts.strength_percent = s as u8;
                    self.touch();
                }
                if ui.button("推荐强度").clicked() {
                    self.opts.strength_percent = core::recommended_strength(&self.opts.preset_name);
                    self.touch();
                }
                ui.separator();
                // 暗角
                ui.label("暗角");
                let vmodes = ["不处理", "自动", "手动"];
                if egui::ComboBox::from_id_source("vig_mode").selected_text(vmodes[self.vignette_mode_idx]).show_ui(ui, |ui| {
                    for (i, name) in vmodes.iter().enumerate() {
                        ui.selectable_value(&mut self.vignette_mode_idx, i, *name);
                    }
                }).response.changed() {
                    self.opts.vignette_mode = match self.vignette_mode_idx {
                        1 => VignetteMode::Auto,
                        2 => VignetteMode::Manual,
                        _ => VignetteMode::Off
                    };
                    self.touch();
                }
                if matches!(self.opts.vignette_mode, VignetteMode::Manual) {
                    let mut v = self.opts.vignette_amount as i32;
                    if ui.add(egui::Slider::new(&mut v, 0..=100)).changed() {
                        self.opts.vignette_amount = v as u8;
                        self.touch();
                    }
                }
            });
            ui.separator();
            // 手动滑条区（聚合 changed，统一 touch，避免可变借用冲突）
            let mut any_changed = false;
            egui::Grid::new("tone_color_grid").num_columns(4).spacing([16.0, 8.0]).show(ui, |ui| {
                any_changed |= slider_i16_simple(ui, "曝光", &mut self.opts.exposure_ev_x100, -200, 200);
                any_changed |= slider_i16_simple(ui, "色温", &mut self.opts.temp_bias, -100, 100);
                any_changed |= slider_i16_simple(ui, "对比", &mut self.opts.contrast, -100, 100);
                any_changed |= slider_i16_simple(ui, "清晰", &mut self.opts.clarity, -100, 100);
                any_changed |= slider_i16_simple(ui, "高光", &mut self.opts.highlights, -100, 100);
                any_changed |= slider_i16_simple(ui, "阴影", &mut self.opts.shadows, -100, 100);
                any_changed |= slider_i16_simple(ui, "鲜艳", &mut self.opts.vibrance, -100, 100);
                any_changed |= slider_i16_simple(ui, "饱和", &mut self.opts.saturation, -100, 100);
                ui.end_row();
            });
            if any_changed {
                self.touch();
            }
            ui.separator();
            // 分色器（8 段）
            egui::CollapsingHeader::new("分色器（8 段）").default_open(false).show(ui, |ui| {
                let bands = ["红","橙","黄","绿","青","蓝","紫","品红"];
                ui.label("饱和（-100..100）");
                let mut sat_changed = false;
                for i in 0..8 {
                    ui.horizontal(|ui| {
                        ui.label(bands[i]);
                        let mut v = self.opts.hsl_sat8[i] as i32;
                        if ui.add(egui::Slider::new(&mut v, -100..=100).clamp_to_range(true)).changed() {
                            self.opts.hsl_sat8[i] = v as i16;
                            sat_changed = true;
                        }
                    });
                }
                ui.separator();
                ui.label("明度（-100..100）");
                let mut lum_changed = false;
                for i in 0..8 {
                    ui.horizontal(|ui| {
                        ui.label(bands[i]);
                        let mut v = self.opts.hsl_lum8[i] as i32;
                        if ui.add(egui::Slider::new(&mut v, -100..=100).clamp_to_range(true)).changed() {
                            self.opts.hsl_lum8[i] = v as i16;
                            lum_changed = true;
                        }
                    });
                }
                ui.horizontal(|ui| {
                    if ui.button("分色器复位").clicked() {
                        self.opts.hsl_sat8 = [0; 8];
                        self.opts.hsl_lum8 = [0; 8];
                        self.touch();
                    }
                });
                if sat_changed || lum_changed {
                    self.opts.enable_color_splitter = true;
                    self.touch();
                }
            });
            ui.separator();
            // 颗粒
            ui.horizontal(|ui| {
                let mut enable = self.opts.enable_grain;
                if ui.checkbox(&mut enable, "颗粒").changed() {
                    self.opts.enable_grain = enable;
                    self.touch();
                }
                let names = ["Silver halide", "Modern fine", "Coarse push"];
                let mut typ_idx = match self.opts.grain_type {
                    GrainType::SilverHalide => 0,
                    GrainType::ModernFine => 1,
                    GrainType::CoarsePush => 2,
                };
                if egui::ComboBox::from_id_source("grain_type").selected_text(names[typ_idx]).show_ui(ui, |ui| {
                    for (i, name) in names.iter().enumerate() {
                        ui.selectable_value(&mut typ_idx, i, *name);
                    }
                }).response.changed() {
                    self.opts.grain_type = match typ_idx {
                        1 => GrainType::ModernFine,
                        2 => GrainType::CoarsePush,
                        _ => GrainType::SilverHalide,
                    };
                    self.touch();
                }
                if slider_u8_inline_simple(ui, "大小", &mut self.opts.grain_size) { self.touch(); }
                if slider_u8_inline_simple(ui, "密度", &mut self.opts.grain_density) { self.touch(); }
                if slider_u8_inline_simple(ui, "粗糙", &mut self.opts.grain_roughness) { self.touch(); }
                if slider_u8_inline_simple(ui, "彩混", &mut self.opts.grain_chroma_mix) { self.touch(); }
            });
            ui.separator();
            // 特色 FX（镜头老化、划痕、过期、漏光）
            ui.horizontal(|ui| {
                let mut lens = self.opts.enable_lens_aging;
                if ui.checkbox(&mut lens, "镜头老化").changed() {
                    self.opts.enable_lens_aging = lens;
                    self.touch();
                }
                let mut lv = self.opts.lens_aging;
                if ui.add(egui::Slider::new(&mut lv, 0..=100).text("强度")).changed() {
                    self.opts.lens_aging = lv;
                    if lv > 0 { self.opts.enable_lens_aging = true; }
                    self.touch();
                }
                ui.separator();
                let mut s = self.opts.enable_scratches;
                if ui.checkbox(&mut s, "镜片划伤").changed() {
                    self.opts.enable_scratches = s;
                    self.touch();
                }
                let mut sv = self.opts.scratches;
                if ui.add(egui::Slider::new(&mut sv, 0..=100).text("强度")).changed() {
                    self.opts.scratches = sv;
                    if sv > 0 { self.opts.enable_scratches = true; }
                    self.touch();
                }
                ui.separator();
                let mut e = self.opts.enable_expired_film;
                if ui.checkbox(&mut e, "胶片过期").changed() {
                    self.opts.enable_expired_film = e;
                    self.touch();
                }
                let mut d = self.opts.expired_film;
                if ui.add(egui::Slider::new(&mut d, 0..=100).text("强度")).changed() {
                    self.opts.expired_film = d;
                    if d > 0 { self.opts.enable_expired_film = true; }
                    self.touch();
                }
                ui.separator();
                let mut l = self.opts.enable_light_leak;
                if ui.checkbox(&mut l, "胶片漏光").changed() {
                    self.opts.enable_light_leak = l;
                    self.touch();
                }
                let mut pl = self.opts.light_leak;
                if ui.add(egui::Slider::new(&mut pl, 0..=100).text("强度")).changed() {
                    self.opts.light_leak = pl;
                    if pl > 0 { self.opts.enable_light_leak = true; }
                    self.touch();
                }
            });
            ui.separator();
            // 预览区域
            ui.label("预览：参数变更 200ms 防抖实时更新；导出为全分辨率 JPG（95）");
            ui.separator();
            let avail = ui.available_size();
            let rect = egui::Rect::from_min_size(ui.min_rect().min, avail);
            if let Some(tex) = &self.preview_tex {
                let ratio = tex.size()[0] as f32 / tex.size()[1] as f32;
                let mut size = egui::vec2(avail.x, avail.x / ratio);
                if size.y > avail.y {
                    size = egui::vec2(avail.y * ratio, avail.y);
                }
                ui.image((tex.id(), size));
            } else {
                ui.allocate_space(avail);
            }
        });

        // 防抖渲染
        if self.need_render && self.last_change.elapsed() >= Duration::from_millis(200) {
            self.render_preview(ctx);
            self.need_render = false;
        }
    // 导出完成检测（简易轮询）
    if self.exporting {
        let done = self.export_cur.load(Ordering::Relaxed) >= self.export_total.load(Ordering::Relaxed);
        if done || self.export_cancel.load(Ordering::Relaxed) {
            self.exporting = false;
        }
    }
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

impl AppState {
    fn add_files(&mut self, files: Vec<PathBuf>) {
        for p in files {
            if !is_supported(&p) {
                continue;
            }
            if let Some(dir) = p.parent() {
                write_last_dir(dir);
            }
            self.items.push(QueueItem { path: p, thumb: None });
        }
        if self.current.is_none() && !self.items.is_empty() {
            self.current = Some(0);
        }
    }

    fn set_current(&mut self, idx: usize, ctx: &egui::Context) {
        self.current = Some(idx);
        self.src_rgb8 = None;
        self.preview_tex = None;
        self.load_current_source();
        self.render_preview(ctx);
    }

    fn load_current_source(&mut self) {
        let Some(i) = self.current else { return; };
        let Some(item) = self.items.get(i) else { return; };
        let lower = item.path.to_string_lossy().to_lowercase();
        if lower.ends_with(".nef") {
            #[cfg(feature = "nef")]
            {
                match core::load_raw_preview_bgr8(&item.path.to_string_lossy(), 1600) {
                    Ok(img) => self.src_rgb8 = Some(img),
                    Err(e) => eprintln!("RAW 预览失败: {}", e),
                }
            }
            #[cfg(not(feature = "nef"))]
            {
                eprintln!("未启用 RAW 支持（编译特性 nef 关闭）");
            }
        } else {
            match core::load_image_bgr_or_rgb8(item.path.to_string_lossy().as_ref()) {
                Ok(core::InputImage::Rgb8(rgb)) => {
                    self.src_rgb8 = Some(rgb);
                }
                Err(e) => {
                    eprintln!("加载失败: {}", e);
                }
            }
        }
    }

    fn touch(&mut self) {
        // 参数变化：记录时间，稍后统一渲染
        self.last_change = Instant::now();
        self.need_render = true;
        // 固定种子：根据路径生成可重复颗粒/FX
        if let Some(i) = self.current {
            if let Some(item) = self.items.get(i) {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                item.path.to_string_lossy().hash(&mut hasher);
                let seed = hasher.finish();
                self.opts.grain_seed = Some(seed);
                self.opts.fx_seed = Some(seed ^ 0xa5a5_5a5a_dead_beefu64);
            }
        }
    }

    fn render_preview(&mut self, ctx: &egui::Context) {
        if self.src_rgb8.is_none() {
            self.load_current_source();
        }
        let Some(src) = &self.src_rgb8 else { return; };
        let processed = core::process_rgb8(src, &self.opts);
        self.preview_tex = Some(upload_rgb8_as_texture(ctx, &processed));
    }
}

fn upload_rgb8_as_texture(ctx: &egui::Context, img: &RgbImage) -> TextureHandle {
    let size = [img.width() as usize, img.height() as usize];
    // egui 纹理为 RGBA
    let mut rgba = Vec::with_capacity((size[0] * size[1]) * 4);
    for p in img.pixels() {
        rgba.push(p[2]); // R
        rgba.push(p[1]); // G
        rgba.push(p[0]); // B
        rgba.push(255);  // A
    }
    let color_image = ColorImage::from_rgba_unmultiplied(size, &rgba);
    ctx.load_texture("preview", color_image, Default::default())
}

fn is_supported(p: &Path) -> bool {
    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
        let e = ext.to_ascii_lowercase();
        return matches!(e.as_str(), "jpg" | "jpeg" | "nef");
    }
    false
}

fn slider_i16_simple(ui: &mut egui::Ui, label: &str, v: &mut i16, lo: i16, hi: i16) -> bool {
    let mut tmp = *v as i32;
    ui.horizontal(|ui| {
        ui.label(label);
    }).response; // label rendered
    let changed = ui.add(egui::Slider::new(&mut tmp, lo as i32..=hi as i32)).changed();
    if changed { *v = tmp as i16; }
    changed
}

fn slider_u8_inline_simple(ui: &mut egui::Ui, label: &str, v: &mut u8) -> bool {
    ui.label(label);
    let mut tmp = *v as i32;
    let changed = ui.add(egui::Slider::new(&mut tmp, 0..=100).show_value(false).clamp_to_range(true)).changed();
    if changed { *v = tmp as u8; }
    changed
}

// ---------- 拖放、缩略与导出 ----------

fn dropped_files(ctx: &egui::Context) -> Option<Vec<PathBuf>> {
    let files = ctx.input(|i| i.raw.dropped_files.clone());
    if files.is_empty() { return None; }
    let mut out = vec![];
    for f in files {
        if let Some(p) = f.path {
            out.push(p);
        }
    }
    Some(out)
}

fn load_thumb_rgb8(path: &Path, max_side: u32) -> Option<RgbImage> {
    if !is_supported(path) { return None; }
    let lower = path.to_string_lossy().to_lowercase();
    if lower.ends_with(".nef") {
        #[cfg(feature = "nef")]
        {
            return core::load_raw_preview_bgr8(&path.to_string_lossy(), max_side).ok();
        }
        #[cfg(not(feature = "nef"))]
        {
            return None;
        }
    } else {
        let img = image::open(path).ok()?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let scale = (max_side as f32 / w.max(h) as f32).clamp(0.0, 1.0);
        let (nw, nh) = if scale < 1.0 {
            ((w as f32 * scale) as u32, (h as f32 * scale) as u32)
        } else { (w, h) };
        return Some(image::imageops::resize(&rgb, nw.max(1), nh.max(1), image::imageops::FilterType::Triangle));
    }
}

fn image_to_retained(img: RgbImage) -> RetainedImage {
    // Convert to RGBA for egui_extras
    let mut rgba = Vec::with_capacity((img.width() * img.height()) as usize * 4);
    for p in img.pixels() {
        rgba.push(p[0]);
        rgba.push(p[1]);
        rgba.push(p[2]);
        rgba.push(255);
    }
    let ci = ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], &rgba);
    RetainedImage::from_color_image("thumb", ci)
}

fn ellipsize_path(s: &str, max_len: usize) -> String {
    if s.len() <= max_len { return s.to_string(); }
    let keep = max_len / 2;
    format!("{}…{}", &s[..keep], &s[s.len()-keep..])
}

fn config_path() -> Option<PathBuf> {
    let base = dirs::config_dir()?;
    let dir = base.join("nikon-film-lab-rs");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("last_dir.txt"))
}

fn read_last_dir() -> Option<PathBuf> {
    let p = config_path()?;
    let s = std::fs::read_to_string(p).ok()?;
    let s = s.trim();
    if s.is_empty() { return None; }
    Some(PathBuf::from(s))
}

fn write_last_dir(dir: &Path) {
    if let Some(p) = config_path() {
        let _ = std::fs::write(p, dir.to_string_lossy().as_bytes());
    }
}

impl AppState {
    fn export_current(&mut self) {
        let Some(i) = self.current else { return; };
        if let Some(item) = self.items.get(i) {
            // 始终全分辨率重载
            let lower = item.path.to_string_lossy().to_lowercase();
            let full = if lower.ends_with(".nef") {
                #[cfg(feature = "nef")]
                {
                    match core::load_raw_fullres_bgr8(&item.path.to_string_lossy()) {
                        Ok(img) => img,
                        Err(e) => { eprintln!("RAW 导出解码失败: {}", e); return; }
                    }
                }
                #[cfg(not(feature = "nef"))]
                {
                    eprintln!("未启用 RAW 支持（编译特性 nef 关闭）"); return;
                }
            } else {
                match core::load_image_bgr_or_rgb8(item.path.to_string_lossy().as_ref()) {
                    Ok(core::InputImage::Rgb8(rgb)) => rgb,
                    Err(e) => { eprintln!("载入失败: {}", e); return; }
                }
            };
            let processed = core::process_rgb8(&full, &self.opts);
            let name = item.path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
            let out_path = self.export_dir.join(format!("{name}_film.jpg"));
            // RAW→JPG 暂不复制 EXIF；JPG→JPG 尝试 EXIF 透传
            let exif = if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
                extract_exif_app1(&item.path).ok()
            } else { None };
            if let Err(e) = write_jpeg_with_optional_exif(&processed, &out_path, exif.as_deref(), 95) {
                eprintln!("导出失败: {e}");
            }
        }
    }
    fn start_export_all(&mut self) {
        // 只拷贝路径列表，避免缩略图克隆
        let items: Vec<PathBuf> = self.items.iter().map(|it| it.path.clone()).collect();
        let export_dir = self.export_dir.clone();
        let opts = self.opts.clone();
        let cur = self.export_cur.clone();
        let tot = self.export_total.clone();
        let cancel = self.export_cancel.clone();
        cur.store(0, Ordering::Relaxed);
        tot.store(items.len(), Ordering::Relaxed);
        cancel.store(false, Ordering::Relaxed);
        self.exporting = true;
        thread::spawn(move || {
            for (idx, p) in items.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) { break; }
                let lower = p.to_string_lossy().to_lowercase();
                let full = if lower.ends_with(".nef") {
                    #[cfg(feature = "nef")]
                    {
                        match core::load_raw_fullres_bgr8(&p.to_string_lossy()) {
                            Ok(img) => img,
                            Err(_) => { cur.store(idx+1, Ordering::Relaxed); continue; }
                        }
                    }
                    #[cfg(not(feature = "nef"))]
                    {
                        cur.store(idx+1, Ordering::Relaxed);
                        continue;
                    }
                } else {
                    match core::load_image_bgr_or_rgb8(p.to_string_lossy().as_ref()) {
                        Ok(core::InputImage::Rgb8(rgb)) => rgb,
                        Err(_) => { cur.store(idx+1, Ordering::Relaxed); continue; }
                    }
                };
                let processed = core::process_rgb8(&full, &opts);
                let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
                let out_path = export_dir.join(format!("{name}_film.jpg"));
                let exif = if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
                    extract_exif_app1(&p).ok()
                } else { None };
                let _ = write_jpeg_with_optional_exif(&processed, &out_path, exif.as_deref(), 95);
                cur.store(idx+1, Ordering::Relaxed);
            }
        });
    }
}

fn extract_exif_app1(path: &Path) -> std::io::Result<Vec<u8>> {
    let data = fs::read(path)?;
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 { return Err(std::io::Error::new(std::io::ErrorKind::Other, "not jpeg")); }
    let mut i = 2;
    while i + 4 <= data.len() {
        if data[i] != 0xFF { break; }
        let marker = data[i + 1];
        i += 2;
        if i + 2 > data.len() { break; }
        let len = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
        if i + len > data.len() { break; }
        let payload = &data[i + 2..i + len];
        if marker == 0xE1 && payload.starts_with(b"Exif\0\0") {
            return Ok(payload.to_vec());
        }
        i += len;
    }
    Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no exif"))
}

fn write_jpeg_with_optional_exif(img: &RgbImage, path: &Path, exif_app1: Option<&[u8]>, quality: u8) -> anyhow::Result<()> {
    use image::codecs::jpeg::JpegEncoder;
    let mut buf = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut buf, quality);
        enc.encode_image(&image::DynamicImage::ImageRgb8(img.clone()))?;
    }
    if let Some(app1) = exif_app1 {
        if buf.len() >= 2 && buf[0] == 0xFF && buf[1] == 0xD8 {
            let mut out = Vec::with_capacity(buf.len() + app1.len() + 4);
            out.extend_from_slice(&buf[0..2]); // SOI
            out.push(0xFF);
            out.push(0xE1);
            let len = (app1.len() as u16 + 2).to_be_bytes();
            out.push(len[0]);
            out.push(len[1]);
            out.extend_from_slice(app1);
            out.extend_from_slice(&buf[2..]);
            fs::write(path, out)?;
            return Ok(());
        }
    }
    fs::write(path, buf)?;
    Ok(())
}
