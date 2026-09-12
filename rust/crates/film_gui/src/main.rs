use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::{egui, egui::{ColorImage, TextureHandle}};
use egui_extras::RetainedImage;
use film_core::{self as core, ProcessOptions, GrainType, VignetteMode};
use image::{RgbImage, DynamicImage, GenericImageView, ImageBuffer, Rgb};

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
        Box::new(|_cc| Box::new(AppState::default())),
    )
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
                    if let Some(files) = rfd::FileDialog::new().add_filter("Images", &["jpg", "jpeg"]).pick_files() {
                        self.add_files(files);
                    }
                }
                if ui.button("添加文件夹").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
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
                ui.label("导出先略：MVP 展示预览与参数连通（导出随后补齐）");
            });
        });

        egui::SidePanel::left("queue_panel").resizable(true).default_width(240.0).show(ctx, |ui| {
            queue_width = ui.available_width();
            let mut select_idx: Option<usize> = None;
            egui::ScrollArea::vertical().show(ui, |ui| {
                for (idx, it) in self.items.iter().enumerate() {
                    let name = it.path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
                    if ui.selectable_label(self.current == Some(idx), name).clicked() {
                        select_idx = Some(idx);
                    }
                    ui.separator();
                }
            });
            if let Some(i) = select_idx {
                self.set_current(i, ctx);
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
            // 特色 FX（MVP：仅显示开关，功能已在内核实现 2 项）
            ui.horizontal(|ui| {
                let mut e = self.opts.enable_expired_film;
                if ui.checkbox(&mut e, "胶片过期").changed() {
                    self.opts.enable_expired_film = e;
                    self.touch();
                }
                let mut d = self.opts.expired_film;
                if ui.add(egui::Slider::new(&mut d, 0..=100).text("强度")).changed() {
                    self.opts.expired_film = d;
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
                    self.touch();
                }
            });
            ui.separator();
            // 预览区域
            ui.label("预览：参数变更 200ms 防抖实时更新；导出功能稍后补齐");
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
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

impl AppState {
    fn add_files(&mut self, files: Vec<PathBuf>) {
        for p in files {
            if !is_supported(&p) {
                continue;
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
        match core::load_image_bgr_or_rgb8(item.path.to_string_lossy().as_ref()) {
            Ok(core::InputImage::Rgb8(rgb)) => {
                // 核心输出为 BGR 约定，这里统一用 RGB8，process 接口同样使用 RGB8
                // 由于 load_image_bgr_or_rgb8 返回的是 RGB（由 image crate 解码），直接使用
                self.src_rgb8 = Some(rgb);
            }
            Err(e) => {
                eprintln!("加载失败: {}", e);
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
        return matches!(e.as_str(), "jpg" | "jpeg");
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
