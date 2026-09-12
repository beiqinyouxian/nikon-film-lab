//! film_core: Nikon Film Lab 核心图像处理（Rust）
//! - 目标：与 Python 版本功能对齐（分阶段），优先正确性与 CPU 性能
//! - 该 crate 提供参数结构与处理 API，GUI 在 `film_gui`

use anyhow::{bail, Context, Result};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use once_cell::sync::Lazy;
use palette::{FromColor, Hsv, Srgb, IntoColor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand::prelude::SliceRandom;
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[repr(u8)]
pub enum GrainType {
    SilverHalide,
    ModernFine,
    CoarsePush,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessOptions {
    pub preset_name: String,          // 例如 "不处理"、"Kodak Portra 400"
    pub strength_percent: u8,         // 0..100
    // 颗粒
    pub enable_grain: bool,
    pub grain_type: GrainType,
    pub grain_size: u8,               // 0..100
    pub grain_density: u8,            // 0..100
    pub grain_roughness: u8,          // 0..100
    pub grain_chroma_mix: u8,         // 0..100
    pub grain_seed: Option<u64>,
    // 暗角
    pub vignette_mode: VignetteMode,  // off | auto | manual
    pub vignette_amount: u8,          // 0..100 (manual)
    // 自动基线（轻微曝光/WB）
    pub enable_auto_baseline: bool,
    // 手动控制
    pub exposure_ev_x100: i16,        // -200..200
    pub temp_bias: i16,               // -100..100
    pub clarity: i16,                 // -100..100
    pub contrast: i16,                // -100..100
    pub highlights: i16,              // -100..100
    pub shadows: i16,                 // -100..100
    pub vibrance: i16,                // -100..100
    pub saturation: i16,              // -100..100
    // 分色器（8 段）
    pub enable_color_splitter: bool,
    pub hsl_sat8: [i16; 8],           // -100..100
    pub hsl_lum8: [i16; 8],           // -100..100
    // 独立特效
    pub enable_lens_aging: bool,
    pub lens_aging: u8,               // 0..100
    pub enable_scratches: bool,
    pub scratches: u8,                // 0..100
    pub enable_expired_film: bool,
    pub expired_film: u8,             // 0..100
    pub enable_light_leak: bool,
    pub light_leak: u8,               // 0..100
    pub fx_seed: Option<u64>,
}

impl Default for ProcessOptions {
    fn default() -> Self {
        Self {
            preset_name: "不处理".to_string(),
            strength_percent: 0,
            enable_grain: false,
            grain_type: GrainType::SilverHalide,
            grain_size: 30,
            grain_density: 30,
            grain_roughness: 40,
            grain_chroma_mix: 0,
            grain_seed: None,
            vignette_mode: VignetteMode::Off,
            vignette_amount: 0,
            enable_auto_baseline: false,
            exposure_ev_x100: 0,
            temp_bias: 0,
            clarity: 0,
            contrast: 0,
            highlights: 0,
            shadows: 0,
            vibrance: 0,
            saturation: 0,
            enable_color_splitter: true,
            hsl_sat8: [0; 8],
            hsl_lum8: [0; 8],
            enable_lens_aging: false,
            lens_aging: 0,
            enable_scratches: false,
            scratches: 0,
            enable_expired_film: false,
            expired_film: 0,
            enable_light_leak: false,
            light_leak: 0,
            fx_seed: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum VignetteMode {
    Off,
    Auto,
    Manual,
}

/// 预设条目：process 仅负责“胶片风格”部分（不含手动参数与独立特效）
pub struct FilmPreset {
    pub name: &'static str,
    pub recommended_strength: u8,
    pub default_vignette01: f32,
    pub process: fn(&RgbImage, strength01: f32) -> RgbImage,
}

/// 目前内置少量演示预设；后续按 Python 版参数对齐
static PRESETS: Lazy<Vec<FilmPreset>> = Lazy::new(|| {
    vec![
        FilmPreset {
            name: "不处理",
            recommended_strength: 0,
            default_vignette01: 0.0,
            process: |img, _| img.clone(),
        },
        FilmPreset {
            name: "Chrome 经典",
            recommended_strength: 80,
            default_vignette01: 0.14,
            process: preset_chrome_classic,
        },
        FilmPreset {
            name: "Kodak Portra 400",
            recommended_strength: 75,
            default_vignette01: 0.20,
            process: preset_portra_400,
        },
    ]
});

pub fn list_presets() -> Vec<&'static str> {
    PRESETS.iter().map(|p| p.name).collect()
}

pub fn recommended_strength(name: &str) -> u8 {
    PRESETS
        .iter()
        .find(|p| p.name == name)
        .map(|p| p.recommended_strength)
        .unwrap_or(0)
}

fn find_preset(name: &str) -> &'static FilmPreset {
    PRESETS
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| &PRESETS[0])
}

/// 管线：输入 RGB8，返回 RGB8
pub fn process_rgb8(input: &RgbImage, opts: &ProcessOptions) -> RgbImage {
    // 1) 胶片预设（仅风格/曲线/色彩）按强度混合
    let strength01 = (opts.strength_percent as f32 / 100.0).clamp(0.0, 1.0);
    let preset = find_preset(&opts.preset_name);
    let styled = (preset.process)(input, strength01);
    // 2) 手动曝光/色温/清晰/对比/高光阴影/鲜艳/饱和
    let mut out = styled;
    if opts.enable_auto_baseline {
        out = auto_baseline(&out);
    }
    if opts.exposure_ev_x100 != 0 {
        out = apply_exposure(&out, opts.exposure_ev_x100 as f32 / 100.0);
    }
    if opts.temp_bias != 0 {
        out = apply_color_temp(&out, opts.temp_bias as f32 / 100.0);
    }
    if opts.clarity != 0 {
        out = apply_clarity(&out, opts.clarity as f32 / 100.0);
    }
    if opts.contrast != 0 {
        out = adjust_contrast_mid(&out, opts.contrast as f32 / 100.0 * 0.8);
    }
    if opts.highlights != 0 || opts.shadows != 0 {
        out = apply_highlights_shadows(&out, opts.highlights as f32 / 100.0, opts.shadows as f32 / 100.0);
    }
    if opts.vibrance != 0 {
        out = apply_vibrance(&out, opts.vibrance as f32 / 100.0);
    }
    if opts.saturation != 0 {
        out = adjust_saturation(&out, opts.saturation as f32 / 100.0 * 0.7);
    }
    // 3) 分色器（HSV 8 段）
    if opts.enable_color_splitter && (opts.hsl_sat8.iter().any(|&v| v != 0) || opts.hsl_lum8.iter().any(|&v| v != 0)) {
        out = apply_color_splitter_8band(&out, &opts.hsl_sat8, &opts.hsl_lum8);
    }
    // 4) 颗粒
    if opts.enable_grain && opts.grain_density > 0 {
        out = apply_grain(&out, opts);
    }
    // 5) 暗角
    let enable_vignette = match opts.vignette_mode {
        VignetteMode::Off => false,
        VignetteMode::Auto => true,
        VignetteMode::Manual => opts.vignette_amount > 0,
    };
    if enable_vignette {
        let amount = if matches!(opts.vignette_mode, VignetteMode::Manual) {
            (opts.vignette_amount as f32 / 100.0).clamp(0.0, 1.0)
        } else {
            preset.default_vignette01
        };
        out = apply_vignette(&out, amount);
    }
    // 6) 独立特效（镜头老化/划痕/过期/漏光）— MVP：先放置可见的“过期”和“漏光”
    if opts.enable_expired_film && opts.expired_film > 0 {
        out = fx_expired_film(&out, opts.fx_seed.unwrap_or(29), opts.expired_film as f32 / 100.0);
    }
    if opts.enable_light_leak && opts.light_leak > 0 {
        out = fx_light_leak(&out, opts.fx_seed.unwrap_or(41), opts.light_leak as f32 / 100.0);
    }
    out
}

// ---------------- 基础工具 ----------------

fn clamp01(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

fn to_rgb01(px: &Rgb<u8>) -> [f32; 3] {
    [px[0] as f32 / 255.0, px[1] as f32 / 255.0, px[2] as f32 / 255.0]
}
fn from_rgb01(v: [f32; 3]) -> Rgb<u8> {
    Rgb([
        (clamp01(v[0]) * 255.0 + 0.5) as u8,
        (clamp01(v[1]) * 255.0 + 0.5) as u8,
        (clamp01(v[2]) * 255.0 + 0.5) as u8,
    ])
}

// ---------------- 胶片预设（简版） ----------------

fn preset_chrome_classic(img: &RgbImage, strength01: f32) -> RgbImage {
    // 简化：适度对比 + 轻微增饱和 + 近似色矩阵偏色
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let mut v = to_rgb01(p);
        // 颜色矩阵（RGB）
        let r = v[2];
        let g = v[1];
        let b = v[0];
        let r2 = (1.06 * r + 0.01 * g - 0.02 * b) as f32;
        let g2 = (-0.01 * r + 1.05 * g - 0.02 * b) as f32;
        let b2 = (-0.01 * r + 0.00 * g + 1.04 * b) as f32;
        v = [b2 as f32, g2 as f32, r2 as f32];
        // 对比（围绕 0.5）
        let c = 0.15 * strength01;
        v = [
            (v[0] - 0.5) * (1.0 + c) + 0.5,
            (v[1] - 0.5) * (1.0 + c) + 0.5,
            (v[2] - 0.5) * (1.0 + c) + 0.5,
        ];
        // 饱和
        v = adjust_saturation01(v, 0.14 * strength01);
        *p = from_rgb01(v);
    }
    out
}

fn preset_portra_400(img: &RgbImage, strength01: f32) -> RgbImage {
    // 柔和对比、轻微负饱和、暖高光/青阴影分离色
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let mut v = to_rgb01(p);
        // 轻微对比
        let c = 0.10 * strength01;
        v = [
            (v[0] - 0.5) * (1.0 + c) + 0.5,
            (v[1] - 0.5) * (1.0 + c) + 0.5,
            (v[2] - 0.5) * (1.0 + c) + 0.5,
        ];
        // 负饱和
        v = adjust_saturation01(v, -0.03 * strength01);
        *p = from_rgb01(v);
    }
    // 简单高光卷曲
    out = highlight_rolloff(&out, 0.18 * strength01);
    // 轻微暗角
    out
}

// ---------------- 手动控制 ----------------

pub fn auto_baseline(img: &RgbImage) -> RgbImage {
    // 灰世界近似 + V 中位曝光归一
    let (w, h) = img.dimensions();
    let mut sum = [0f64; 3];
    for p in img.pixels() {
        sum[0] += p[0] as f64;
        sum[1] += p[1] as f64;
        sum[2] += p[2] as f64;
    }
    let n = (w as f64) * (h as f64) + 1e-6;
    let mean = (sum[0] + sum[1] + sum[2]) / 3.0;
    let gb = [
        (mean / (sum[0] / n + 1e-6)) as f32,
        (mean / (sum[1] / n + 1e-6)) as f32,
        (mean / (sum[2] / n + 1e-6)) as f32,
    ];
    let mut tmp = img.clone();
    for p in tmp.pixels_mut() {
        let v = to_rgb01(p);
        *p = from_rgb01([v[0] * gb[0], v[1] * gb[1], v[2] * gb[2]]);
    }
    // V 近似（max channel）
    let mut vals = Vec::with_capacity((w as usize) * (h as usize));
    for p in tmp.pixels() {
        let v = [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0];
        vals.push(v[0].max(v[1]).max(v[2]));
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = vals[vals.len() / 2];
    let gain = (0.5 / med).clamp(0.5, 2.0);
    let mut out = tmp.clone();
    for p in out.pixels_mut() {
        let v = to_rgb01(p);
        *p = from_rgb01([v[0] * gain, v[1] * gain, v[2] * gain]);
    }
    out
}

pub fn apply_exposure(img: &RgbImage, ev: f32) -> RgbImage {
    let gain = 2f32.powf(ev);
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let v = to_rgb01(p);
        *p = from_rgb01([v[0] * gain, v[1] * gain, v[2] * gain]);
    }
    out
}

pub fn apply_color_temp(img: &RgbImage, temp01: f32) -> RgbImage {
    if temp01.abs() < 1e-6 {
        return img.clone();
    }
    let r_gain = 1.0 + 0.4 * temp01.max(0.0) - 0.2 * (-temp01).max(0.0);
    let b_gain = 1.0 + 0.4 * (-temp01).max(0.0) - 0.2 * temp01.max(0.0);
    let g_gain = 1.0;
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let v = to_rgb01(p);
        *p = from_rgb01([v[0] * b_gain, v[1] * g_gain, v[2] * r_gain]);
    }
    out
}

pub fn adjust_contrast_mid(img: &RgbImage, amount: f32) -> RgbImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let v = to_rgb01(p);
        let o = [
            (v[0] - 0.5) * (1.0 + amount) + 0.5,
            (v[1] - 0.5) * (1.0 + amount) + 0.5,
            (v[2] - 0.5) * (1.0 + amount) + 0.5,
        ];
        *p = from_rgb01(o);
    }
    out
}

pub fn adjust_saturation(img: &RgbImage, amount: f32) -> RgbImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let rgb = Srgb::new(p[2] as f32 / 255.0, p[1] as f32 / 255.0, p[0] as f32 / 255.0);
        let mut hsv = Hsv::from_color(rgb.into_linear());
        hsv.saturation = (hsv.saturation * (1.0 + amount)).clamp(0.0, 1.0);
        let rgb2: Srgb = Srgb::from_linear(hsv.into_color());
        *p = Rgb([
            (rgb2.blue.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            (rgb2.green.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            (rgb2.red.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        ]);
    }
    out
}

pub fn apply_vibrance(img: &RgbImage, vibrance: f32) -> RgbImage {
    let k = 0.55;
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let rgb = Srgb::new(p[2] as f32 / 255.0, p[1] as f32 / 255.0, p[0] as f32 / 255.0);
        let mut hsv = Hsv::from_color(rgb.into_linear());
        let w = 1.0 - hsv.saturation;
        let s2 = (hsv.saturation + vibrance * k * w * hsv.saturation.max(0.05)).clamp(0.0, 1.0);
        hsv.saturation = s2;
        let rgb2: Srgb = Srgb::from_linear(hsv.into_color());
        *p = Rgb([
            (rgb2.blue.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            (rgb2.green.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            (rgb2.red.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        ]);
    }
    out
}

pub fn apply_highlights_shadows(img: &RgbImage, highlights: f32, shadows: f32) -> RgbImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let mut v = to_rgb01(p);
        let l = 0.114 * v[0] + 0.587 * v[1] + 0.299 * v[2];
        let hmask = ((l - 0.55) / 0.45).clamp(0.0, 1.0).powf(1.2);
        let smask = ((0.45 - l) / 0.45).clamp(0.0, 1.0).powf(1.2);
        v[0] += 0.35 * (shadows * smask + highlights * hmask);
        v[1] += 0.35 * (shadows * smask + highlights * hmask);
        v[2] += 0.35 * (shadows * smask + highlights * hmask);
        *p = from_rgb01(v);
    }
    out
}

fn adjust_saturation01(rgb: [f32; 3], amount: f32) -> [f32; 3] {
    let rgbp = Srgb::new(rgb[2], rgb[1], rgb[0]);
    let mut hsv = Hsv::from_color(rgbp.into_linear());
    hsv.saturation = (hsv.saturation * (1.0 + amount)).clamp(0.0, 1.0);
    let rgb2: Srgb = Srgb::from_linear(hsv.into_color());
    [rgb2.blue, rgb2.green, rgb2.red]
}

pub fn apply_color_splitter_8band(img: &RgbImage, sat8: &[i16; 8], lum8: &[i16; 8]) -> RgbImage {
    // 使用 8 个扇区中心（度）：0, 30, 60, 120, 180, 240, 270, 300
    let centers = [0.0f32, 30.0, 60.0, 120.0, 180.0, 240.0, 270.0, 300.0];
    let width = 40.0f32;
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let rgb = Srgb::new(p[2] as f32 / 255.0, p[1] as f32 / 255.0, p[0] as f32 / 255.0);
        let mut hsv = Hsv::from_color(rgb.into_linear());
        // hue 0..1 -> deg
        let mut h_deg = hsv.hue.into_degrees() as f32;
        if h_deg < 0.0 {
            h_deg += 360.0;
        }
        let mut total_sat = 0.0f32;
        let mut total_lum = 0.0f32;
        for i in 0..8 {
            let d = (h_deg - centers[i]).abs().min(360.0 - (h_deg - centers[i]).abs());
            let mut w = (1.0f32 - d / width).clamp(0.0, 1.0);
            w *= w;
            total_sat += w * (sat8[i].clamp(-100, 100) as f32 / 100.0);
            total_lum += w * (lum8[i].clamp(-100, 100) as f32 / 100.0);
        }
        hsv.saturation = (hsv.saturation * (1.0 + total_sat)).clamp(0.0, 1.0);
        // luminance 近似通过 V 调整：将 HSV 的 v 当作亮度
        hsv.value = (hsv.value * (1.0 + total_lum)).clamp(0.0, 1.0);
        let rgb2: Srgb = Srgb::from_linear(hsv.into_color());
        *p = Rgb([
            (rgb2.blue.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            (rgb2.green.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            (rgb2.red.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        ]);
    }
    out
}

pub fn apply_clarity(img: &RgbImage, clarity: f32) -> RgbImage {
    if clarity.abs() < 1e-6 {
        return img.clone();
    }
    // 简化版：全局 USM + 中灰掩模
    let sigma = 1.2;
    // 转为 Dynamic 方便模糊（imageproc 高斯需要单通道，这里简单用 box 模糊近似）
    let blur = box_blur_rgb(img, 3);
    let mut out = img.clone();
    for (o, b) in out.pixels_mut().zip(blur.pixels()) {
        let v = to_rgb01(o);
        let vb = to_rgb01(b);
        let detail = [v[0] - vb[0], v[1] - vb[1], v[2] - vb[2]];
        let l = (0.114 * v[0] + 0.587 * v[1] + 0.299 * v[2]) as f32;
        let midmask = (-(l - 0.5).powi(2) / (2.0 * 0.22f32.powi(2))).exp();
        let amt = clarity;
        let newv = if amt >= 0.0 {
            [
                v[0] + amt * 0.6 * detail[0] * midmask,
                v[1] + amt * 0.6 * detail[1] * midmask,
                v[2] + amt * 0.6 * detail[2] * midmask,
            ]
        } else {
            let t = (-amt * 0.6).min(1.0);
            [
                v[0] * (1.0 - t * midmask) + vb[0] * (t * midmask),
                v[1] * (1.0 - t * midmask) + vb[1] * (t * midmask),
                v[2] * (1.0 - t * midmask) + vb[2] * (t * midmask),
            ]
        };
        *o = from_rgb01(newv);
    }
    let _ = sigma; // 占位以备后续替换为高斯
    out
}

fn box_blur_rgb(img: &RgbImage, radius: u32) -> RgbImage {
    // 极简 box 模糊（两次一维积分近似），MVP 先保证观感
    let (w, h) = img.dimensions();
    let mut tmp = img.clone();
    let mut out = img.clone();
    let r = radius as i32;
    // 水平
    for y in 0..h as i32 {
        let mut acc = [0u32; 3];
        for x in -r..=r {
            let xx = x.clamp(0, w as i32 - 1) as u32;
            let p = img.get_pixel(xx, y as u32);
            acc[0] += p[0] as u32;
            acc[1] += p[1] as u32;
            acc[2] += p[2] as u32;
        }
        for x in 0..w as i32 {
            let left = (x - r - 1).clamp(0, w as i32 - 1) as u32;
            let right = (x + r).clamp(0, w as i32 - 1) as u32;
            if x > 0 {
                let pl = img.get_pixel(left, y as u32);
                let pr = img.get_pixel(right, y as u32);
                acc[0] = acc[0] - pl[0] as u32 + pr[0] as u32;
                acc[1] = acc[1] - pl[1] as u32 + pr[1] as u32;
                acc[2] = acc[2] - pl[2] as u32 + pr[2] as u32;
            }
            let n = (2 * r + 1) as u32;
            tmp.put_pixel(
                x as u32,
                y as u32,
                Rgb([(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8]),
            );
        }
    }
    // 垂直
    for x in 0..w as i32 {
        let mut acc = [0u32; 3];
        for y in -r..=r {
            let yy = y.clamp(0, h as i32 - 1) as u32;
            let p = tmp.get_pixel(x as u32, yy);
            acc[0] += p[0] as u32;
            acc[1] += p[1] as u32;
            acc[2] += p[2] as u32;
        }
        for y in 0..h as i32 {
            let top = (y - r - 1).clamp(0, h as i32 - 1) as u32;
            let bottom = (y + r).clamp(0, h as i32 - 1) as u32;
            if y > 0 {
                let pt = tmp.get_pixel(x as u32, top);
                let pb = tmp.get_pixel(x as u32, bottom);
                acc[0] = acc[0] - pt[0] as u32 + pb[0] as u32;
                acc[1] = acc[1] - pt[1] as u32 + pb[1] as u32;
                acc[2] = acc[2] - pt[2] as u32 + pb[2] as u32;
            }
            let n = (2 * r + 1) as u32;
            out.put_pixel(
                x as u32,
                y as u32,
                Rgb([(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8]),
            );
        }
    }
    out
}

// ---------------- 颗粒 ----------------

pub fn apply_grain(img: &RgbImage, opts: &ProcessOptions) -> RgbImage {
    let (w, h) = img.dimensions();
    let seed = opts.grain_seed.unwrap_or(12345);
    let mut rng = StdRng::seed_from_u64(seed);
    let density = (opts.grain_density as f32 / 100.0).clamp(0.0, 1.0);
    let chroma_mix = (opts.grain_chroma_mix as f32 / 100.0).clamp(0.0, 1.0);
    let size01 = (opts.grain_size as f32 / 100.0).clamp(0.0, 1.0);
    let rough01 = (opts.grain_roughness as f32 / 100.0).clamp(0.0, 1.0);

    // 基于下采样-上采样控制频率
    let factor = (1.0 + 24.0 * size01).round().max(1.0) as u32;
    let low_w = (w / factor).max(1);
    let low_h = (h / factor).max(1);

    // 鲁棒噪声
    let mut noise_l = vec![0f32; (low_w * low_h) as usize];
    for v in &mut noise_l {
        *v = rng.sample::<f32, _>(StandardNormal);
    }
    // 轻微模糊模拟粗糙/结团
    let blur_passes = (rough01 * 3.0).round() as u32;
    for _ in 0..blur_passes {
        box_blur_1c_inplace(&mut noise_l, low_w as usize, low_h as usize, 1);
    }
    // 上采样
    let up = resize_1c(&noise_l, low_w as usize, low_h as usize, w as usize, h as usize);

    let mut out = img.clone();
    for (i, p) in out.pixels_mut().enumerate() {
        let n = up[i] as f32;
        // 亮度通道添加
        let mut v = to_rgb01(p);
        let scale = density * 0.04; // MVP 适中
        v[0] = v[0] + scale * n;
        v[1] = v[1] + scale * n;
        v[2] = v[2] + scale * n;
        // 彩色混合
        if chroma_mix > 0.0 {
            let nb = rng.sample::<f32, _>(StandardNormal) * chroma_mix * density * 0.02;
            let ng = rng.sample::<f32, _>(StandardNormal) * chroma_mix * density * 0.02;
            let nr = rng.sample::<f32, _>(StandardNormal) * chroma_mix * density * 0.02;
            v[0] += nb;
            v[1] += ng;
            v[2] += nr;
        }
        *p = from_rgb01(v);
    }
    out
}

fn box_blur_1c_inplace(buf: &mut [f32], w: usize, h: usize, radius: usize) {
    // 一维两次 box 模糊
    let mut tmp = vec![0f32; buf.len()];
    // 水平
    for y in 0..h {
        let mut acc = 0f32;
        for x in 0..=(radius * 2) {
            let xx = x.min(w - 1);
            acc += buf[y * w + xx];
        }
        for x in 0..w {
            if x > 0 {
                let l = x.saturating_sub(radius + 1);
                let r = (x + radius).min(w - 1);
                acc += buf[y * w + r] - buf[y * w + l];
            }
            tmp[y * w + x] = acc / (2 * radius + 1) as f32;
        }
    }
    // 垂直
    for x in 0..w {
        let mut acc = 0f32;
        for y in 0..=(radius * 2) {
            let yy = y.min(h - 1);
            acc += tmp[yy * w + x];
        }
        for y in 0..h {
            if y > 0 {
                let t = y.saturating_sub(radius + 1);
                let b = (y + radius).min(h - 1);
                acc += tmp[b * w + x] - tmp[t * w + x];
            }
            buf[y * w + x] = acc / (2 * radius + 1) as f32;
        }
    }
}

fn resize_1c(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0f32; dw * dh];
    for y in 0..dh {
        let fy = y as f32 * (sh as f32 - 1.0) / (dh as f32 - 1.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = fy - y0 as f32;
        for x in 0..dw {
            let fx = x as f32 * (sw as f32 - 1.0) / (dw as f32 - 1.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = fx - x0 as f32;
            let v00 = src[y0 * sw + x0];
            let v01 = src[y0 * sw + x1];
            let v10 = src[y1 * sw + x0];
            let v11 = src[y1 * sw + x1];
            let v0 = v00 * (1.0 - tx) + v01 * tx;
            let v1 = v10 * (1.0 - tx) + v11 * tx;
            out[y * dw + x] = v0 * (1.0 - ty) + v1 * ty;
        }
    }
    out
}

// ---------------- 暗角 ----------------

pub fn apply_vignette(img: &RgbImage, strength01: f32) -> RgbImage {
    if strength01 <= 0.0 {
        return img.clone();
    }
    let (w, h) = img.dimensions();
    let edge_dark = 0.35 + 0.57 * strength01;
    let power = 1.35 + 1.1 * strength01;
    let mut out = img.clone();
    for y in 0..h {
        for x in 0..w {
            let cy = (h as f32 - 1.0) * 0.5;
            let cx = (w as f32 - 1.0) * 0.5;
            let ry = (h as f32 * 0.5).max(1.0);
            let rx = (w as f32 * 0.5).max(1.0);
            let dy = (y as f32 - cy) / ry;
            let dx = (x as f32 - cx) / rx;
            let dist = (dx * dx + dy * dy).sqrt().clamp(0.0, 1.0);
            let atten = 1.0 - edge_dark * dist.powf(power);
            let p = out.get_pixel_mut(x, y);
            let v = to_rgb01(p);
            *p = from_rgb01([v[0] * atten, v[1] * atten, v[2] * atten]);
        }
    }
    out
}

// ---------------- 特效（MVP 可见版） ----------------

pub fn fx_expired_film(img: &RgbImage, seed: u64, t: f32) -> RgbImage {
    if t <= 1e-6 {
        return img.clone();
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = img.clone();
    // 基底雾化 + 轻微对比损失
    let fog = 0.05 + 0.14 * t;
    out = adjust_contrast_mid(&out, -0.25 * t);
    for p in out.pixels_mut() {
        let v = to_rgb01(p);
        let mut v2 = [v[0] * (1.0 - 0.55 * t) + fog, v[1] * (1.0 - 0.55 * t) + fog, v[2] * (1.0 - 0.55 * t) + fog];
        v2 = [1.0 - (1.0 - v2[0]) * (1.0 - 0.08 * t), 1.0 - (1.0 - v2[1]) * (1.0 - 0.08 * t), 1.0 - (1.0 - v2[2]) * (1.0 - 0.08 * t)];
        *p = from_rgb01(v2);
    }
    // 随机色偏
    let warm = rng.gen_bool(0.5);
    let cast = if warm { [0.00, 0.05 + 0.12 * t, 0.08 * t] } else { [0.04 + 0.10 * t, 0.03 * t, 0.00] };
    for p in out.pixels_mut() {
        let v = to_rgb01(p);
        *p = from_rgb01([v[0] + cast[0], v[1] + cast[1], v[2] + cast[2]]);
    }
    out
}

pub fn fx_light_leak(img: &RgbImage, seed: u64, t: f32) -> RgbImage {
    if t <= 1e-6 {
        return img.clone();
    }
    let (w, h) = img.dimensions();
    let mut rng = StdRng::seed_from_u64(seed);
    // 选择 1-2 个角
    let corners = [(0.0, 0.0), (w as f32 - 1.0, 0.0), (0.0, h as f32 - 1.0), (w as f32 - 1.0, h as f32 - 1.0)];
    let count = if rng.gen_bool(0.5) { 1 } else { 2 };
    let picks = {
        let mut v = vec![0usize, 1, 2, 3];
        v.shuffle(&mut rng);
        v.truncate(count);
        v
    };
    let mut mask = vec![0f32; (w * h) as usize];
    for idx in picks {
        let (cx, cy) = corners[idx];
        for y in 0..h {
            for x in 0..w {
                let dx = (x as f32 - cx).abs() / (w as f32).max(1.0);
                let dy = (y as f32 - cy).abs() / (h as f32).max(1.0);
                let d = (dx * dx + dy * dy).sqrt();
                let pwr = 2.6 + rng.gen::<f32>() * 0.9;
                let v = (1.0 - d.powf(pwr)).clamp(0.0, 1.0).powf(3.0);
                let i = (y * w + x) as usize;
                mask[i] = mask[i].max(v);
            }
        }
    }
    // 颜色从橙/琥珀到洋红/红
    let r1: f32 = rng.gen_range(0.25..0.85);
    let r2: f32 = rng.gen_range(0.25..0.95);
    let orange = [0.03, 0.22, 0.56];
    let amber = [0.05, 0.30, 0.50];
    let magenta_red = [0.14, 0.06, 0.62];
    let warm = [
        orange[0] * (1.0 - r1) + amber[0] * r1,
        orange[1] * (1.0 - r1) + amber[1] * r1,
        orange[2] * (1.0 - r1) + amber[2] * r1,
    ];
    let color = [
        warm[0] * (1.0 - r2) + magenta_red[0] * r2,
        warm[1] * (1.0 - r2) + magenta_red[1] * r2,
        warm[2] * (1.0 - r2) + magenta_red[2] * r2,
    ];
    let k = 0.28 + 0.60 * t;
    let mut out = img.clone();
    for (i, p) in out.pixels_mut().enumerate() {
        let m = mask[i].clamp(0.0, 1.0).powf(0.95);
        let add = [k * m * color[0], k * m * color[1], k * m * color[2]];
        let v = to_rgb01(p);
        let scr = [1.0 - (1.0 - v[0]) * (1.0 - add[0]), 1.0 - (1.0 - v[1]) * (1.0 - add[1]), 1.0 - (1.0 - v[2]) * (1.0 - add[2])];
        *p = from_rgb01(scr);
    }
    out
}

// ---------------- I/O ----------------

pub enum InputImage {
    Rgb8(RgbImage),
}

pub fn load_image_bgr_or_rgb8(path: &str) -> Result<InputImage> {
    let dyn_img = image::open(path).with_context(|| format!("无法打开图像: {}", path))?;
    let rgb8 = dyn_to_rgb8(&dyn_img);
    Ok(InputImage::Rgb8(rgb8))
}

#[cfg(feature = "nef")]
pub fn load_nef_rgb8(_path: &str) -> Result<RgbImage> {
    // TODO: 使用 rawloader 解码 NEF 到 8bit sRGB。MVP 暂未实现。
    bail!("NEF 解码尚未在 MVP 中实现（请使用 JPG/JPEG 测试）");
}

fn dyn_to_rgb8(img: &DynamicImage) -> RgbImage {
    img.to_rgb8()
}

// ---------------- 其他辅助 ----------------

fn highlight_rolloff(img: &RgbImage, amount: f32) -> RgbImage {
    if amount <= 1e-6 {
        return img.clone();
    }
    let t = 0.7f32;
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let v = to_rgb01(p);
        let mut y = [v[0].max(t) - t, v[1].max(t) - t, v[2].max(t) - t];
        y[0] = 1.0 - (1.0 - y[0] / (1.0 - t)).powf(1.0 + 2.5 * amount);
        y[1] = 1.0 - (1.0 - y[1] / (1.0 - t)).powf(1.0 + 2.5 * amount);
        y[2] = 1.0 - (1.0 - y[2] / (1.0 - t)).powf(1.0 + 2.5 * amount);
        let low = [v[0].min(t), v[1].min(t), v[2].min(t)];
        let newv = [low[0] + (1.0 - t) * y[0], low[1] + (1.0 - t) * y[1], low[2] + (1.0 - t) * y[2]];
        *p = from_rgb01(newv);
    }
    out
}

// ---------------- 测试 ----------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_test_img(w: u32, h: u32, c: u8) -> RgbImage {
        ImageBuffer::from_fn(w, h, |_x, _y| Rgb([c, c, c]))
    }

    #[test]
    fn shape_preserved() {
        let img = make_test_img(640, 480, 128);
        let opts = ProcessOptions::default();
        let out = process_rgb8(&img, &opts);
        assert_eq!(out.dimensions(), img.dimensions());
    }

    #[test]
    fn manual_exposure_changes_brightness() {
        let img = make_test_img(64, 64, 64);
        let mut opts = ProcessOptions::default();
        opts.exposure_ev_x100 = 100; // +1 EV
        let out = process_rgb8(&img, &opts);
        assert!(out.get_pixel(0, 0)[0] > img.get_pixel(0, 0)[0]);
    }

    #[test]
    fn vibrance_not_affecting_grayscale_heavily() {
        let img = make_test_img(64, 64, 128);
        let mut opts = ProcessOptions::default();
        opts.vibrance = 100;
        let out = process_rgb8(&img, &opts);
        // 灰阶几乎不变
        assert!((out.get_pixel(0, 0)[0] as i32 - 128).abs() <= 2);
    }
}

