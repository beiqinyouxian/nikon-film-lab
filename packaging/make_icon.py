from __future__ import annotations

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter


def generate_icon_png(dst: Path, size: int = 256) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (size, size), (30, 30, 30, 255))
    draw = ImageDraw.Draw(img, "RGBA")
    # Rounded square base
    r = int(size * 0.18)
    draw.rounded_rectangle([r, r, size - r, size - r], radius=int(size * 0.08), fill=(38, 38, 38, 255))
    # Warm glow center
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow, "RGBA")
    cx, cy = size // 2, size // 2
    for i in range(180, 0, -1):
        alpha = int(220 * (i / 180) ** 2)
        col = (255, 165, 66, alpha)
        gdraw.ellipse([cx - i, cy - i, cx + i, cy + i], fill=col)
    glow = glow.filter(ImageFilter.GaussianBlur(radius=size * 0.06))
    img.alpha_composite(glow)
    # Simple aperture symbol
    ap = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    apd = ImageDraw.Draw(ap, "RGBA")
    apd.ellipse([cx - size * 0.14, cy - size * 0.14, cx + size * 0.14, cy + size * 0.14], fill=(32, 32, 32, 255))
    img.alpha_composite(ap)
    img.save(dst, "PNG")


def convert_to_ico(png_path: Path, ico_path: Path) -> None:
    img = Image.open(png_path).convert("RGBA")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    img.save(ico_path, format="ICO", sizes=[(s, s) for s in sizes])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assets = repo_root / "assets"
    assets.mkdir(exist_ok=True)
    png = assets / "app_icon.png"
    ico = assets / "app_icon.ico"
    if not png.exists():
        generate_icon_png(png, size=256)
    convert_to_ico(png, ico)


if __name__ == "__main__":
    main()

