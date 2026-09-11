# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Collect all data/binaries/hiddenimports for critical packages
datas = []
binaries = []
hiddenimports = []
for pkg in ("PySide6", "cv2", "rawpy"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
# Ensure our application package and its submodules are included
hiddenimports += collect_submodules("nikon_film_app")

block_cipher = None

app_name = "NikonFilmLab"
# Resolve repo root relative to this spec file (which lives in packaging/)
# Note: In PyInstaller spec execution, __file__ is NOT defined; use SPECPATH.
SPEC_DIR = Path(SPECPATH).resolve()
REPO_ROOT = SPEC_DIR.parent
entry_script = str(REPO_ROOT / "src" / "nikon_film_app" / "main.py")
assets_dir = REPO_ROOT / "assets"
assets_dir.mkdir(exist_ok=True)
icon_png = assets_dir / "app_icon.png"
icon_ico = assets_dir / "app_icon.ico"
datas.append((str(icon_png), "assets"))

a = Analysis(
    [entry_script],
    pathex=[str(REPO_ROOT), str(REPO_ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=(str(icon_ico) if icon_ico.exists() else None),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=app_name,
)

