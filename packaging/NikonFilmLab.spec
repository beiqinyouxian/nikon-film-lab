# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all

# Collect all data/binaries/hiddenimports for critical packages
datas = []
binaries = []
hiddenimports = []
for pkg in ("PySide6", "cv2", "rawpy"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

block_cipher = None

app_name = "NikonFilmLab"
entry_script = os.path.join("src", "nikon_film_app", "main.py")

a = Analysis(
    [entry_script],
    pathex=["."],
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
    icon=None,
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

