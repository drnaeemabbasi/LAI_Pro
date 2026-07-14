# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the standalone (no-Python) laipro desktop app.

CPU-ONLY by design: torch/CUDA are excluded. The full DHP pipeline and the
default RandomForest segmenter run on CPU, so end users need no GPU and the
bundle stays far smaller and more reliable. (The optional MLP/SAM extras are
not available in the bundle; those users install from source.)

Build (on a real Windows machine, in the project venv):
    pip install pyinstaller
    pyinstaller packaging/laipro.spec --noconfirm --clean
Result: dist/laipro/laipro.exe  (ship the whole dist/laipro folder)

Note: freezing napari (Qt + OpenGL/vispy) is finicky. If the built app fails to
start, the usual fixes are adding the offending package to `collect_pkgs`
below, or adding a missing module to `hiddenimports`. See packaging/README.md.
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# Packages that ship data files / plugins / dynamic imports and must be fully collected.
collect_pkgs = [
    "napari", "vispy", "magicgui", "superqt", "app_model", "in_n_out",
    "psygnal", "napari_svg", "napari_builtins", "imageio", "skimage",
    "sklearn", "scipy", "rawpy", "matplotlib", "PIL", "pandas", "laipro",
]
for pkg in collect_pkgs:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

hiddenimports += collect_submodules("napari")
hiddenimports += collect_submodules("skimage")

a = Analysis(
    ["laipro_app.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=["torch", "torchvision", "tensorflow", "PyQt5", "PySide2", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="laipro",
    console=False,        # windowed GUI app (no console)
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="laipro")
