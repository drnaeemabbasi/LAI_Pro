# Building the standalone laipro app (no Python for end users)

This builds a Windows desktop app your colleagues run **without installing Python**. They get a folder with `laipro.exe`; double-clicking it opens the napari GUI.

## Honest expectations

- **CPU-only.** The bundle excludes PyTorch/CUDA on purpose. The whole DHP pipeline and the default RandomForest segmenter run fine on CPU, so end users need no GPU and the bundle is much smaller and more reliable. Colleagues who want the optional GPU MLP/SAM extras install from source instead (see the main README).
- **Freezing napari is finicky.** napari uses Qt + OpenGL (vispy) and lots of dynamic plugin loading, which PyInstaller doesn't always trace automatically. Expect to possibly iterate once or twice on a real machine (see Troubleshooting).
- **Build on the target OS.** A Windows app must be built on Windows. The result is ~1–2 GB (napari + scientific stack).

## Build steps (Windows, in the project venv)

```powershell
# from the project root, with the venv active
.\packaging\build.ps1
```

or manually:

```powershell
pip install -e ".[gui]"
pip install pyinstaller
pyinstaller packaging/laipro.spec --noconfirm --clean
```

Output: `dist/laipro/laipro.exe`. **Ship the entire `dist/laipro` folder** (zip it). Colleagues unzip and run `laipro.exe`.

## Test it before sharing

1. On the build machine: run `dist/laipro/laipro.exe` — the GUI should open.
2. Better: copy the `dist/laipro` folder to a **clean machine without Python** and run it there. That's the real test that nothing external is required.

## Troubleshooting

- **"Failed to execute script" / blank window / OpenGL error on start:** usually a missing napari/vispy resource. Rebuild after adding the missing package name to `collect_pkgs` in `laipro.spec`.
- **`ModuleNotFoundError` at runtime:** add that module to `hiddenimports` in `laipro.spec`.
- **rawpy / RAW files fail:** ensure `rawpy` is in `collect_pkgs` (it is) so its compiled `libraw` is bundled.
- **App starts but a plugin errors:** run the exe from a console (`laipro.exe gui`) to see the traceback, then map the missing piece as above.

## Fallback if freezing proves too troublesome

If the napari freeze fights back, the reliable alternative that still needs **no manual Python steps** from colleagues is a one-shot installer script that provisions a private environment and creates a desktop shortcut. Ask and this can be added as `packaging/install.ps1`.
