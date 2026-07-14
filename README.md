# laipro

**Reproducible leaf-area index (PAI / LAI / FCOVER) processing from downward-looking fish-eye (DHP) crop photographs.**

An open, Python replacement for the manual steps in CAN_EYE. It computes, from a folder of level, straight-down fish-eye photos of a plot:

- **FCOVER** — fractional vegetation cover (nadir)
- **PAI** — plant area index (effective and clumping-corrected "true")
- **ALA** — average leaf inclination angle
- **Clumping index**, **LAI57**, and **FAPAR** (black-sky + white-sky)

Everything is built on published radiative-transfer methods (Poisson gap-fraction, ellipsoidal leaf-angle distribution, Lang & Yueqin clumping), and every run records its configuration and input file hashes so results are exactly reproducible.

---

## Quick start (command line)

```bash
laipro process  path/to/PlotFolder  --latitude 49.9 --doy 181
```

Outputs land in `PlotFolder/laipro_results/`:

| File | What |
|---|---|
| `report.html` | Self-contained visual report (open in any browser) |
| `results.json` | All plot-level variables |
| `per_image.csv` | Per-image FCOVER, vegetation %, LAI57 |
| `qc/*.png` | Per-image 4-panel diagnostics |
| `provenance.json` | Config + input hashes (reproducibility) |

Speed up repeated runs by decoding the RAW files once into a downscaled cache:

```bash
laipro prepare  path/to/PlotFolder          # run once
laipro process  path/to/PlotFolder          # now uses the cache automatically
```

Desktop app (paint corrections, draw masks, review, process):

```bash
laipro gui
```

All commands: `laipro --help`, or `laipro <command> --help`.

---

## Installation (developers / from source)

Requires Python 3.10+. On Windows:

```powershell
git clone <repo-url> laipro
cd laipro
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .            # core CLI
pip install -e ".[gui]"     # add the napari desktop app
```

### Optional GPU (only for the learned MLP segmenter / SAM)

The default RandomForest segmenter and the whole DHP pipeline run on CPU — **no GPU needed**. The GPU only accelerates the optional `--backend mlp` segmenter and SAM masking.

To enable it, install the CUDA build of PyTorch that matches your Python. **Note:** on Python 3.14 the CUDA wheels are on the `cu128` index (the default `cu124` index has no 3.14 wheels):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # True when the GPU is usable
```

You do **not** need to install the CUDA Toolkit — only an up-to-date NVIDIA driver.

---

## Commands

| Command | Purpose |
|---|---|
| `laipro process FOLDER` | Full DHP processing → report + CSVs |
| `laipro prepare FOLDER` | Decode-once downscaled cache (speed) |
| `laipro gui` | napari desktop app |
| `laipro train-seg` | Train the learned vegetation segmenter from painted labels |
| `laipro train-mask` | Train the learned operator/boot mask from polygons |
| `laipro calibrate` | Build a fish-eye lens calibration profile |

Common `process` options: `--latitude`/`--doy` (needed for correct FAPAR), `--max-side` (working resolution, default 2500), `--model`/`--mask-model` (use trained models), `--no-report`/`--no-qc-panels` (faster bulk runs).

---

## Notes on accuracy

- Results are stable between ~1800 and 3000 px working resolution; **2500 px (default) is the sweet spot** and equivalent to full resolution. Use one resolution consistently across plots for comparable results.
- Outputs are **PAI**, not pure LAI (photos cannot separate leaves from stems). This matches how satellite "LAI" products behave, so they are directly comparable.
- The engine is validated by synthetic self-recovery; it is **not yet field-validated** against destructive LAI. Treat absolute values as good optical estimates and excellent for relative comparison (plots, dates, treatments).

---

## Building the standalone app

See [`packaging/README.md`](packaging/README.md) to build the one-click Windows executable (no Python required for end users) with PyInstaller.

---

## License

MIT.
