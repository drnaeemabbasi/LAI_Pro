"""
Unified `laipro` command-line interface.

One command with subcommands replaces the old scattered scripts:

  laipro process     FOLDER   [options]     # full DHP: FCOVER, PAI, ALA, clumping, FAPAR
  laipro prepare     FOLDER   [--target]    # decode-once downscaled cache (speed)
  laipro gui                                # napari desktop app
  laipro train-seg   --model D --pairs ...  # train the learned segmenter
  laipro train-mask  --model D --pairs ...  # train the learned foreign-object mask
  laipro calibrate   --mode auto|measure    # fish-eye lens calibration

Heavy optional deps (napari, torch) are imported lazily inside the relevant
handlers, so `laipro --help` and `laipro process` stay light.
"""

from __future__ import annotations
import argparse
import os


def _progress_bar():
    """Return a progress callback backed by tqdm, or None."""
    try:
        from tqdm import tqdm
    except Exception:
        return None, None
    state = {"bar": None}

    def cb(i, n, name):
        if state["bar"] is None:
            state["bar"] = tqdm(total=n, desc="Processing", unit="img")
        state["bar"].update(1); state["bar"].set_postfix_str(name)
    return cb, state


# ----------------------------------------------------------------- handlers

def cmd_process(a):
    from .config import DHPConfig
    from .pipeline import process_folder, build_calibration
    from .learned_segment import LearnedSegmenter
    from .masking import ForeignObjectDetector

    calib = build_calibration(a.input, a.calibration, a.auto_calibrate, a.diagonal_fov, a.max_side)
    cfg = DHPConfig(veg_index=a.veg_index, coi_deg=a.coi, fcover_cone_deg=a.fcover_cone,
                    latitude_deg=a.latitude, day_of_year=a.doy, max_side=a.max_side,
                    calibration=calib)
    seg = LearnedSegmenter.load(a.model) if a.model else None
    if seg:
        print(f"Using learned segmenter ({seg.backend}) from {a.model}")
    mask = ForeignObjectDetector.load(a.mask_model) if a.mask_model else None
    if mask:
        print(f"Using learned foreign-object mask from {a.mask_model}")

    cb, state = _progress_bar()
    process_folder(a.input, cfg, segmenter=seg, mask_detector=mask,
                   make_report=not a.no_report, make_qc_panels=not a.no_qc_panels,
                   use_cache=not a.no_cache, progress=cb)
    if state and state["bar"] is not None:
        state["bar"].close()


def cmd_prepare(a):
    from .preprocess import prepare_folder
    cb, state = _progress_bar()
    print(f"Preparing '{a.input}' at {a.target}px (decoding RAW once)...")
    s = prepare_folder(a.input, target_side=a.target, force=a.force, progress=cb)
    if state and state["bar"] is not None:
        state["bar"].close()
    print(f"Cache ready: {s['converted']} converted, {s['reused']} reused, "
          f"{s['n_images']} total -> {s['cache_dir']}")


def cmd_gui(a):
    from .gui import launch
    launch(a.input)


def cmd_train_seg(a):
    import numpy as np
    from PIL import Image
    from .io import load_rgb
    from .learned_segment import LearnedSegmenter, CorrectionStore
    store = CorrectionStore(a.model)
    for pair in a.pairs:
        img, lab = pair.split("=", 1)
        rgb = load_rgb(img, a.max_side)
        L = np.asarray(Image.open(lab))
        if L.shape[:2] != rgb.shape[:2]:
            raise SystemExit(f"label {lab} {L.shape[:2]} != image {rgb.shape[:2]} (match --max-side)")
        n = store.add(rgb, L, source_name=img, context_window=a.context_window)
        print(f"added {img}: cache now {n} labeled pixels")
    X, y, contribs = store.load_samples()
    seg = LearnedSegmenter(backend=a.backend, context_window=a.context_window).fit(X, y, contributing=contribs)
    seg.save(a.model)
    print(f"trained {a.backend} on {len(y)} px -> {a.model}")


def cmd_train_mask(a):
    import json
    import numpy as np
    from .io import load_rgb
    from .masking import ForeignObjectDetector, MaskCorrectionStore
    det = ForeignObjectDetector(context_window=a.context_window)
    store = MaskCorrectionStore(a.model)
    rng = np.random.default_rng(0)
    for pair in a.pairs:
        img, poly = pair.split("=", 1)
        rgb = load_rgb(img, a.max_side)
        with open(poly) as fh:
            polys = json.load(fh)["polygons"]
        n = store.add(rgb, polys, img, det, rng=rng)
        print(f"added {img}: cache now {n} samples")
    X, y, contribs = store.load_samples()
    det.fit(X, y, contributing=contribs)
    det.save(a.model)
    print(f"trained mask detector on {len(y)} px -> {a.model}")


def cmd_calibrate(a):
    import numpy as np
    from .io import load_rgb
    from .calibrate import (CalibrationProfile, fit_optical_center, fit_projection,
                            auto_profile, detect_image_circle, is_full_frame)
    if a.mode == "auto":
        prof = auto_profile(load_rgb(a.image, a.max_side), diagonal_fov_deg=a.fov, kind=a.kind)
        prof.save(a.out)
        print(f"[auto] {prof.method} center={tuple(round(v,1) for v in prof.center_xy)} "
              f"deg/px={prof.diagnostics['deg_per_pixel']:.5f} -> {a.out}")
    else:  # measure
        import pandas as pd
        hdf = pd.read_csv(a.holes)
        holes = {str(h): list(zip(g["x"], g["y"])) for h, g in hdf.groupby("hole")}
        center, cdiag = fit_optical_center(holes)
        pdf = pd.read_csv(a.projection)
        coeffs, pdiag = fit_projection(pdf["radius_px"].values, pdf["angle_deg"].values, a.degree)
        rgb = load_rgb(a.image, a.max_side)
        H, W = rgb.shape[:2]
        r_edge = 0.5 * float(np.hypot(W, H)) if is_full_frame(rgb) else detect_image_circle(rgb)[2]
        max_fov = 2.0 * float(sum(c * r_edge ** i for i, c in enumerate(coeffs)))
        prof = CalibrationProfile(center_xy=center, proj_coeffs=coeffs, max_fov_deg=max_fov,
                                  image_shape=(H, W), method="measured",
                                  diagnostics={"optical_center": cdiag, "projection": pdiag})
        prof.save(a.out)
        print(f"[measure] center_spread={cdiag['center_spread_px']:.2f}px "
              f"proj_r2={pdiag['r2']:.4f} maxFOV={max_fov:.1f} -> {a.out}")


# ----------------------------------------------------------------- parser

def build_parser():
    p = argparse.ArgumentParser(prog="laipro",
                                description="Reproducible DHP leaf-area (PAI/LAI/FCOVER) processing.")
    p.add_argument("--version", action="store_true", help="print version and exit")
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("process", help="Full DHP processing of a plot folder")
    pr.add_argument("input", help="Plot folder of fish-eye images")
    pr.add_argument("--veg-index", default="exg", choices=["exg", "exgr"])
    pr.add_argument("--coi", type=float, default=60.0)
    pr.add_argument("--fcover-cone", type=float, default=10.0)
    pr.add_argument("--diagonal-fov", type=float, default=180.0)
    pr.add_argument("--latitude", type=float, default=43.0, help="Site latitude for FAPAR")
    pr.add_argument("--doy", type=int, default=167, help="Day of year for FAPAR")
    pr.add_argument("--max-side", type=int, default=2500)
    pr.add_argument("--calibration", default=None, help="Lens profile JSON (from `laipro calibrate`)")
    pr.add_argument("--auto-calibrate", action="store_true")
    pr.add_argument("--model", default=None, help="Learned segmenter dir")
    pr.add_argument("--mask-model", default=None, help="Learned foreign-object mask dir")
    pr.add_argument("--no-report", action="store_true")
    pr.add_argument("--no-qc-panels", action="store_true")
    pr.add_argument("--no-cache", action="store_true")
    pr.set_defaults(func=cmd_process)

    pp = sub.add_parser("prepare", help="Decode-once downscaled cache (speeds up processing)")
    pp.add_argument("input", help="Plot folder")
    pp.add_argument("--target", type=int, default=2500, help="Working resolution, long side (px)")
    pp.add_argument("--force", action="store_true")
    pp.set_defaults(func=cmd_prepare)

    pg = sub.add_parser("gui", help="Launch the napari desktop app")
    pg.add_argument("--input", default=None, help="Plot folder to open on launch")
    pg.set_defaults(func=cmd_gui)

    ps = sub.add_parser("train-seg", help="Train/update the learned vegetation segmenter")
    ps.add_argument("--model", required=True)
    ps.add_argument("--pairs", nargs="+", required=True, help="image=label.png (0 ignore,1 soil,2 veg)")
    ps.add_argument("--backend", default="rf", choices=["rf", "mlp"])
    ps.add_argument("--context-window", type=int, default=9)
    ps.add_argument("--max-side", type=int, default=2500)
    ps.set_defaults(func=cmd_train_seg)

    pm = sub.add_parser("train-mask", help="Train/update the learned foreign-object detector")
    pm.add_argument("--model", required=True)
    pm.add_argument("--pairs", nargs="+", required=True, help="image=polygons.json")
    pm.add_argument("--context-window", type=int, default=9)
    pm.add_argument("--max-side", type=int, default=2500)
    pm.set_defaults(func=cmd_train_mask)

    pc = sub.add_parser("calibrate", help="Fish-eye lens calibration profile")
    pc.add_argument("--mode", default="auto", choices=["auto", "measure"])
    pc.add_argument("--out", required=True)
    pc.add_argument("--image", default=None, help="An image (required for auto; sets shape/FOV for measure)")
    pc.add_argument("--fov", type=float, default=180.0)
    pc.add_argument("--kind", default="auto", choices=["auto", "fullframe", "circular"])
    pc.add_argument("--holes", default=None, help="measure: CSV hole,x,y")
    pc.add_argument("--projection", default=None, help="measure: CSV radius_px,angle_deg")
    pc.add_argument("--degree", type=int, default=3, choices=[1, 2, 3])
    pc.add_argument("--max-side", type=int, default=2000)
    pc.set_defaults(func=cmd_calibrate)
    return p


def main(argv=None):
    import laipro
    p = build_parser()
    a = p.parse_args(argv)
    if getattr(a, "version", False):
        print(f"laipro {laipro.__version__}")
        return
    if not getattr(a, "cmd", None):
        p.print_help()
        return
    a.func(a)


if __name__ == "__main__":
    main()
