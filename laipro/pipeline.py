"""
Plot processing pipeline: the top-level orchestration that turns a folder of
fish-eye photos into the full DHP variable set + outputs.

This is the single entry point used by both the CLI (laipro.cli) and the GUI
(laipro.gui_controller), so there is one code path and one place to maintain.
"""

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

from .config import DHPConfig, FisheyeCalibration, write_provenance
from .io import load_rgb, find_images
from .geometry import zenith_azimuth_maps, coi_mask
from .segment import segment_vegetation
from .gapfraction import ring_gap_fraction, fcover_from_cone, gap_at_angle
from .biophysical import lai57_from_gap, invert_gap_fraction
from .clumping import clumping_index
from .fapar import compute_fapar
from .calibrate import CalibrationProfile, auto_profile
from .report import build_report, qc_overlay_uri, save_qc_panels
from .preprocess import cached_images


def build_calibration(folder, calibration_path=None, auto_calibrate=False,
                      diagonal_fov=180.0, max_side=2500):
    """Resolve a FisheyeCalibration from CLI/GUI options (saved profile,
    auto-detect from the first image, or the zero-config default)."""
    if calibration_path:
        return CalibrationProfile.load(calibration_path).to_fisheye_calibration()
    if auto_calibrate:
        first = find_images(folder)[0]
        prof = auto_profile(load_rgb(first, max_side), diagonal_fov_deg=diagonal_fov)
        os.makedirs(os.path.join(folder, "laipro_results"), exist_ok=True)
        prof.save(os.path.join(folder, "laipro_results", "lens_auto.json"))
        return prof.to_fisheye_calibration()
    return FisheyeCalibration(mode="fullframe180", diagonal_fov_deg=diagonal_fov)


def process_folder(folder: str, cfg: DHPConfig, segmenter=None, mask_detector=None,
                   make_report=True, make_qc_panels=True, use_cache=True, progress=None):
    """Process a plot folder. `progress`, if given, is called as
    progress(i, n, name) before each image is processed (for progress bars)."""
    out_dir = os.path.join(folder, "laipro_results")
    os.makedirs(out_dir, exist_ok=True)
    qc_dir = os.path.join(out_dir, "qc")
    if make_qc_panels:
        os.makedirs(qc_dir, exist_ok=True)

    files = (cached_images(folder, cfg.max_side) if use_cache else None) or find_images(folder)
    if not files:
        raise SystemExit(f"No images found in {folder}")
    if use_cache and cached_images(folder, cfg.max_side):
        print(f"Using preprocessed cache ({len(files)} images at {cfg.max_side}px)")

    rows = []
    ring_table = {}
    overlays = []
    sum_po_w = None
    sum_w = None
    clump_vals, clump_w, sat_fracs = [], [], []

    for i, f in enumerate(files):
        if progress:
            progress(i, len(files), os.path.basename(f))
        rgb = load_rgb(f, cfg.max_side)
        theta, azimuth = zenith_azimuth_maps(rgb.shape[:2], cfg.calibration)
        valid = ~coi_mask(theta, cfg.coi_deg)
        obj = np.zeros(rgb.shape[:2], bool)
        obj_frac = 0.0
        if mask_detector is not None:
            obj = mask_detector.predict(rgb)
            valid = valid & ~obj
            obj_frac = float(obj.mean())
        # always compute the classical index + threshold (cheap) so the QC
        # histogram is available even when a learned model makes the decision.
        veg_cls, t_hist, idx = segment_vegetation(rgb, cfg.veg_index, cfg.manual_thresh, valid)
        if segmenter is not None:
            veg, t = segmenter.predict(rgb), float("nan")
        else:
            veg, t = veg_cls, t_hist

        centers, po, counts = ring_gap_fraction(
            veg, theta, valid, cfg.coi_deg, cfg.zenith_res_deg)
        fcover, ncone = fcover_from_cone(veg, theta, valid, cfg.fcover_cone_deg)
        n_analyzed = int(valid.sum())
        veg_pct_coi = 100.0 * int((veg & valid).sum()) / n_analyzed if n_analyzed else float("nan")
        po57 = gap_at_angle(centers, po, cfg.lai57_zenith_deg)
        lai57 = lai57_from_gap(po57)

        cl = clumping_index(veg, theta, azimuth, valid, cfg.coi_deg, cfg.zenith_res_deg,
                            cfg.n_azimuth_cells, cfg.min_cell_px, cfg.pai_sat)
        if np.isfinite(cl.omega):
            clump_vals.append(cl.omega); clump_w.append(int(np.nansum(counts)))
        sat_fracs.append(cl.saturated_frac)

        w = np.where(np.isfinite(po), counts.astype(float), 0.0)
        contrib = np.where(np.isfinite(po), po, 0.0) * w
        sum_po_w = contrib if sum_po_w is None else sum_po_w + contrib
        sum_w = w if sum_w is None else sum_w + w

        name = os.path.basename(f)
        ring_table[name] = po
        panel_title = f"FCOVER(nadir)={fcover*100:.1f}%  masked={obj_frac*100:.1f}%"
        if make_report:
            overlays.append((name, qc_overlay_uri(
                rgb, veg, valid, obj, panel_title, veg_pct=veg_pct_coi)))
        if make_qc_panels:
            save_qc_panels(rgb, veg, valid, obj, idx, t_hist,
                           os.path.join(qc_dir, name + "_qc.png"),
                           panel_title, veg_pct=veg_pct_coi)
        rows.append(dict(image=name, threshold=round(t, 4),
                         FCOVER=round(fcover, 4), FCOVER_pct=round(fcover * 100, 1),
                         veg_pct_COI=round(veg_pct_coi, 1),
                         Po57=round(po57, 4), LAI57=round(lai57, 3), cone_px=ncone,
                         masked_pct=round(obj_frac * 100, 2)))
        print(f"{name:20s} veg={veg_pct_coi:4.1f}%  FCOVER(nadir)={fcover*100:5.1f}%  "
              f"Po(57.5)={po57:5.3f}  LAI57={lai57:5.3f}  masked={obj_frac*100:4.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "per_image.csv"), index=False)
    pd.DataFrame(ring_table, index=np.round(centers, 2)).rename_axis("zenith_deg") \
        .to_csv(os.path.join(out_dir, "gap_fraction_by_ring.csv"))

    po_avg = np.divide(sum_po_w, sum_w, out=np.full_like(sum_po_w, np.nan), where=sum_w > 0)
    inv = invert_gap_fraction(centers, po_avg, sum_w)
    omega = float(np.average(clump_vals, weights=clump_w)) if clump_vals else float("nan")
    pai_true = inv.pai_eff / omega if (np.isfinite(inv.pai_eff) and np.isfinite(omega) and omega > 0) else float("nan")
    fap = compute_fapar(inv.pai_eff, inv.x, cfg.latitude_deg, cfg.day_of_year, cfg.coi_deg)

    results = {
        "n_images": len(df),
        "mean_FCOVER_pct": round(float(df["FCOVER_pct"].mean()), 2),
        "std_FCOVER_pct": round(float(df["FCOVER_pct"].std(ddof=1)), 2) if len(df) > 1 else None,
        "mean_LAI57": round(float(pd.to_numeric(df["LAI57"], errors="coerce").dropna().mean()), 3),
        "PAI_effective": round(inv.pai_eff, 3),
        "ALA_effective_deg": round(inv.ala_eff, 1),
        "ellipsoidal_x": round(inv.x, 3),
        "inversion_rmse_gap": round(inv.rmse, 4),
        "clumping_index": round(omega, 3) if np.isfinite(omega) else None,
        "PAI_true": round(pai_true, 3) if np.isfinite(pai_true) else None,
        "saturated_cell_frac": round(float(np.mean(sat_fracs)), 4),
        "FAPAR_black_sky_daily": round(fap.daily_black_sky, 3) if np.isfinite(fap.daily_black_sky) else None,
        "FAPAR_white_sky": round(fap.white_sky, 3) if np.isfinite(fap.white_sky) else None,
        "latitude_deg": cfg.latitude_deg, "day_of_year": cfg.day_of_year,
        "segmenter": ("learned:" + segmenter.backend) if segmenter is not None else f"classical:{cfg.veg_index}",
        "mask_detector": "learned" if mask_detector is not None else "none",
        "mean_masked_pct": round(float(df["masked_pct"].mean()), 2) if "masked_pct" in df else 0.0,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as fp:
        json.dump(results, fp, indent=2)
    prov = write_provenance(out_dir, cfg, files, extra={"results": results})

    report_path = None
    if make_report:
        report_path = build_report(out_dir, results, df, centers, po_avg, inv, cfg, prov, overlays)

    print("\nPLOT RESULTS")
    for k, v in results.items():
        print(f"  {k:24s}: {v}")
    print(f"\n  results -> {out_dir}")
    if report_path:
        print(f"  report  -> {report_path}")
    print("  Note: PAI/ALA/clumping/FAPAR are plot-level (from the averaged gap-fraction "
          "profile); FAPAR depends on latitude & day-of-year.")
    return df, results
