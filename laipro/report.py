"""
Self-contained HTML report for a processed plot.

Produces a single report.html with every figure embedded as a base64 data URI,
so it can be shared as one file (no sidecar images to lose). Mirrors and extends
CAN_EYE's report: general info, configuration (incl. which segmenter/mask were
used), per-image table, plot-level biophysical variables, the measured-vs-
modelled gap-fraction profile, QC overlays, and a provenance section with input
hashes for reproducibility.
"""

from __future__ import annotations
import base64
import datetime as _dt
import io
import os
import html as _html
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .biophysical import modelled_gap


def _fig_uri(fig, dpi=90):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def qc_overlay_uri(rgb, veg, valid, obj_mask, title, veg_pct=None, max_side=1600,
                   veg_color=(0.0, 0.9, 0.8)):
    """Two-panel QC overlay (original | classified+masked) as a data URI.

    Vegetation is drawn in a bright teal at high opacity (teal does not cancel
    green the way magenta did, so classified blades stay vivid instead of turning
    muddy). Rendered at max_side so thin blades don't vanish. The classified
    vegetation % of the analysed area is stamped on the panel, computed at full
    resolution so it exactly matches the value used in the numbers.
    """
    analyzed = valid & ~obj_mask
    if veg_pct is None:
        n = int(analyzed.sum())
        veg_pct = 100.0 * int((veg & analyzed).sum()) / n if n else float("nan")

    im = rgb
    v, va, om = veg, valid, obj_mask
    if max(im.shape[:2]) > max_side:                    # downscale for report size
        step = int(np.ceil(max(im.shape[:2]) / max_side))
        im = im[::step, ::step]
        v, va, om = veg[::step, ::step], valid[::step, ::step], obj_mask[::step, ::step]
    over = im.copy()
    vv = v & va & ~om
    over[vv] = 0.4 * over[vv] + 0.6 * np.array(veg_color)          # bright teal = vegetation
    over[om] = 0.4 * over[om] + 0.6 * np.array([1.0, 0.2, 0.2])    # red = masked object
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].imshow(np.clip(im, 0, 1)); ax[0].set_title("original"); ax[0].axis("off")
    ax[1].imshow(np.clip(over, 0, 1))
    ax[1].set_title(f"{title}\nvegetation = {veg_pct:.1f}% of analysed area")
    ax[1].axis("off")
    return _fig_uri(fig)


def save_qc_panels(rgb, veg, valid, obj_mask, idx, thresh, out_path, title,
                   veg_pct=None, margin=0.03, max_side=1800, veg_color=(0.0, 0.9, 0.8)):
    """Save a 4-panel diagnostic PNG per image: original; teal classification
    overlay; neutral binary-mask audit (white=veg, black=soil, grey=excluded);
    and the vegetation-index histogram with the threshold and +/-margin
    sensitivity band. This is the deep audit view, written to laipro_results/qc/.
    """
    analyzed_full = valid & ~obj_mask
    if veg_pct is None:
        n = int(analyzed_full.sum())
        veg_pct = 100.0 * int((veg & analyzed_full).sum()) / n if n else float("nan")

    im, v, va, om = rgb, veg, valid, obj_mask
    if max(im.shape[:2]) > max_side:
        step = int(np.ceil(max(im.shape[:2]) / max_side))
        im, v, va, om = rgb[::step, ::step], veg[::step, ::step], valid[::step, ::step], obj_mask[::step, ::step]
    an = va & ~om

    over = im.copy()
    over[v & an] = 0.4 * over[v & an] + 0.6 * np.array(veg_color)
    over[om] = 0.4 * over[om] + 0.6 * np.array([1.0, 0.2, 0.2])

    bmask = np.zeros(im.shape[:2] + (3,))
    bmask[v & an] = 1.0            # white = vegetation
    bmask[~an] = 0.5              # grey = excluded (outside circle of interest or masked)

    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    ax[0, 0].imshow(np.clip(im, 0, 1)); ax[0, 0].set_title("original"); ax[0, 0].axis("off")
    ax[0, 1].imshow(np.clip(over, 0, 1))
    ax[0, 1].set_title(f"{title}\nvegetation = {veg_pct:.1f}% of analysed area"); ax[0, 1].axis("off")
    ax[1, 0].imshow(bmask)
    ax[1, 0].set_title("audit: binary mask (white=veg, black=soil, grey=excluded)"); ax[1, 0].axis("off")

    if idx is not None:
        ax[1, 1].hist(idx.ravel(), bins=200, color="0.6")
        if thresh is not None and np.isfinite(thresh):
            ax[1, 1].axvline(thresh, color="black", lw=1.5, label=f"threshold={thresh:.3f}")
            ax[1, 1].axvline(thresh - margin, color="black", ls="--", lw=1)
            ax[1, 1].axvline(thresh + margin, color="black", ls="--", lw=1,
                             label=f"+/-{margin:.3f} sensitivity band")
            ax[1, 1].legend(fontsize=8)
        ax[1, 1].set_title("vegetation-index histogram + threshold")
        ax[1, 1].set_xlabel("index value")
    else:
        ax[1, 1].axis("off")
        ax[1, 1].set_title("(histogram n/a for learned segmenter)")

    fig.tight_layout(); fig.savefig(out_path, dpi=95); plt.close(fig)
    return out_path


def _gap_profile_uri(centers, po_avg, inv, coi_deg):
    fig, ax = plt.subplots(figsize=(6, 4))
    ok = np.isfinite(po_avg)
    ax.plot(centers[ok], po_avg[ok], "o", color="#1b7837", label="measured (plot avg)")
    if np.isfinite(inv.pai_eff) and np.isfinite(inv.x):
        th = np.linspace(0, coi_deg, 200)
        ax.plot(th, modelled_gap(th, inv.pai_eff, inv.x), "-", color="#762a83",
                label=f"model: PAI={inv.pai_eff:.2f}, ALA={inv.ala_eff:.0f}deg")
    ax.axvline(57.5, ls="--", color="0.6", lw=1)
    ax.set_xlabel("view zenith angle (deg)"); ax.set_ylabel("gap fraction Po")
    ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.set_title("Gap fraction: measured vs modelled")
    return _fig_uri(fig)


def _fcover_bar_uri(df):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(range(len(df)), df["FCOVER_pct"], color="#1b7837")
    ax.axhline(df["FCOVER_pct"].mean(), color="#762a83", ls="--",
               label=f"mean {df['FCOVER_pct'].mean():.1f}%")
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df["image"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("FCOVER (%)"); ax.legend(fontsize=8); ax.set_title("FCOVER per image")
    return _fig_uri(fig)


def _table(rows, header=None):
    h = "<tr>" + "".join(f"<th>{_html.escape(str(c))}</th>" for c in header) + "</tr>" if header else ""
    body = "".join("<tr>" + "".join(f"<td>{_html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{h}{body}</table>"


_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f7f7f5;color:#1a1a1a}
.wrap{max-width:1000px;margin:0 auto;padding:24px}
h1{color:#1b5e20;margin-bottom:2px} h2{color:#1b5e20;border-bottom:2px solid #cfe3cf;padding-bottom:4px;margin-top:32px}
.sub{color:#666;margin-top:0}
table{border-collapse:collapse;margin:8px 0;font-size:14px;width:100%}
th,td{border:1px solid #d5ddd5;padding:5px 9px;text-align:left}
th{background:#e7f0e7}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}
.kpi{background:#fff;border:1px solid #d5ddd5;border-radius:8px;padding:12px 16px;min-width:120px}
.kpi .v{font-size:22px;font-weight:700;color:#1b5e20} .kpi .l{font-size:12px;color:#666}
img{max-width:100%;border:1px solid #d5ddd5;border-radius:6px;background:#fff}
.gallery img{margin:8px 0}
details{margin:8px 0} summary{cursor:pointer;color:#1b5e20;font-weight:600}
.note{color:#666;font-size:13px}
"""


def build_report(out_dir, results, df, centers, po_avg, inv, cfg, provenance, overlays):
    """Assemble and write out_dir/report.html. `overlays` is a list of
    (image_name, data_uri)."""
    def kpi(label, value):
        return f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>'

    kpis = "".join([
        kpi("PAI effective", results.get("PAI_effective")),
        kpi("PAI true", results.get("PAI_true")),
        kpi("ALA (deg)", results.get("ALA_effective_deg")),
        kpi("Clumping", results.get("clumping_index")),
        kpi("FCOVER", f"{results.get('mean_FCOVER_pct')}%"),
        kpi("FAPAR (daily)", results.get("FAPAR_black_sky_daily")),
    ])

    general = _table([
        ["laipro version", provenance.get("laipro_version", "")],
        ["Folder", out_dir],
        ["Processed", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Images", results.get("n_images")],
        ["Segmenter", results.get("segmenter")],
        ["Mask detector", results.get("mask_detector")],
    ])

    calib = cfg.calibration
    config = _table([
        ["Vegetation index", cfg.veg_index],
        ["Circle of interest", f"{cfg.coi_deg} deg"],
        ["FCOVER cone", f"{cfg.fcover_cone_deg} deg"],
        ["Calibration mode", calib.mode],
        ["Latitude / DoY (FAPAR)", f"{cfg.latitude_deg} / {cfg.day_of_year}"],
        ["Mean masked area", f"{results.get('mean_masked_pct', 0)}%"],
    ])

    plot_vars = _table([
        ["Effective PAI", results.get("PAI_effective")],
        ["True PAI", results.get("PAI_true")],
        ["Effective ALA (deg)", results.get("ALA_effective_deg")],
        ["Clumping index", results.get("clumping_index")],
        ["Gap-fraction inversion RMSE", results.get("inversion_rmse_gap")],
        ["Mean LAI57", results.get("mean_LAI57")],
        ["Mean FCOVER (%)", results.get("mean_FCOVER_pct")],
        ["FAPAR black-sky daily", results.get("FAPAR_black_sky_daily")],
        ["FAPAR white-sky", results.get("FAPAR_white_sky")],
    ], header=["Variable", "Value"])

    per_img = _table(
        [[r["image"], r.get("veg_pct_COI"), r["FCOVER_pct"], r.get("LAI57"), r.get("masked_pct", 0)]
         for _, r in df.iterrows()],
        header=["Image", "Vegetation %", "FCOVER (nadir) %", "LAI57", "masked %"])

    gap_img = _gap_profile_uri(centers, po_avg, inv, cfg.coi_deg)
    bar_img = _fcover_bar_uri(df)
    gallery = "".join(f'<div><b>{_html.escape(n)}</b><br><img src="{u}"></div>' for n, u in overlays)

    inputs = _table([[i["file"], i["sha256"][:16] + "..."] for i in provenance.get("inputs", [])],
                    header=["Input file", "SHA-256 (truncated)"])

    plot_name = os.path.basename(os.path.dirname(os.path.abspath(out_dir.rstrip(os.sep)))) or "plot"

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>laipro report - {_html.escape(plot_name)}</title>
<style>{_CSS}</style></head><body><div class="wrap">
<h1>laipro DHP Report &mdash; {_html.escape(plot_name)}</h1>
<p class="sub">{_html.escape(out_dir)}</p>
<div class="kpis">{kpis}</div>
<p class="note">PAI / ALA / clumping / FAPAR are plot-level (from the gap-fraction profile averaged over all images).
FAPAR depends on latitude &amp; day-of-year. Results derived from published radiative-transfer models; not tuned to any reference tool.</p>

<h2>General information</h2>{general}
<h2>Configuration</h2>{config}
<h2>Plot biophysical variables</h2>{plot_vars}
<h2>Gap fraction &amp; cover</h2>
<img src="{gap_img}"><img src="{bar_img}">
<h2>Per-image results</h2>{per_img}
<h2>QC overlays</h2>
<p class="note">magenta = detected vegetation, red = masked foreign object.</p>
<div class="gallery">{gallery}</div>
<details><summary>Provenance (input hashes)</summary>{inputs}</details>
</div></body></html>"""

    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
