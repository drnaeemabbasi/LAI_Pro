"""
Masked-aware bi-directional gap fraction.

Gap fraction Po(theta) = fraction of a zenith ring that is NOT vegetation
(i.e. soil/background seen through the canopy), computed only over valid,
unmasked pixels. This is the quantity every DHP output is derived from:
  FCOVER = 1 - Po(0..cone)          (nadir cover)
  LAI57  = -ln(Po(57.5)) / 0.93     (single-angle LAI)
  LAIeff, ALAeff, clumping, FAPAR   (Poisson-LUT inversion; phase 2)
"""

from __future__ import annotations
import numpy as np


def ring_gap_fraction(veg, theta_deg, valid, coi_deg=60.0, res_deg=2.5):
    """Azimuth-averaged gap fraction per zenith ring.

    Returns (ring_centers_deg, po, ring_valid_counts). po[i] is NaN where a ring
    has no valid pixels. Vectorised via binning (one O(pixels) pass) rather than
    a per-ring scan.
    """
    edges = np.arange(0.0, coi_deg + 1e-9, res_deg)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nrings = len(centers)

    in_coi = valid & (theta_deg < coi_deg)
    ri = np.clip((theta_deg[in_coi] / res_deg).astype(np.intp), 0, nrings - 1)
    vin = veg[in_coi].astype(np.float64)

    counts = np.bincount(ri, minlength=nrings).astype(np.int64)
    vegc = np.bincount(ri, weights=vin, minlength=nrings)
    po = np.full(nrings, np.nan)
    nz = counts > 0
    po[nz] = 1.0 - vegc[nz] / counts[nz]          # gap = non-vegetation
    return centers, po, counts


def fcover_from_cone(veg, theta_deg, valid, cone_deg=10.0):
    """FCOVER = vegetation fraction within the central nadir cone (theta<=cone),
    i.e. 1 - Po(0..cone). Matches CAN_EYE's default 0-10 deg integration."""
    cone = valid & (theta_deg <= cone_deg)
    n = int(cone.sum())
    if not n:
        return float("nan"), 0
    return float((veg & cone).sum()) / n, n


def gap_at_angle(centers, po, target_deg):
    """Gap fraction of the ring nearest target_deg (skipping NaN rings)."""
    ok = ~np.isnan(po)
    if not ok.any():
        return float("nan")
    idx = np.where(ok)[0]
    j = idx[np.argmin(np.abs(centers[idx] - target_deg))]
    return float(po[j])
