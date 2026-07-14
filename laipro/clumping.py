"""
Clumping index and true PAI (Lang & Yueqin 1986 logarithm-averaging method).

Foliage is rarely randomly placed, so the Poisson-based effective PAI
under-estimates the true PAI. Lang & Yueqin's method divides each zenith ring
into azimuthal cells, and compares the log of the mean gap fraction to the mean
of the log:

    Omega(theta) = ln(<Po_cell>) / <ln(Po_cell)>            (Omega <= 1)
    PAI_true     = PAI_eff / Omega

Cells with no gap (saturated, Po=0) are assigned a small Po_sat from a prescribed
saturating PAI so the logarithm stays finite; the saturated fraction is reported
as a sensitivity indicator (as CAN_EYE does).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ClumpingResult:
    omega: float                 # plot-average clumping index (weighted over rings)
    per_ring_omega: np.ndarray
    ring_centers_deg: np.ndarray
    saturated_frac: float        # fraction of cells that were saturated (Po=0)


def clumping_index(veg, theta_deg, azimuth_deg, valid,
                   coi_deg=60.0, res_deg=2.5, n_azimuth_cells=36,
                   min_cell_px=40, pai_sat=8.0):
    """Compute per-ring and plot-average clumping via log-averaging over azimuth
    cells.

    Po_sat = exp(-pai_sat) stands in for saturated (gap-free) cells.
    """
    po_sat = float(np.exp(-pai_sat))
    edges = np.arange(0.0, coi_deg + 1e-9, res_deg)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nrings = len(centers)

    # Bin every valid pixel into a (zenith ring, azimuth cell) cell in one pass,
    # instead of scanning all pixels for each of the rings x cells combinations.
    in_coi = valid & (theta_deg < coi_deg)
    ri = np.clip((theta_deg[in_coi] / res_deg).astype(np.intp), 0, nrings - 1)
    ai = np.clip(((azimuth_deg[in_coi] % 360.0) / (360.0 / n_azimuth_cells)).astype(np.intp),
                 0, n_azimuth_cells - 1)
    flat = ri * n_azimuth_cells + ai
    nbin = nrings * n_azimuth_cells
    total = np.bincount(flat, minlength=nbin).reshape(nrings, n_azimuth_cells)
    vegc = np.bincount(flat, weights=veg[in_coi].astype(np.float64),
                       minlength=nbin).reshape(nrings, n_azimuth_cells)

    per_ring = np.full(nrings, np.nan)
    ring_weight = np.zeros(nrings)
    sat_cells = 0
    tot_cells = 0

    for i in range(nrings):
        keep = total[i] >= min_cell_px
        if int(keep.sum()) < 3:
            continue
        po_cells = 1.0 - vegc[i][keep] / total[i][keep]
        tot_cells += po_cells.size
        sat = po_cells <= 0.0
        sat_cells += int(sat.sum())
        po_cells = np.where(sat, po_sat, po_cells)
        mean_po = po_cells.mean()
        mean_ln = np.log(po_cells).mean()
        if mean_ln < 0:                           # guard
            per_ring[i] = np.log(mean_po) / mean_ln
            ring_weight[i] = po_cells.size

    ok = np.isfinite(per_ring) & (ring_weight > 0)
    omega = (float(np.average(per_ring[ok], weights=ring_weight[ok]))
             if ok.any() else float("nan"))
    return ClumpingResult(omega=omega, per_ring_omega=per_ring,
                          ring_centers_deg=centers,
                          saturated_frac=(sat_cells / tot_cells if tot_cells else 0.0))
