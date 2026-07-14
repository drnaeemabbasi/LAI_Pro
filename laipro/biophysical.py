"""
Biophysical variables derived from gap fraction.

Model (standard, published; no tuning to any reference tool):
  * Ellipsoidal leaf-angle distribution (Campbell 1986, 1990), parameter x =
    ratio of horizontal to vertical ellipsoid axes. x=1 is spherical (mean leaf
    angle 57.3 deg).
  * Extinction coefficient K(theta, x) = sqrt(x^2 + tan^2 theta) /
    (x + 1.774 (x + 1.182)^-0.733). Gap fraction Po(theta) = exp(-K * PAI)
    (Poisson / turbid-medium model).
  * PAIeff, ALAeff by inverting this model against the measured per-zenith-ring
    gap fraction with a look-up table (Weiss et al. 2004 approach).
  * LAI57 (single angle, leaf-angle-independent; Warren-Wilson 1963).

Clumping -> true PAI is in clumping.py; FAPAR in fapar.py.

Everything is validated by synthetic self-recovery, i.e. generate Po from a
known (PAI, ALA) and check the inversion returns it -- correctness proven from
first principles, not by matching CAN_EYE.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------- leaf angle

def ellipsoidal_density(theta_l_rad, x):
    """Unnormalised ellipsoidal leaf-inclination density g(theta_l) (Campbell 1990)."""
    s, c = np.sin(theta_l_rad), np.cos(theta_l_rad)
    return 2.0 * x ** 3 * s / (c ** 2 + (x ** 2) * s ** 2) ** 2


def ala_from_x(x, n=4000):
    """Mean leaf inclination angle (deg) of the ellipsoidal distribution for
    parameter x, by numerical integration. x=1 -> 57.3 deg (spherical)."""
    th = np.linspace(1e-6, np.pi / 2 - 1e-6, n)
    g = ellipsoidal_density(th, x)
    w = g / np.trapz(g, th) if hasattr(np, "trapz") else g / np.trapezoid(g, th)
    return float(np.degrees(_trapz(th * w, th)))


def _trapz(y, x):
    # numpy renamed trapz->trapezoid; support both.
    return (np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def campbell_K(theta_rad, x):
    """Extinction coefficient K(theta, x) for the ellipsoidal distribution.
    Po(theta) = exp(-K * PAI). G(theta) = K * cos(theta)."""
    t = np.tan(theta_rad)
    num = np.sqrt(x ** 2 + t ** 2)
    den = x + 1.774 * (x + 1.182) ** (-0.733)
    return num / den


def x_grid_for_ala(ala_min=8.0, ala_max=82.0, n=75):
    """Build an x grid whose mean leaf angles span [ala_min, ala_max], returned
    sorted by ALA. (x and ALA are monotonically inversely related.)"""
    xs = np.logspace(np.log10(0.06), np.log10(16.0), 400)
    alas = np.array([ala_from_x(x) for x in xs])
    keep = (alas >= ala_min) & (alas <= ala_max)
    xs, alas = xs[keep], alas[keep]
    order = np.argsort(alas)
    xs, alas = xs[order], alas[order]
    # resample to n roughly-even ALA points
    target = np.linspace(alas.min(), alas.max(), n)
    idx = np.searchsorted(alas, target)
    idx = np.clip(idx, 0, len(xs) - 1)
    idx = np.unique(idx)
    return xs[idx], alas[idx]


# ---------------------------------------------------------------- LAI57

def lai57_from_gap(po_57: float) -> float:
    """LAI57 = -ln(Po(57.5)) / 0.93 (G ~ 0.5, leaf-angle independent)."""
    if not (0.0 < po_57 < 1.0):
        return float("nan")
    return float(-np.log(po_57) / 0.93)


# ---------------------------------------------------------------- LUT inversion

@dataclass
class InversionResult:
    pai_eff: float
    ala_eff: float
    x: float
    rmse: float          # weighted RMSE of gap fraction (model vs measured)


def build_lut(theta_centers_deg, lai_grid, x_grid):
    """LUT of modelled gap fraction, shape (n_lai, n_x, n_theta)."""
    th = np.radians(theta_centers_deg)
    K = np.stack([campbell_K(th, x) for x in x_grid], axis=0)   # (n_x, n_theta)
    lai = np.asarray(lai_grid)[:, None, None]                   # (n_lai,1,1)
    return np.exp(-lai * K[None, :, :])                         # (n_lai, n_x, n_theta)


def invert_gap_fraction(theta_centers_deg, po_measured, weights,
                        lai_grid=None, x_grid=None, ala_grid=None):
    """Retrieve (PAIeff, ALAeff) by matching modelled to measured ring gap
    fraction with a weighted RMSE cost (Weiss et al. 2004).

    Only finite, positive-weight rings are used.
    """
    if lai_grid is None:
        lai_grid = np.arange(0.0, 10.0 + 1e-9, 0.02)
    if x_grid is None or ala_grid is None:
        x_grid, ala_grid = x_grid_for_ala()

    po = np.asarray(po_measured, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(po) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return InversionResult(float("nan"), float("nan"), float("nan"), float("nan"))

    th = np.asarray(theta_centers_deg)[ok]
    po = po[ok]
    w = w[ok] / w[ok].sum()

    lut = build_lut(th, lai_grid, x_grid)                      # (n_lai,n_x,n_theta)
    diff2 = (lut - po[None, None, :]) ** 2
    cost = np.sqrt(np.tensordot(diff2, w, axes=([2], [0])))    # (n_lai, n_x)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return InversionResult(pai_eff=float(lai_grid[i]), ala_eff=float(ala_grid[j]),
                           x=float(x_grid[j]), rmse=float(cost[i, j]))


def modelled_gap(theta_deg, pai, x):
    """Modelled gap fraction at arbitrary zenith angle(s) for (pai, x)."""
    return np.exp(-pai * campbell_K(np.radians(np.asarray(theta_deg, float)), x))
