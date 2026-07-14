"""
FAPAR: fraction of absorbed photosynthetically active radiation.

In the PAR domain leaves are strongly absorbing, so fAPAR is well approximated
by fIPAR = 1 - gap fraction (Andrieu & Baret 1993). We use the modelled gap
fraction from the retrieved (PAI, x) so it extends to any sun zenith angle.

  * Instantaneous black-sky:  fAPAR_BS(theta_s) = 1 - Po(theta_s)
  * Daily black-sky:          integral over the sun's daily course, weighted by
                              cos(theta_s) (Weiss et al.).
  * White-sky (diffuse):      hemispherical integral of (1 - Po) weighted by the
                              isotropic-sky term 2 sin(theta) cos(theta).

Solar geometry uses the standard declination + hour-angle formulae; latitude and
day-of-year come from the config (they must be set for meaningful FAPAR).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .biophysical import modelled_gap


@dataclass
class FaparResult:
    daily_black_sky: float
    white_sky: float
    instantaneous: dict          # solar_hour -> fAPAR_BS


def solar_declination_deg(doy: int) -> float:
    return 23.45 * np.sin(np.radians(360.0 * (284 + doy) / 365.0))


def solar_zenith_deg(latitude_deg, doy, hour_angle_deg):
    lat = np.radians(latitude_deg)
    dec = np.radians(solar_declination_deg(doy))
    h = np.radians(hour_angle_deg)
    cos_z = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(h)
    return np.degrees(np.arccos(np.clip(cos_z, -1.0, 1.0)))


def compute_fapar(pai, x, latitude_deg, doy, coi_deg=60.0):
    """Black-sky (daily) and white-sky FAPAR from the modelled gap fraction."""
    if not np.isfinite(pai) or not np.isfinite(x):
        return FaparResult(float("nan"), float("nan"), {})

    # --- daily black sky: sample the half-day sun course (symmetric at noon) ---
    hours = np.arange(0.0, 12.01, 0.5)                 # solar hours from noon
    ha = hours * 15.0                                  # deg per hour
    theta_s = solar_zenith_deg(latitude_deg, doy, ha)
    day = theta_s < 90.0
    inst = {}
    for hh, ts, up in zip(hours, theta_s, day):
        if up:
            inst[f"{12+hh:.1f}"] = float(1.0 - modelled_gap(ts, pai, x))
    if day.any():
        ts_d = theta_s[day]
        num = np.trapezoid((1.0 - modelled_gap(ts_d, pai, x)) * np.cos(np.radians(ts_d)),
                           np.radians(ha[day]))
        den = np.trapezoid(np.cos(np.radians(ts_d)), np.radians(ha[day]))
        daily_bs = float(num / den) if den else float("nan")
    else:
        daily_bs = float("nan")

    # --- white sky: isotropic diffuse over the hemisphere (to COI) ---
    th = np.radians(np.linspace(0.0, coi_deg, 200))
    integrand = (1.0 - modelled_gap(np.degrees(th), pai, x)) * 2.0 * np.sin(th) * np.cos(th)
    ws = float(np.trapezoid(integrand, th) / np.trapezoid(2.0 * np.sin(th) * np.cos(th), th))

    return FaparResult(daily_black_sky=daily_bs, white_sky=ws, instantaneous=inst)
