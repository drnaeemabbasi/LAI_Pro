"""
Fish-eye calibration: optical centre + projection function.

Two independent characteristics fully define the fish-eye mapping used by the
geometry module (laipro.geometry):

  1. Optical centre (cx, cy): the pixel the optical axis projects onto. Found by
     the drilled-cap method (CAN_EYE section 6): a hole in the lens cap is
     rotated to many positions; its imaged point traces a circle whose centre is
     the optical centre. We fit that circle algebraically (Kasa). Multiple holes
     give independent estimates whose spread is a built-in consistency check.

  2. Projection function theta = P(r): view zenith angle (deg) as a polynomial of
     radial pixel distance r from the optical centre. Found by the ruler
     experiment: known angles theta = arctan(x/H) are paired with measured pixel
     radii, and a degree-1/2/3 polynomial (through the origin, since theta(0)=0)
     is least-squares fit. Degree 1 == ideal equidistant lens.

A CalibrationProfile bundles both, serialises to JSON, and converts to the
FisheyeCalibration the geometry uses. For the common case with no calibration
frames, `auto_profile` derives a principled default from image geometry (no
clicks), and `detect_image_circle` handles circular fish-eyes.

Design goals: flexible (measured OR auto), reproducible (everything saved), and
zero mandatory user interaction unless precision is wanted.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
import json
import numpy as np

from .config import FisheyeCalibration


# --------------------------------------------------------------------------
# Profile container
# --------------------------------------------------------------------------

@dataclass
class CalibrationProfile:
    """A complete, serialisable fish-eye calibration."""
    center_xy: tuple[float, float]
    proj_coeffs: tuple[float, ...]        # ascending powers of r (pixels); index 0 is 0.0
    max_fov_deg: float                    # full field of view across the imaged diameter
    image_shape: tuple[int, int]          # (H, W) the calibration applies to
    method: str = "auto"                  # "measured", "auto_fullframe", "auto_circular"
    notes: str = ""
    diagnostics: dict = field(default_factory=dict)  # rmse, r2, center spread, etc.

    def to_fisheye_calibration(self) -> FisheyeCalibration:
        return FisheyeCalibration(
            mode="explicit",
            center_xy=(float(self.center_xy[0]), float(self.center_xy[1])),
            proj_coeffs=tuple(float(c) for c in self.proj_coeffs),
            ref_shape=(int(self.image_shape[0]), int(self.image_shape[1])),
        )

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "CalibrationProfile":
        with open(path) as f:
            d = json.load(f)
        d["center_xy"] = tuple(d["center_xy"])
        d["proj_coeffs"] = tuple(d["proj_coeffs"])
        d["image_shape"] = tuple(d["image_shape"])
        return CalibrationProfile(**d)


# --------------------------------------------------------------------------
# Optical centre: algebraic circle fit (Kasa)
# --------------------------------------------------------------------------

def circle_fit(points) -> tuple[float, float, float, float]:
    """Least-squares circle through points. Returns (cx, cy, radius, rms_residual).

    Solves x^2 + y^2 + D x + E y + F = 0 linearly; centre = (-D/2, -E/2)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ValueError("need >= 3 points to fit a circle")
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -D / 2.0, -E / 2.0
    radius = float(np.sqrt(max(cx * cx + cy * cy - F, 0.0)))
    resid = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius
    return float(cx), float(cy), radius, float(np.sqrt(np.mean(resid ** 2)))


def fit_optical_center(holes: dict[str, list]) -> tuple[tuple[float, float], dict]:
    """Fit the optical centre from drilled-cap hole tracks.

    holes: {hole_id: [(x, y), ...positions...]} - each hole's imaged positions as
    the cap rotates lie on a circle centred on the optical centre.

    Returns ((cx, cy), diagnostics). Multiple holes are averaged; their spread is
    reported as a consistency check (should be < ~1 px for a good calibration).
    """
    centers = []
    per_hole = {}
    for hid, pts in holes.items():
        cx, cy, r, rms = circle_fit(pts)
        centers.append((cx, cy))
        per_hole[hid] = {"center": [cx, cy], "radius": r, "fit_rms_px": rms, "n": len(pts)}
    centers = np.asarray(centers)
    cx, cy = centers.mean(axis=0)
    spread = float(np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1))) if len(centers) > 1 else 0.0
    return (float(cx), float(cy)), {"per_hole": per_hole, "center_spread_px": spread,
                                    "n_holes": len(centers)}


# --------------------------------------------------------------------------
# Projection function: polynomial theta(r) through the origin
# --------------------------------------------------------------------------

def fit_projection(radii_px, angles_deg, degree: int = 3) -> tuple[tuple[float, ...], dict]:
    """Least-squares fit theta_deg = c1*r + c2*r^2 + ... + cd*r^d (no constant:
    theta(0)=0). degree=1 is the ideal equidistant lens.

    Returns (coeffs_ascending_with_leading_zero, diagnostics{rmse_deg, r2}).
    """
    r = np.asarray(radii_px, dtype=np.float64)
    th = np.asarray(angles_deg, dtype=np.float64)
    if r.size <= degree:
        raise ValueError(f"need > {degree} measurements for degree {degree}")
    X = np.column_stack([r ** k for k in range(1, degree + 1)])   # no intercept
    coeffs, *_ = np.linalg.lstsq(X, th, rcond=None)
    pred = X @ coeffs
    resid = th - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((th - th.mean()) ** 2)) or 1.0
    diag = {"degree": degree, "rmse_deg": float(np.sqrt(np.mean(resid ** 2))),
            "r2": 1.0 - ss_res / ss_tot, "n": int(r.size)}
    return (0.0, *(float(c) for c in coeffs)), diag


# --------------------------------------------------------------------------
# Auto calibration (no calibration frames needed)
# --------------------------------------------------------------------------

def detect_image_circle(rgb, rel_thresh: float = 0.06):
    """Detect the imaged disk of a CIRCULAR fish-eye (dark border around a bright
    disk). Returns (cx, cy, radius). Raises if no clear disk (e.g. full-frame)."""
    lum = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
    mask = lum > (rel_thresh * lum.max())
    ys, xs = np.nonzero(mask)
    if xs.size < 100:
        raise ValueError("no imaged disk detected")
    cx, cy = xs.mean(), ys.mean()
    # radius as the median extent -> robust to stray bright pixels
    radius = float(np.median(np.hypot(xs - cx, ys - cy)) * 2.0 / np.sqrt(2))
    return float(cx), float(cy), radius


def is_full_frame(rgb, corner_frac: float = 0.04, dark_rel: float = 0.10) -> bool:
    """True if the image fills the frame (bright/typical corners) rather than
    being a circular fish-eye with dark corners."""
    lum = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
    H, W = lum.shape
    ch, cw = max(1, int(H * corner_frac)), max(1, int(W * corner_frac))
    corners = np.concatenate([
        lum[:ch, :cw].ravel(), lum[:ch, -cw:].ravel(),
        lum[-ch:, :cw].ravel(), lum[-ch:, -cw:].ravel()])
    return float(corners.mean()) > dark_rel * lum.max()


def auto_profile(rgb, diagonal_fov_deg: float = 180.0, kind: str = "auto") -> CalibrationProfile:
    """Derive a principled calibration with no calibration frames.

    kind="fullframe": diagonal of the frame spans diagonal_fov_deg (corners at
      the max angle); optical centre = image centre. Resolution-independent.
    kind="circular": detect the imaged disk; its diameter spans diagonal_fov_deg.
    kind="auto": pick fullframe vs circular from corner darkness.
    """
    H, W = rgb.shape[:2]
    if kind == "auto":
        kind = "fullframe" if is_full_frame(rgb) else "circular"

    if kind == "fullframe":
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        r_edge = 0.5 * float(np.hypot(W, H))          # corner radius
        method = "auto_fullframe"
    elif kind == "circular":
        cx, cy, radius = detect_image_circle(rgb)
        r_edge = radius                               # disk edge radius
        method = "auto_circular"
    else:
        raise ValueError(f"unknown kind: {kind}")

    coeff = (diagonal_fov_deg / 2.0) / r_edge          # equidistant deg/pixel
    return CalibrationProfile(
        center_xy=(cx, cy), proj_coeffs=(0.0, coeff),
        max_fov_deg=diagonal_fov_deg, image_shape=(H, W), method=method,
        notes="equidistant model derived from image geometry (no calibration frames)",
        diagnostics={"deg_per_pixel": coeff, "edge_radius_px": r_edge, "kind": kind},
    )
