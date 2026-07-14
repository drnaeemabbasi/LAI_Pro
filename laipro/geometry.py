"""
Fish-eye geometry: per-pixel zenith and azimuth angles.

This is the piece CAN_EYE hides behind its calibration menu. We expose it
explicitly and default to a resolution-independent full-frame-180 model that
reproduces CAN_EYE's measured lens coefficient for this camera.
"""

from __future__ import annotations
import numpy as np
from .config import FisheyeCalibration


def zenith_azimuth_maps(shape, calib: FisheyeCalibration):
    """Return (theta_deg, azimuth_deg) arrays of the given (H, W) shape.

    theta_deg  : view zenith angle from the optical axis (0 at nadir centre).
    azimuth_deg: 0..360, atan2 around the optical centre.
    """
    H, W = shape

    # Rescale an explicit calibration measured at a different resolution so it
    # applies correctly here (pixel-based coeffs are resolution-specific).
    scale = 1.0
    if calib.mode == "explicit" and calib.ref_shape is not None:
        Href, Wref = calib.ref_shape
        scale = float(np.hypot(Wref, Href) / np.hypot(W, H))   # ref px per current px

    if calib.center_xy is not None:
        cx, cy = calib.center_xy[0] / (Wref / W if calib.ref_shape else 1.0), \
                 calib.center_xy[1] / (Href / H if calib.ref_shape else 1.0)
    else:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    ys, xs = np.mgrid[0:H, 0:W]
    dx = xs - cx
    dy = ys - cy
    r = np.sqrt(dx * dx + dy * dy)

    if calib.mode == "fullframe180":
        # diagonal spans diagonal_fov_deg; corner radius maps to half of it.
        r_corner = 0.5 * np.hypot(W, H)
        coeff = (calib.diagonal_fov_deg / 2.0) / r_corner   # deg per pixel (equidistant)
        theta = coeff * r
    elif calib.mode == "explicit":
        if calib.proj_coeffs is None:
            raise ValueError("explicit calibration requires proj_coeffs")
        r_ref = r * scale                                   # radius in reference pixels
        theta = np.zeros_like(r)
        for i, c in enumerate(calib.proj_coeffs):
            theta = theta + c * np.power(r_ref, i)
    else:
        raise ValueError(f"unknown calibration mode: {calib.mode}")

    azimuth = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    return theta, azimuth


def coi_mask(theta_deg, coi_deg: float):
    """Boolean array: True where the pixel is OUTSIDE the circle of interest
    (i.e. should be excluded, matching CAN_EYE's COI>60 exclusion)."""
    return theta_deg > coi_deg
