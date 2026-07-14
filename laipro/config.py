"""
Configuration + provenance for reproducible DHP processing.

Everything that affects a result lives in DHPConfig. A run serialises the fully
resolved config plus the SHA-256 of every input image, so any output can be
re-derived exactly. This is the backbone of the "reproducible by construction"
goal.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib
import json
import os


@dataclass
class FisheyeCalibration:
    """Fish-eye optical model: maps a pixel's radial distance from the optical
    centre to a zenith (view) angle.

    Two modes:
      * mode="fullframe180" (default): a full-frame fish-eye whose diagonal
        spans a `diagonal_fov_deg` field of view (180 by default), with the
        optical centre at the image centre. The equidistant coefficient is
        derived from the image shape, so it is resolution-independent and needs
        no manual calibration. For 6000x4000 this yields 0.0250 deg/pixel,
        matching CAN_EYE's measured 0.0249 for this camera.
      * mode="explicit": user-supplied optical centre (cx, cy) in pixels and a
        polynomial projection theta_deg = sum(proj_coeffs[i] * r**i), matching
        CAN_EYE's degree-1..3 polar projection. Use this once a proper lens
        calibration (see CAN_EYE section 6) is available.
    """
    mode: str = "fullframe180"
    diagonal_fov_deg: float = 180.0
    # explicit-mode parameters (ignored in fullframe180 mode):
    center_xy: tuple[float, float] | None = None
    proj_coeffs: tuple[float, ...] | None = None  # ascending powers of r (pixels)
    # Resolution the explicit params were measured at (H, W). If set, geometry
    # rescales pixel radii/centre to the processing resolution, so a profile
    # calibrated at full res stays correct when images are downscaled.
    ref_shape: tuple[int, int] | None = None


@dataclass
class DHPConfig:
    """All parameters that determine a DHP result."""
    veg_index: str = "exg"
    manual_thresh: float | None = None
    calibration: FisheyeCalibration = field(default_factory=FisheyeCalibration)

    coi_deg: float = 60.0              # circle of interest (zenith limit) processed
    zenith_res_deg: float = 2.5        # ring width for gap fraction
    fcover_cone_deg: float = 10.0      # nadir cone used for FCOVER = 1 - Po(0..cone)
    lai57_zenith_deg: float = 57.5     # angle where G is ~independent of leaf angle

    # clumping (Lang & Yueqin) -> true PAI
    n_azimuth_cells: int = 36
    min_cell_px: int = 40
    pai_sat: float = 8.0               # saturating PAI for gap-free cells

    # FAPAR solar geometry (must be set for meaningful FAPAR)
    latitude_deg: float = 43.0
    day_of_year: int = 167

    max_side: int | None = 2500        # downscale longest side for speed (fractions preserved); None = full res

    def resolved(self) -> dict:
        return asdict(self)


def sha256_file(path: str, _buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_buf), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(out_dir: str, cfg: DHPConfig, image_paths: list[str],
                     extra: dict | None = None) -> str:
    """Write a JSON provenance record (config + input hashes) for reproducibility."""
    rec = {
        "laipro_version": __import__("laipro").__version__,
        "config": cfg.resolved(),
        "inputs": [{"file": os.path.basename(p), "sha256": sha256_file(p)}
                   for p in image_paths],
    }
    if extra:
        rec.update(extra)
    path = os.path.join(out_dir, "provenance.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    return rec
