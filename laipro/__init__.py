"""
laipro - a reproducible, open Python engine for LAI / PAI / FCOVER estimation
from downward-looking fish-eye (DHP) crop photographs.

Goal: a next-generation, better-validated replacement for CAN_EYE's DHP pipeline
(Weiss & Baret, INRA), built in Python with:

  * fish-eye geometry (optical centre + projection function -> per-pixel zenith
    and azimuth), matching CAN_EYE's equidistant polar model.
  * masked-aware bi-directional gap fraction Po(zenith, azimuth).
  * FCOVER (nadir cover fraction) and LAI57 now; Poisson-LUT inversion for
    effective/true PAI + ALA, clumping, and FAPAR in the next build phase.
  * hybrid segmentation (deterministic vegetation indices now; optional learned
    GPU model later) and learned polygon-seeded masking (later).

Reproducibility is a first-class goal: every run records its full resolved
configuration and the content hashes of its inputs (see laipro.config).

This module is the scientific core. The napari desktop GUI and HTML reporting
are separate layers built on top of it.
"""

from .config import DHPConfig, FisheyeCalibration
from .geometry import zenith_azimuth_maps
from .segment import segment_vegetation, VEG_INDEX_FUNCS
from .gapfraction import ring_gap_fraction, fcover_from_cone
from .biophysical import (lai57_from_gap, invert_gap_fraction, modelled_gap,
                          campbell_K, ala_from_x, x_grid_for_ala)
from .clumping import clumping_index
from .fapar import compute_fapar
from .calibrate import (CalibrationProfile, fit_optical_center, fit_projection,
                        auto_profile)
from .features import extract_features, FEATURE_NAMES
from .learned_segment import LearnedSegmenter, CorrectionStore, samples_from_label_image
from .masking import (ForeignObjectDetector, MaskCorrectionStore, polygon_to_mask,
                      clean_object_mask, refine_with_sam)
from .report import build_report, qc_overlay_uri, save_qc_panels
from .preprocess import prepare_folder, cached_images
from .pipeline import process_folder, build_calibration

__all__ = [
    "DHPConfig", "FisheyeCalibration",
    "zenith_azimuth_maps",
    "segment_vegetation", "VEG_INDEX_FUNCS",
    "ring_gap_fraction", "fcover_from_cone",
    "lai57_from_gap", "invert_gap_fraction", "modelled_gap",
    "campbell_K", "ala_from_x", "x_grid_for_ala",
    "clumping_index", "compute_fapar",
    "CalibrationProfile", "fit_optical_center", "fit_projection", "auto_profile",
    "extract_features", "FEATURE_NAMES",
    "LearnedSegmenter", "CorrectionStore", "samples_from_label_image",
    "ForeignObjectDetector", "MaskCorrectionStore", "polygon_to_mask",
    "clean_object_mask", "refine_with_sam",
    "build_report", "qc_overlay_uri", "save_qc_panels",
    "prepare_folder", "cached_images",
    "process_folder", "build_calibration",
]

__version__ = "0.1.0"
