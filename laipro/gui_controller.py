"""
GUI controller: all the state and logic behind the napari GUI, with no Qt/napari
dependency so it can be unit-tested headlessly.

The napari view (laipro.gui) is a thin layer that binds layers and buttons to
this controller. Keeping the logic here means the interactive workflow --
loading a plot, painting vegetation labels, drawing foreign-object polygons,
training the learned models from those corrections, previewing, and running the
full DHP computation -- is all exercisable without a display.

Label conventions (shared with the CLIs):
  vegetation labels : 0 = unlabeled, 1 = soil, 2 = vegetation
  foreign-object    : list of polygons (each a list of (x, y) vertices)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os
import numpy as np

from .config import DHPConfig, FisheyeCalibration
from .io import load_rgb, find_images
from .geometry import zenith_azimuth_maps, coi_mask
from .segment import segment_vegetation
from .learned_segment import LearnedSegmenter, CorrectionStore, LABEL_SOIL, LABEL_VEG
from .masking import ForeignObjectDetector, MaskCorrectionStore, polygon_to_mask


@dataclass
class GuiController:
    cfg: DHPConfig = field(default_factory=DHPConfig)
    folder: str | None = None
    files: list = field(default_factory=list)
    index: int = 0
    # loaded per-image state
    rgb: np.ndarray | None = None
    theta: np.ndarray | None = None
    valid: np.ndarray | None = None
    # models (optional; None => classical / no mask)
    segmenter: LearnedSegmenter | None = None
    mask_detector: ForeignObjectDetector | None = None
    # model directories (for accumulating corrections)
    seg_model_dir: str | None = None
    mask_model_dir: str | None = None

    # ---------------------------------------------------------------- loading
    def open_folder(self, folder: str):
        self.folder = folder
        self.files = find_images(folder)
        if not self.files:
            raise ValueError(f"No images found in {folder}")
        self.index = 0
        self.load_current()
        return len(self.files)

    def load_current(self):
        path = self.files[self.index]
        self.rgb = load_rgb(path, self.cfg.max_side)
        self.theta, _ = zenith_azimuth_maps(self.rgb.shape[:2], self.cfg.calibration)
        self.valid = ~coi_mask(self.theta, self.cfg.coi_deg)
        return self.rgb

    def step(self, delta: int):
        self.index = int(np.clip(self.index + delta, 0, len(self.files) - 1))
        return self.load_current()

    @property
    def current_name(self):
        return os.path.basename(self.files[self.index]) if self.files else None

    # ---------------------------------------------------------------- previews
    def vegetation_mask(self):
        """Current vegetation prediction (learned model if set, else classical),
        restricted to the valid circle of interest."""
        if self.segmenter is not None:
            veg = self.segmenter.predict(self.rgb)
        else:
            veg, _, _ = segment_vegetation(self.rgb, self.cfg.veg_index,
                                           self.cfg.manual_thresh, self.valid)
        return veg & self.valid

    def object_mask_from_polygons(self, polygons):
        """Exact mask from user polygons (what the Shapes layer holds)."""
        if not polygons:
            return np.zeros(self.rgb.shape[:2], bool)
        return polygon_to_mask(self.rgb.shape[:2], polygons)

    def predicted_object_mask(self):
        """Auto-mask from the learned detector (editable by the user afterwards)."""
        if self.mask_detector is None:
            return np.zeros(self.rgb.shape[:2], bool)
        return self.mask_detector.predict(self.rgb)

    # ---------------------------------------------------------------- training
    def train_segmenter_from_labels(self, label_img, model_dir=None, backend="rf"):
        """Add the current painted labels as a correction and retrain on ALL
        accumulated corrections."""
        model_dir = model_dir or self.seg_model_dir
        if not model_dir:
            raise ValueError("no segmenter model directory set")
        store = CorrectionStore(model_dir)
        store.add(self.rgb, np.asarray(label_img), self.current_name)
        X, y, contribs = store.load_samples()
        seg = LearnedSegmenter(backend=backend).fit(X, y, contributing=contribs)
        seg.save(model_dir)
        self.segmenter, self.seg_model_dir = seg, model_dir
        return int(len(y))

    def train_mask_from_polygons(self, polygons, model_dir=None):
        model_dir = model_dir or self.mask_model_dir
        if not model_dir:
            raise ValueError("no mask model directory set")
        det = self.mask_detector or ForeignObjectDetector()
        store = MaskCorrectionStore(model_dir)
        store.add(self.rgb, polygons, self.current_name, det, valid=self.valid)
        X, y, contribs = store.load_samples()
        det.fit(X, y, contributing=contribs)
        det.save(model_dir)
        self.mask_detector, self.mask_model_dir = det, model_dir
        return int(len(y))

    # ---------------------------------------------------------------- models
    def load_segmenter(self, model_dir):
        self.segmenter = LearnedSegmenter.load(model_dir)
        self.seg_model_dir = model_dir

    def load_mask_detector(self, model_dir):
        self.mask_detector = ForeignObjectDetector.load(model_dir)
        self.mask_model_dir = model_dir

    def use_classical(self):
        self.segmenter = None

    # ---------------------------------------------------------------- prepare
    def prepare_cache(self, target_side, progress=None):
        """Decode-once + downscale cache for the folder; also sets the working
        resolution so processing uses the cache."""
        from laipro.preprocess import prepare_folder
        summary = prepare_folder(self.folder, target_side=target_side, progress=progress)
        self.cfg.max_side = target_side
        return summary

    # ---------------------------------------------------------------- process
    def process_folder(self, progress=None):
        """Run the full DHP computation on the loaded folder with the current
        models/settings (same code path as the CLI)."""
        from laipro.pipeline import process_folder
        return process_folder(self.folder, self.cfg,
                               segmenter=self.segmenter,
                               mask_detector=self.mask_detector,
                               progress=progress)
