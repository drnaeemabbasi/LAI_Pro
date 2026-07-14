"""
Learned foreign-object masking (operator, boot, monopod, ...).

Motivation
----------
A fixed "mask the bottom 15%" throws away real vegetation. Instead we learn the
foreign object from polygons the user draws, and exclude only the object -
wherever it appears. This preserves every vegetation pixel the object does not
actually cover.

How it learns
-------------
The user draws polygon(s) around the object on a few images (CAN_EYE-style, but
persistent). Pixels inside are positive (object); a balanced random sample
outside is negative. A RandomForest is trained on colour/texture features PLUS
normalised position (x, y, radius) - position is a strong, honest cue because a
pole/tripod rig puts the operator in a consistent part of the frame. Corrections
accumulate across images and retrain on everything (like the segmenter).

At inference the detector predicts an object probability per pixel; a
morphological clean-up (remove specks, close, fill, keep large blobs) yields the
final exclusion mask. An OPTIONAL SAM refinement (refine_with_sam) can crisp the
boundary when torch + a SAM checkpoint are available; it is lazy and never
required.

Polygons are exact and reproducible; the learned detector generalises them; the
GUI will let the user edit any auto-mask before it is applied.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import os
import numpy as np
from skimage.draw import polygon as sk_polygon
from skimage import morphology
from scipy import ndimage as ndi

from .features import extract_features, FEATURE_VERSION, FEATURE_NAMES
from .learned_segment import SEED, _sha256_array

MASK_FEATURE_NAMES = list(FEATURE_NAMES) + ["x_norm", "y_norm", "r_norm"]


# --------------------------------------------------------------- polygon raster

def polygon_to_mask(shape, polygons) -> np.ndarray:
    """Rasterise polygons (each a list of (x, y) vertices) to a boolean mask."""
    H, W = shape[:2]
    mask = np.zeros((H, W), bool)
    for poly in polygons:
        pts = np.asarray(poly, dtype=float)
        rr, cc = sk_polygon(pts[:, 1], pts[:, 0], shape=(H, W))   # (row=y, col=x)
        mask[rr, cc] = True
    return mask


def mask_features(rgb, context_window=9):
    """Colour/texture features + normalised position (scale-invariant)."""
    feats, _ = extract_features(rgb, context_window)
    H, W, _ = feats.shape
    ys, xs = np.mgrid[0:H, 0:W]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    xn = (xs / (W - 1)).astype(np.float32)
    yn = (ys / (H - 1)).astype(np.float32)
    rn = (np.hypot(xs - cx, ys - cy) / np.hypot(cx, cy)).astype(np.float32)
    return np.concatenate([feats, np.stack([xn, yn, rn], -1)], axis=-1)


def clean_object_mask(prob, thresh=0.5, min_area_frac=0.004, close_radius=4):
    """Turn a per-pixel object probability into a clean exclusion mask."""
    H, W = prob.shape
    m = prob >= thresh
    if not m.any():
        return m
    # drop specks below the area threshold (version-robust: no deprecated skimage args)
    lbl, n = ndi.label(m)
    if n:
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        m = (counts >= int(min_area_frac * H * W))[lbl]
    if m.any():
        m = morphology.closing(m, morphology.disk(close_radius))
        m = ndi.binary_fill_holes(m)
    return m


# --------------------------------------------------------------- detector

@dataclass
class MaskModelCard:
    laipro_version: str
    feature_version: int
    feature_names: list
    seed: int
    context_window: int
    hyperparams: dict
    n_samples: int
    class_balance: dict
    contributing_masks: list

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


class ForeignObjectDetector:
    """Trainable, accumulating detector that excludes learned foreign objects."""

    def __init__(self, context_window=9, hyperparams=None,
                 thresh=0.5, min_area_frac=0.004, close_radius=4):
        self.context_window = context_window
        self.hyperparams = hyperparams or dict(n_estimators=300, max_depth=None,
                                               min_samples_leaf=2, n_jobs=-1)
        self.thresh = thresh
        self.min_area_frac = min_area_frac
        self.close_radius = close_radius
        self.model = None
        self.card = None

    def samples_from_polygons(self, rgb, polygons, valid=None, neg_ratio=2.0, rng=None):
        """Positives = pixels inside polygons; negatives = balanced random sample
        outside (optionally restricted to `valid`)."""
        rng = rng or np.random.default_rng(SEED)
        obj = polygon_to_mask(rgb.shape[:2], polygons)
        feats = mask_features(rgb, self.context_window)
        F = feats.shape[-1]
        Xpos = feats[obj]
        neg_pool = ~obj if valid is None else (~obj & valid)
        neg_idx = np.flatnonzero(neg_pool.ravel())
        n_neg = min(neg_idx.size, int(neg_ratio * len(Xpos)))
        pick = rng.choice(neg_idx, size=n_neg, replace=False)
        Xneg = feats.reshape(-1, F)[pick]
        X = np.concatenate([Xpos, Xneg])
        y = np.concatenate([np.ones(len(Xpos), np.int64), np.zeros(len(Xneg), np.int64)])
        return X, y, int(obj.sum())

    def fit(self, X, y, contributing=None):
        import laipro
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(random_state=SEED, **self.hyperparams)
        self.model.fit(X, y)
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        self.card = MaskModelCard(
            laipro_version=laipro.__version__, feature_version=FEATURE_VERSION,
            feature_names=list(MASK_FEATURE_NAMES), seed=SEED,
            context_window=self.context_window, hyperparams=self.hyperparams,
            n_samples=int(len(y)), class_balance={"background": n0, "object": n1},
            contributing_masks=contributing or [])
        return self

    def predict(self, rgb):
        feats = mask_features(rgb, self.context_window)
        H, W, F = feats.shape
        prob = self.model.predict_proba(feats.reshape(-1, F))[:, 1].reshape(H, W)
        return clean_object_mask(prob, self.thresh, self.min_area_frac, self.close_radius)

    def save(self, model_dir):
        import joblib
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, "mask_model.joblib"))
        cfg = dict(context_window=self.context_window, hyperparams=self.hyperparams,
                   thresh=self.thresh, min_area_frac=self.min_area_frac,
                   close_radius=self.close_radius)
        with open(os.path.join(model_dir, "mask_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        if self.card:
            self.card.save(os.path.join(model_dir, "mask_model_card.json"))

    @staticmethod
    def load(model_dir):
        import joblib
        with open(os.path.join(model_dir, "mask_config.json")) as f:
            cfg = json.load(f)
        det = ForeignObjectDetector(**cfg)
        det.model = joblib.load(os.path.join(model_dir, "mask_model.joblib"))
        return det


class MaskCorrectionStore:
    """Append-only cache of (features, labels) for foreign-object corrections."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.cache = os.path.join(model_dir, "mask_samples.npz")
        self.manifest = os.path.join(model_dir, "mask_contributions.json")

    def add(self, rgb, polygons, source_name, detector: ForeignObjectDetector,
            valid=None, rng=None):
        X, y, n_obj = detector.samples_from_polygons(rgb, polygons, valid, rng=rng)
        if os.path.exists(self.cache):
            d = np.load(self.cache)
            X = np.concatenate([d["X"], X]); y = np.concatenate([d["y"], y])
        np.savez_compressed(self.cache, X=X, y=y)
        contribs = self._load()
        contribs.append({"file": source_name, "n_object_px": n_obj,
                         "sha256": _sha256_array(np.asarray(polygons, dtype=object).astype(str))})
        with open(self.manifest, "w") as f:
            json.dump(contribs, f, indent=2)
        return int(len(y))

    def _load(self):
        if os.path.exists(self.manifest):
            with open(self.manifest) as f:
                return json.load(f)
        return []

    def load_samples(self):
        d = np.load(self.cache)
        return d["X"], d["y"], self._load()


# --------------------------------------------------------------- optional SAM

def refine_with_sam(rgb, seed_mask, checkpoint=None, model_type="vit_h"):
    """OPTIONAL crisp-boundary refinement via Segment Anything. Lazy and safe:
    if torch / segment_anything / a checkpoint are unavailable, returns seed_mask
    unchanged so the pipeline never depends on SAM being installed."""
    try:
        import torch  # noqa
        from segment_anything import sam_model_registry, SamPredictor
    except Exception:
        return seed_mask
    if not checkpoint or not os.path.exists(checkpoint):
        return seed_mask
    ys, xs = np.nonzero(seed_mask)
    if xs.size == 0:
        return seed_mask
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to("cuda" if __import__("torch").cuda.is_available() else "cpu")
    pred = SamPredictor(sam)
    pred.set_image((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    masks, _, _ = pred.predict(box=box[None, :], multimask_output=False)
    return masks[0].astype(bool)
