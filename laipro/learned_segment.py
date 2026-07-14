"""
Learnable vegetation segmentation that improves from manual corrections.

Design goals
------------
* Hybrid: the classical ExG/Otsu path (laipro.segment) stays the deterministic
  default. A trained LearnedSegmenter is an OPTIONAL drop-in that returns the
  same thing -- a boolean vegetation mask.
* Learns from sparse labels: the user (via the future napari GUI, or any label
  image) marks some pixels soil vs vegetation; the model learns the colour/texture
  of each class and classifies every pixel of every image -- CAN_EYE's interactive
  idea, but persistent.
* Accumulates: each new set of corrections is appended to a sample cache in the
  model directory and the model is retrained on ALL accumulated corrections, so
  it genuinely "improves future processing" instead of forgetting.
* Reproducible: fixed random seed, a model card recording the feature version,
  backend, hyperparameters, class balance, and the SHA-256 of every contributing
  label image. Re-training from the same cache yields the same model.

Backends
--------
* "rf"  (default): scikit-learn RandomForest. Robust with sparse labels, fast,
  CPU, fully reproducible via random_state. Recommended default.
* "mlp" (optional): a small PyTorch MLP on GPU (RTX 4080 / 3080 Ti). Used only if
  `torch` is installed; enable when you have abundant labels. Lazy-imported.

Label convention (single-channel image): 0 = unlabeled/ignore, 1 = soil,
2 = vegetation. This matches how the GUI will export brush strokes.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import os
import numpy as np

from .features import extract_features, FEATURE_VERSION, FEATURE_NAMES

SEED = 20260101
LABEL_IGNORE, LABEL_SOIL, LABEL_VEG = 0, 1, 2


# ------------------------------------------------------------- sample extraction

def samples_from_label_image(rgb, label_img, context_window=9):
    """Return (X, y) for annotated pixels only. label_img uses the 0/1/2 codes."""
    feats, _ = extract_features(rgb, context_window)
    lab = np.asarray(label_img)
    ann = lab != LABEL_IGNORE
    X = feats[ann]
    y = (lab[ann] == LABEL_VEG).astype(np.int64)   # 1 = vegetation, 0 = soil
    return X, y


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ------------------------------------------------------------- model card

@dataclass
class ModelCard:
    laipro_version: str
    feature_version: int
    feature_names: list
    backend: str
    seed: int
    context_window: int
    hyperparams: dict
    n_samples: int
    class_balance: dict            # {"soil": n0, "veg": n1}
    contributing_labels: list      # [{"file":..., "sha256":..., "n":...}]

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ------------------------------------------------------------- segmenter

class LearnedSegmenter:
    """Trainable, persistable, accumulating vegetation classifier."""

    def __init__(self, backend="rf", context_window=9, hyperparams=None):
        self.backend = backend
        self.context_window = context_window
        self.hyperparams = hyperparams or self._default_hyperparams(backend)
        self.model = None
        self._scaler = None          # (mean, std) for mlp
        self.card: ModelCard | None = None

    @staticmethod
    def _default_hyperparams(backend):
        if backend == "rf":
            return dict(n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1)
        if backend == "mlp":
            return dict(hidden=(64, 64), epochs=60, lr=1e-3, batch=8192)
        raise ValueError(f"unknown backend {backend}")

    # ---- training ----
    def fit(self, X, y, contributing=None):
        import laipro
        if self.backend == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(random_state=SEED, **self.hyperparams)
            self.model.fit(X, y)
        elif self.backend == "mlp":
            self._fit_mlp(X, y)
        else:
            raise ValueError(self.backend)

        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        self.card = ModelCard(
            laipro_version=laipro.__version__, feature_version=FEATURE_VERSION,
            feature_names=list(FEATURE_NAMES), backend=self.backend, seed=SEED,
            context_window=self.context_window, hyperparams=self.hyperparams,
            n_samples=int(len(y)), class_balance={"soil": n0, "veg": n1},
            contributing_labels=contributing or [],
        )
        return self

    def _fit_mlp(self, X, y):
        import torch, torch.nn as nn
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mean, std = X.mean(0), X.std(0) + 1e-6
        self._scaler = (mean.astype(np.float32), std.astype(np.float32))
        Xs = (X - mean) / std
        h = self.hyperparams
        layers, prev = [], X.shape[1]
        for w in h["hidden"]:
            layers += [nn.Linear(prev, w), nn.ReLU()]; prev = w
        layers += [nn.Linear(prev, 2)]
        net = nn.Sequential(*layers).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=h["lr"])
        lossf = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
        yt = torch.tensor(y, dtype=torch.long, device=dev)
        n = len(y)
        for _ in range(h["epochs"]):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, h["batch"]):
                idx = perm[i:i + h["batch"]]
                opt.zero_grad()
                lossf(net(Xt[idx]), yt[idx]).backward()
                opt.step()
        net.eval()
        self.model = net

    # ---- prediction ----
    def predict(self, rgb):
        feats, _ = extract_features(rgb, self.context_window)
        H, W, F = feats.shape
        X = feats.reshape(-1, F)
        if self.backend == "rf":
            prob = self.model.predict_proba(X)[:, 1]
        else:
            import torch
            mean, std = self._scaler
            dev = next(self.model.parameters()).device
            with torch.no_grad():
                logits = self.model(torch.tensor((X - mean) / std, dtype=torch.float32, device=dev))
                prob = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        return (prob.reshape(H, W) >= 0.5)

    # ---- persistence ----
    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        if self.backend == "rf":
            import joblib
            joblib.dump(self.model, os.path.join(model_dir, "model.joblib"))
        else:
            import torch
            torch.save({"state": self.model.state_dict(), "scaler": self._scaler,
                        "hyperparams": self.hyperparams}, os.path.join(model_dir, "model.pt"))
        if self.card:
            self.card.save(os.path.join(model_dir, "model_card.json"))

    @staticmethod
    def load(model_dir):
        with open(os.path.join(model_dir, "model_card.json")) as f:
            card = json.load(f)
        seg = LearnedSegmenter(backend=card["backend"],
                               context_window=card["context_window"],
                               hyperparams=card["hyperparams"])
        if card["backend"] == "rf":
            import joblib
            seg.model = joblib.load(os.path.join(model_dir, "model.joblib"))
        else:
            import torch, torch.nn as nn
            blob = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
            h = blob["hyperparams"]
            layers, prev = [], len(card["feature_names"])
            for w in h["hidden"]:
                layers += [nn.Linear(prev, w), nn.ReLU()]; prev = w
            layers += [nn.Linear(prev, 2)]
            net = nn.Sequential(*layers)
            net.load_state_dict(blob["state"]); net.eval()
            seg.model, seg._scaler = net, blob["scaler"]
        if card["feature_version"] != FEATURE_VERSION:
            raise RuntimeError("feature version mismatch; retrain the model")
        return seg


# ------------------------------------------------------------- accumulating store

class CorrectionStore:
    """Append-only cache of (features, labels) samples in a model directory, so
    retraining uses ALL past corrections. This is the 'learn from previous manual
    fine-tunes' mechanism, decoupled from the original images."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.cache = os.path.join(model_dir, "samples.npz")
        self.manifest = os.path.join(model_dir, "contributions.json")

    def add(self, rgb, label_img, source_name, context_window=9):
        X, y = samples_from_label_image(rgb, label_img, context_window)
        if os.path.exists(self.cache):
            d = np.load(self.cache)
            X = np.concatenate([d["X"], X]); y = np.concatenate([d["y"], y])
        np.savez_compressed(self.cache, X=X, y=y)
        contribs = self._load_manifest()
        contribs.append({"file": source_name, "sha256": _sha256_array(np.asarray(label_img)),
                         "n": int((np.asarray(label_img) != LABEL_IGNORE).sum())})
        with open(self.manifest, "w") as f:
            json.dump(contribs, f, indent=2)
        return int(len(y))

    def _load_manifest(self):
        if os.path.exists(self.manifest):
            with open(self.manifest) as f:
                return json.load(f)
        return []

    def load_samples(self):
        d = np.load(self.cache)
        return d["X"], d["y"], self._load_manifest()
