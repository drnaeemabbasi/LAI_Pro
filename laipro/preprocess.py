"""
Preliminary processing: decode-once, downscale, and cache.

The slow part of processing your plots is decoding the full-resolution RAW
(.NEF) files -- and that cost is paid on every run regardless of the analysis
resolution. This layer pays it ONCE: it decodes each source image, downscales to
a chosen working resolution (default 2500 px on the long side, which the
resolution test showed is accuracy-equivalent to full res), and writes a fast-
loading PNG cache under <folder>/laipro_cache/. Later runs load the cache and
skip the RAW decode entirely.

It also standardises the working resolution across a plot, which keeps results
comparable between images, dates, and sites.

A manifest records the source file hash + target size, so the cache is rebuilt
only when a source changes or the target resolution changes.
"""

from __future__ import annotations
import json
import os
import numpy as np
from PIL import Image

from .io import load_rgb, find_images
from .config import sha256_file

CACHE_DIRNAME = "laipro_cache"
DEFAULT_TARGET = 2500


def prepare_folder(folder, target_side=DEFAULT_TARGET, force=False, progress=None):
    """Build/refresh the downscaled cache. Returns a summary dict.

    progress: optional callable(i, n, name) for progress reporting.
    """
    cache = os.path.join(folder, CACHE_DIRNAME)
    os.makedirs(cache, exist_ok=True)
    manifest_path = os.path.join(cache, "manifest.json")
    prev = {}
    if os.path.exists(manifest_path) and not force:
        with open(manifest_path) as fh:
            prev = json.load(fh)
    prev_files = prev.get("files", {}) if prev.get("target_side") == target_side else {}

    src = find_images(folder)      # RAW-preferring; ignores the cache subfolder
    out = {"target_side": target_side, "files": {}}
    converted = 0
    for i, s in enumerate(src):
        stem = os.path.splitext(os.path.basename(s))[0]
        dst = os.path.join(cache, stem + ".png")
        h = sha256_file(s)
        rec = prev_files.get(stem)
        if not force and os.path.exists(dst) and rec and rec.get("sha256") == h:
            out["files"][stem] = rec                      # unchanged -> reuse
        else:
            rgb = load_rgb(s, target_side)                # decode + downscale (the slow step)
            Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(dst)
            out["files"][stem] = {"source": os.path.basename(s), "sha256": h,
                                  "cache": stem + ".png"}
            converted += 1
        if progress:
            progress(i + 1, len(src), stem)

    with open(manifest_path, "w") as fh:
        json.dump(out, fh, indent=2)
    return {"cache_dir": cache, "n_images": len(src), "converted": converted,
            "reused": len(src) - converted, "target_side": target_side}


def cached_images(folder, target_side=None):
    """Return the sorted cached image paths if a matching cache exists, else None."""
    cache = os.path.join(folder, CACHE_DIRNAME)
    mp = os.path.join(cache, "manifest.json")
    if not os.path.exists(mp):
        return None
    with open(mp) as fh:
        m = json.load(fh)
    if target_side is not None and m.get("target_side") != target_side:
        return None
    paths = [os.path.join(cache, v["cache"]) for v in m.get("files", {}).values()]
    paths = [p for p in paths if os.path.exists(p)]
    return sorted(paths) if paths else None
