"""
napari view layer for laipro.

A thin binding between napari layers/widgets and GuiController (which holds all
logic). napari is imported lazily so importing `laipro` never requires Qt.

Workflow in the window:
  * Open a plot folder; navigate images (Prev/Next).
  * Paint the "veg labels" layer (brush value 1 = soil, 2 = vegetation) on a few
    images, then "Train segmenter" -> learned segmentation, accumulating.
  * Draw polygons on the "object polygons" layer around the operator/boot, then
    "Train mask" -> learned foreign-object detector. "Auto-mask" predicts the
    object so you can edit the polygons before applying.
  * "Toggle vegetation" previews the current segmentation; "Process folder" runs
    the full DHP computation (FCOVER, PAI, ALA, clumping, FAPAR).

Layers use napari's (row, col) = (y, x) order; polygons are converted to (x, y)
for the laipro functions.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from .gui_controller import GuiController
from .learned_segment import LABEL_SOIL, LABEL_VEG


def _shapes_to_polygons(shapes_data):
    # napari polygon vertices are (row=y, col=x); laipro wants (x, y)
    return [[(float(v[1]), float(v[0])) for v in poly] for poly in shapes_data]


class LaiViewer:
    def __init__(self, controller: GuiController | None = None):
        import napari
        self.ctrl = controller or GuiController()
        self.viewer = napari.Viewer(title="laipro - DHP LAI")
        self._img = None
        self._veg_lbl = None
        self._polys = None
        self._veg_preview = None
        self._mask_preview = None
        self._build_widgets()

    # ---- layer refresh ----
    def _refresh_image(self):
        rgb = self.ctrl.rgb
        if self._img is None:
            self._img = self.viewer.add_image(rgb, name="photo", rgb=True)
            self._veg_lbl = self.viewer.add_labels(
                np.zeros(rgb.shape[:2], np.uint8), name="veg labels (1=soil,2=veg)")
            self._polys = self.viewer.add_shapes(
                name="object polygons", shape_type="polygon",
                edge_color="red", face_color="transparent", edge_width=3)
            self._veg_preview = self.viewer.add_labels(
                np.zeros(rgb.shape[:2], np.uint8), name="vegetation", opacity=0.5)
            self._veg_preview.visible = False
            self._mask_preview = self.viewer.add_labels(
                np.zeros(rgb.shape[:2], np.uint8), name="object mask", opacity=0.5)
            self._mask_preview.visible = False
        else:
            self._img.data = rgb
            self._veg_lbl.data = np.zeros(rgb.shape[:2], np.uint8)
            self._polys.data = []
            self._veg_preview.data = np.zeros(rgb.shape[:2], np.uint8)
            self._mask_preview.data = np.zeros(rgb.shape[:2], np.uint8)
        self.viewer.title = f"laipro - {self.ctrl.current_name} " \
                            f"[{self.ctrl.index+1}/{len(self.ctrl.files)}]"

    def _update_veg_preview(self):
        self._veg_preview.data = self.ctrl.vegetation_mask().astype(np.uint8) * LABEL_VEG
        self._veg_preview.visible = True

    def _update_mask_preview(self, mask):
        self._mask_preview.data = mask.astype(np.uint8)
        self._mask_preview.visible = True

    # ---- widgets ----
    def _build_widgets(self):
        from magicgui import magicgui
        from magicgui.widgets import Container, PushButton

        @magicgui(folder={"mode": "d", "label": "plot folder"}, call_button="Open")
        def open_folder(folder=Path(".")):     # value-inferred FileEdit (annotation-free: __future__ annotations stringifies types)
            n = self.ctrl.open_folder(str(folder))
            self._refresh_image()
            self.viewer.status = f"opened {n} images"

        @magicgui(target_side={"label": "working resolution (px)", "min": 800, "max": 8000, "step": 100},
                  call_button="Prepare (decode + cache)")
        def prepare(target_side=2500):
            if not self.ctrl.files:
                self.viewer.status = "open a folder first"; return
            self._run_with_progress(
                work=lambda cb: self.ctrl.prepare_cache(int(target_side), progress=cb),
                total=len(self.ctrl.files), desc="Preparing",
                on_done=lambda s: setattr(self.viewer, "status",
                    f"cache ready: {s['converted']} converted, {s['reused']} reused @ {s['target_side']}px"))

        def nav(delta):
            if self.ctrl.files:
                self.ctrl.step(delta); self._refresh_image()

        prev_b, next_b = PushButton(text="Prev"), PushButton(text="Next")
        prev_b.clicked.connect(lambda: nav(-1))
        next_b.clicked.connect(lambda: nav(+1))

        veg_b = PushButton(text="Toggle vegetation preview")
        veg_b.clicked.connect(lambda: self._update_veg_preview())

        @magicgui(model_dir={"label": "segmenter dir"},
                  backend={"choices": ["rf", "mlp"]}, call_button="Train segmenter")
        def train_seg(model_dir="models/segmenter", backend="rf"):
            n = self.ctrl.train_segmenter_from_labels(
                self._veg_lbl.data, model_dir=str(model_dir), backend=backend)
            self._update_veg_preview()
            self.viewer.status = f"trained segmenter on {n} labeled pixels"

        automask_b = PushButton(text="Auto-mask (predict object)")
        def do_automask():
            mask = self.ctrl.predicted_object_mask()
            self._update_mask_preview(mask)
            self.viewer.status = f"predicted object mask: {100*mask.mean():.1f}% of frame"
        automask_b.clicked.connect(do_automask)

        @magicgui(model_dir={"label": "mask dir"}, call_button="Train mask")
        def train_mask(model_dir="models/mask"):
            polys = _shapes_to_polygons(self._polys.data)
            if not polys:
                self.viewer.status = "draw at least one object polygon first"
                return
            n = self.ctrl.train_mask_from_polygons(polys, model_dir=str(model_dir))
            self._update_mask_preview(self.ctrl.predicted_object_mask())
            self.viewer.status = f"trained mask detector on {n} samples"

        proc_b = PushButton(text="Process folder (DHP)")
        def do_process():
            if not self.ctrl.files:
                self.viewer.status = "open a folder first"; return
            self._run_with_progress(
                work=lambda cb: self.ctrl.process_folder(progress=cb),
                total=len(self.ctrl.files), desc="Processing",
                on_done=lambda res: setattr(self.viewer, "status",
                    f"PAIeff={res[1]['PAI_effective']}  PAItrue={res[1]['PAI_true']}  "
                    f"FCOVER={res[1]['mean_FCOVER_pct']}%  (report in laipro_results)"))
        proc_b.clicked.connect(do_process)

        panel = Container(widgets=[open_folder, prepare, prev_b, next_b, veg_b,
                                   train_seg, automask_b, train_mask, proc_b])
        self.viewer.window.add_dock_widget(panel, area="right", name="laipro")

    def _run_with_progress(self, work, total, desc, on_done):
        """Run `work(progress_cb)` on a background thread (keeps the window
        responsive) with a napari progress bar. Progress from the worker thread
        is marshalled to the GUI thread via a Qt signal."""
        from napari.qt.threading import thread_worker
        from napari.utils import progress
        from qtpy.QtCore import QObject, Signal

        class _Bridge(QObject):
            tick = Signal(int, int, str)

        pbar = progress(total=total, desc=desc)
        bridge = _Bridge()

        def on_tick(i, n, name):
            try:
                pbar.set_description(f"{desc}: {name}")
                pbar.update(1)
            except Exception:
                pass
        bridge.tick.connect(on_tick)          # queued -> runs on GUI thread

        def cb(i, n, name):                   # called from worker thread
            bridge.tick.emit(i, n, name)

        def _close():
            try:
                pbar.close()
            except Exception:
                pass

        @thread_worker
        def run():
            return work(cb)

        self.viewer.status = f"{desc}..."
        w = run()
        w.returned.connect(lambda res: (_close(), on_done(res)))
        w.errored.connect(lambda e: (_close(), setattr(self.viewer, "status", f"error: {e}")))
        w.start()
        self._worker = w                      # keep a reference so it isn't GC'd

    def run(self):
        import napari
        napari.run()


def launch(folder: str | None = None):
    from .config import DHPConfig
    ctrl = GuiController(cfg=DHPConfig())
    view = LaiViewer(ctrl)
    if folder:
        ctrl.open_folder(folder)
        view._refresh_image()
    view.run()
