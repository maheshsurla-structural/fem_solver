"""Nonlinear pushover UI (plan §14 GUI-5): run the fiber pushover on a worker
thread — the main window never freezes — with a live progress bar, a
convergence log, a Cancel button, and a base-shear vs displacement curve that
draws as the analysis advances.

The heavy work runs in :class:`PushoverWorker` (a ``QThread``); it streams each
committed step via the engine's ``step_callback`` (through
``nonlinear.run_pushover``'s ``on_step``) and cooperatively stops on Cancel.
Pure Qt + matplotlib (Agg canvas) — headless-constructible under
``QT_QPA_PLATFORM=offscreen``; the worker's logic is exercisable by calling
``run()`` directly (no thread).
"""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QSlider, QSpinBox, QTabWidget, QVBoxLayout,
                               QWidget)

import nonlinear as NL
from nl_results import NonlinearResults

_DOFS_2D = [("Ux", 0), ("Uy", 1), ("Rz", 2)]
_DOFS_3D = [("Ux", 0), ("Uy", 1), ("Uz", 2),
            ("Rx", 3), ("Ry", 4), ("Rz", 5)]


def _dof_items(ndf: int):
    return _DOFS_3D if ndf >= 6 else _DOFS_2D


class PushoverWorker(QThread):
    """Runs ``nonlinear.run_pushover`` off the UI thread, streaming progress."""

    progress = Signal(dict)          # {step, num_steps, disp, shear}
    done = Signal(dict)              # {disp: [...], shear: [...]}
    failed = Signal(str)

    def __init__(self, project, kwargs: dict, case=None):
        super().__init__()
        self._project = project
        self._kwargs = kwargs
        self._case = case                # a NonlinearCase -> run_case, else manual
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:                          # QThread entry point
        try:
            if self._case is not None:
                res = NL.run_case(
                    self._project, self._case,
                    on_step=lambda info: self.progress.emit(info),
                    should_cancel=lambda: self._cancel,
                    capture_fibers=self._kwargs.get("capture_fibers", False),
                    capture_shape=self._kwargs.get("capture_shape", False))
            else:
                res = NL.run_pushover(
                    self._project,
                    on_step=lambda info: self.progress.emit(info),
                    should_cancel=lambda: self._cancel,
                    **self._kwargs,
                )
            self.done.emit(res)
        except Exception as exc:                     # noqa: BLE001
            self.failed.emit(str(exc))


class PushoverDialog(QDialog):
    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Nonlinear pushover")
        self._project = project
        self._worker: PushoverWorker | None = None
        self._disp: list[float] = []
        self._shear: list[float] = []
        self._protocol: str = "monotonic"
        outer = QHBoxLayout(self)

        # ---- left: inputs + progress ----
        left = QWidget()
        form = QFormLayout(left)
        node_ids = [n.id for n in project.nodes]
        free = [n.id for n in project.nodes if not (n.supports and any(n.supports))]
        self.node = self._combo([(str(i), i) for i in node_ids],
                                default=(free[-1] if free else
                                         (node_ids[-1] if node_ids else None)))
        dof_items = _dof_items(project.ndf)
        self.dof = self._combo(dof_items, default=1)
        self.target = self._spin(0.05, unit=project.length_unit, decimals=4)
        self.n_steps = QSpinBox()
        self.n_steps.setRange(2, 2000)
        self.n_steps.setValue(40)
        self.tol = QComboBox()
        for t in (1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8):
            self.tol.addItem(f"{t:.0e}", t)
        self.tol.setCurrentIndex(2)                      # 1e-6 default
        self.max_iter = QSpinBox()
        self.max_iter.setRange(10, 1000)
        self.max_iter.setValue(60)
        self.capture = QCheckBox("Record fiber response")
        self.capture.setChecked(True)
        self.axial = self._spin(0.0, unit=project.force_unit, decimals=1,
                                big=True)
        self.axial_node = self._combo([(str(i), i) for i in node_ids],
                                      default=(free[-1] if free else None))
        self.axial_dof = self._combo(dof_items, default=0)
        # saved nonlinear cases (GUI-4): pick one to run it (monotonic/cyclic/
        # staged), or "— manual —" to use the quick inputs below.
        self.case_combo = self._combo(
            [("— manual —", None)]
            + [(f"{c.id}: {c.name}", c.id)
               for c in getattr(project, "nonlinear_cases", [])])
        self.case_combo.currentIndexChanged.connect(self._on_case_changed)
        form.addRow("Case", self.case_combo)
        form.addRow("Control node", self.node)
        form.addRow("Push DOF", self.dof)
        form.addRow(f"Target [{project.length_unit}]", self.target)
        form.addRow("Steps", self.n_steps)
        form.addRow("Convergence tol", self.tol)
        form.addRow("Max iterations", self.max_iter)
        form.addRow(f"Axial preload [{project.force_unit}]", self.axial)
        form.addRow("Axial node", self.axial_node)
        form.addRow("Axial DOF", self.axial_dof)
        form.addRow(self.capture)
        # manual inputs disabled while a saved case drives the run
        self._manual = [self.node, self.dof, self.target, self.n_steps,
                        self.tol, self.max_iter, self.axial, self.axial_node,
                        self.axial_dof]

        self.bar = QProgressBar()
        form.addRow(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        form.addRow(self.log)

        row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        row.addWidget(self.close_btn)
        form.addRow(row)
        outer.addWidget(left, 0)

        # ---- right: tabs (curve | fiber stress | deformed shape) + shared step controls ----
        # Step-indexed results model (GUI-I1); populated on run completion.
        self._results: NonlinearResults | None = None
        right = QWidget()
        rv = QVBoxLayout(right)
        tabs = QTabWidget()

        self._fig = Figure(figsize=(4.2, 3.4), layout="constrained")
        self._ax = self._fig.add_subplot(111)
        self._canvas = Canvas(self._fig)
        tabs.addTab(self._canvas, "Pushover curve")

        self._ffig = Figure(figsize=(4.2, 3.4), layout="constrained")
        self._fax = self._ffig.add_subplot(111)
        self._fcanvas = Canvas(self._ffig)
        tabs.addTab(self._fcanvas, "Fiber stress")

        self._sfig = Figure(figsize=(4.2, 3.4), layout="constrained")
        self._sax = self._sfig.add_subplot(111)
        self._scanvas = Canvas(self._sfig)
        tabs.addTab(self._scanvas, "Deformed shape")
        self._tabs = tabs
        rv.addWidget(tabs, 1)

        # shared step controls (drive whichever tab is showing)
        crow = QHBoxLayout()
        self.play_btn = QPushButton("▶")               # ▶ play/pause
        self.play_btn.setEnabled(False)
        self.play_btn.setFixedWidth(32)
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._toggle_play)
        self._timer = QTimer(self)
        self._timer.setInterval(120)                        # ms per step
        self._timer.timeout.connect(self._advance_step)
        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setEnabled(False)
        self.step_slider.valueChanged.connect(self._on_step)
        self.step_lbl = QLabel("step —")
        self.fiber_mode = QComboBox()
        self.fiber_mode.addItems(["stress", "strain"])
        self.fiber_mode.currentIndexChanged.connect(self._draw_fibers)
        self.shape_scale = QDoubleSpinBox()
        self.shape_scale.setRange(0.0, 1.0e6)
        self.shape_scale.setValue(20.0)
        self.shape_scale.valueChanged.connect(self._draw_shape)
        crow.addWidget(self.play_btn)
        crow.addWidget(QLabel("Step"))
        crow.addWidget(self.step_slider, 1)
        crow.addWidget(self.step_lbl)
        crow.addWidget(QLabel("fiber:"))
        crow.addWidget(self.fiber_mode)
        crow.addWidget(QLabel("shape ×"))
        crow.addWidget(self.shape_scale)
        rv.addLayout(crow)

        # export row (GUI-7): enabled once a run has results
        erow = QHBoxLayout()
        erow.addWidget(QLabel("Export:"))
        self._export_btns = []
        for label, cb in (("Curve CSV…", self._export_curve_csv),
                          ("Fibers CSV…", self._export_fibers_csv),
                          ("Image…", self._save_plot),
                          ("Report…", self._export_report)):
            b = QPushButton(label)
            b.setEnabled(False)
            b.clicked.connect(cb)
            erow.addWidget(b)
            self._export_btns.append(b)
        erow.addStretch(1)
        rv.addLayout(erow)

        self._canvas.setMinimumWidth(420)
        outer.addWidget(right, 1)
        self._draw_curve()
        self._draw_fibers()
        self._draw_shape()

    def _on_step(self, *_) -> None:
        self._draw_fibers()
        self._draw_shape()

    # ------------------------------------------------ step animation
    def _toggle_play(self, on: bool) -> None:
        self.play_btn.setText("❚❚" if on else "▶")   # ❚❚ / ▶
        if on:
            self._timer.start()
        else:
            self._timer.stop()

    def _advance_step(self) -> None:
        n = self.step_slider.maximum()
        if n <= 0:
            self.play_btn.setChecked(False)
            return
        nxt = self.step_slider.value() + 1
        self.step_slider.setValue(0 if nxt > n else nxt)   # loop

    # ------------------------------------------------ small widget helpers
    @staticmethod
    def _combo(items, default=None):
        c = QComboBox()
        for label, data in items:
            c.addItem(label, data)
        if default is not None:
            i = c.findData(default)
            if i >= 0:
                c.setCurrentIndex(i)
        return c

    @staticmethod
    def _spin(value, *, unit="", decimals=4, big=False):
        s = QDoubleSpinBox()
        s.setRange(-1e15 if big else -1e6, 1e15 if big else 1e6)
        s.setDecimals(decimals)
        s.setValue(value)
        return s

    @staticmethod
    def _set_combo(combo, value) -> None:
        i = combo.findData(value)
        if i >= 0:
            combo.setCurrentIndex(i)

    # ------------------------------------------------ saved cases (GUI-4)
    def _selected_case(self):
        cid = self.case_combo.currentData()
        return self._project.nonlinear_case(cid) if cid else None

    def _on_case_changed(self, *_) -> None:
        case = self._selected_case()
        for w in self._manual:
            w.setEnabled(case is None)                # case drives the run
        if case is None:
            return
        self._set_combo(self.node, case.control_node)
        self._set_combo(self.dof, case.control_dof)
        self.target.setValue(case.target)
        self.n_steps.setValue(case.n_steps)
        self._set_combo(self.tol, case.tol)
        self.max_iter.setValue(case.max_iter)
        self.axial.setValue(case.axial)
        if case.axial_node:
            self._set_combo(self.axial_node, case.axial_node)
        self._set_combo(self.axial_dof, case.axial_dof)

    # ------------------------------------------------ run lifecycle
    def _kwargs(self) -> dict:
        axial = float(self.axial.value())
        return dict(
            control_node=self.node.currentData(),
            control_dof=self.dof.currentData(),
            target=float(self.target.value()),
            n_steps=int(self.n_steps.value()),
            axial=axial,
            axial_node=(self.axial_node.currentData() if axial else None),
            axial_dof=self.axial_dof.currentData(),
            tol=float(self.tol.currentData()),
            max_iter=int(self.max_iter.value()),
            capture_fibers=self.capture.isChecked(),
            capture_shape=self.capture.isChecked(),
        )

    def _start(self) -> None:
        self.play_btn.setChecked(False)                      # stop any animation
        self.play_btn.setEnabled(False)
        self._disp, self._shear = [], []
        self.log.clear()
        case = self._selected_case()
        total = (NL.case_total_steps(self._project, case) if case is not None
                 else int(self.n_steps.value()))
        self.bar.setRange(0, total)
        self.bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._worker = PushoverWorker(self._project, self._kwargs(), case=case)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.log.appendPlainText("cancelling…")

    def _on_progress(self, info: dict) -> None:
        self._disp.append(info["disp"])
        self._shear.append(info["shear"])
        self.bar.setValue(int(info["step"]))
        self.log.appendPlainText(
            f"step {info['step']}/{info['num_steps']}: "
            f"d={info['disp']:.4g}, V={info['shear']:.4g}")
        self._draw_curve()

    def _on_done(self, res: dict) -> None:
        self._results = NonlinearResults.from_run(res)
        self._disp = res.get("disp", self._disp)
        self._shear = res.get("shear", self._shear)
        self._protocol = res.get("protocol", "monotonic")
        n = self._results.n_steps
        if n:
            self.step_slider.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.step_slider.setRange(0, n - 1)
            self.step_slider.setValue(n - 1)                   # show last step
            self._on_step()
        for b in self._export_btns:
            b.setEnabled(bool(self._disp))
        ms = self._results.accept_milestones if self._results else {}
        if ms:
            parts = [f"{lvl} @ d={ms[lvl]['disp']:.4g}"
                     for lvl in ("IO", "LS", "CP") if lvl in ms]
            self.log.appendPlainText("ASCE 41: " + "; ".join(parts))
        self._finish(f"done — {len(self._disp)} steps, "
                     f"V_max = {max(self._shear) if self._shear else 0:.4g}")

    def _on_failed(self, msg: str) -> None:
        self._finish(f"FAILED: {msg}")

    def _finish(self, msg: str) -> None:
        self.log.appendPlainText(msg)
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._draw_curve()

    def _draw_curve(self) -> None:
        self._ax.clear()
        cyclic = self._protocol == "cyclic"
        if self._disp:
            self._ax.plot(self._disp, self._shear,
                          "-" if cyclic else "-o",
                          ms=3, lw=1.2 if cyclic else 1.6)
            if cyclic:                              # origin cross-hairs for loops
                self._ax.axhline(0, color="0.6", lw=0.6)
                self._ax.axvline(0, color="0.6", lw=0.6)
        self._ax.set_xlabel(f"displacement [{self._project.length_unit}]")
        self._ax.set_ylabel(f"base shear [{self._project.force_unit}]")
        self._ax.set_title("Cyclic hysteresis" if cyclic else "Pushover",
                           fontsize=9)
        self._ax.grid(True, alpha=0.25)
        self._canvas.draw_idle()

    def _draw_fibers(self, *_) -> None:
        self._ffig.clear()
        ax = self._ffig.add_subplot(111)
        self._fax = ax
        if self._results is None or not self._results.has_fibers:
            ax.text(0.5, 0.5,
                    "(run with 'Record fiber response' to see fiber stresses)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="0.5")
            ax.set_axis_off()
            self._fcanvas.draw_idle()
            return
        n_steps = self._results.n_steps
        step = max(0, min(self.step_slider.value(), n_steps - 1))
        frame = self._results.step(step).fibers
        y = np.array([t[0] for t in frame])
        z = np.array([t[1] for t in frame])
        if self.fiber_mode.currentText() == "strain":
            val = np.array([t[3] for t in frame])
            label = "fiber strain"
        else:
            val = np.array([t[2] for t in frame]) / 1.0e6        # MPa
            label = "fiber stress (MPa)"
        vmax = max(float(np.abs(val).max()), 1e-12)
        sc = ax.scatter(z, y, c=val, cmap="coolwarm", s=10,
                        vmin=-vmax, vmax=vmax)
        self._ffig.colorbar(sc, ax=ax, label=label)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(f"z [{self._project.length_unit}]")
        ax.set_ylabel(f"y [{self._project.length_unit}]")
        d = self._disp[step] if step < len(self._disp) else 0.0
        ax.set_title(f"Fiber stress · step {step + 1}/{n_steps} "
                     f"(d = {d:.4g})", fontsize=9)
        self.step_lbl.setText(f"step {step + 1}/{n_steps}")
        self._fcanvas.draw_idle()

    def _draw_shape(self, *_) -> None:
        self._sfig.clear()
        ax = self._sfig.add_subplot(111)
        self._sax = ax
        if self._results is None or not self._results.has_shape:
            ax.text(0.5, 0.5,
                    "(run with 'Record fiber response' to see the deformed shape)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="0.5")
            ax.set_axis_off()
            self._scanvas.draw_idle()
            return
        n_steps = self._results.n_steps
        step = max(0, min(self.step_slider.value(), n_steps - 1))
        st = self._results.step(step)
        frame = st.node_disp
        dmg = st.member_damage or {}
        scale = float(self.shape_scale.value())
        nodes = {n.id: (n.x, n.y) for n in self._project.nodes}

        # undeformed reference (light gray)
        for mb in self._project.members:
            (x1, y1), (x2, y2) = nodes[mb.n1], nodes[mb.n2]
            ax.plot([x1, x2], [y1, y2], "-", color="0.85", lw=1.0, zorder=1)

        # deformed, members colored by peak fiber strain (damage)
        vmax = max([abs(v) for v in dmg.values()] + [1e-9])
        cmap = colormaps["YlOrRd"]
        norm = Normalize(0.0, vmax)

        def _d(nid):
            d = frame.get(nid) or (0.0, 0.0)          # 2-D (dx,dy) or 3-D (…,dz)
            dx, dy = d[0], d[1]                        # x-y projection
            return nodes[nid][0] + dx * scale, nodes[nid][1] + dy * scale

        for mb in self._project.members:
            (dx1, dy1), (dx2, dy2) = _d(mb.n1), _d(mb.n2)
            color = cmap(norm(dmg[mb.id])) if mb.id in dmg else "0.4"
            ax.plot([dx1, dx2], [dy1, dy2], "-", color=color, lw=3.0, zorder=2)
        px = [_d(n.id)[0] for n in self._project.nodes]
        py = [_d(n.id)[1] for n in self._project.nodes]
        ax.scatter(px, py, s=14, color="k", zorder=3)

        if dmg:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            self._sfig.colorbar(sm, ax=ax, label="peak fiber strain")
        ax.set_aspect("equal", "datalim")
        ax.set_xlabel(f"x [{self._project.length_unit}]")
        ax.set_ylabel(f"y [{self._project.length_unit}]")
        ax.set_title(f"Deformed shape · step {step + 1}/{n_steps} "
                     f"(×{scale:.0f})", fontsize=9)
        self._scanvas.draw_idle()

    # ------------------------------------------------ export / report (GUI-7)
    def _result(self) -> dict:
        if self._results is not None:
            return self._results.to_dict()
        return {"disp": self._disp, "shear": self._shear,
                "protocol": self._protocol}

    def _report_meta(self) -> dict:
        lu, fu = self._project.length_unit, self._project.force_unit
        dofs = {0: "Ux", 1: "Uy", 2: "Rz"}
        meta = {"Project": self._project.name, "Steps": str(len(self._disp))}
        case = self._selected_case()
        if case is not None:
            meta["Case"] = f"{case.id}: {case.name}"
            meta["Control"] = (f"node {case.control_node} "
                               f"{dofs.get(case.control_dof, '?')}")
            meta["Target"] = f"{case.target:g} {lu}"
            if case.axial:
                meta["Axial preload"] = f"{case.axial:g} {fu}"
            if case.continue_from:
                meta["Continues from"] = f"case {case.continue_from}"
        else:
            meta["Control"] = (f"node {self.node.currentData()} "
                               f"{dofs.get(self.dof.currentData(), '?')}")
            meta["Target"] = f"{self.target.value():g} {lu}"
        return meta

    def _curve_png_b64(self) -> str:
        import base64
        import io
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", dpi=130)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _export_curve_csv(self) -> None:
        if not self._disp:
            return
        import nl_report
        default = ("hysteresis.csv" if self._protocol == "cyclic"
                   else "pushover_curve.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export curve", default,
                                              "CSV (*.csv)")
        if not path:
            return
        self._write(path, nl_report.curve_csv(
            self._disp, self._shear, length_unit=self._project.length_unit,
            force_unit=self._project.force_unit))

    def _export_fibers_csv(self) -> None:
        if self._results is None or not self._results.has_fibers:
            self.log.appendPlainText("no fiber frames to export "
                                     "(run with 'Record fiber response')")
            return
        import nl_report
        step = max(0, min(self.step_slider.value(), self._results.n_steps - 1))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export fibers", f"fibers_step{step + 1}.csv", "CSV (*.csv)")
        if not path:
            return
        self._write(path, nl_report.fibers_csv(
            self._results.step(step).fibers,
            length_unit=self._project.length_unit))

    def _save_plot(self) -> None:
        fig = {0: self._fig, 1: self._ffig,
               2: self._sfig}.get(self._tabs.currentIndex(), self._fig)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", "plot.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not path:
            return
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self.log.appendPlainText(f"saved {path}")
        except Exception as exc:                          # noqa: BLE001
            self.log.appendPlainText(f"save failed: {exc}")

    def _export_report(self) -> None:
        if not self._disp:
            return
        import nl_report
        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", "nonlinear_report.html", "HTML (*.html)")
        if not path:
            return
        html = nl_report.report_html(
            self._result(), self._report_meta(),
            length_unit=self._project.length_unit,
            force_unit=self._project.force_unit,
            curve_png_b64=self._curve_png_b64())
        self._write(path, html)

    def _write(self, path: str, text: str) -> None:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            self.log.appendPlainText(f"saved {path}")
        except Exception as exc:                          # noqa: BLE001
            self.log.appendPlainText(f"save failed: {exc}")
