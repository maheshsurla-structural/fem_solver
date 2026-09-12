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
                               QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit,
                               QProgressBar, QPushButton, QSlider, QSpinBox,
                               QTabWidget, QVBoxLayout, QWidget)

import nonlinear as NL

_DOFS = [("Ux", 0), ("Uy", 1), ("Rz", 2)]


class PushoverWorker(QThread):
    """Runs ``nonlinear.run_pushover`` off the UI thread, streaming progress."""

    progress = Signal(dict)          # {step, num_steps, disp, shear}
    done = Signal(dict)              # {disp: [...], shear: [...]}
    failed = Signal(str)

    def __init__(self, project, kwargs: dict):
        super().__init__()
        self._project = project
        self._kwargs = kwargs
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:                          # QThread entry point
        try:
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
        outer = QHBoxLayout(self)

        # ---- left: inputs + progress ----
        left = QWidget()
        form = QFormLayout(left)
        node_ids = [n.id for n in project.nodes]
        free = [n.id for n in project.nodes if not (n.supports and any(n.supports))]
        self.node = self._combo([(str(i), i) for i in node_ids],
                                default=(free[-1] if free else
                                         (node_ids[-1] if node_ids else None)))
        self.dof = self._combo(_DOFS, default=1)
        self.target = self._spin(0.05, unit=project.length_unit, decimals=4)
        self.n_steps = QSpinBox()
        self.n_steps.setRange(2, 2000)
        self.n_steps.setValue(40)
        self.capture = QCheckBox("Record fiber response")
        self.capture.setChecked(True)
        self.axial = self._spin(0.0, unit=project.force_unit, decimals=1,
                                big=True)
        self.axial_node = self._combo([(str(i), i) for i in node_ids],
                                      default=(free[-1] if free else None))
        self.axial_dof = self._combo(_DOFS, default=0)
        form.addRow("Control node", self.node)
        form.addRow("Push DOF", self.dof)
        form.addRow(f"Target [{project.length_unit}]", self.target)
        form.addRow("Steps", self.n_steps)
        form.addRow(f"Axial preload [{project.force_unit}]", self.axial)
        form.addRow("Axial node", self.axial_node)
        form.addRow("Axial DOF", self.axial_dof)
        form.addRow(self.capture)

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
        self._frames: list = []
        self._shape_frames: list = []
        self._damage_frames: list = []
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
            capture_fibers=self.capture.isChecked(),
            capture_shape=self.capture.isChecked(),
        )

    def _start(self) -> None:
        self.play_btn.setChecked(False)                      # stop any animation
        self.play_btn.setEnabled(False)
        self._disp, self._shear = [], []
        self.log.clear()
        self.bar.setRange(0, int(self.n_steps.value()))
        self.bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._worker = PushoverWorker(self._project, self._kwargs())
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
        self._disp = res.get("disp", self._disp)
        self._shear = res.get("shear", self._shear)
        self._frames = res.get("fiber_frames", [])
        self._shape_frames = res.get("shape_frames", [])
        self._damage_frames = res.get("damage_frames", [])
        n = max(len(self._frames), len(self._shape_frames))
        if n:
            self.step_slider.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.step_slider.setRange(0, n - 1)
            self.step_slider.setValue(n - 1)                   # show last step
            self._on_step()
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
        if self._disp:
            self._ax.plot(self._disp, self._shear, "-o", ms=3, lw=1.6)
        self._ax.set_xlabel(f"displacement [{self._project.length_unit}]")
        self._ax.set_ylabel(f"base shear [{self._project.force_unit}]")
        self._ax.set_title("Pushover", fontsize=9)
        self._ax.grid(True, alpha=0.25)
        self._canvas.draw_idle()

    def _draw_fibers(self, *_) -> None:
        self._ffig.clear()
        ax = self._ffig.add_subplot(111)
        self._fax = ax
        if not self._frames:
            ax.text(0.5, 0.5,
                    "(run with 'Record fiber response' to see fiber stresses)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="0.5")
            ax.set_axis_off()
            self._fcanvas.draw_idle()
            return
        step = max(0, min(self.step_slider.value(), len(self._frames) - 1))
        frame = self._frames[step]
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
        ax.set_title(f"Fiber stress · step {step + 1}/{len(self._frames)} "
                     f"(d = {d:.4g})", fontsize=9)
        self.step_lbl.setText(f"step {step + 1}/{len(self._frames)}")
        self._fcanvas.draw_idle()

    def _draw_shape(self, *_) -> None:
        self._sfig.clear()
        ax = self._sfig.add_subplot(111)
        self._sax = ax
        if not self._shape_frames:
            ax.text(0.5, 0.5,
                    "(run with 'Record fiber response' to see the deformed shape)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="0.5")
            ax.set_axis_off()
            self._scanvas.draw_idle()
            return
        step = max(0, min(self.step_slider.value(), len(self._shape_frames) - 1))
        frame = self._shape_frames[step]
        dmg = self._damage_frames[step] if step < len(self._damage_frames) else {}
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
            dx, dy = frame.get(nid, (0.0, 0.0))
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
        ax.set_title(f"Deformed shape · step {step + 1}/{len(self._shape_frames)} "
                     f"(×{scale:.0f})", fontsize=9)
        self._scanvas.draw_idle()
