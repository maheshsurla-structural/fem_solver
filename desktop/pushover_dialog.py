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
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
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

        # ---- right: tabs (pushover curve | fiber-stress contour) ----
        self._frames: list = []
        self._cbar = None
        tabs = QTabWidget()

        self._fig = Figure(figsize=(4.2, 3.4), layout="constrained")
        self._ax = self._fig.add_subplot(111)
        self._canvas = Canvas(self._fig)
        tabs.addTab(self._canvas, "Pushover curve")

        fib = QWidget()
        fibv = QVBoxLayout(fib)
        self._ffig = Figure(figsize=(4.2, 3.4), layout="constrained")
        self._fax = self._ffig.add_subplot(111)
        self._fcanvas = Canvas(self._ffig)
        fibv.addWidget(self._fcanvas, 1)
        srow = QHBoxLayout()
        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setEnabled(False)
        self.step_slider.valueChanged.connect(self._draw_fibers)
        self.step_lbl = QLabel("step —")
        self.fiber_mode = QComboBox()
        self.fiber_mode.addItems(["stress", "strain"])
        self.fiber_mode.currentIndexChanged.connect(self._draw_fibers)
        srow.addWidget(QLabel("Step"))
        srow.addWidget(self.step_slider, 1)
        srow.addWidget(self.step_lbl)
        srow.addWidget(self.fiber_mode)
        fibv.addLayout(srow)
        tabs.addTab(fib, "Fiber stress")
        self._tabs = tabs

        self._canvas.setMinimumWidth(400)
        outer.addWidget(tabs, 1)
        self._draw_curve()
        self._draw_fibers()

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
        )

    def _start(self) -> None:
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
        if self._frames:
            self.step_slider.setEnabled(True)
            self.step_slider.setRange(0, len(self._frames) - 1)
            self.step_slider.setValue(len(self._frames) - 1)   # show last step
            self._draw_fibers()
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
