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

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QLabel, QPlainTextEdit,
                               QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

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

        # ---- right: pushover curve ----
        self._fig = Figure(figsize=(4.2, 3.4), layout="constrained")
        self._ax = self._fig.add_subplot(111)
        self._canvas = Canvas(self._fig)
        self._canvas.setMinimumWidth(380)
        outer.addWidget(self._canvas, 1)
        self._draw_curve()

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
