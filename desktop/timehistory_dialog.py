"""Nonlinear dynamic time-history UI (plan §16 C3 — GUI half).

Imports a ground-motion acceleration record, runs the fiber model under
rigid-base excitation on a worker thread (`nonlinear.run_time_history`), and
plots the monitored DOF's response history (displacement / velocity /
acceleration vs time) — with a live progress bar and Cancel.

The heavy work runs in :class:`TimeHistoryWorker` (a ``QThread``); it streams
each committed step through the engine's transient ``step_callback``. Pure Qt +
matplotlib (Agg canvas), headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit,
                               QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

import nonlinear as NL

# direction label -> (direction string, translational DOF index)
_DIRS = [("X", ("x", 0)), ("Y", ("y", 1)), ("Z", ("z", 2))]
_G = 9.80665                                    # m/s^2 (record-in-g scaling)


def load_accel_record(path: str) -> np.ndarray:
    """Parse an acceleration record: free-format whitespace/comma-separated
    numbers, skipping header lines that contain letters (handles single- or
    multi-column plain text and PEER-style headers)."""
    vals: list[float] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or any(c.isalpha() for c in s):
                continue
            for tok in s.replace(",", " ").split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
    return np.asarray(vals, dtype=float)


class TimeHistoryWorker(QThread):
    progress = Signal(dict)          # {step, num_steps, time, disp}
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, project, accel, dt, kwargs: dict):
        super().__init__()
        self._project = project
        self._accel = accel
        self._dt = dt
        self._kwargs = kwargs
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            res = NL.run_time_history(
                self._project, self._accel, self._dt,
                on_step=lambda info: self.progress.emit(info),
                should_cancel=lambda: self._cancel, **self._kwargs)
            self.done.emit(res)
        except Exception as exc:                     # noqa: BLE001
            self.failed.emit(str(exc))


class TimeHistoryDialog(QDialog):
    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Nonlinear time history")
        self._project = project
        self._worker: TimeHistoryWorker | None = None
        self._accel: np.ndarray | None = None
        self._result: dict | None = None
        outer = QHBoxLayout(self)

        # ---- left: inputs ----
        left = QWidget()
        form = QFormLayout(left)
        node_ids = [n.id for n in project.nodes]
        free = [n.id for n in project.nodes
                if not (n.supports and any(n.supports))]
        self.node = self._combo([(str(i), i) for i in node_ids],
                                default=(free[-1] if free else
                                         (node_ids[-1] if node_ids else None)))
        self.direction = self._combo(_DIRS, default=("y", 1))
        form.addRow("Monitor node", self.node)
        form.addRow("Direction", self.direction)

        self.rec_btn = QPushButton("Load record…")
        self.rec_btn.clicked.connect(self._load_record)
        self.rec_lbl = QLabel("(no record)")
        form.addRow(self.rec_btn, self.rec_lbl)

        self.dt = self._spin(0.01, decimals=5, step=0.001)
        self.scale = self._spin(1.0, decimals=4, step=0.1)
        self.in_g = QComboBox()
        self.in_g.addItems(["record in m/s²", "record in g"])
        self.zeta = self._spin(0.05, decimals=3, step=0.01)
        self.density = self._spin(2400.0, decimals=1, step=100.0, big=True)
        form.addRow("dt [s]", self.dt)
        form.addRow("Scale factor", self.scale)
        form.addRow("Units", self.in_g)
        form.addRow("Damping ζ", self.zeta)
        form.addRow(f"Density [kg/{project.length_unit}³]", self.density)

        self.bar = QProgressBar()
        form.addRow(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        form.addRow(self.log)

        row = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._start)
        self.run_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        row.addWidget(close_btn)
        form.addRow(row)
        outer.addWidget(left, 0)

        # ---- right: response-history plot ----
        rightw = QWidget()
        rv = QVBoxLayout(rightw)
        self._fig = Figure(figsize=(4.6, 3.4), layout="constrained")
        self._ax = self._fig.add_subplot(111)
        self._canvas = Canvas(self._fig)
        self._canvas.setMinimumWidth(420)
        rv.addWidget(self._canvas, 1)
        qrow = QHBoxLayout()
        self.quantity = QComboBox()
        self.quantity.addItems(["displacement", "velocity", "acceleration"])
        self.quantity.currentIndexChanged.connect(self._draw)
        qrow.addWidget(QLabel("Plot"))
        qrow.addWidget(self.quantity)
        qrow.addStretch(1)
        rv.addLayout(qrow)
        outer.addWidget(rightw, 1)
        self._draw()

    # ------------------------------------------------ helpers
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
    def _spin(value, *, decimals=3, step=0.1, big=False):
        s = QDoubleSpinBox()
        s.setRange(0.0, 1e15 if big else 1e6)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(value)
        return s

    def _load_record(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load acceleration record", "",
            "Records (*.txt *.csv *.at2 *.acc *.dat);;All files (*)")
        if not path:
            return
        try:
            self._accel = load_accel_record(path)
        except Exception as exc:                       # noqa: BLE001
            self.rec_lbl.setText(f"(error: {exc})")
            return
        n = self._accel.size
        self.rec_lbl.setText(f"{path.split('/')[-1].split(chr(92))[-1]} "
                             f"— {n} pts")
        self.run_btn.setEnabled(n >= 2)

    # ------------------------------------------------ run lifecycle
    def _kwargs(self) -> dict:
        direction, dof = self.direction.currentData()
        return dict(control_node=self.node.currentData(), control_dof=dof,
                    direction=direction, zeta=float(self.zeta.value()),
                    density=float(self.density.value()))

    def _scaled_accel(self):
        a = self._accel * float(self.scale.value())
        if self.in_g.currentIndex() == 1:
            a = a * _G
        return a

    def _start(self) -> None:
        if self._accel is None or self._accel.size < 2:
            return
        self.log.clear()
        self.bar.setRange(0, max(1, self._accel.size - 1))
        self.bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._worker = TimeHistoryWorker(self._project, self._scaled_accel(),
                                         float(self.dt.value()), self._kwargs())
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.log.appendPlainText("cancelling…")

    def _on_progress(self, info: dict) -> None:
        self.bar.setValue(int(info["step"]))
        if info["step"] % 20 == 0:
            self.log.appendPlainText(
                f"t={info['time']:.3f}s  d={info['disp']:.4g}")

    def _on_done(self, res: dict) -> None:
        self._result = res
        self._draw()
        self.log.appendPlainText(
            f"done — peak |d| = {res.get('peak_disp', 0.0):.4g} "
            f"{self._project.length_unit}")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_failed(self, msg: str) -> None:
        self.log.appendPlainText(f"FAILED: {msg}")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _draw(self, *_) -> None:
        self._ax.clear()
        r = self._result
        if r:
            q = self.quantity.currentText()
            key = {"displacement": "disp", "velocity": "velocity",
                   "acceleration": "acceleration"}[q]
            t = r.get("times", [])
            y = r.get(key, [])
            self._ax.plot(t, y, "-", lw=1.2, color="#1f5f8b")
            self._ax.set_xlabel("time (s)")
            self._ax.set_ylabel(q)
            self._ax.set_title("Response history", fontsize=9)
        else:
            self._ax.text(0.5, 0.5, "(load a record and Run)", ha="center",
                          va="center", transform=self._ax.transAxes,
                          fontsize=8, color="0.5")
        self._ax.grid(True, alpha=0.25)
        self._canvas.draw_idle()
