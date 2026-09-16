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
                               QFormLayout, QHBoxLayout, QLabel, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

import analysis_ui as ui
import nonlinear as NL
import style
from pick import PickDialog

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


_DIR_DOF = {"x": 0, "y": 1, "z": 2}


class TimeHistoryDialog(PickDialog):
    def __init__(self, parent, project, seed: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Nonlinear time history")
        self._project = project
        self._worker: TimeHistoryWorker | None = None
        self._accel: np.ndarray | None = None
        self._result: dict | None = None
        self._initial_case = None            # E2: source nonlinear-case id
        self._hold_source_loads = False
        outer = QHBoxLayout(self)

        # ---- left: grouped inputs + run controls (plan A3) ----
        left = QWidget()
        left.setMaximumWidth(380)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(style.SP_MD)
        node_ids = [n.id for n in project.nodes]
        free = [n.id for n in project.nodes
                if not (n.supports and any(n.supports))]
        self.node = self._combo([(str(i), i) for i in node_ids],
                                default=(free[-1] if free else
                                         (node_ids[-1] if node_ids else None)))
        self.register_pick_field("node", self.node)   # click monitor node in model
        self.direction = self._combo(_DIRS, default=("y", 1))
        self.rec_btn = QPushButton("Load record…")
        self.rec_btn.clicked.connect(self._load_record)
        self.rec_lbl = QLabel("(no record)")
        self.dt = self._spin(0.01, decimals=5, step=0.001)
        self.scale = self._spin(1.0, decimals=4, step=0.1)
        self.in_g = QComboBox()
        self.in_g.addItems(["record in m/s²", "record in g"])
        self.zeta = self._spin(0.05, decimals=3, step=0.01)
        self.density = self._spin(2400.0, decimals=1, step=100.0, big=True)
        self.ic_lbl = QLabel("unstressed")     # E2: initial-state summary

        monitor = ui.GroupCard("Monitor")
        monitor.add_row("Monitor node", self.node)
        monitor.add_row("Direction", self.direction)

        gm = ui.GroupCard("Ground motion")
        rec_host = QWidget()
        rh = QHBoxLayout(rec_host)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.addWidget(self.rec_btn)
        rh.addWidget(self.rec_lbl, 1)
        gm.add_row("Record", rec_host)
        gm.add_row("dt [s]", self.dt)
        gm.add_row("Scale factor", self.scale)
        gm.add_row("Units", self.in_g)
        gm.add_row("Initial state", self.ic_lbl)

        model = ui.GroupCard("Damping & mass")
        model.add_row("Damping ζ", self.zeta)
        model.add_row(f"Density [kg/{project.length_unit}³]", self.density)

        for card in (monitor, gm, model):
            lv.addWidget(card)

        self.bar = QProgressBar()
        lv.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        lv.addWidget(self.log)

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
        lv.addLayout(row)
        lv.addStretch(1)
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
        if seed:
            self._apply_seed(seed)
        self._draw()
        style.apply(self)

    def _apply_seed(self, seed: dict) -> None:
        """Pre-fill the runner from a saved Time-History case (its referenced
        function's record + the case settings), ready for the user to Run."""
        vals = seed.get("values") or []
        self._accel = np.asarray(vals, dtype=float)
        self.dt.setValue(float(seed.get("dt", 0.01)))
        self.in_g.setCurrentIndex(1 if seed.get("in_g") else 0)
        ni = self.node.findData(seed.get("control_node"))
        if ni >= 0:
            self.node.setCurrentIndex(ni)
        direction = seed.get("direction", "y")
        di = self.direction.findData((direction, _DIR_DOF.get(direction, 1)))
        if di >= 0:
            self.direction.setCurrentIndex(di)
        self.scale.setValue(float(seed.get("scale", 1.0)))
        self.zeta.setValue(float(seed.get("zeta", 0.05)))
        self.density.setValue(float(seed.get("density", 2400.0)))
        self.rec_lbl.setText(f"{seed.get('name', 'function')} "
                             f"— {self._accel.size} pts")
        self.run_btn.setEnabled(self._accel.size >= 2)

        ic = seed.get("initial_condition") or ("zero",)
        self._hold_source_loads = bool(seed.get("hold_source_loads", False))
        if ic and ic[0] == "state":
            self._initial_case = ic[1]
            src = self._project.nonlinear_case(ic[1])
            nm = src.name if src else f"case {ic[1]}"
            self.ic_lbl.setText(nm + (" · loads held" if self._hold_source_loads
                                      else " · state only"))
        else:
            self._initial_case = None
            self.ic_lbl.setText("unstressed")

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
                    density=float(self.density.value()),
                    initial_case=self._initial_case,
                    hold_source_loads=self._hold_source_loads)

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


class TimeHistoryCaseDialog(PickDialog):
    """Config-only editor for a saved Time-History :class:`project.AnalysisCase`
    (analysis-cases-manager TH-2). It picks a **function** from the project's
    time-history library plus the monitor / scale / damping / mass — no run, no
    file import. The interactive solve happens later in :class:`TimeHistoryDialog`
    (opened seeded from these params). Built on the L1 card scaffold; pure Qt,
    headless-constructible.
    """

    def __init__(self, parent, project, *, initial: dict | None = None,
                 initial_ic: tuple = ("zero",),
                 name: str = "Time History", notes: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Time-history case")
        self._project = project
        node_ids = [n.id for n in project.nodes]
        free = [n.id for n in project.nodes
                if not (n.supports and any(n.supports))]
        default_node = (free[-1] if free else
                        (node_ids[-1] if node_ids else None))

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)
        self.header = ui.CaseHeader(name=name, type_label="Time History",
                                    notes=notes)
        root.addWidget(self.header)

        monitor = ui.GroupCard("Monitor")
        self.node = self._combo([(str(i), i) for i in node_ids],
                                default=default_node)
        self.register_pick_field("node", self.node)   # click monitor node in model
        self.direction = self._combo(_DIRS, default=("y", 1))
        monitor.add_row("Monitor node", self.node)
        monitor.add_row("Direction", self.direction)

        gm = ui.GroupCard("Ground motion")
        self.function = QComboBox()
        for f in project.th_functions:
            self.function.addItem(
                f"{f.name}  ({f.npts} pts · {f.duration:.3g}s"
                f"{' · g' if f.in_g else ''})", f.id)
        gm.add_row("Function", self.function)
        self._no_fn = QLabel("No time-history functions yet — define them in "
                             "Analysis ▸ Functions.")
        self._no_fn.setObjectName("hintLabel")
        self._no_fn.setWordWrap(True)
        self._no_fn.setVisible(self.function.count() == 0)
        gm.add_full_row(self._no_fn)
        self.scale = self._spin(1.0, decimals=4, step=0.1)
        gm.add_row("Scale factor", self.scale)

        model = ui.GroupCard("Damping & mass")
        self.zeta = self._spin(0.05, decimals=3, step=0.01)
        self.density = self._spin(2400.0, decimals=1, step=100.0, big=True)
        model.add_row("Damping ζ", self.zeta)
        model.add_row(f"Density [kg/{project.length_unit}³]", self.density)

        sources = [(c.id, c.name) for c in project.nonlinear_cases]
        self.initial = ui.InitialConditionCard(sources)

        root.addWidget(ui.two_column(monitor, gm, model))
        root.addWidget(self.initial)
        root.addWidget(ui.dialog_buttons(self))
        if initial:
            self._seed(initial)
        self.initial.set_value(
            initial_ic, bool((initial or {}).get("hold_source_loads", True)))
        style.apply(self)

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

    def _seed(self, p: dict) -> None:
        ni = self.node.findData(p.get("control_node"))
        if ni >= 0:
            self.node.setCurrentIndex(ni)
        want = p.get("direction", "y")
        for i in range(self.direction.count()):
            if self.direction.itemData(i)[0] == want:
                self.direction.setCurrentIndex(i)
                break
        fi = self.function.findData(p.get("function_id"))
        if fi >= 0:
            self.function.setCurrentIndex(fi)
        self.scale.setValue(float(p.get("scale", 1.0)))
        self.zeta.setValue(float(p.get("zeta", 0.05)))
        self.density.setValue(float(p.get("density", 2400.0)))

    def accept(self) -> None:
        if self.function.count() == 0 or self.function.currentData() is None:
            QMessageBox.warning(
                self, "Time history",
                "Define a time-history function (Analysis ▸ Functions) and "
                "select it first.")
            return
        super().accept()

    def initial_condition(self) -> tuple:
        return self.initial.value()

    def params(self) -> dict:
        return {"function_id": self.function.currentData(),
                "control_node": self.node.currentData(),
                "direction": self.direction.currentData()[0],
                "scale": float(self.scale.value()),
                "zeta": float(self.zeta.value()),
                "density": float(self.density.value()),
                "hold_source_loads": self.initial.hold()}
