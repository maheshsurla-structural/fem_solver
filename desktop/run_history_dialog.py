"""Nonlinear run-history browser (plan §16 G-S2).

Lists the nonlinear runs saved on the project (``Project.runs`` — lean
:class:`nl_runs.RunRecord`s) and re-plots a selected run's curve, so a pushover
/ cyclic / time-history result can be *re-viewed after the project is closed and
re-opened*. Runs can be renamed and deleted here; the edited list is returned to
the caller (applied as one undoable edit, like the other managers).

Pure Qt + matplotlib (Agg canvas) — headless-constructible under
``QT_QPA_PLATFORM=offscreen`` (:meth:`manage` shown separately).
"""
from __future__ import annotations

import copy

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QInputDialog, QLabel,
                               QListWidget, QPushButton, QVBoxLayout, QWidget)

import style
from analysis_ui import dialog_buttons
from femsolver.performance.acceptance import LEVEL_COLORS, LEVELS


class RunHistoryDialog(QDialog):
    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Nonlinear run history")
        self._project = project
        self._runs = copy.deepcopy(list(project.runs))
        outer = QHBoxLayout(self)

        # ---- left: the list + actions ----
        left = QWidget()
        lv = QVBoxLayout(left)
        hdr = QLabel("Saved runs")
        hdr.setObjectName("h3")
        lv.addWidget(hdr)
        self.list = QListWidget()
        self.list.setMinimumWidth(240)
        self.list.currentRowChanged.connect(lambda *_: self._draw())
        lv.addWidget(self.list, 1)
        row = QHBoxLayout()
        for label, cb in (("Rename…", self._rename), ("Delete", self._delete)):
            b = QPushButton(label)
            b.clicked.connect(cb)
            row.addWidget(b)
        row.addStretch(1)
        lv.addLayout(row)
        outer.addWidget(left, 0)

        # ---- right: the selected run's curve + summary ----
        rightw = QWidget()
        rv = QVBoxLayout(rightw)
        self._fig = Figure(figsize=(4.8, 3.6), layout="constrained")
        self._ax = self._fig.add_subplot(111)
        self._canvas = Canvas(self._fig)
        self._canvas.setMinimumWidth(440)
        rv.addWidget(self._canvas, 1)
        self._summary = QLabel("")
        self._summary.setObjectName("sub")
        self._summary.setWordWrap(True)
        rv.addWidget(self._summary)
        outer.addWidget(rightw, 1)

        rv.addWidget(dialog_buttons(self))
        style.apply(self)
        self._refresh()

    # ---------------------------------------------------------- list
    def _refresh(self, select: int | None = None) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for r in self._runs:
            self.list.addItem(f"{r.name}  ·  {r.kind}")
        self.list.blockSignals(False)
        if self._runs:
            self.list.setCurrentRow(
                min(select if select is not None else 0, len(self._runs) - 1))
        self._draw()

    def _current(self):
        r = self.list.currentRow()
        return self._runs[r] if 0 <= r < len(self._runs) else None

    def _rename(self) -> None:
        run = self._current()
        if run is None:
            return
        name, ok = QInputDialog.getText(self, "Rename run", "Name:",
                                        text=run.name)
        if ok and name.strip():
            run.name = name.strip()
            self._refresh(self.list.currentRow())

    def _delete(self) -> None:
        r = self.list.currentRow()
        if 0 <= r < len(self._runs):
            del self._runs[r]
            self._refresh(r)

    # ---------------------------------------------------------- plot
    def _draw(self) -> None:
        self._ax.clear()
        run = self._current()
        if run is None or not run.x:
            self._ax.text(0.5, 0.5, "(no run selected)", ha="center",
                          va="center", transform=self._ax.transAxes,
                          fontsize=9, color=style.MUTED)
            self._summary.setText("")
        else:
            cyclic = run.protocol == "cyclic"
            self._ax.plot(run.x, run.y, "-" if cyclic else "-o",
                          ms=3, lw=1.3, color=style.C_PRIMARY)
            if cyclic:
                self._ax.axhline(0, color=style.AX_SPINE, lw=0.6)
                self._ax.axvline(0, color=style.AX_SPINE, lw=0.6)
            # ASCE 41 first-reach markers (IO/LS/CP), if captured
            for lvl, d in run.milestones.items():
                if lvl in LEVELS:
                    rgb = LEVEL_COLORS[LEVELS.index(lvl)]
                    self._ax.axvline(d, color=rgb, lw=1.0, ls="--")
                    self._ax.text(d, 0.98, lvl, rotation=90, va="top",
                                  ha="right", fontsize=7,
                                  color=rgb, transform=self._ax.get_xaxis_transform())
            lu, fu = self._project.length_unit, self._project.force_unit
            self._ax.set_xlabel(f"{run.x_label} [{lu if run.x_label!='time' else 's'}]")
            self._ax.set_ylabel(f"{run.y_label} "
                                f"[{fu if run.y_label=='base shear' else lu}]")
            self._ax.set_title(run.name, fontsize=9)
            meta = "  ".join(f"{k}: {v}" for k, v in run.meta.items())
            self._summary.setText(f"{run.summary()}\n{meta}"
                                  f"\nrun {run.created}")
        style.beautify_axes(self._ax)
        self._canvas.draw_idle()

    # ---------------------------------------------------------- result
    def result_runs(self) -> list:
        return self._runs

    @classmethod
    def manage(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result_runs() if dlg.exec() else None
