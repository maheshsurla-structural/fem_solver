"""Building quick-template dialog (wall plan W8c).

The ETABS "New Model Quick Templates" idiom — a uniform grid + a simple story
stack — but with a **live plan + elevation preview** that redraws as you type
(our differentiator; ETABS shows only a static thumbnail). Returns
``(stories, grid_lines)`` for the caller to apply.
"""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QSpinBox,
                               QVBoxLayout, QWidget)

import building_template as bt
import style
from unit_widgets import UnitSpin, labeled
from units import Quantity, UnitSystem


def _int(v, lo, hi):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(v)
    return s


class GridStoryTemplateDialog(QDialog):
    """Uniform grid + simple stories, with a live plan/elevation preview."""

    def __init__(self, parent, units: UnitSystem | None = None):
        super().__init__(parent)
        self.setWindowTitle("New grid & stories (quick template)")
        self.resize(720, 520)
        self._us = units or UnitSystem()
        self.result = None

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)
        head = QLabel("New grid & stories")
        head.setObjectName("h2")
        root.addWidget(head)

        body = QHBoxLayout()
        root.addLayout(body, 1)

        # ---- inputs ----
        form = QFormLayout()
        self.nx = _int(4, 1, 200)
        self.ny = _int(4, 1, 200)
        self.sx = self._len(6.0)
        self.sy = self._len(6.0)
        self.x_lab = QLineEdit("A")
        self.y_lab = QLineEdit("1")
        self.n_st = _int(4, 1, 300)
        self.typ_h = self._len(3.5)
        self.bot_h = self._len(4.5)
        form.addRow("Grid lines X", self.nx)
        form.addRow("Grid lines Y", self.ny)
        form.addRow(labeled("Spacing X", self.sx), self.sx)
        form.addRow(labeled("Spacing Y", self.sy), self.sy)
        form.addRow("X label starts", self.x_lab)
        form.addRow("Y label starts", self.y_lab)
        form.addRow("Number of stories", self.n_st)
        form.addRow(labeled("Typical height", self.typ_h), self.typ_h)
        form.addRow(labeled("Bottom height", self.bot_h), self.bot_h)
        fw = QWidget()
        fw.setLayout(form)
        fw.setMaximumWidth(300)
        body.addWidget(fw)

        # ---- live preview ----
        self._fig = Figure(figsize=(5.0, 4.0), layout="constrained")
        self._canvas = Canvas(self._fig)
        body.addWidget(self._canvas, 1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)
        style.apply(self)

        for w in (self.nx, self.ny, self.n_st):
            w.valueChanged.connect(self._redraw)
        for w in (self.sx, self.sy, self.typ_h, self.bot_h):
            w.valueChanged.connect(self._redraw)
        for w in (self.x_lab, self.y_lab):
            w.textChanged.connect(self._redraw)
        self._redraw()

    def _len(self, si):
        return UnitSpin(Quantity.LENGTH, self._us, si=si, decimals=2, step=0.5,
                        rng=(0.0, 1.0e9))

    # ------------------------------------------------------------- preview
    def _params_display(self):
        """Grid coords + story elevations in *display* units for the preview."""
        us = self._us
        sx = us.to_display(self.sx.si_value(), Quantity.LENGTH)
        sy = us.to_display(self.sy.si_value(), Quantity.LENGTH)
        th = us.to_display(self.typ_h.si_value(), Quantity.LENGTH)
        bh = us.to_display(self.bot_h.si_value(), Quantity.LENGTH)
        xs = [i * sx for i in range(self.nx.value())]
        ys = [j * sy for j in range(self.ny.value())]
        xlab = bt._label_seq(self.x_lab.text() or "A", self.nx.value())
        ylab = bt._label_seq(self.y_lab.text() or "1", self.ny.value())
        elevs, names, z = [0.0], ["Base"], 0.0
        for i in range(1, self.n_st.value() + 1):
            z += bh if i == 1 else th
            elevs.append(z)
            names.append(f"Story {i}")
        return xs, ys, xlab, ylab, elevs, names

    def _redraw(self, *_) -> None:
        xs, ys, xlab, ylab, elevs, names = self._params_display()
        lu = self._us.label(Quantity.LENGTH)
        self._fig.clear()
        axp = self._fig.add_subplot(121)
        axe = self._fig.add_subplot(122)

        # plan: const-X lines (vertical), const-Y lines (horizontal) + bubbles
        y0, y1 = (min(ys) if ys else 0), (max(ys) if ys else 0)
        x0, x1 = (min(xs) if xs else 0), (max(xs) if xs else 0)
        for x, nm in zip(xs, xlab):
            axp.plot([x, x], [y0, y1], color=style.C_PRIMARY, lw=0.8)
            axp.annotate(nm, (x, y1), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=7)
        for y, nm in zip(ys, ylab):
            axp.plot([x0, x1], [y, y], color=style.C_PRIMARY, lw=0.8)
            axp.annotate(nm, (x0, y), textcoords="offset points",
                         xytext=(-8, 0), va="center", fontsize=7)
        axp.set_title("Plan", fontsize=9)
        axp.set_aspect("equal", "datalim")
        axp.set_xlabel(f"X ({lu})", fontsize=8)

        # elevation: story levels
        for e, nm in zip(elevs, names):
            axe.axhline(e, color="#c0392b" if nm == "Base" else style.C_PRIMARY,
                        lw=0.9)
            axe.annotate(f"{nm}  {e:g}", (0.02, e), fontsize=7, va="bottom")
        axe.set_title("Elevation", fontsize=9)
        axe.set_xticks([])
        axe.set_ylabel(f"Elev ({lu})", fontsize=8)
        if elevs:
            axe.set_ylim(min(elevs) - 1, max(elevs) + max(elevs) * 0.1 + 1)
        for ax in (axp, axe):
            try:
                style.beautify_axes(ax)
            except Exception:                          # noqa: BLE001
                pass
        self._canvas.draw_idle()

    # ------------------------------------------------------------- accept
    def params(self) -> dict:
        return dict(nx=self.nx.value(), ny=self.ny.value(),
                    sx=self.sx.si_value(), sy=self.sy.si_value(),
                    n_stories=self.n_st.value(), typ_h=self.typ_h.si_value(),
                    bot_h=self.bot_h.si_value(),
                    x_label=self.x_lab.text() or "A",
                    y_label=self.y_lab.text() or "1")

    def _accept(self) -> None:
        p = self.params()
        self.result = bt.build_grid_stories(
            p["nx"], p["ny"], p["sx"], p["sy"], p["n_stories"], p["typ_h"],
            p["bot_h"], x_label=p["x_label"], y_label=p["y_label"])
        self.accept()

    @classmethod
    def get(cls, parent, units: UnitSystem | None = None):
        dlg = cls(parent, units)
        return dlg.result if dlg.exec() else None
