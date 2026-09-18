"""Pier-forces results dialog (wall plan W2).

Shows the integrated wall-pier **P / V / M** up the height of each pier — the
headline ETABS wall output — as a table plus a shear/moment elevation diagram,
in the project's display units, with a CSV export. Reads the solved ``model``
and the project pier labels (wall plan W0); the integration is done by
:mod:`piers`.
"""
from __future__ import annotations

import csv

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

import piers
import style
from units import Quantity, UnitSystem


class PierForcesDialog(QDialog):
    """Per-pier P/V/M table + shear/moment elevation diagram."""

    def __init__(self, parent, project, model, *, unitsys=None):
        super().__init__(parent)
        self.setWindowTitle("Pier forces")
        self.resize(620, 620)
        self._project = project
        self._model = model
        self._us = unitsys or UnitSystem()
        self._piers = project.pier_names()

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel("Wall pier forces")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel("Internal P (axial, + tension), V (in-plane shear) and M "
                     "(about the pier centroid) integrated from the shell "
                     "membrane stresses at each mesh level.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        pick = QHBoxLayout()
        pick.addWidget(QLabel("Pier"))
        self.pier = QComboBox()
        for name in self._piers:
            self.pier.addItem(name, name)
        self.pier.currentIndexChanged.connect(self._refresh)
        pick.addWidget(self.pier)
        pick.addStretch(1)
        export = QPushButton("Export CSV…")
        export.clicked.connect(self._export)
        pick.addWidget(export)
        root.addLayout(pick)

        self._fig = Figure(figsize=(5.6, 3.0), layout="constrained")
        self._canvas = Canvas(self._fig)
        self._canvas.setMinimumHeight(230)
        root.addWidget(self._canvas, 1)

        self.tbl = QTableWidget(0, 5)
        fu, mu, lu = (self._us.label(Quantity.FORCE),
                      self._us.label(Quantity.MOMENT),
                      self._us.label(Quantity.LENGTH))
        self.tbl.setHorizontalHeaderLabels(
            [f"Elev ({lu})", f"P ({fu})", f"V ({fu})", f"M ({mu})",
             f"Width ({lu})"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(self.tbl.EditTrigger.NoEditTriggers)
        self.tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.tbl)

        self._empty = QLabel("")
        self._empty.setObjectName("hintLabel")
        self._empty.setWordWrap(True)
        root.addWidget(self._empty)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

        if self._piers:
            self._refresh()
        else:
            self._empty.setText(
                "No piers defined. Draw a wall (Draw ▸ Wall) with a pier label, "
                "or set the Pier field on a wall area, then run the analysis.")

    # -------------------------------------------------------------- internals
    def _current_forces(self) -> list:
        name = self.pier.currentData()
        if name is None:
            return []
        return piers.pier_forces(self._project, self._model, name)

    def _refresh(self) -> None:
        forces = self._current_forces()
        us = self._us
        # table — top of wall first
        rows = sorted(forces, key=lambda f: f.z, reverse=True)
        self.tbl.setRowCount(len(rows))
        for r, f in enumerate(rows):
            cells = [us.to_display(f.z, Quantity.LENGTH),
                     us.to_display(f.axial, Quantity.FORCE),
                     us.to_display(f.shear, Quantity.FORCE),
                     us.to_display(f.moment, Quantity.MOMENT),
                     us.to_display(f.width, Quantity.LENGTH)]
            for c, v in enumerate(cells):
                self.tbl.setItem(r, c, QTableWidgetItem(f"{v:.4g}"))
        self._draw(forces)

    def _draw(self, forces) -> None:
        us = self._us
        self._fig.clear()
        if not forces:
            self._canvas.draw_idle()
            return
        z = [us.to_display(f.z, Quantity.LENGTH) for f in forces]
        V = [us.to_display(f.shear, Quantity.FORCE) for f in forces]
        M = [us.to_display(f.moment, Quantity.MOMENT) for f in forces]
        axV = self._fig.add_subplot(121)
        axM = self._fig.add_subplot(122, sharey=axV)
        axV.plot(V, z, "-o", color=style.C_PRIMARY, ms=3)
        axV.axvline(0, color="#888", lw=0.8)
        axV.set_xlabel(f"V ({us.label(Quantity.FORCE)})")
        axV.set_ylabel(f"elevation ({us.label(Quantity.LENGTH)})")
        axV.set_title("Shear")
        axM.plot(M, z, "-o", color="#c0392b", ms=3)
        axM.axvline(0, color="#888", lw=0.8)
        axM.set_xlabel(f"M ({us.label(Quantity.MOMENT)})")
        axM.set_title("Moment")
        for ax in (axV, axM):
            try:
                style.beautify_axes(ax)
            except Exception:                          # noqa: BLE001
                pass
        self._canvas.draw_idle()

    def _export(self) -> None:
        us = self._us
        path, _ = QFileDialog.getSaveFileName(
            self, "Export pier forces", "pier_forces.csv", "CSV files (*.csv)")
        if not path:
            return
        fu, mu, lu = (self._us.label(Quantity.FORCE),
                      self._us.label(Quantity.MOMENT),
                      self._us.label(Quantity.LENGTH))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["pier", f"elevation[{lu}]", f"P[{fu}]", f"V[{fu}]",
                        f"M[{mu}]", f"width[{lu}]"])
            for name in self._piers:
                for f in sorted(piers.pier_forces(self._project, self._model,
                                                  name), key=lambda f: f.z):
                    w.writerow([name,
                                f"{us.to_display(f.z, Quantity.LENGTH):.6g}",
                                f"{us.to_display(f.axial, Quantity.FORCE):.6g}",
                                f"{us.to_display(f.shear, Quantity.FORCE):.6g}",
                                f"{us.to_display(f.moment, Quantity.MOMENT):.6g}",
                                f"{us.to_display(f.width, Quantity.LENGTH):.6g}"])
        if self.parent() is not None:
            try:
                self.parent().statusBar().showMessage(
                    f"Exported pier forces → {path}")
            except Exception:                          # noqa: BLE001
                pass

    @classmethod
    def show_results(cls, parent, project, model, **kw):
        dlg = cls(parent, project, model, **kw)
        dlg.show()
        return dlg
