"""Cable-tuning results — the solved stay tensions and the before/after deck
profile (bridge GUI plan G6).

Fed the tuned tensions (from :func:`femsolver.bridges.unknown_load_factors`)
and the node deflection profiles under dead load with / without the stays.
Non-modal. Pure Qt + matplotlib (Agg) — headless-constructible under
``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHeaderView, QLabel,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

import style


class CableTuningResultsDialog(QDialog):
    """Tuned stay tensions + before/after deck deflection profile."""

    def __init__(self, parent, cable_labels, tensions, xs, before, after, *,
                 max_residual=0.0):
        super().__init__(parent)
        self.setWindowTitle("Cable-tuning results")
        self.resize(600, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel("Cable-stayed tuning (unknown load factor)")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel(f"{len(tensions)} stay(s) tuned · "
                     f"target residual ≤ {abs(max_residual) * 1e3:.3g} mm")
        sub.setObjectName("sub")
        root.addWidget(sub)

        table = QTableWidget(len(tensions), 2)
        table.setHorizontalHeaderLabels(["Stay", "Tension [kN]"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        for r, (lab, T) in enumerate(zip(cable_labels, tensions)):
            table.setItem(r, 0, QTableWidgetItem(str(lab)))
            table.setItem(r, 1, QTableWidgetItem(f"{T / 1e3:.1f}"))
        table.setMaximumHeight(150)
        root.addWidget(table)

        fig = Figure(figsize=(5.4, 3.2), layout="constrained")
        ax = fig.add_subplot(111)
        x = np.asarray(xs)
        ax.plot(x, -np.asarray(before) * 1e3, "-o", color=style.C_SECONDARY,
                lw=2, ms=3, label="dead load only")
        ax.plot(x, -np.asarray(after) * 1e3, "-o", color=style.C_PRIMARY,
                lw=2.2, ms=3, label="after tuning")
        ax.axhline(0.0, color=style.AX_SPINE, lw=0.8)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("vertical deflection (mm, ↓)")
        ax.set_title("deck profile — stays tune the dead-load shape to target")
        ax.legend(fontsize=8)
        try:
            style.beautify_axes(ax)
        except Exception:                              # noqa: BLE001
            pass
        canvas = Canvas(fig)
        canvas.setMinimumHeight(240)
        root.addWidget(canvas, 1)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

    @classmethod
    def show_results(cls, parent, cable_labels, tensions, xs, before, after,
                     **kw):
        dlg = cls(parent, cable_labels, tensions, xs, before, after, **kw)
        dlg.show()
        return dlg
