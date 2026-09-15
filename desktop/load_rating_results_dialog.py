"""Load-rating (LRFR) results — the rating-factor chart + per-level table.

``LoadRatingResultsDialog`` shows the rating factor for each computed level
(design inventory / operating, and optional legal / permit) as a bar chart with
the RF = 1 acceptance threshold, plus a table with the live-load factor, RF and
status.  Non-modal, so it stays open beside the model.  Fed by the engine's
:class:`femsolver.bridges.BridgeRating`.  Pure Qt + matplotlib (Agg) —
headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHeaderView, QLabel,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

import style
from units import Quantity, UnitSystem

_ORDER = ["inventory", "operating", "legal", "permit"]
_TITLE = {"inventory": "Inventory", "operating": "Operating",
          "legal": "Legal", "permit": "Permit"}


class LoadRatingResultsDialog(QDialog):
    """RF bar chart + rating table, in the project's display units."""

    def __init__(self, parent, rating, *, effect_label="effect", ll_im=0.0,
                 unitsys=None, quantity=Quantity.MOMENT):
        super().__init__(parent)
        self.setWindowTitle("Load-rating results (LRFR)")
        self.resize(560, 520)
        us = unitsys or UnitSystem()
        r_unit = us.label(quantity)

        levels = [lv for lv in _ORDER if lv in rating.results]
        results = [rating.results[lv] for lv in levels]

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel(f"Load rating — {effect_label}")
        head.setObjectName("h2")
        root.addWidget(head)
        ctrl = rating.controlling
        ll_disp = us.to_display(ll_im, quantity)
        sub = QLabel(
            f"Controlling: {_TITLE.get(ctrl.level, ctrl.level)} "
            f"RF = {ctrl.rf:.2f} "
            f"({'ADEQUATE' if ctrl.adequate else 'DEFICIENT'})"
            f"   ·   HL-93 LL+IM = {ll_disp:.4g} {r_unit}")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # ---- RF bar chart with the RF = 1 threshold ----
        fig = Figure(figsize=(5.2, 3.0), layout="constrained")
        ax = fig.add_subplot(111)
        xs = list(range(len(levels)))
        rfs = [min(r.rf, 3.0) for r in results]        # clip inf/huge for view
        colors = [style.C_PRIMARY if r.adequate else "#c0392b"
                  for r in results]
        ax.bar(xs, rfs, color=colors, alpha=0.85, width=0.6)
        ax.axhline(1.0, color="#c0392b", lw=1.2, ls="--")
        ax.text(len(levels) - 0.5, 1.02, "RF = 1", color="#c0392b",
                ha="right", va="bottom", fontsize=8)
        for x, r in zip(xs, results):
            txt = "∞" if r.rf == float("inf") else f"{r.rf:.2f}"
            ax.text(x, min(r.rf, 3.0) + 0.03, txt, ha="center", va="bottom",
                    fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([_TITLE.get(lv, lv) for lv in levels])
        ax.set_ylabel("rating factor RF")
        ax.set_ylim(0, max(3.0, max(rfs) * 1.15) if rfs else 3.0)
        ax.set_title("LRFR rating factors")
        try:
            style.beautify_axes(ax)
        except Exception:                              # noqa: BLE001
            pass
        canvas = Canvas(fig)
        canvas.setMinimumHeight(220)
        root.addWidget(canvas, 1)

        # ---- table ----
        tbl = QTableWidget(len(levels), 5)
        tbl.setHorizontalHeaderLabels(
            ["Level", "γLL", "RF", "Status", f"Capacity ({r_unit})"])
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(tbl.EditTrigger.NoEditTriggers)
        tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        for row, (lv, r) in enumerate(zip(levels, results)):
            rf_txt = "∞" if r.rf == float("inf") else f"{r.rf:.3f}"
            cap = us.to_display(r.capacity, quantity)
            cells = [_TITLE.get(lv, lv), f"{r.factors.gamma_LL:.2f}", rf_txt,
                     "OK" if r.adequate else "DEFICIENT", f"{cap:.4g}"]
            for col, text in enumerate(cells):
                it = QTableWidgetItem(text)
                if col == 3 and not r.adequate:
                    from PySide6.QtGui import QBrush, QColor
                    it.setForeground(QBrush(QColor("#c0392b")))
                tbl.setItem(row, col, it)
        tbl.setMaximumHeight(40 + 28 * len(levels))
        root.addWidget(tbl)

        hint = QLabel("RF ≥ 1 means the bridge carries that live-load level; "
                      "rating in tons = RF × rating-vehicle weight.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

    @classmethod
    def show_results(cls, parent, rating, **kw):
        dlg = cls(parent, rating, **kw)
        dlg.show()
        return dlg
