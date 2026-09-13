"""Response-spectrum results — the per-mode participation table and the
participating-mass summary, beside the combined peak response drawn on the
main view.

``ResponseSpectrumResultsDialog`` is fed
:class:`femsolver.analysis.response_spectrum.ResponseSpectrumAnalysis`'s result
dict plus the total in-direction mass (so effective modal masses become
percentages). It is non-modal so the peak deformed shape stays visible on the
3-D view while the table is read. Pure Qt — headless-constructible under
``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import math

from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                               QHeaderView, QLabel, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

import style


def _fmt(x, spec=".4g"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return format(float(x), spec)


class ResponseSpectrumResultsDialog(QDialog):
    """Show modal participation + total participating mass."""

    def __init__(self, parent, info: dict, total_dir_mass: float):
        super().__init__(parent)
        self.setWindowTitle("Response-spectrum results")
        modal = list(info.get("modal_results", []))
        total = float(total_dir_mass) if total_dir_mass else 0.0
        captured = float(info.get("total_participating_mass", 0.0))
        pct = (100.0 * captured / total) if total > 0 else 0.0
        self.resize(560, 460)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel(
            f"Direction {str(info.get('direction', '?')).upper()} · "
            f"{str(info.get('combination', '?')).upper()} · "
            f"ζ = {_fmt(info.get('damping_ratio', 0.05))}")
        head.setObjectName("h2")
        root.addWidget(head)

        summary = QLabel(f"Participating mass: {pct:.1f}%")
        summary.setObjectName("sub")
        if total > 0 and pct < 90.0:
            summary.setText(f"Participating mass: {pct:.1f}%  "
                            f"— below 90%, add more modes")
            summary.setStyleSheet(f"color: {style.WARN};")
        root.addWidget(summary)

        self.table = QTableWidget(len(modal), 6)
        self.table.setHorizontalHeaderLabels(
            ["Mode", "T [s]", "Sa [m/s²]", "Γ", "Mass %", "Cumulative %"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        for c in range(6):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        cum = 0.0
        for r, md in enumerate(modal):
            m_eff = float(md.get("modal_mass_eff", 0.0))
            frac = (100.0 * m_eff / total) if total > 0 else 0.0
            cum += frac
            cells = (str(md.get("mode", r + 1)), _fmt(md.get("period")),
                     _fmt(md.get("Sa")), _fmt(md.get("Gamma")),
                     f"{frac:.1f}", f"{cum:.1f}")
            for c, txt in enumerate(cells):
                self.table.setItem(r, c, QTableWidgetItem(txt))
        root.addWidget(self.table, 1)

        hint = QLabel("The combined peak response is shown on the model.")
        hint.setObjectName("hintLabel")
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

    @classmethod
    def show_results(cls, parent, info: dict, total_dir_mass: float):
        dlg = cls(parent, info, total_dir_mass)
        dlg.show()
        return dlg
