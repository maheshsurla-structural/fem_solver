"""Linear-buckling results — the critical load-factor table plus live
buckling-shape preview.

``BucklingResultsDialog`` lists one row per buckling mode (load factor λ, the
multiplier of the reference load at which the structure buckles). The first
row is the critical mode. Selecting a row calls the ``on_show_mode`` callback
the owning window supplies, which paints that buckling shape on the main view.
Non-modal, so the view updates live. Fed by
:class:`femsolver.analysis.buckling.LinearBucklingAnalysis`'s result dict
(``load_factors`` / ``critical_load_factor``). Headless-constructible under
``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                               QHeaderView, QLabel, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

import style


class BucklingResultsDialog(QDialog):
    """Show buckling load factors; drive the buckling-shape preview."""

    def __init__(self, parent, info: dict, ref_label: str = "reference load",
                 on_show_mode=None):
        super().__init__(parent)
        self.setWindowTitle("Buckling results")
        self._on_show_mode = on_show_mode
        self._factors = list(info.get("load_factors", []))
        self.resize(440, 400)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel(f"{len(self._factors)} buckling modes")
        head.setObjectName("h2")
        root.addWidget(head)
        if self._factors:
            sub = QLabel(f"Critical λ = {self._factors[0]:.4g} × ({ref_label})")
        else:
            sub = QLabel("No buckling modes — the reference load creates no "
                         "compression.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        self.table = QTableWidget(len(self._factors), 2)
        self.table.setHorizontalHeaderLabels(["Mode", "Load factor λ"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for r, lam in enumerate(self._factors):
            self.table.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            self.table.setItem(r, 1, QTableWidgetItem(f"{lam:.6g}"))
        self.table.itemSelectionChanged.connect(self._emit_mode)
        root.addWidget(self.table, 1)

        hint = QLabel("Select a mode to preview its buckling shape.")
        hint.setObjectName("hintLabel")
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

        if self._factors:
            self.table.setCurrentCell(0, 0)

    def _emit_mode(self) -> None:
        r = self.table.currentRow()
        if 0 <= r < len(self._factors) and self._on_show_mode:
            self._on_show_mode(r)

    @classmethod
    def show_results(cls, parent, info: dict, ref_label: str = "reference load",
                     on_show_mode=None):
        dlg = cls(parent, info, ref_label, on_show_mode)
        dlg.show()
        return dlg
