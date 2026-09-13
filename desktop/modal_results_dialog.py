"""Modal-analysis results — the period / frequency table plus live mode-shape
preview, the *Modal* counterpart of the linear-static deformed view.

``ModalResultsDialog`` lists one row per extracted mode (period, cyclic
frequency, circular frequency). Selecting a row calls the ``on_show_mode``
callback the owning window supplies, which paints that mode's shape on the
main 3-D view. It is **non-modal** so the view updates live as the user
clicks through modes; the owning window keeps a reference so it survives.

Fed by :class:`femsolver.analysis.eigen.EigenAnalysis`'s result dict
(``frequencies_hz`` / ``periods_s``). Pure Qt — headless-constructible under
``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import math

from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                               QHeaderView, QLabel, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

import style


def _fmt(x: float, spec: str = ".4g") -> str:
    if x is None or (isinstance(x, float) and (math.isinf(x) or math.isnan(x))):
        return "—"
    return format(float(x), spec)


class ModalResultsDialog(QDialog):
    """Show modal periods / frequencies; drive the mode-shape preview."""

    def __init__(self, parent, info: dict, on_show_mode=None):
        super().__init__(parent)
        self.setWindowTitle("Modal results")
        self._on_show_mode = on_show_mode
        self._freqs = list(info.get("frequencies_hz", []))
        self._periods = list(info.get("periods_s", []))
        self.resize(460, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel(f"{len(self._periods)} modes")
        head.setObjectName("h2")
        root.addWidget(head)
        # fundamental (first mode with a finite, non-zero period)
        t1 = next((t for t in self._periods
                   if t and math.isfinite(t)), None)
        if t1:
            sub = QLabel(f"Fundamental T₁ = {_fmt(t1)} s "
                         f"(f = {_fmt(1.0 / t1)} Hz)")
        else:
            sub = QLabel("No finite modes — check supports and density.")
        sub.setObjectName("sub")
        root.addWidget(sub)

        self.table = QTableWidget(len(self._periods), 4)
        self.table.setHorizontalHeaderLabels(
            ["Mode", "Period T [s]", "Freq f [Hz]", "ω [rad/s]"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        for c in range(4):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        for r, (T, f) in enumerate(zip(self._periods, self._freqs)):
            omega = 2.0 * math.pi * f if f else 0.0
            for c, txt in enumerate((str(r + 1), _fmt(T), _fmt(f),
                                     _fmt(omega))):
                self.table.setItem(r, c, QTableWidgetItem(txt))
        self.table.itemSelectionChanged.connect(self._emit_mode)
        root.addWidget(self.table, 1)

        hint = QLabel("Select a mode to preview its shape on the model.")
        hint.setObjectName("hintLabel")
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

        if self._periods:                    # preview mode 1 straight away
            self.table.setCurrentCell(0, 0)

    def _emit_mode(self) -> None:
        r = self.table.currentRow()
        if 0 <= r < len(self._periods) and self._on_show_mode:
            self._on_show_mode(r)

    @classmethod
    def show_results(cls, parent, info: dict, on_show_mode=None):
        """Create and show the dialog non-modally; return it so the caller
        can keep a reference (otherwise Qt garbage-collects it)."""
        dlg = cls(parent, info, on_show_mode)
        dlg.show()
        return dlg
