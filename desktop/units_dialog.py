"""Units preferences dialog (plan U4) — the MIDAS-style Force × Length picker.

Two combos choose the primitives; a live preview shows every derived unit
label (moment, stress, distributed load, …) so the user sees exactly how the
whole model will read before committing. Returns ``(force, length)`` or None.

Pure Qt, headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QLabel, QVBoxLayout)

import analysis_ui as ui
import style
from units import FORCE_UNITS, LENGTH_UNITS, Quantity, UnitSystem


class UnitsDialog(QDialog):
    """Pick the project's force + length display units."""

    # quantities worth previewing, in reading order
    _PREVIEW = [
        ("Length", Quantity.LENGTH),
        ("Force", Quantity.FORCE),
        ("Moment", Quantity.MOMENT),
        ("Stress", Quantity.STRESS),
        ("Distributed load", Quantity.DIST_LOAD),
    ]

    def __init__(self, parent, force: str, length: str):
        super().__init__(parent)
        self.setWindowTitle("Units")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(style.SP_LG, style.SP_LG,
                                 style.SP_LG, style.SP_LG)
        outer.setSpacing(style.SP_MD)

        pick = ui.GroupCard("Display units")
        self.force = QComboBox()
        for u in FORCE_UNITS:
            self.force.addItem(u, u)
        self.length = QComboBox()
        for u in LENGTH_UNITS:
            self.length.addItem(u, u)
        self._select(self.force, force)
        self._select(self.length, length)
        pick.add_row("Force", self.force)
        pick.add_row("Length", self.length)

        preview = ui.GroupCard("Everything else follows", form=False)
        self._preview = QLabel()
        self._preview.setObjectName("hintLabel")
        self._preview.setWordWrap(True)
        preview.body_layout().addWidget(self._preview)

        note = QLabel("The model is stored in SI base units — changing this "
                      "only changes how values are shown and entered, never the "
                      "structure itself.")
        note.setObjectName("hintLabel")
        note.setWordWrap(True)

        outer.addWidget(pick)
        outer.addWidget(preview)
        outer.addWidget(note)
        outer.addWidget(ui.dialog_buttons(self))
        style.apply(self)

        self.force.currentIndexChanged.connect(self._refresh)
        self.length.currentIndexChanged.connect(self._refresh)
        self._refresh()

    @staticmethod
    def _select(combo: QComboBox, value: str) -> None:
        i = combo.findData(value)
        if i >= 0:
            combo.setCurrentIndex(i)

    def _refresh(self, *_) -> None:
        us = UnitSystem(self.force.currentData(), self.length.currentData())
        rows = "     ".join(f"{name}: {us.label(q)}" for name, q in self._PREVIEW)
        self._preview.setText(rows)

    def result_units(self) -> tuple[str, str]:
        return self.force.currentData(), self.length.currentData()

    @classmethod
    def get(cls, parent, force: str, length: str):
        """Run modally; returns the chosen ``(force, length)`` or None."""
        dlg = cls(parent, force, length)
        if dlg.exec():
            return dlg.result_units()
        return None
