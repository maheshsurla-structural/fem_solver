"""Slab reinforcement contour dialog (slab plan S9).

Collects the design inputs needed to turn Wood-Armer design moments into a
required-steel-area (As per width) contour: which bar layer to show, the cover
(to bar centroid), and the steel / concrete strengths. Pure Qt, headless-
constructible; ``configure()`` returns ``(quantity, cover, fy, fc)`` or ``None``.
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout

import style
from analysis_ui import dialog_buttons
from model_geometry import AREA_REBAR_QUANTITIES
from unit_widgets import UnitSpin, labeled
from units import Quantity, UnitSystem


class ReinforcementDialog(QDialog):
    """Pick the bar layer + design inputs for the As contour."""

    def __init__(self, parent, project, *, last=None):
        super().__init__(parent)
        self.setWindowTitle("Slab reinforcement contour")
        self._us = UnitSystem.from_project(project)
        last = last or {}
        form = QFormLayout(self)

        self.quantity = QComboBox()
        for key, (label, _comp) in AREA_REBAR_QUANTITIES.items():
            self.quantity.addItem(label, key)
        if last.get("quantity"):
            i = self.quantity.findData(last["quantity"])
            if i >= 0:
                self.quantity.setCurrentIndex(i)
        form.addRow("Bar layer", self.quantity)

        self.cover = UnitSpin(Quantity.LENGTH, self._us,
                              si=last.get("cover", 0.025), decimals=4,
                              step=0.005, rng=(0.0, 1.0))
        form.addRow(labeled("Cover (to bar)", self.cover), self.cover)
        self.fy = UnitSpin(Quantity.STRESS, self._us,
                           si=last.get("fy", 420e6), decimals=3, step=10e6,
                           rng=(1.0, 1.0e12))
        form.addRow(labeled("fy", self.fy), self.fy)
        self.fc = UnitSpin(Quantity.STRESS, self._us,
                           si=last.get("fc", 30e6), decimals=3, step=5e6,
                           rng=(1.0, 1.0e12))
        form.addRow(labeled("f'c", self.fc), self.fc)

        form.addRow(dialog_buttons(self))
        style.apply(self)

    def data(self) -> dict:
        return {"quantity": self.quantity.currentData(),
                "cover": float(self.cover.si_value()),
                "fy": float(self.fy.si_value()),
                "fc": float(self.fc.si_value())}

    @classmethod
    def configure(cls, parent, project, *, last=None):
        dlg = cls(parent, project, last=last)
        return dlg.data() if dlg.exec() else None
