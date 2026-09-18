"""Area (surface / pressure) load dialog (slab plan S5b).

A :class:`project.AreaLoad` is a uniform load on an ``Area`` object — either a
downward gravity area load (``kind="gravity"``, applied as global −Z) or a
pressure normal to the surface (``kind="pressure"``), magnitude ``w`` in
pressure units (Pa). The surface analogue of :class:`member_load_dialog`, in the
grouped-card scaffold.

Pure Qt, headless-constructible under ``QT_QPA_PLATFORM=offscreen``; ``.edit()``
runs it modally and returns the ``AreaLoad`` (or ``None`` if cancelled).
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout

import analysis_ui as ui
import style
from pick import PickDialog
from project import AreaLoad
from unit_widgets import UnitSpin
from units import Quantity, UnitSystem


def _combo(entries) -> QComboBox:
    c = QComboBox()
    for label, value in entries:
        c.addItem(label, value)
    return c


def _select(combo: QComboBox, value) -> None:
    i = combo.findData(value)
    if i >= 0:
        combo.setCurrentIndex(i)


class AreaLoadDialog(PickDialog):
    """Add / edit a uniform area (surface / pressure) load."""

    def __init__(self, parent, project, aload=None):
        super().__init__(parent)
        self.setWindowTitle("Edit area load" if aload else "Add area load")
        self._us = UnitSystem.from_project(project)
        unit = self._us.label(Quantity.STRESS)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(style.SP_LG, style.SP_LG,
                                 style.SP_LG, style.SP_LG)
        outer.setSpacing(style.SP_MD)

        self.area = _combo([(f"{a.id}:  {'-'.join(map(str, a.nodes))}", a.id)
                            for a in project.areas])
        if aload is not None:
            _select(self.area, aload.area)
        self.case = _combo([(c.name, c.id) for c in project.load_cases])
        if aload is not None:
            _select(self.case, getattr(aload, "case", project.default_case_id()))
        self.kind = _combo([("Gravity (global −Z)", "gravity"),
                            ("Pressure (normal to surface)", "pressure"),
                            ("Self-weight (ρ·t·g, auto)", "selfweight")])
        if aload is not None:
            _select(self.kind, getattr(aload, "kind", "gravity"))
        self.kind.currentIndexChanged.connect(self._on_kind)

        applied = ui.GroupCard("Applied to")
        applied.add_row("Area", self.area)
        applied.add_row("Load pattern", self.case)
        applied.add_row("Type", self.kind)

        loads = ui.GroupCard(f"Uniform area load  [{unit}]")
        self.w = UnitSpin(Quantity.STRESS, self._us,
                          si=aload.w if aload else 0.0, decimals=3,
                          step=1000.0, rng=(-1.0e12, 1.0e12))
        loads.add_row(f"w  [{self.w.unit_label()}]", self.w)
        note = QLabel("Gravity acts downward (global −Z); pressure acts along "
                      "the surface normal. Magnitude is force per unit area.")
        note.setObjectName("hintLabel")
        note.setWordWrap(True)
        loads.add_full_row(note)

        outer.addWidget(applied)
        outer.addWidget(loads)
        outer.addWidget(ui.dialog_buttons(self))
        style.apply(self)
        self._on_kind()

    def _on_kind(self) -> None:
        # self-weight is computed from ρ·t·g, so the magnitude field is disabled
        self.w.setEnabled(self.kind.currentData() != "selfweight")

    def data(self) -> AreaLoad:
        return AreaLoad(area=self.area.currentData(),
                        w=0.0 if self.kind.currentData() == "selfweight"
                        else float(self.w.si_value()),
                        kind=self.kind.currentData(),
                        case=self.case.currentData())

    @classmethod
    def edit(cls, parent, project, aload=None):
        dlg = cls(parent, project, aload)
        return dlg.data() if dlg.exec() else None
