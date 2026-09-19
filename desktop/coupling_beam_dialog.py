"""Coupling-beam dialog (wall plan W6).

Pick two wall panels, an elevation and a beam section/material; returns the
params for ``coupling.add_coupling_beam``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout, QLabel)

import style
from editing import _buttons, _combo, _grid_len_spin
from units import UnitSystem


class CouplingBeamDialog(QDialog):
    """Two wall panels + elevation + section/material for a coupling beam."""

    def __init__(self, parent, project, units: UnitSystem | None = None,
                 seed=None):
        super().__init__(parent)
        self.setWindowTitle("Coupling beam")
        self._us = units or UnitSystem()
        form = QFormLayout(self)

        walls = [a for a in project.areas if a.role == "wall"
                 and len(a.nodes) == 4]

        def _wall_items():
            return [(f"Area {a.id}" + (f" · {a.pier}" if a.pier else ""), a.id)
                    for a in walls]

        self.wall_a = _combo(_wall_items())
        self.wall_b = _combo(_wall_items())
        seed = list(seed or [])
        if len(seed) >= 2:
            self.wall_a.setCurrentIndex(max(self.wall_a.findData(seed[0]), 0))
            self.wall_b.setCurrentIndex(max(self.wall_b.findData(seed[1]), 0))
        elif len(walls) > 1:
            self.wall_b.setCurrentIndex(1)

        self.elev = _grid_len_spin(self._us, 3.0)
        self.sec = _combo([(f"{s.id}: {s.name}", s.id) for s in project.sections])
        self.mat = _combo([(f"{m.id}: {m.name}", m.id) for m in project.materials])

        form.addRow("Wall A", self.wall_a)
        form.addRow("Wall B", self.wall_b)
        form.addRow("Elevation", self.elev)
        form.addRow("Beam section", self.sec)
        form.addRow("Material", self.mat)
        hint = QLabel("The beam spans each wall's facing (inner) edge; the "
                      "elevation snaps to each wall's nearest mesh row so the "
                      "ends tie into the shell.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        form.addRow("", hint)
        form.addRow(_buttons(self))
        style.apply(self)

    def params(self):
        return dict(area_a_id=self.wall_a.currentData(),
                    area_b_id=self.wall_b.currentData(),
                    elev=self.elev.si_value(),
                    section=self.sec.currentData(),
                    material=self.mat.currentData())

    @classmethod
    def get(cls, parent, project, units=None, seed=None):
        dlg = cls(parent, project, units, seed)
        return dlg.params() if dlg.exec() else None
