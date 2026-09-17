"""Punching-shear check dialog (slab plan S9 slice 3).

Pick a column node, enter the column size / effective depth / f'c, and check the
ACI 318-19 punching capacity against the demand read from that node's vertical
reaction (after a solve). Pure Qt + the ``slab_punching`` calc core; the demand
is pulled from the solved ``model`` passed in.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout, QLabel,
                               QPushButton)

import style
from editing import _buttons, _combo, _select
from slab_punching import aci_punching_check, punching_demand_from_reaction
from unit_widgets import UnitSpin, labeled
from units import Quantity, UnitSystem


class SlabPunchingDialog(QDialog):
    """Interior/edge/corner punching check at a chosen column node."""

    _POSITIONS = [("Interior", "interior"), ("Edge", "edge"),
                  ("Corner", "corner")]

    def __init__(self, parent, project, model):
        super().__init__(parent)
        self.setWindowTitle("Punching-shear check (ACI 318-19)")
        self._model = model
        self._us = UnitSystem.from_project(project)
        self._last = None                          # last result dict (for tests)
        form = QFormLayout(self)

        self.node = _combo([(str(n.id), n.id) for n in project.nodes])
        form.addRow("Column node", self.node)
        self.c_x = UnitSpin(Quantity.LENGTH, self._us, si=0.40, decimals=3,
                            step=0.05, rng=(0.01, 100.0))
        self.c_y = UnitSpin(Quantity.LENGTH, self._us, si=0.40, decimals=3,
                            step=0.05, rng=(0.01, 100.0))
        self.d = UnitSpin(Quantity.LENGTH, self._us, si=0.20, decimals=3,
                          step=0.01, rng=(0.01, 10.0))
        self.fc = UnitSpin(Quantity.STRESS, self._us, si=30e6, decimals=3,
                           step=5e6, rng=(1.0, 1.0e12))
        form.addRow(labeled("Column c_x", self.c_x), self.c_x)
        form.addRow(labeled("Column c_y", self.c_y), self.c_y)
        form.addRow(labeled("Effective depth d", self.d), self.d)
        form.addRow(labeled("f'c", self.fc), self.fc)
        self.position = _combo(self._POSITIONS)
        form.addRow("Column position", self.position)

        check = QPushButton("Check")
        check.clicked.connect(self._run_check)
        form.addRow(check)
        self.result = QLabel("Pick a column node and click Check.")
        self.result.setWordWrap(True)
        self.result.setObjectName("hintLabel")
        form.addRow(self.result)
        form.addRow(_buttons(self))
        style.apply(self)

    def _run_check(self) -> None:
        node = self.node.currentData()
        if node is None:
            return
        V_u = punching_demand_from_reaction(self._model, node)
        res = aci_punching_check(
            V_u, c_x=self.c_x.si_value(), c_y=self.c_y.si_value(),
            d=self.d.si_value(), f_c=self.fc.si_value(),
            position=self.position.currentData())
        self._last = res
        verdict = "✓ OK" if res["ok"] else "✗ needs shear reinforcement"
        color = style.OK if res["ok"] else style.BAD
        self.result.setText(
            f"<b style='color:{color}'>{verdict}</b> — DCR = {res['dcr']:.2f}"
            f"<br>V_u = {res['V_u']/1e3:.1f} kN · "
            f"v_u = {res['v_u']/1e6:.3f} MPa · "
            f"φv_c = {res['phi_v_c']/1e6:.3f} MPa · "
            f"b₀ = {res['b_0']:.3f} m")

    def result_data(self):
        return self._last

    @classmethod
    def run(cls, parent, project, model):
        dlg = cls(parent, project, model)
        dlg.exec()
        return dlg.result_data()
