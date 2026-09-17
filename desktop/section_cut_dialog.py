"""Section-cut (design-strip) dialog (slab plan S7).

Pick two nodes that span a cut line across a slab and integrate a shell result
along it — the total bending moment about the cut (design-strip moment) or the
total vertical shear crossing it (SAP-style 'section cut'). Uses the solved
``model`` passed in; draws the cut on the viewport for feedback.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout, QLabel,
                               QPushButton)

import model_geometry as mg
import style
from editing import _buttons, _combo, _select


class SectionCutDialog(QDialog):
    """Integrate a shell resultant along a two-node cut line."""

    _QUANTITIES = [("Moment about cut (design strip)", "M"),
                   ("Shear across cut", "V")]

    def __init__(self, parent, project, model):
        super().__init__(parent)
        self.setWindowTitle("Section cut (design strip)")
        self._model = model
        self._view = getattr(parent, "view", None)
        self._last = None
        form = QFormLayout(self)

        items = [(str(n.id), n.id) for n in project.nodes]
        self.n1 = _combo(items)
        self.n2 = _combo(items)
        if len(items) > 1:
            _select(self.n2, items[1][1])
        form.addRow("From node", self.n1)
        form.addRow("To node", self.n2)
        self.quantity = _combo(self._QUANTITIES)
        form.addRow("Quantity", self.quantity)

        run = QPushButton("Compute")
        run.clicked.connect(self._compute)
        form.addRow(run)
        self.result = QLabel("Pick two nodes spanning the cut, then Compute.")
        self.result.setWordWrap(True)
        self.result.setObjectName("hintLabel")
        form.addRow(self.result)
        form.addRow(_buttons(self))
        style.apply(self)

    def _node_xyz(self, tag):
        return mg.to_xyz(self._model.node(tag).coords)

    def _compute(self) -> None:
        a, b = self.n1.currentData(), self.n2.currentData()
        if a is None or b is None or a == b:
            self.result.setText("Choose two different nodes.")
            return
        p0, p1 = self._node_xyz(a), self._node_xyz(b)
        q = self.quantity.currentData()
        total = mg.section_cut(self._model, p0, p1, q)
        self._last = total
        if q == "M":
            self.result.setText(
                f"<b>Design-strip moment</b> = {total / 1e3:.2f} kN·m "
                f"(total about the cut)")
        else:
            self.result.setText(
                f"<b>Shear across cut</b> = {total / 1e3:.2f} kN (total)")
        if self._view is not None:
            try:
                self._view.show_cut_line(p0, p1)
            except Exception:
                pass

    def result_value(self):
        return self._last

    @classmethod
    def run(cls, parent, project, model):
        dlg = cls(parent, project, model)
        dlg.exec()
        return dlg.result_value()
