"""Add / edit dialogs for the model-building UI (nodes, members, loads).

Each dialog edits a Project dataclass: pass an existing item to edit it, or
``None`` to add (the id is pre-filled with the next free value). ``edit()``
runs the dialog modally and returns the resulting dataclass, or None if
cancelled. No solver / OpenGL here — pure Qt, so it is headless-constructible.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QMessageBox, QSpinBox, QWidget)

from project import Load, Member, Node


def dof_labels(ndm: int, ndf: int) -> list[str]:
    """Human labels for each DOF, e.g. 2-D frame -> ['Dx', 'Dy', 'Rz']."""
    if ndm == 2 and ndf <= 3:
        return ["Dx", "Dy", "Rz"][:ndf]
    if ndm == 3 and ndf <= 6:
        return ["Dx", "Dy", "Dz", "Rx", "Ry", "Rz"][:ndf]
    return [f"D{i + 1}" for i in range(ndf)]


def _next_id(ids) -> int:
    return (max(ids) + 1) if ids else 1


def _id_spin(value: int, editing: bool) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(1, 10_000_000)
    spin.setValue(value)
    spin.setEnabled(not editing)          # id is fixed while editing
    return spin


def _coord_spin(value: float = 0.0) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-1.0e6, 1.0e6)
    spin.setDecimals(4)
    spin.setSingleStep(0.1)
    spin.setValue(value)
    return spin


def _force_spin(value: float = 0.0) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(-1.0e12, 1.0e12)
    spin.setDecimals(3)
    spin.setSingleStep(1000.0)
    spin.setValue(value)
    return spin


class NodeDialog(QDialog):
    def __init__(self, parent, project, node=None):
        super().__init__(parent)
        self.setWindowTitle("Edit node" if node else "Add node")
        form = QFormLayout(self)

        self.id_spin = _id_spin(
            node.id if node else _next_id([n.id for n in project.nodes]),
            editing=node is not None)
        form.addRow("Node id", self.id_spin)

        self.x = _coord_spin(node.x if node else 0.0)
        self.y = _coord_spin(node.y if node else 0.0)
        form.addRow(f"x [{project.length_unit}]", self.x)
        form.addRow(f"y [{project.length_unit}]", self.y)

        self.fix = []
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        sup = tuple(node.supports) if (node and node.supports) else ()
        for k, lbl in enumerate(dof_labels(project.ndm, project.ndf)):
            cb = QCheckBox(lbl)
            cb.setChecked(k < len(sup) and bool(sup[k]))
            self.fix.append(cb)
            hl.addWidget(cb)
        form.addRow("Fixity", row)
        form.addRow(_buttons(self))

    def data(self) -> Node:
        supports = tuple(1 if cb.isChecked() else 0 for cb in self.fix)
        return Node(id=self.id_spin.value(), x=self.x.value(), y=self.y.value(),
                    supports=supports if any(supports) else ())

    @classmethod
    def edit(cls, parent, project, node=None):
        dlg = cls(parent, project, node)
        return dlg.data() if dlg.exec() else None


class MemberDialog(QDialog):
    def __init__(self, parent, project, member=None):
        super().__init__(parent)
        self.setWindowTitle("Edit member" if member else "Add member")
        form = QFormLayout(self)

        self.id_spin = _id_spin(
            member.id if member else _next_id([m.id for m in project.members]),
            editing=member is not None)
        form.addRow("Member id", self.id_spin)

        self.n1 = _combo([(str(n.id), n.id) for n in project.nodes])
        self.n2 = _combo([(str(n.id), n.id) for n in project.nodes])
        self.sec = _combo([(f"{s.id}: {s.name}", s.id) for s in project.sections])
        self.mat = _combo([(f"{m.id}: {m.name}", m.id) for m in project.materials])

        if member:
            _select(self.n1, member.n1)
            _select(self.n2, member.n2)
            _select(self.sec, member.section)
            _select(self.mat, member.material)
        elif self.n2.count() > 1:
            self.n2.setCurrentIndex(1)

        form.addRow("Start node", self.n1)
        form.addRow("End node", self.n2)
        form.addRow("Section", self.sec)
        form.addRow("Material", self.mat)
        form.addRow(_buttons(self))

    def accept(self) -> None:
        if self.n1.currentData() == self.n2.currentData():
            QMessageBox.warning(self, "Invalid member",
                                "Start and end nodes must differ.")
            return
        super().accept()

    def data(self) -> Member:
        return Member(id=self.id_spin.value(), n1=self.n1.currentData(),
                      n2=self.n2.currentData(), section=self.sec.currentData(),
                      material=self.mat.currentData())

    @classmethod
    def edit(cls, parent, project, member=None):
        dlg = cls(parent, project, member)
        return dlg.data() if dlg.exec() else None


class LoadDialog(QDialog):
    def __init__(self, parent, project, load=None):
        super().__init__(parent)
        self.setWindowTitle("Edit load" if load else "Add load")
        form = QFormLayout(self)

        self.node = _combo([(str(n.id), n.id) for n in project.nodes])
        if load:
            _select(self.node, load.node)
        form.addRow("Node", self.node)

        self.vals = []
        vec = tuple(load.values) if load else ()
        for k, lbl in enumerate(dof_labels(project.ndm, project.ndf)):
            spin = _force_spin(vec[k] if k < len(vec) else 0.0)
            self.vals.append(spin)
            form.addRow(f"{lbl}  [{project.force_unit}]", spin)
        form.addRow(_buttons(self))

    def data(self) -> Load:
        return Load(node=self.node.currentData(),
                    values=tuple(s.value() for s in self.vals))

    @classmethod
    def edit(cls, parent, project, load=None):
        dlg = cls(parent, project, load)
        return dlg.data() if dlg.exec() else None


def _combo(entries) -> QComboBox:
    combo = QComboBox()
    for label, value in entries:
        combo.addItem(label, value)
    return combo


def _select(combo: QComboBox, value) -> None:
    i = combo.findData(value)
    if i >= 0:
        combo.setCurrentIndex(i)


def _buttons(dialog: QDialog) -> QDialogButtonBox:
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                          | QDialogButtonBox.StandardButton.Cancel)
    bb.accepted.connect(dialog.accept)
    bb.rejected.connect(dialog.reject)
    return bb
