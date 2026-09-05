"""Add / edit dialogs for the model-building UI (nodes, members, loads).

Each dialog edits a Project dataclass: pass an existing item to edit it, or
``None`` to add (the id is pre-filled with the next free value). ``edit()``
runs the dialog modally and returns the resulting dataclass, or None if
cancelled. No solver / OpenGL here — pure Qt, so it is headless-constructible.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from project import (Load, LoadCase, Member, NATURE_LABELS, Node, Section)


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
        self.z = None
        if project.ndm == 3:
            self.z = _coord_spin(node.z if node else 0.0)
            form.addRow(f"z [{project.length_unit}]", self.z)

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
        z = self.z.value() if self.z is not None else 0.0
        return Node(id=self.id_spin.value(), x=self.x.value(), y=self.y.value(),
                    z=z, supports=supports if any(supports) else ())

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

        self.case = _combo([(c.name, c.id) for c in project.load_cases])
        if load is not None:
            _select(self.case, getattr(load, "case", project.default_case_id()))
        form.addRow("Load case", self.case)

        self.vals = []
        vec = tuple(load.values) if load else ()
        for k, lbl in enumerate(dof_labels(project.ndm, project.ndf)):
            spin = _force_spin(vec[k] if k < len(vec) else 0.0)
            self.vals.append(spin)
            form.addRow(f"{lbl}  [{project.force_unit}]", spin)
        form.addRow(_buttons(self))

    def data(self) -> Load:
        return Load(node=self.node.currentData(),
                    values=tuple(s.value() for s in self.vals),
                    case=self.case.currentData())

    @classmethod
    def edit(cls, parent, project, load=None):
        dlg = cls(parent, project, load)
        return dlg.data() if dlg.exec() else None


class LoadCaseDialog(QDialog):
    """Manage the project's load cases (name + nature). Returns the edited list
    of ``LoadCase`` via ``result_cases``; ids are preserved for existing cases
    and assigned fresh for new rows so load ownership stays intact."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Load cases")
        self.resize(380, 300)
        self._project = project
        self.result_cases = None
        # working rows: [id, name, nature]; id 0 => new (assigned on accept)
        self._rows = [[c.id, c.name, c.nature] for c in project.load_cases]

        v = QVBoxLayout(self)
        self.tbl = QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Name", "Nature"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().setVisible(False)
        v.addWidget(self.tbl)
        row = QHBoxLayout()
        add = QPushButton("+ Case")
        add.clicked.connect(self._add)
        rem = QPushButton("Remove")
        rem.clicked.connect(self._remove)
        row.addWidget(add)
        row.addWidget(rem)
        v.addLayout(row)
        v.addWidget(_buttons(self))
        self._reload()

    def _reload(self) -> None:
        self.tbl.setRowCount(len(self._rows))
        for r, (_id, name, nature) in enumerate(self._rows):
            self.tbl.setItem(r, 0, QTableWidgetItem(name))
            combo = QComboBox()
            for key, lbl in NATURE_LABELS.items():
                combo.addItem(lbl, key)
            i = combo.findData(nature)
            combo.setCurrentIndex(i if i >= 0 else 0)
            self.tbl.setCellWidget(r, 1, combo)

    def _add(self) -> None:
        self._sync_names()
        self._rows.append([0, f"Case {len(self._rows) + 1}", "live"])
        self._reload()

    def _remove(self) -> None:
        r = self.tbl.currentRow()
        if 0 <= r < len(self._rows) and len(self._rows) > 1:
            self._sync_names()
            del self._rows[r]
            self._reload()

    def _sync_names(self) -> None:
        for r in range(self.tbl.rowCount()):
            it = self.tbl.item(r, 0)
            if it and r < len(self._rows):
                self._rows[r][1] = it.text().strip() or self._rows[r][1]
            w = self.tbl.cellWidget(r, 1)
            if w and r < len(self._rows):
                self._rows[r][2] = w.currentData()

    def accept(self) -> None:
        self._sync_names()
        used = {row[0] for row in self._rows if row[0]}
        nxt = (max(used) if used else 0) + 1
        out = []
        for _id, name, nature in self._rows:
            if not _id:
                _id, nxt = nxt, nxt + 1
            out.append(LoadCase(id=_id, name=name, nature=nature))
        self.result_cases = out
        super().accept()

    @classmethod
    def edit(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result_cases if dlg.exec() else None


def _designations() -> list[str]:
    try:
        from femsolver.design.steel.sections import all_designations
        return list(all_designations())
    except Exception:
        return []


def _shape_props(shape: str):
    """(A, Ix, Iy, J) for an AISC designation, or None if not in the catalog."""
    try:
        from femsolver.design.steel.sections import get_section
        ss = get_section(shape.replace("X", "x"))
        return ss.A, ss.Ix, ss.Iy, ss.J
    except Exception:
        return None


def _prop_spin(value: float, decimals: int, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 1.0e3)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(value)
    return spin


class SectionDialog(QDialog):
    def __init__(self, parent, project, section=None):
        super().__init__(parent)
        self.setWindowTitle("Edit section" if section else "Add section")
        form = QFormLayout(self)

        self.id_spin = _id_spin(
            section.id if section else _next_id([s.id for s in project.sections]),
            editing=section is not None)
        form.addRow("Section id", self.id_spin)

        self.name = QLineEdit(section.name if section else "")
        form.addRow("Name", self.name)

        self.shape = QComboBox()
        self.shape.addItem("(custom A, Iz)", "")
        for d in _designations():
            self.shape.addItem(d, d)
        if section and section.shape:
            _select(self.shape, section.shape.replace("X", "x"))
        form.addRow("AISC shape", self.shape)

        self.A = _prop_spin(section.A if section else 6.0e-3, 6, 1.0e-4)
        self.Iz = _prop_spin(section.Iz if section else 2.0e-4, 8, 1.0e-5)
        form.addRow("A [m²]", self.A)
        form.addRow("Iz [m⁴]", self.Iz)
        self.Iy = self.J = None
        if project.ndm == 3:
            self.Iy = _prop_spin(section.Iy if section else 1.0e-4, 8, 1.0e-5)
            self.J = _prop_spin(section.J if section else 1.0e-5, 9, 1.0e-6)
            form.addRow("Iy [m⁴]", self.Iy)
            form.addRow("J [m⁴]", self.J)

        self.shape.currentIndexChanged.connect(self._on_shape)
        self._on_shape()                     # set initial fill / enabled state
        form.addRow(_buttons(self))

    def _on_shape(self) -> None:
        shape = self.shape.currentData()
        if shape:                            # catalog W-shape drives A/Iz/Iy/J
            props = _shape_props(shape)
            if props:
                self.A.setValue(props[0])
                self.Iz.setValue(props[1])
                if self.Iy is not None:
                    self.Iy.setValue(props[2])
                if self.J is not None:
                    self.J.setValue(props[3])
            for spin in (self.A, self.Iz, self.Iy, self.J):
                if spin is not None:
                    spin.setEnabled(False)
            if not self.name.text().strip():
                self.name.setText(shape)
        else:                                # custom section: type the values
            for spin in (self.A, self.Iz, self.Iy, self.J):
                if spin is not None:
                    spin.setEnabled(True)

    def data(self) -> Section:
        shape = self.shape.currentData() or ""
        name = (self.name.text().strip() or shape
                or f"Section {self.id_spin.value()}")
        return Section(id=self.id_spin.value(), name=name,
                       A=self.A.value(), Iz=self.Iz.value(), shape=shape,
                       Iy=self.Iy.value() if self.Iy is not None else 0.0,
                       J=self.J.value() if self.J is not None else 0.0)

    @classmethod
    def edit(cls, parent, project, section=None):
        dlg = cls(parent, project, section)
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


def _int_spin(value, lo, hi):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


def _len_spin(value):
    s = QDoubleSpinBox()
    s.setRange(0.1, 1000.0)
    s.setDecimals(2)
    s.setSingleStep(0.5)
    s.setValue(value)
    return s


class FrameDialog(QDialog):
    """Parameters for a regular building frame (2-D or 3-D space frame)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Generate frame")
        form = QFormLayout(self)

        from PySide6.QtWidgets import QCheckBox
        self.threed = QCheckBox("3-D space frame")
        self.threed.toggled.connect(self._toggle)
        form.addRow(self.threed)

        self.bays_x = _int_spin(3, 1, 500)
        self.bay_x = _len_spin(6.0)
        form.addRow("Bays (X)", self.bays_x)
        form.addRow(f"Bay width X [m]", self.bay_x)
        self.bays_y = _int_spin(2, 1, 500)
        self.bay_y = _len_spin(6.0)
        form.addRow("Bays (Y)", self.bays_y)
        form.addRow("Bay width Y [m]", self.bay_y)
        self.storeys = _int_spin(3, 1, 500)
        self.storey_h = _len_spin(3.5)
        form.addRow("Storeys", self.storeys)
        form.addRow("Storey height [m]", self.storey_h)

        self.shape = QComboBox()
        for d in _designations():
            self.shape.addItem(d, d)
        _select(self.shape, "W12x65")
        form.addRow("AISC shape", self.shape)

        form.addRow(_buttons(self))
        self._toggle(False)

    def _toggle(self, on) -> None:
        self.bays_y.setEnabled(on)
        self.bay_y.setEnabled(on)

    def params(self) -> dict:
        td = self.threed.isChecked()
        return dict(bays_x=self.bays_x.value(), bay_x=self.bay_x.value(),
                    storeys=self.storeys.value(), storey_h=self.storey_h.value(),
                    bays_y=self.bays_y.value() if td else 0,
                    bay_y=self.bay_y.value(), shape=self.shape.currentData())

    @classmethod
    def get(cls, parent):
        dlg = cls(parent)
        return dlg.params() if dlg.exec() else None


class LoadGenDialog(QDialog):
    """Parameters for a parametric load pattern (gravity or lateral)."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Generate loads")
        form = QFormLayout(self)
        self.kind = QComboBox()
        self.kind.addItem("Gravity (downward, per node)", "gravity")
        self.kind.addItem("Lateral X (storey forces)", "lateral_x")
        if project.ndm == 3:
            self.kind.addItem("Lateral Y (storey forces)", "lateral_y")
        self.kind.currentIndexChanged.connect(self._relabel)
        form.addRow("Pattern", self.kind)
        self.mag = _force_spin(50000.0)
        self._label = QLabel()
        form.addRow(self._label, self.mag)
        form.addRow(_buttons(self))
        self._relabel()

    def _relabel(self) -> None:
        gravity = self.kind.currentData() == "gravity"
        self._label.setText("Load per node [N]" if gravity else "Base shear [N]")

    def params(self):
        return self.kind.currentData(), self.mag.value()

    @classmethod
    def get(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.params() if dlg.exec() else None


class MoveDialog(QDialog):
    """Translation vector for moving the selection."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Move selection")
        form = QFormLayout(self)
        self.dx, self.dy = _coord_spin(0.0), _coord_spin(0.0)
        form.addRow(f"dx [{project.length_unit}]", self.dx)
        form.addRow(f"dy [{project.length_unit}]", self.dy)
        self.dz = None
        if project.ndm == 3:
            self.dz = _coord_spin(0.0)
            form.addRow(f"dz [{project.length_unit}]", self.dz)
        form.addRow(_buttons(self))

    def data(self):
        return (self.dx.value(), self.dy.value(),
                self.dz.value() if self.dz is not None else 0.0)

    @classmethod
    def get(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.data() if dlg.exec() else None


class CopyDialog(QDialog):
    """Offset vector + copy count for arraying (or extruding) the selection."""

    def __init__(self, parent, project, title="Copy / array selection"):
        super().__init__(parent)
        self.setWindowTitle(title)
        form = QFormLayout(self)
        self.dx, self.dy = _coord_spin(0.0), _coord_spin(0.0)
        form.addRow(f"dx [{project.length_unit}]", self.dx)
        form.addRow(f"dy [{project.length_unit}]", self.dy)
        self.dz = None
        if project.ndm == 3:
            self.dz = _coord_spin(0.0)
            form.addRow(f"dz [{project.length_unit}]", self.dz)
        self.count = _int_spin(1, 1, 500)
        form.addRow("Copies", self.count)
        form.addRow(_buttons(self))

    def data(self):
        return (self.dx.value(), self.dy.value(),
                self.dz.value() if self.dz is not None else 0.0,
                self.count.value())

    @classmethod
    def get(cls, parent, project, title="Copy / array selection"):
        dlg = cls(parent, project, title)
        return dlg.data() if dlg.exec() else None


class MirrorDialog(QDialog):
    """Mirror plane (axis = coord) for reflecting the selection."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Mirror selection")
        form = QFormLayout(self)
        self.axis = QComboBox()
        for a in (["X", "Y"] if project.ndm == 2 else ["X", "Y", "Z"]):
            self.axis.addItem(f"{a} = constant plane", a)
        form.addRow("Mirror plane", self.axis)
        self.coord = _coord_spin(0.0)
        form.addRow(f"Plane coordinate [{project.length_unit}]", self.coord)
        form.addRow(_buttons(self))

    def data(self):
        return (self.axis.currentData(), self.coord.value())

    @classmethod
    def get(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.data() if dlg.exec() else None


class RotateDialog(QDialog):
    """Axis, center and angle for rotating the selection in place."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Rotate selection")
        self._three_d = project.ndm == 3
        form = QFormLayout(self)
        self.axis = QComboBox()
        if self._three_d:
            for a in ("Z", "X", "Y"):
                self.axis.addItem(f"about {a} axis", a)
            form.addRow("Axis", self.axis)
        self.cx = _coord_spin(0.0)
        self.cy = _coord_spin(0.0)
        form.addRow(f"Center x [{project.length_unit}]", self.cx)
        form.addRow(f"Center y [{project.length_unit}]", self.cy)
        self.cz = None
        if self._three_d:
            self.cz = _coord_spin(0.0)
            form.addRow(f"Center z [{project.length_unit}]", self.cz)
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-360.0, 360.0)
        self.angle.setDecimals(1)
        self.angle.setSuffix(" °")
        form.addRow("Angle", self.angle)
        form.addRow(_buttons(self))

    def data(self):
        axis = self.axis.currentData() if self._three_d else "Z"
        center = (self.cx.value(), self.cy.value(),
                  self.cz.value() if self.cz is not None else 0.0)
        return (axis, center, self.angle.value())

    @classmethod
    def get(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.data() if dlg.exec() else None
