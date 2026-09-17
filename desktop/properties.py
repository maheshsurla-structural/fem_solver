"""The Properties inspector — a docked panel that edits the selected item live.

Reuses the widget helpers from ``editing`` and commits each change through the
main window's apply callback, so edits made here are undoable like any other.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QVBoxLayout,
                               QWidget)

from editing import (_coord_spin, _designations, _force_spin, _prop_spin,
                     _shape_props, dof_labels)
from project import Load, Member, Node, Section
from units import Quantity, UnitSystem


class PropertiesPanel(QWidget):
    def __init__(self, on_apply, on_bulk, parent=None):
        super().__init__(parent)
        self._on_apply = on_apply           # callback(kind, key, new_item)
        self._on_bulk = on_bulk             # callback(kind, ids, payload)
        self._project = None
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(8, 8, 8, 8)
        self._content = QLabel()
        self._outer.addWidget(self._content)
        self._outer.addStretch(1)
        self.clear_selection()

    # ----------------------------------------------------------------- public
    def clear_selection(self) -> None:
        self._project = None
        lbl = QLabel("Select a node, member, section, or load to edit.")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color:#888;")
        self._swap(lbl)

    def show_item(self, project, kind, key) -> None:
        self._project = project
        item = self._find(kind, key)
        if item is None:
            self.clear_selection()
            return
        builder = {"node": self._node_form, "member": self._member_form,
                   "section": self._section_form, "load": self._load_form,
                   "area": self._area_form}.get(kind)
        if builder is None:
            self.clear_selection()
            return
        self._swap(builder(key, item))

    def show_multi(self, project, refs) -> None:
        """Summary + bulk-edit controls for a multi-selection."""
        self._project = project
        node_ids = [k for (t, k) in refs if t == "node"]
        member_ids = [k for (t, k) in refs if t == "member"]
        counts = [f"{len(ids)} {label}" for label, ids in (
            ("nodes", node_ids), ("members", member_ids),
            ("sections", [k for t, k in refs if t == "section"]),
            ("loads", [k for t, k in refs if t == "load"])) if ids]

        w = QWidget()
        form = QFormLayout(w)
        head = QLabel("Selected: " + ", ".join(counts))
        head.setStyleSheet("font-weight:600;")
        form.addRow(head)

        if member_ids:
            sec = _pair_combo([(s.id, f"{s.id}: {s.name}")
                               for s in project.sections], None)
            mat = _pair_combo([(m.id, f"{m.id}: {m.name}")
                               for m in project.materials], None)
            form.addRow(QLabel("Assign to members:"))
            form.addRow("Section", sec)
            form.addRow("Material", mat)
            btn = QPushButton(f"Apply to {len(member_ids)} members")
            btn.clicked.connect(lambda: self._on_bulk(
                "member", list(member_ids),
                {"section": sec.currentData(), "material": mat.currentData()}))
            form.addRow(btn)

        if node_ids:
            boxes, row = [], QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            for lbl in dof_labels(project.ndm, project.ndf):
                cb = QCheckBox(lbl)
                boxes.append(cb)
                hl.addWidget(cb)
            form.addRow(QLabel("Set node fixity:"))
            form.addRow("Fixity", row)
            btn = QPushButton(f"Apply to {len(node_ids)} nodes")
            btn.clicked.connect(lambda: self._on_bulk(
                "node", list(node_ids),
                {"supports": tuple(1 if b.isChecked() else 0 for b in boxes)}))
            form.addRow(btn)
        self._swap(w)

    # -------------------------------------------------------------- internals
    def _swap(self, widget) -> None:
        self._outer.replaceWidget(self._content, widget)
        self._content.deleteLater()
        self._content = widget

    def _find(self, kind, key):
        p = self._project
        if kind == "node":
            return next((n for n in p.nodes if n.id == key), None)
        if kind == "member":
            return next((m for m in p.members if m.id == key), None)
        if kind == "section":
            return next((s for s in p.sections if s.id == key), None)
        if kind == "load":
            return p.loads[key] if 0 <= key < len(p.loads) else None
        if kind == "area":
            return next((a for a in getattr(p, "areas", []) if a.id == key),
                        None)
        return None

    def _frame(self, title):
        w = QWidget()
        form = QFormLayout(w)
        form.setContentsMargins(0, 0, 0, 0)
        head = QLabel(title)
        head.setStyleSheet("font-weight:600;")
        form.addRow(head)
        return w, form

    @staticmethod
    def _apply_button(form, commit) -> None:
        btn = QPushButton("Apply")
        btn.clicked.connect(commit)
        form.addRow(btn)

    def _area_form(self, key, area):
        """Read-only summary of a surface (slab / shell) area (slab plan S10)."""
        p = self._project
        w, form = self._frame(f"Area {area.id}")
        ss = next((s for s in getattr(p, "shell_sections", [])
                   if s.id == area.shell_section), None)
        mat = next((m for m in p.materials if m.id == area.material), None)
        form.addRow("Corners", QLabel(", ".join(str(n) for n in area.nodes)))
        form.addRow("Thickness", QLabel(ss.name if ss else str(area.shell_section)))
        form.addRow("Material", QLabel(mat.name if mat else str(area.material)))
        form.addRow("Mesh", QLabel(f"{area.mesh[0]} × {area.mesh[1]}"))
        form.addRow("Type", QLabel(ss.kind if ss else "—"))
        return w

    # ------------------------------------------------------------------ forms
    def _node_form(self, key, node):
        p = self._project
        us = UnitSystem.from_project(p)
        w, form = self._frame(f"Node {node.id}")
        x, y = _coord_spin(us, node.x), _coord_spin(us, node.y)
        form.addRow(f"x [{x.unit_label()}]", x)
        form.addRow(f"y [{y.unit_label()}]", y)
        z = None
        if p.ndm == 3:
            z = _coord_spin(us, node.z)
            form.addRow(f"z [{z.unit_label()}]", z)
        boxes, row = [], QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        sup = tuple(node.supports) if node.supports else ()
        for k, lbl in enumerate(dof_labels(p.ndm, p.ndf)):
            cb = QCheckBox(lbl)
            cb.setChecked(k < len(sup) and bool(sup[k]))
            boxes.append(cb)
            hl.addWidget(cb)
        form.addRow("Fixity", row)

        def commit():
            supports = tuple(1 if b.isChecked() else 0 for b in boxes)
            self._on_apply("node", key, Node(
                id=node.id, x=x.si_value(), y=y.si_value(),
                z=z.si_value() if z is not None else node.z,
                supports=supports if any(supports) else ()))
        self._apply_button(form, commit)
        return w

    def _member_form(self, key, member):
        p = self._project
        w, form = self._frame(f"Member {member.id}")
        n1 = _id_combo([n.id for n in p.nodes], member.n1)
        n2 = _id_combo([n.id for n in p.nodes], member.n2)
        sec = _pair_combo([(s.id, f"{s.id}: {s.name}") for s in p.sections],
                          member.section)
        mat = _pair_combo([(m.id, f"{m.id}: {m.name}") for m in p.materials],
                          member.material)
        form.addRow("Start node", n1)
        form.addRow("End node", n2)
        form.addRow("Section", sec)
        form.addRow("Material", mat)

        def commit():
            self._on_apply("member", key, Member(
                id=member.id, n1=n1.currentData(), n2=n2.currentData(),
                section=sec.currentData(), material=mat.currentData(),
                kind=member.kind))
        self._apply_button(form, commit)
        return w

    def _section_form(self, key, section):
        p = self._project
        us = UnitSystem.from_project(p)
        w, form = self._frame(f"Section {section.id}")
        name = QLineEdit(section.name)
        form.addRow("Name", name)
        shape = QComboBox()
        shape.addItem("(custom A, Iz)", "")
        for d in _designations():
            shape.addItem(d, d)
        if section.shape:
            _sel(shape, section.shape.replace("X", "x"))
        form.addRow("AISC shape", shape)
        A = _prop_spin(us, Quantity.AREA, section.A, decimals=6, step=1.0e-4)
        Iz = _prop_spin(us, Quantity.INERTIA, section.Iz, decimals=8, step=1.0e-5)
        form.addRow(f"A [{A.unit_label()}]", A)
        form.addRow(f"Iz [{Iz.unit_label()}]", Iz)
        Iy = Jt = None
        if p.ndm == 3:
            Iy = _prop_spin(us, Quantity.INERTIA, section.Iy or 1.0e-4,
                            decimals=8, step=1.0e-5)
            Jt = _prop_spin(us, Quantity.INERTIA, section.J or 1.0e-5,
                            decimals=9, step=1.0e-6)
            form.addRow(f"Iy [{Iy.unit_label()}]", Iy)
            form.addRow(f"J [{Jt.unit_label()}]", Jt)

        def on_shape():
            s = shape.currentData()
            if s:
                props = _shape_props(s)          # catalog values are SI
                if props:
                    A.set_si(props[0])
                    Iz.set_si(props[1])
                    if Iy is not None:
                        Iy.set_si(props[2])
                    if Jt is not None:
                        Jt.set_si(props[3])
            for sp in (A, Iz, Iy, Jt):
                if sp is not None:
                    sp.setEnabled(not s)
        shape.currentIndexChanged.connect(on_shape)
        on_shape()

        def commit():
            s = shape.currentData() or ""
            self._on_apply("section", key, Section(
                id=section.id, name=name.text().strip() or s or section.name,
                A=A.si_value(), Iz=Iz.si_value(), shape=s,
                Iy=Iy.si_value() if Iy is not None else section.Iy,
                J=Jt.si_value() if Jt is not None else section.J))
        self._apply_button(form, commit)
        return w

    def _load_form(self, key, load):
        p = self._project
        us = UnitSystem.from_project(p)
        w, form = self._frame(f"Load on node {load.node}")
        node = _id_combo([n.id for n in p.nodes], load.node)
        form.addRow("Node", node)
        vals, vec = [], tuple(load.values)
        for k, lbl in enumerate(dof_labels(p.ndm, p.ndf)):
            # translational DOFs are forces, rotational ones moments (F·L)
            spin = _force_spin(us, vec[k] if k < len(vec) else 0.0,
                               quantity=us.dof_quantity(lbl))
            vals.append(spin)
            form.addRow(f"{lbl} [{spin.unit_label()}]", spin)

        def commit():
            self._on_apply("load", key, Load(
                node=node.currentData(),
                values=tuple(s.si_value() for s in vals)))
        self._apply_button(form, commit)
        return w


def _id_combo(ids, current):
    c = QComboBox()
    for i in ids:
        c.addItem(str(i), i)
    _sel(c, current)
    return c


def _pair_combo(pairs, current):
    c = QComboBox()
    for value, label in pairs:
        c.addItem(label, value)
    _sel(c, current)
    return c


def _sel(combo, value):
    i = combo.findData(value)
    if i >= 0:
        combo.setCurrentIndex(i)
