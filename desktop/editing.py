"""Add / edit dialogs for the model-building UI (nodes, members, loads).

Each dialog edits a Project dataclass: pass an existing item to edit it, or
``None`` to add (the id is pre-filled with the next free value). ``edit()``
runs the dialog modally and returns the resulting dataclass, or None if
cancelled. No solver / OpenGL here — pure Qt, so it is headless-constructible.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

import analysis_ui as ui
import icons
import style
from pick import PickDialog
from project import (Area, Diaphragm, Load, LoadCase, Member, NATURE_ASCE,
                     NATURE_LABELS, Node, Section)
from unit_widgets import UnitSpin, labeled
from units import Quantity, UnitSystem


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


def _coord_spin(units: UnitSystem, si: float = 0.0) -> UnitSpin:
    """A length spin (plan U3): displays in the project's length unit, stores
    SI metres. ``si`` is the stored (SI) seed value."""
    return UnitSpin(Quantity.LENGTH, units, si=si, decimals=4, step=0.1,
                    rng=(-1.0e6, 1.0e6))


def _force_spin(units: UnitSystem, si: float = 0.0,
                quantity: Quantity = Quantity.FORCE) -> UnitSpin:
    """A force/moment spin (plan U3): displays in the project's force (or
    force·length) unit, stores SI. ``quantity`` lets a load vector's rotational
    DOFs be moments while translational ones are forces."""
    return UnitSpin(quantity, units, si=si, decimals=3, step=1000.0,
                    rng=(-1.0e12, 1.0e12))


class NodeDialog(QDialog):
    def __init__(self, parent, project, node=None):
        super().__init__(parent)
        self.setWindowTitle("Edit node" if node else "Add node")
        form = QFormLayout(self)

        self._us = UnitSystem.from_project(project)
        self.id_spin = _id_spin(
            node.id if node else _next_id([n.id for n in project.nodes]),
            editing=node is not None)
        form.addRow("Node id", self.id_spin)

        self.x = _coord_spin(self._us, node.x if node else 0.0)
        self.y = _coord_spin(self._us, node.y if node else 0.0)
        form.addRow(labeled("x", self.x), self.x)
        form.addRow(labeled("y", self.y), self.y)
        self.z = None
        if project.ndm == 3:
            self.z = _coord_spin(self._us, node.z if node else 0.0)
            form.addRow(labeled("z", self.z), self.z)

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
        z = self.z.si_value() if self.z is not None else 0.0
        return Node(id=self.id_spin.value(), x=self.x.si_value(),
                    y=self.y.si_value(), z=z,
                    supports=supports if any(supports) else ())

    @classmethod
    def edit(cls, parent, project, node=None):
        dlg = cls(parent, project, node)
        return dlg.data() if dlg.exec() else None


class MemberDialog(PickDialog):
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
        # Pick start/end node by clicking in the model (whichever field has focus).
        self.register_pick_field("node", self.n1)
        self.register_pick_field("node", self.n2)
        self.sec = _combo([(f"{s.id}: {s.name}", s.id) for s in project.sections])
        self.mat = _combo([(f"{m.id}: {m.name}", m.id) for m in project.materials])
        self.kind = _combo([("Beam / column", "beamcolumn2d"),
                            ("Cable / truss (axial only)", "cable")])
        self.hinge = _combo([("— none —", None)]
                            + [(f"{h.id}: {h.name}", h.id)
                               for h in getattr(project, "hinges", [])])

        if member:
            _select(self.n1, member.n1)
            _select(self.n2, member.n2)
            _select(self.sec, member.section)
            _select(self.mat, member.material)
            _select(self.kind, getattr(member, "kind", "beamcolumn2d"))
            _select(self.hinge, getattr(member, "hinge", None))
        elif self.n2.count() > 1:
            self.n2.setCurrentIndex(1)

        form.addRow("Start node", self.n1)
        form.addRow("End node", self.n2)
        form.addRow("", _pick_hint("Tip: click a node in the model to fill the "
                                   "focused end."))
        form.addRow("Section", self.sec)
        form.addRow("Material", self.mat)
        form.addRow("Type", self.kind)
        if getattr(project, "hinges", []):
            form.addRow("Hinge", self.hinge)
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
                      material=self.mat.currentData(),
                      kind=self.kind.currentData(),
                      hinge=self.hinge.currentData())

    @classmethod
    def edit(cls, parent, project, member=None):
        dlg = cls(parent, project, member)
        return dlg.data() if dlg.exec() else None


class AreaDialog(PickDialog):
    """Add / edit a surface (shell / plate) *area object* — the 2-D analogue of
    ``MemberDialog`` (slab plan S2). Pick 3 (triangle) or 4 (quad) corner nodes,
    a thickness (shell section) and a material. Corners can be filled by clicking
    nodes in the model (the focused corner field). ``seed_nodes`` pre-selects
    corners (e.g. from the current selection)."""

    def __init__(self, parent, project, area=None, seed_nodes=None):
        super().__init__(parent)
        self.setWindowTitle("Edit area" if area else "Add area")
        form = QFormLayout(self)

        self.id_spin = _id_spin(
            area.id if area else _next_id([a.id for a in project.areas]),
            editing=area is not None)
        form.addRow("Area id", self.id_spin)

        node_items = [(str(n.id), n.id) for n in project.nodes]
        none_item = ("— none (triangle) —", None)
        self.corners = []
        for i in range(4):
            # corner 4 is optional (None -> triangle); corners 1-3 required
            items = node_items if i < 3 else [none_item] + node_items
            combo = _combo(items)
            self.register_pick_field("node", combo)
            self.corners.append(combo)

        self.sec = _combo([(f"{s.id}: {s.name}", s.id)
                           for s in project.shell_sections])
        self.mat = _combo([(f"{m.id}: {m.name}", m.id) for m in project.materials])

        # mesh divisions (slab S3): a quad is subdivided n1 × n2 at solve time
        default_mesh = area.mesh if area else (2, 2)
        self.n1 = QSpinBox()
        self.n1.setRange(1, 200)
        self.n1.setValue(int(default_mesh[0]))
        self.n2 = QSpinBox()
        self.n2.setRange(1, 200)
        self.n2.setValue(int(default_mesh[1]))

        seed = list(area.nodes) if area else list(seed_nodes or [])
        for i, combo in enumerate(self.corners):
            if i < len(seed):
                _select(combo, seed[i])
            elif i < 3 and combo.count() > i:
                combo.setCurrentIndex(min(i, combo.count() - 1))
        if area:
            _select(self.sec, area.shell_section)
            _select(self.mat, area.material)

        for i, combo in enumerate(self.corners, start=1):
            form.addRow(f"Corner {i}", combo)
        form.addRow("", _pick_hint("Tip: click a node in the model to fill the "
                                   "focused corner. Order corners around the "
                                   "panel (no crossing)."))
        form.addRow("Thickness", self.sec)
        form.addRow("Material", self.mat)
        mesh_row = QWidget()
        mh = QHBoxLayout(mesh_row)
        mh.setContentsMargins(0, 0, 0, 0)
        mh.addWidget(self.n1)
        mh.addWidget(QLabel("×"))
        mh.addWidget(self.n2)
        mh.addStretch(1)
        form.addRow("Mesh (quad)", mesh_row)
        form.addRow(_buttons(self))

    def _node_list(self):
        return [c.currentData() for c in self.corners if c.currentData() is not None]

    def accept(self) -> None:
        nodes = self._node_list()
        if len(nodes) < 3:
            QMessageBox.warning(self, "Invalid area",
                                "An area needs 3 or 4 corner nodes.")
            return
        if len(set(nodes)) != len(nodes):
            QMessageBox.warning(self, "Invalid area",
                                "Corner nodes must all differ.")
            return
        if self.sec.currentData() is None:
            QMessageBox.warning(self, "No thickness",
                                "Define a thickness (shell section) first.")
            return
        super().accept()

    def data(self) -> Area:
        return Area(id=self.id_spin.value(), nodes=self._node_list(),
                    shell_section=self.sec.currentData(),
                    material=self.mat.currentData(),
                    mesh=(self.n1.value(), self.n2.value()))

    @classmethod
    def edit(cls, parent, project, area=None, seed_nodes=None):
        dlg = cls(parent, project, area, seed_nodes=seed_nodes)
        return dlg.data() if dlg.exec() else None


class DiaphragmDialog(QDialog):
    """Add / edit a rigid floor diaphragm (slab plan S8): a name, the tied node
    ids (seeded from the current selection, editable) and the diaphragm plane.
    A master node is auto-created at the joints' centroid at build time."""

    _PLANES = [("Horizontal floor (normal Z)", 2),
               ("Vertical, normal Y", 1),
               ("Vertical, normal X", 0)]

    def __init__(self, parent, project, diaphragm=None, seed_nodes=None):
        super().__init__(parent)
        self.setWindowTitle("Edit diaphragm" if diaphragm else "Add diaphragm")
        form = QFormLayout(self)

        self.id_spin = _id_spin(
            diaphragm.id if diaphragm
            else _next_id([d.id for d in project.diaphragms]),
            editing=diaphragm is not None)
        form.addRow("Diaphragm id", self.id_spin)

        self.name = QLineEdit(diaphragm.name if diaphragm else "")
        form.addRow("Name", self.name)

        nodes = (diaphragm.nodes if diaphragm else list(seed_nodes or []))
        self.nodes = QLineEdit(", ".join(str(n) for n in nodes))
        form.addRow("Nodes", self.nodes)
        form.addRow("", _pick_hint("Comma-separated joint ids to tie. Tip: "
                                   "select the joints first — they seed here."))

        self.plane = _combo(self._PLANES)
        if diaphragm:
            _select(self.plane, diaphragm.perp_dir)
        form.addRow("Plane", self.plane)
        form.addRow(_buttons(self))

    def _node_list(self):
        out, seen = [], set()
        for tok in self.nodes.text().replace(",", " ").split():
            try:
                v = int(tok)
            except ValueError:
                continue
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def accept(self) -> None:
        if len(self._node_list()) < 2:
            QMessageBox.warning(self, "Invalid diaphragm",
                                "A diaphragm needs at least two joint nodes.")
            return
        super().accept()

    def data(self) -> Diaphragm:
        name = self.name.text().strip() or f"Diaphragm {self.id_spin.value()}"
        return Diaphragm(id=self.id_spin.value(), name=name,
                         nodes=self._node_list(),
                         perp_dir=self.plane.currentData())

    @classmethod
    def edit(cls, parent, project, diaphragm=None, seed_nodes=None):
        dlg = cls(parent, project, diaphragm, seed_nodes=seed_nodes)
        return dlg.data() if dlg.exec() else None


class LoadDialog(PickDialog):
    """Add / edit a nodal load, laid out as grouped cards (plan L4): an
    *Applied to* card (node + load case) over a *Components* card whose rows
    each pair a force/moment spin with a sign-convention hint. Built from the
    L1 scaffold; ``.edit()`` return contract unchanged. The node can be chosen
    by clicking it in the model while the dialog is open (see :mod:`pick`)."""

    def __init__(self, parent, project, load=None, nodes=None):
        super().__init__(parent)
        # ``nodes`` (a list of already-selected node ids, e.g. from a window
        # select) turns this into a *bulk* assign: the same components go to
        # every selected node (the SAP/MIDAS select-then-assign idiom). One
        # node, or none, keeps the single-node picker.
        self._targets = [int(n) for n in (nodes or [])
                         if any(nd.id == int(n) for nd in project.nodes)]
        multi = len(self._targets) > 1
        self.setWindowTitle(
            f"Add load — {len(self._targets)} nodes" if multi
            else ("Edit load" if load else "Add load"))
        self._us = UnitSystem.from_project(project)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(style.SP_LG, style.SP_LG,
                                 style.SP_LG, style.SP_LG)
        outer.setSpacing(style.SP_MD)

        self.node = _combo([(str(n.id), n.id) for n in project.nodes])
        if load:
            _select(self.node, load.node)
        elif len(self._targets) == 1:
            _select(self.node, self._targets[0])
        self.case = _combo([(c.name, c.id) for c in project.load_cases])
        if load is not None:
            _select(self.case, getattr(load, "case", project.default_case_id()))

        applied = ui.GroupCard("Applied to")
        if multi:
            shown = ", ".join(str(n) for n in self._targets[:12])
            if len(self._targets) > 12:
                shown += " …"
            summary = QLabel(f"{len(self._targets)} selected nodes: {shown}")
            summary.setWordWrap(True)
            applied.add_row("Nodes", summary)
        else:
            self.register_pick_field("node", self.node)  # click a node to set it
            applied.add_row("Node", self.node)
        applied.add_row("Load case", self.case)
        if not multi:
            applied.add_full_row(_pick_hint(
                "Tip: click a node in the model to set it here — no need to "
                "close this window. Or window-select several nodes first to "
                "load them all at once."))

        # Components mix forces (translational DOFs) and moments (rotational),
        # so units live per-row, not in one card header.
        comps = ui.GroupCard("Components")
        self.vals = []
        vec = tuple(load.values) if load else ()
        for k, lbl in enumerate(dof_labels(project.ndm, project.ndf)):
            qty = self._us.dof_quantity(lbl)
            spin = _force_spin(self._us, vec[k] if k < len(vec) else 0.0,
                               quantity=qty)
            self.vals.append(spin)
            host = QWidget()
            h = QHBoxLayout(host)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(style.SP_SM)
            h.addWidget(spin, 1)
            h.addWidget(ui.direction_glyph(lbl))
            comps.add_row(f"{lbl}  [{spin.unit_label()}]", host)

        outer.addWidget(applied)
        outer.addWidget(comps)
        outer.addWidget(_buttons(self))
        style.apply(self)

    def data(self) -> Load:
        return Load(node=self.node.currentData(),
                    values=tuple(s.si_value() for s in self.vals),
                    case=self.case.currentData())

    def loads(self) -> list:
        """One :class:`Load` per target node (the whole selection in bulk mode,
        else the single chosen node)."""
        values = tuple(s.si_value() for s in self.vals)
        case = self.case.currentData()
        targets = self._targets if len(self._targets) > 1 else [
            self.node.currentData()]
        return [Load(node=n, values=values, case=case)
                for n in targets if n is not None]

    @classmethod
    def edit(cls, parent, project, load=None):
        dlg = cls(parent, project, load)
        return dlg.data() if dlg.exec() else None

    @classmethod
    def create(cls, parent, project, nodes=None):
        """Add nodal load(s): returns a list of :class:`Load` (one per selected
        node) or ``None`` if cancelled. Pass ``nodes`` (selected node ids) to
        bulk-assign the same load to all of them."""
        dlg = cls(parent, project, nodes=nodes)
        return dlg.loads() if dlg.exec() else None


class LoadCaseDialog(QDialog):
    """Manage the project's load cases (name + nature). Returns the edited list
    of ``LoadCase`` via ``result_cases``; ids are preserved for existing cases
    and assigned fresh for new rows so load ownership stays intact."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Load cases")
        self.resize(440, 360)
        self._project = project
        self.result_cases = None
        # working rows: [id, name, nature, self_weight_factor]; id 0 => new
        self._rows = [[c.id, c.name, c.nature,
                       float(getattr(c, "self_weight_factor", 0.0))]
                      for c in project.load_cases]

        outer = QVBoxLayout(self)
        outer.setContentsMargins(style.SP_LG, style.SP_LG,
                                 style.SP_LG, style.SP_LG)
        outer.setSpacing(style.SP_MD)
        card = ui.GroupCard("Load cases", form=False)
        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["Name", "Nature", "Self-weight ×"])
        self.tbl.horizontalHeader().setStretchLastSection(False)
        self.tbl.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        card.body_layout().addWidget(self.tbl)
        row = QHBoxLayout()
        add = QPushButton("＋ Case")
        add.clicked.connect(self._add)
        sw = QPushButton("＋ Self weight")
        sw.setToolTip("Add a 'Self weight' dead case with the multiplier at 1.0")
        sw.clicked.connect(self.add_self_weight_case)
        rem = QPushButton("Remove")
        rem.clicked.connect(self._remove)
        row.addWidget(add)
        row.addWidget(sw)
        row.addWidget(rem)
        row.addStretch(1)
        card.body_layout().addLayout(row)
        outer.addWidget(card)

        hint = QLabel("Each case's nature (Dead / Live / Wind …) drives the "
                      "ASCE 7-22 generator in Analysis ▸ Load combinations. "
                      "Set Self-weight × to 1.0 to include the model's own "
                      "gravity weight (needs a material density).")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        outer.addWidget(hint)
        outer.addWidget(_buttons(self))
        style.apply(self)
        self._reload()

    @staticmethod
    def _nature_icon(nature: str):
        key = NATURE_ASCE.get(nature) or "·"        # ASCE load key badge
        return icons.letter_icon(key, style.ACCENT)

    def _on_nature(self, r: int, combo: QComboBox) -> None:
        it = self.tbl.item(r, 0)
        if it is not None:
            it.setIcon(self._nature_icon(combo.currentData()))

    def _reload(self) -> None:
        self.tbl.setRowCount(len(self._rows))
        for r, (_id, name, nature, sw) in enumerate(self._rows):
            item = QTableWidgetItem(name)
            item.setIcon(self._nature_icon(nature))
            self.tbl.setItem(r, 0, item)
            combo = QComboBox()
            for key, lbl in NATURE_LABELS.items():
                combo.addItem(lbl, key)
            i = combo.findData(nature)
            combo.setCurrentIndex(i if i >= 0 else 0)
            combo.currentIndexChanged.connect(
                lambda _i, rr=r, cc=combo: self._on_nature(rr, cc))
            self.tbl.setCellWidget(r, 1, combo)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setSingleStep(1.0)
            spin.setDecimals(3)
            spin.setValue(float(sw))
            spin.setToolTip("Self-weight multiplier — 1.0 applies the model's "
                            "full gravity weight to this case (0 = off).")
            self.tbl.setCellWidget(r, 2, spin)

    def _add(self) -> None:
        self._sync_names()
        self._rows.append([0, f"Case {len(self._rows) + 1}", "live", 0.0])
        self._reload()

    def add_self_weight_case(self) -> None:
        """Append a ready-made 'Self weight' dead case with the multiplier at
        1.0 — the one-click way to add self-weight to the model."""
        self._sync_names()
        existing = {row[1].strip().lower() for row in self._rows}
        name = "Self weight"
        if name.lower() in existing:
            n = 2
            while f"{name} {n}".lower() in existing:
                n += 1
            name = f"{name} {n}"
        self._rows.append([0, name, "dead", 1.0])
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
            sp = self.tbl.cellWidget(r, 2)
            if sp and r < len(self._rows):
                self._rows[r][3] = float(sp.value())

    def accept(self) -> None:
        self._sync_names()
        used = {row[0] for row in self._rows if row[0]}
        nxt = (max(used) if used else 0) + 1
        out = []
        for _id, name, nature, sw in self._rows:
            if not _id:
                _id, nxt = nxt, nxt + 1
            out.append(LoadCase(id=_id, name=name, nature=nature,
                                self_weight_factor=float(sw)))
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


def _prop_spin(units: UnitSystem, quantity: Quantity, si: float, *,
               decimals: int, step: float) -> UnitSpin:
    """A section-property (area / second moment) spin (plan U6): displays in the
    project's derived unit (m², mm⁴, in⁴, …), stores SI. Non-negative; the range
    is generous so a small SI value is not clamped when shown in a large unit."""
    return UnitSpin(quantity, units, si=si, decimals=decimals, step=step,
                    rng=(0.0, 1.0e15))


class SectionDialog(QDialog):
    def __init__(self, parent, project, section=None):
        super().__init__(parent)
        self.setWindowTitle("Edit section" if section else "Add section")
        form = QFormLayout(self)

        self._us = UnitSystem.from_project(project)
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

        self.A = _prop_spin(self._us, Quantity.AREA,
                            section.A if section else 6.0e-3,
                            decimals=6, step=1.0e-4)
        self.Iz = _prop_spin(self._us, Quantity.INERTIA,
                             section.Iz if section else 2.0e-4,
                             decimals=8, step=1.0e-5)
        form.addRow(labeled("A", self.A), self.A)
        form.addRow(labeled("Iz", self.Iz), self.Iz)
        self.Iy = self.J = None
        if project.ndm == 3:
            self.Iy = _prop_spin(self._us, Quantity.INERTIA,
                                 section.Iy if section else 1.0e-4,
                                 decimals=8, step=1.0e-5)
            self.J = _prop_spin(self._us, Quantity.INERTIA,
                                section.J if section else 1.0e-5,
                                decimals=9, step=1.0e-6)
            form.addRow(labeled("Iy", self.Iy), self.Iy)
            form.addRow(labeled("J", self.J), self.J)

        self.shape.currentIndexChanged.connect(self._on_shape)
        self._on_shape()                     # set initial fill / enabled state
        form.addRow(_buttons(self))

    def _on_shape(self) -> None:
        shape = self.shape.currentData()
        if shape:                            # catalog W-shape drives A/Iz/Iy/J
            props = _shape_props(shape)      # catalog values are SI (m², m⁴)
            if props:
                self.A.set_si(props[0])
                self.Iz.set_si(props[1])
                if self.Iy is not None:
                    self.Iy.set_si(props[2])
                if self.J is not None:
                    self.J.set_si(props[3])
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
                       A=self.A.si_value(), Iz=self.Iz.si_value(), shape=shape,
                       Iy=self.Iy.si_value() if self.Iy is not None else 0.0,
                       J=self.J.si_value() if self.J is not None else 0.0)

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


def _pick_hint(text: str) -> QLabel:
    """A muted one-line hint that the field can be filled by clicking the model
    (the dialog is modeless — see :mod:`pick`)."""
    lbl = QLabel(text)
    lbl.setObjectName("hintLabel")
    lbl.setWordWrap(True)
    return lbl


def _int_spin(value, lo, hi):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


def _grid_len_spin(units: UnitSystem, si: float) -> UnitSpin:
    """A frame-geometry length spin (plan U6b): shows the project's length unit,
    stores SI metres. Generous range so a metre value isn't clamped in mm."""
    return UnitSpin(Quantity.LENGTH, units, si=si, decimals=2, step=0.5,
                    rng=(0.0, 1.0e9))


class FrameDialog(QDialog):
    """Parameters for a regular building frame (2-D or 3-D space frame).

    ``units`` (a :class:`~units.UnitSystem`) makes the bay/storey inputs read
    and store in the project's length unit; omitted, it defaults to metres so
    the generated model's dimensions stay SI (plan U6b)."""

    def __init__(self, parent, units: UnitSystem | None = None):
        super().__init__(parent)
        self.setWindowTitle("Generate frame")
        self._us = units or UnitSystem()
        form = QFormLayout(self)

        from PySide6.QtWidgets import QCheckBox
        self.threed = QCheckBox("3-D space frame")
        self.threed.toggled.connect(self._toggle)
        form.addRow(self.threed)

        self.bays_x = _int_spin(3, 1, 500)
        self.bay_x = _grid_len_spin(self._us, 6.0)
        form.addRow("Bays (X)", self.bays_x)
        form.addRow(labeled("Bay width X", self.bay_x), self.bay_x)
        self.bays_y = _int_spin(2, 1, 500)
        self.bay_y = _grid_len_spin(self._us, 6.0)
        form.addRow("Bays (Y)", self.bays_y)
        form.addRow(labeled("Bay width Y", self.bay_y), self.bay_y)
        self.storeys = _int_spin(3, 1, 500)
        self.storey_h = _grid_len_spin(self._us, 3.5)
        form.addRow("Storeys", self.storeys)
        form.addRow(labeled("Storey height", self.storey_h), self.storey_h)

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
        return dict(bays_x=self.bays_x.value(),
                    bay_x=self.bay_x.si_value(),
                    storeys=self.storeys.value(),
                    storey_h=self.storey_h.si_value(),
                    bays_y=self.bays_y.value() if td else 0,
                    bay_y=self.bay_y.si_value(), shape=self.shape.currentData())

    @classmethod
    def get(cls, parent, units: UnitSystem | None = None):
        dlg = cls(parent, units)
        return dlg.params() if dlg.exec() else None


class LoadGenDialog(QDialog):
    """Parameters for a parametric load pattern (gravity or lateral)."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Generate loads")
        self._project = project
        self._us = UnitSystem.from_project(project)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(style.SP_LG, style.SP_LG,
                                 style.SP_LG, style.SP_LG)
        outer.setSpacing(style.SP_MD)

        card = ui.GroupCard("Load pattern")
        self.kind = QComboBox()
        self.kind.addItem("Gravity (downward, per node)", "gravity")
        self.kind.addItem("Lateral X (storey forces)", "lateral_x")
        if project.ndm == 3:
            self.kind.addItem("Lateral Y (storey forces)", "lateral_y")
        self.kind.currentIndexChanged.connect(self._relabel)
        card.add_row("Pattern", self.kind)
        self.mag = _force_spin(self._us, 50000.0)
        self.mag.valueChanged.connect(self._update_preview)
        self._label = QLabel()               # dynamic label (per-node / shear)
        card.body_layout().addRow(self._label, self.mag)
        self._preview = QLabel()
        self._preview.setObjectName("hintLabel")
        self._preview.setWordWrap(True)
        card.add_full_row(self._preview)

        outer.addWidget(card)
        outer.addWidget(_buttons(self))
        style.apply(self)
        self._relabel()

    def _relabel(self, *_) -> None:
        gravity = self.kind.currentData() == "gravity"
        unit = self._project.force_unit
        self._label.setText(f"Load per node [{unit}]" if gravity
                            else f"Base shear [{unit}]")
        self._update_preview()

    def _update_preview(self, *_) -> None:
        import generators
        kind, mag = self.kind.currentData(), self.mag.si_value()
        fu = self._project.force_unit
        try:
            if kind == "gravity":
                loads = generators.gravity_loads(self._project, mag)
                total = sum(abs(v) for ld in loads for v in ld.values)
                msg = (f"→ {len(loads)} nodal load(s), total ≈ "
                       f"{total:,.0f} {fu} downward")
            else:
                direction = "Y" if kind == "lateral_y" else "X"
                loads = generators.lateral_loads(self._project, mag, direction)
                total = sum(abs(v) for ld in loads for v in ld.values)
                msg = (f"→ {len(loads)} storey force(s), base shear ≈ "
                       f"{total:,.0f} {fu} ({direction}-direction)")
            if not loads:
                msg = "→ no loads (need nodes above the base level)"
        except Exception:                                # noqa: BLE001
            msg = ""
        self._preview.setText(msg)

    def params(self):
        return self.kind.currentData(), self.mag.si_value()

    @classmethod
    def get(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.params() if dlg.exec() else None


class MoveDialog(QDialog):
    """Translation vector for moving the selection."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Move selection")
        self._us = UnitSystem.from_project(project)
        form = QFormLayout(self)
        self.dx, self.dy = _coord_spin(self._us), _coord_spin(self._us)
        form.addRow(labeled("dx", self.dx), self.dx)
        form.addRow(labeled("dy", self.dy), self.dy)
        self.dz = None
        if project.ndm == 3:
            self.dz = _coord_spin(self._us)
            form.addRow(labeled("dz", self.dz), self.dz)
        form.addRow(_buttons(self))

    def data(self):
        return (self.dx.si_value(), self.dy.si_value(),
                self.dz.si_value() if self.dz is not None else 0.0)

    @classmethod
    def get(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.data() if dlg.exec() else None


class CopyDialog(QDialog):
    """Offset vector + copy count for arraying (or extruding) the selection."""

    def __init__(self, parent, project, title="Copy / array selection"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._us = UnitSystem.from_project(project)
        form = QFormLayout(self)
        self.dx, self.dy = _coord_spin(self._us), _coord_spin(self._us)
        form.addRow(labeled("dx", self.dx), self.dx)
        form.addRow(labeled("dy", self.dy), self.dy)
        self.dz = None
        if project.ndm == 3:
            self.dz = _coord_spin(self._us)
            form.addRow(labeled("dz", self.dz), self.dz)
        self.count = _int_spin(1, 1, 500)
        form.addRow("Copies", self.count)
        form.addRow(_buttons(self))

    def data(self):
        return (self.dx.si_value(), self.dy.si_value(),
                self.dz.si_value() if self.dz is not None else 0.0,
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
        self._us = UnitSystem.from_project(project)
        form = QFormLayout(self)
        self.axis = QComboBox()
        for a in (["X", "Y"] if project.ndm == 2 else ["X", "Y", "Z"]):
            self.axis.addItem(f"{a} = constant plane", a)
        form.addRow("Mirror plane", self.axis)
        self.coord = _coord_spin(self._us)
        form.addRow(labeled("Plane coordinate", self.coord), self.coord)
        form.addRow(_buttons(self))

    def data(self):
        return (self.axis.currentData(), self.coord.si_value())

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
        self._us = UnitSystem.from_project(project)
        form = QFormLayout(self)
        self.axis = QComboBox()
        if self._three_d:
            for a in ("Z", "X", "Y"):
                self.axis.addItem(f"about {a} axis", a)
            form.addRow("Axis", self.axis)
        self.cx = _coord_spin(self._us)
        self.cy = _coord_spin(self._us)
        form.addRow(labeled("Center x", self.cx), self.cx)
        form.addRow(labeled("Center y", self.cy), self.cy)
        self.cz = None
        if self._three_d:
            self.cz = _coord_spin(self._us)
            form.addRow(labeled("Center z", self.cz), self.cz)
        self.angle = QDoubleSpinBox()
        self.angle.setRange(-360.0, 360.0)
        self.angle.setDecimals(1)
        self.angle.setSuffix(" °")
        form.addRow("Angle", self.angle)
        form.addRow(_buttons(self))

    def data(self):
        axis = self.axis.currentData() if self._three_d else "Z"
        center = (self.cx.si_value(), self.cy.si_value(),
                  self.cz.si_value() if self.cz is not None else 0.0)
        return (axis, center, self.angle.value())

    @classmethod
    def get(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.data() if dlg.exec() else None
