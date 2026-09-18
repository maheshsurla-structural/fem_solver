"""On-demand spreadsheets for the model tree (nav plan N2 + N7).

The tree is a compact *table of contents*; when the user wants the row-by-row
detail of a category they right-click it and choose **Show Table…**, which opens
:class:`ModelTableDialog` — a columned list of every item in that category.
Safe scalar fields are **editable in place** (nav N7): committing a cell routes
through the main window's ``on_commit`` callback, which applies the change with
undo. Identity / derived columns stay read-only; double-clicking one drills back
to the item's full editor (``on_activate``). This keeps the tree uncluttered
while exposing *all the information* — and quick edits — on demand.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                               QHeaderView, QLabel, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

import style

# Extra item-data roles for the editable cells (nav N7).
_EDIT_ROLE = Qt.ItemDataRole.UserRole + 1        # (ref, field, kind)
_ORIG_ROLE = Qt.ItemDataRole.UserRole + 2        # original text, for revert


class Editable:
    """Marks a builder cell as user-editable: ``field`` names the target
    attribute, ``kind`` ("str" / "float" / "int") how the entry is parsed."""
    __slots__ = ("text", "field", "kind")

    def __init__(self, value, field, kind="float") -> None:
        self.text = _g(value) if kind == "float" else str(value)
        self.field = field
        self.kind = kind


def _g(x) -> str:
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x)


def _supports_str(node, labels) -> str:
    if not node.supports or not any(node.supports):
        return "free"
    mask = [labels[k] for k in range(min(len(node.supports), len(labels)))
            if node.supports[k]]
    return ",".join(mask)


# Each builder returns (headers, rows) where a row is (cells, ref-or-None) and a
# cell is a str (read-only) or an Editable (nav N7).
def _dof_labels(p):
    from editing import dof_labels
    return dof_labels(p.ndm, p.ndf)


def _materials(p):
    headers = ["ID", "Name", "Type", "E", "ν", "ρ", "fy", "fu"]
    rows = []
    for m in p.materials:
        rows.append(([str(m.id), Editable(m.name, "name", "str"),
                      getattr(m, "kind", ""),
                      Editable(m.E, "E"), Editable(m.nu, "nu"),
                      Editable(getattr(m, "rho", 0.0), "rho"),
                      Editable(getattr(m, "fy", 0.0), "fy"),
                      Editable(getattr(m, "fu", 0.0), "fu")],
                     ("material", m.id)))
    return headers, rows


def _sections(p):
    headers = ["ID", "Name", "Shape", "A", "Iz", "Iy", "J", "Designer"]
    rows = []
    for s in p.sections:
        gsd = bool(getattr(s, "gsd_spec", None))     # GSD drives A/Iz/Iy/J
        num = (lambda v, f: str(_g(v)) if gsd else Editable(v, f))
        rows.append(([str(s.id), Editable(s.name, "name", "str"),
                      s.shape or "—",
                      num(s.A, "A"), num(s.Iz, "Iz"),
                      num(getattr(s, "Iy", 0.0), "Iy"),
                      num(getattr(s, "J", 0.0), "J"),
                      "yes" if gsd else "—"],
                     ("section", s.id)))
    return headers, rows


def _hinges(p):
    headers = ["ID", "Name", "Lp (I)", "Lp (J)", "Relative"]
    rows = []
    for h in p.hinges:
        lpj = "same as I" if h.lp_j is None else _g(h.lp_j)
        rows.append(([str(h.id), Editable(h.name, "name", "str"),
                      Editable(h.lp, "lp"), lpj,
                      "yes" if h.relative else "no"], ("hinge", h.id)))
    return headers, rows


def _nodes(p):
    if p.ndm == 2:
        headers = ["ID", "X", "Y", "Supports"]
    else:
        headers = ["ID", "X", "Y", "Z", "Supports"]
    labels = _dof_labels(p)
    rows = []
    for n in p.nodes:
        cells = [str(n.id), Editable(n.x, "x"), Editable(n.y, "y")]
        if p.ndm == 3:
            cells.append(Editable(n.z, "z"))
        cells.append(_supports_str(n, labels))
        rows.append((cells, ("node", n.id)))
    return headers, rows


def _elements(p, type_filter=None):
    from main_window import _element_type
    headers = ["ID", "Type", "Node I", "Node J", "Section", "Material", "Hinge"]
    rows = []
    for m in p.members:
        etype = _element_type(m)
        if type_filter and etype != type_filter:
            continue
        rows.append(([str(m.id), etype, str(m.n1), str(m.n2),
                      Editable(m.section, "section", "int"),
                      Editable(m.material, "material", "int"),
                      "—" if m.hinge is None else str(m.hinge)],
                     ("member", m.id)))
    return headers, rows


def _supports(p):
    labels = _dof_labels(p)
    if p.ndm == 2:
        headers = ["Node", "X", "Y", "Fixity"]
    else:
        headers = ["Node", "X", "Y", "Z", "Fixity"]
    rows = []
    for n in p.nodes:
        if not (n.supports and any(n.supports)):
            continue
        cells = [str(n.id), _g(n.x), _g(n.y)]
        if p.ndm == 3:
            cells.append(_g(n.z))
        cells.append(_supports_str(n, labels))
        rows.append((cells, ("node", n.id)))
    return headers, rows


def _load_cases(p):
    from project import NATURE_LABELS
    headers = ["ID", "Name", "Nature"]
    rows = [([str(c.id), Editable(c.name, "name", "str"),
              NATURE_LABELS.get(c.nature, c.nature)], ("load_case", c.id))
            for c in p.load_cases]
    return headers, rows


def _loads(p):
    labels = _dof_labels(p)
    headers = ["Node", "Pattern"] + labels
    rows = []
    for i, ld in enumerate(p.loads):
        case = p.case(getattr(ld, "case", 1))
        cells = [str(ld.node), case.name if case else str(getattr(ld, "case", 1))]
        cells += [Editable(v, f"v{k}") for k, v in enumerate(ld.values)]
        rows.append((cells, ("load", i)))
    return headers, rows


def _member_loads(p):
    headers = ["Member", "Pattern", "wy", "wz"] if p.ndm == 3 \
        else ["Member", "Pattern", "wy"]
    rows = []
    for i, ml in enumerate(p.member_loads):
        case = p.case(getattr(ml, "case", 1))
        cells = [str(ml.member),
                 case.name if case else str(getattr(ml, "case", 1)),
                 Editable(ml.wy, "wy")]
        if p.ndm == 3:
            cells.append(Editable(ml.wz, "wz"))
        rows.append((cells, ("member_load", i)))
    return headers, rows


def _combinations(p):
    by_id = {c.id: c.name for c in p.load_cases}
    headers = ["ID", "Name", "Combination"]
    rows = []
    for c in p.combinations:
        terms = " + ".join(f"{f:g}·{by_id.get(cid, cid)}"
                           for cid, f in c.factors.items())
        rows.append(([str(c.id), c.name, terms or "—"], None))
    return headers, rows


def _analysis_cases(p):
    headers = ["Name", "Type", "Detail"]
    rows = [(["Linear Static", "Static", "current loads / combinations"], None)]
    for c in p.nonlinear_cases:
        proto = c.protocol + (" · staged" if c.continue_from else "")
        rows.append(([c.name, "Nonlinear Static",
                      f"{proto} · node {c.control_node}"], None))
    for name, detail in [("Modal", "eigen · free vibration"),
                         ("Response Spectrum", "modal superposition · SRSS/CQC"),
                         ("Buckling", "eigenvalue · (K + λ·K_g)"),
                         ("Time History", "ground-motion record")]:
        rows.append(([name, name, detail], None))
    return headers, rows


def _results(p):
    headers = ["Name", "Steps", "Summary"]
    rows = []
    for i, r in enumerate(p.runs):
        name = getattr(r, "name", f"Run {i + 1}")
        curve = getattr(r, "curve", None)
        steps = str(len(curve)) if curve is not None else "—"
        rows.append(([name, steps, getattr(r, "summary", "") or "—"], None))
    return headers, rows


_BUILDERS = {
    "materials": (_materials, "Materials"),
    "sections": (_sections, "Sections"),
    "hinges": (_hinges, "Hinge properties"),
    "nodes": (_nodes, "Nodes"),
    "elements": (_elements, "Elements"),
    "supports": (_supports, "Supports"),
    "load_cases": (_load_cases, "Load patterns"),
    "loads": (_loads, "Nodal loads"),
    "member_loads": (_member_loads, "Line loads"),
    "combinations": (_combinations, "Load combinations"),
    "analysis_cases": (_analysis_cases, "Analysis cases"),
    "results": (_results, "Results"),
}


class ModelTableDialog(QDialog):
    """A spreadsheet of one model category. Safe scalar cells are editable in
    place (``on_commit``); read-only rows drill to their editor (``on_activate``
    via a double-click on a non-editable cell)."""

    def __init__(self, parent, title, headers, rows,
                 on_commit=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{title} — table")
        self.resize(660, 440)
        self._activate_ref = None
        self._on_commit = on_commit
        self._loading = True

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_MD, style.SP_MD,
                                style.SP_MD, style.SP_MD)
        root.setSpacing(style.SP_SM)
        head = QLabel(f"{title}  ·  {len(rows)} "
                      f"{'item' if len(rows) == 1 else 'items'}")
        head.setObjectName("cardTitle")
        root.addWidget(head)

        self.table = QTableWidget(len(rows), len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        ro_flags = (Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        for r, (cells, ref) in enumerate(rows):
            for c, cell in enumerate(cells):
                if isinstance(cell, Editable):
                    it = QTableWidgetItem(cell.text)
                    it.setFlags(ro_flags | Qt.ItemFlag.ItemIsEditable)
                    it.setData(_EDIT_ROLE, (ref, cell.field, cell.kind))
                    it.setData(_ORIG_ROLE, cell.text)
                else:
                    it = QTableWidgetItem(str(cell))
                    it.setFlags(ro_flags)
                if c == 0 and ref is not None:
                    it.setData(Qt.ItemDataRole.UserRole, ref)
                self.table.setItem(r, c, it)
        self._loading = False
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        root.addWidget(self.table)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

    def _on_double_click(self, item) -> None:
        # Editable cell → let it edit; read-only cell → drill to the editor.
        if item.flags() & Qt.ItemFlag.ItemIsEditable:
            return
        first = self.table.item(item.row(), 0)
        ref = first.data(Qt.ItemDataRole.UserRole) if first else None
        if ref:
            self._activate_ref = tuple(ref)
            self.accept()

    def _revert(self, item) -> None:
        self._loading = True
        item.setText(item.data(_ORIG_ROLE))
        self._loading = False

    def _on_item_changed(self, item) -> None:
        if self._loading:
            return
        spec = item.data(_EDIT_ROLE)
        if not spec:
            return
        ref, field, kind = spec
        raw = item.text().strip()
        try:
            value = float(raw) if kind == "float" else (
                int(float(raw)) if kind == "int" else raw)
        except ValueError:
            self._revert(item)
            return
        ok = bool(self._on_commit and self._on_commit(tuple(ref), field, value))
        if not ok:
            self._revert(item)
            return
        self._loading = True                    # normalise the display + baseline
        item.setText(_g(value) if kind == "float" else str(value))
        item.setData(_ORIG_ROLE, item.text())
        self._loading = False

    @classmethod
    def show_category(cls, parent, project, category, on_activate=None,
                      on_commit=None, type_filter=None) -> None:
        entry = _BUILDERS.get(category)
        if entry is None:
            return
        builder, title = entry
        if category == "elements" and type_filter:
            headers, rows = builder(project, type_filter)
            title = f"{type_filter} elements"
        else:
            headers, rows = builder(project)
        dlg = cls(parent, title, headers, rows, on_commit=on_commit)
        dlg.exec()
        if dlg._activate_ref and on_activate is not None:
            on_activate(dlg._activate_ref)
