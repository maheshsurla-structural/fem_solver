"""On-demand spreadsheets for the model tree (nav plan N2).

The tree is a compact *table of contents*; when the user wants the row-by-row
detail of a category they right-click it and choose **Show Table…**, which opens
:class:`ModelTableDialog` — a read-only, columned list of every item in that
category. Double-clicking a row drills back to the item (selecting it in the
tree/viewport and, for editable kinds, opening its editor). This keeps the tree
uncluttered while still exposing *all the information* on demand.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                               QHeaderView, QLabel, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

import style


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


# Each builder returns (headers, rows) where a row is (cells, ref-or-None).
def _dof_labels(p):
    from editing import dof_labels
    return dof_labels(p.ndm, p.ndf)


def _materials(p):
    headers = ["ID", "Name", "Type", "E", "ν", "ρ", "fy", "fu"]
    rows = []
    for m in p.materials:
        rows.append(([str(m.id), m.name, getattr(m, "kind", ""),
                      _g(m.E), _g(m.nu), _g(getattr(m, "rho", 0.0)),
                      _g(getattr(m, "fy", "")), _g(getattr(m, "fu", ""))],
                     None))
    return headers, rows


def _sections(p):
    headers = ["ID", "Name", "Shape", "A", "Iz", "Iy", "J", "Designer"]
    rows = []
    for s in p.sections:
        rows.append(([str(s.id), s.name, s.shape or "—",
                      _g(s.A), _g(s.Iz), _g(getattr(s, "Iy", 0.0)),
                      _g(getattr(s, "J", 0.0)),
                      "yes" if getattr(s, "gsd_spec", None) else "—"],
                     ("section", s.id)))
    return headers, rows


def _hinges(p):
    headers = ["ID", "Name", "Lp (I)", "Lp (J)", "Relative"]
    rows = []
    for h in p.hinges:
        lpj = "same as I" if h.lp_j is None else _g(h.lp_j)
        rows.append(([str(h.id), h.name, _g(h.lp), lpj,
                      "yes" if h.relative else "no"], None))
    return headers, rows


def _nodes(p):
    labels = _dof_labels(p)
    if p.ndm == 2:
        headers = ["ID", "X", "Y", "Supports"]
    else:
        headers = ["ID", "X", "Y", "Z", "Supports"]
    rows = []
    for n in p.nodes:
        cells = [str(n.id), _g(n.x), _g(n.y)]
        if p.ndm == 3:
            cells.append(_g(n.z))
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
                      str(m.section), str(m.material),
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
    rows = [([str(c.id), c.name, NATURE_LABELS.get(c.nature, c.nature)], None)
            for c in p.load_cases]
    return headers, rows


def _loads(p):
    labels = _dof_labels(p)
    headers = ["Node", "Case"] + labels
    rows = []
    for i, ld in enumerate(p.loads):
        case = p.case(getattr(ld, "case", 1))
        cells = [str(ld.node), case.name if case else str(getattr(ld, "case", 1))]
        cells += [_g(v) for v in ld.values]
        rows.append((cells, ("load", i)))
    return headers, rows


def _member_loads(p):
    headers = ["Member", "Case", "wy", "wz"] if p.ndm == 3 \
        else ["Member", "Case", "wy"]
    rows = []
    for i, ml in enumerate(p.member_loads):
        case = p.case(getattr(ml, "case", 1))
        cells = [str(ml.member),
                 case.name if case else str(getattr(ml, "case", 1)), _g(ml.wy)]
        if p.ndm == 3:
            cells.append(_g(ml.wz))
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
    "load_cases": (_load_cases, "Load cases"),
    "loads": (_loads, "Nodal loads"),
    "member_loads": (_member_loads, "Line loads"),
    "combinations": (_combinations, "Load combinations"),
    "analysis_cases": (_analysis_cases, "Analysis cases"),
    "results": (_results, "Results"),
}


class ModelTableDialog(QDialog):
    """A read-only spreadsheet of one model category; double-click a row to
    drill back to the item (via the ``on_activate(ref)`` callback)."""

    def __init__(self, parent, title, headers, rows) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{title} — table")
        self.resize(640, 420)
        self._activate_ref = None

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
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        for r, (cells, ref) in enumerate(rows):
            for c, text in enumerate(cells):
                it = QTableWidgetItem(str(text))
                if c == 0 and ref is not None:
                    it.setData(Qt.ItemDataRole.UserRole, ref)
                self.table.setItem(r, c, it)
        self.table.itemDoubleClicked.connect(self._on_row)
        root.addWidget(self.table)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
        style.apply(self)

    def _on_row(self, item) -> None:
        row = item.row()
        first = self.table.item(row, 0)
        ref = first.data(Qt.ItemDataRole.UserRole) if first else None
        if ref:
            self._activate_ref = tuple(ref)
            self.accept()

    @classmethod
    def show_category(cls, parent, project, category, on_activate=None,
                      type_filter=None) -> None:
        entry = _BUILDERS.get(category)
        if entry is None:
            return
        builder, title = entry
        if category == "elements" and type_filter:
            headers, rows = builder(project, type_filter)
            title = f"{type_filter} elements"
        else:
            headers, rows = builder(project)
        dlg = cls(parent, title, headers, rows)
        dlg.exec()
        if dlg._activate_ref and on_activate is not None:
            on_activate(dlg._activate_ref)
