"""Unified analysis-cases home (plan A1) — the desktop counterpart of
CSiBridge's *Define ▸ Load Cases* list.

One table lists **every** analysis case with a Type column and icon:

* **Linear Static** — the always-available run of the current model under its
  loads / combinations (a built-in launcher, not a stored entity).
* **Nonlinear Static** — each saved :class:`project.NonlinearCase`; add / modify
  / delete these here (delegating to :class:`nonlinear_cases.NonlinearCaseDialog`).
* **Saved analysis cases** — each saved :class:`project.AnalysisCase`, the
  multi-instance, named, editable cases for the migrated built-in types (Modal,
  Buckling, Moving Load, Temperature Gradient, Load Rating, Response Spectrum,
  Vehicle Dynamics, Influence Surface, Cable Tuning, and Time History — see
  :mod:`case_types`). **Add ▾** creates one (seeding the type's setup dialog),
  **Modify** / **Delete** manage it, and **Run** uses its stored params. Time
  History references a :class:`project.TimeHistoryFunction` and, being an
  interactive fiber solve, opens its runner *seeded* rather than headlessly.
* **Construction Stages** — a built-in launcher for the incremental erection
  sequence under self-weight, reporting the camber; operates on the shared,
  already-persistent ``project.stages`` (so left a launcher).

The dialog never runs anything itself: **Run** records a request and closes;
the owning window dispatches it (a linear-static run, a launcher's setup dialog,
or — for a saved :class:`~project.AnalysisCase` — ``("case", id)`` →
:mod:`case_types` ``build_config`` + ``dispatch``). :meth:`manage` returns
``(nonlinear_cases, analysis_cases, run_request)`` or ``None`` if cancelled.
Built from the L1 scaffold; headless-constructible.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout,
                               QHeaderView, QLineEdit, QMenu, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

import case_types
import style
from project import NonlinearCase

_DOF = {0: "Ux", 1: "Uy", 2: "Rz", 3: "Rx", 4: "Ry", 5: "Rz"}

# roadmap placeholders — shown greyed so the list previews the plan
_PLANNED = []


def _icon(name: str):
    try:
        import icons
        return icons.icon(name, style.ICON)
    except Exception:                                          # noqa: BLE001
        from PySide6.QtGui import QIcon
        return QIcon()


class AnalysisCasesDialog(QDialog):
    """List / add / modify / delete / run all analysis cases."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Analysis cases")
        self._project = project
        self._cases = copy.deepcopy(project.nonlinear_cases)
        self._acases = copy.deepcopy(getattr(project, "analysis_cases", []))
        self._row_meta: list[dict] = []          # per table row: {kind, ...}
        self._run_request = None
        self.resize(640, 460)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter cases by name or type…")
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(lambda *_: self._apply_filter())
        root.addWidget(self._filter)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Case", "Type", "Details"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            self.table.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            self.table.SelectionMode.SingleSelection)
        self.table.setEditTriggers(self.table.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._sync_buttons)
        self.table.doubleClicked.connect(lambda *_: self._run())
        root.addWidget(self.table, 1)

        row = QHBoxLayout()
        self._add_btn = QPushButton("Add ▾")
        self._add_btn.setIcon(_icon("run"))
        self._add_btn.setMenu(self._build_add_menu())
        self._mod_btn = QPushButton("Modify…")
        self._mod_btn.clicked.connect(self._modify)
        self._dup_btn = QPushButton("Duplicate")
        self._dup_btn.setToolTip("Create a copy of the selected case to tweak")
        self._dup_btn.clicked.connect(self._duplicate)
        self._del_btn = QPushButton("Delete")
        self._del_btn.clicked.connect(self._delete)
        self._up_btn = QPushButton("↑")
        self._up_btn.setToolTip("Move the selected case up")
        self._up_btn.clicked.connect(lambda: self._move(-1))
        self._down_btn = QPushButton("↓")
        self._down_btn.setToolTip("Move the selected case down")
        self._down_btn.clicked.connect(lambda: self._move(1))
        self._tree_btn = QPushButton("Tree…")
        self._tree_btn.setToolTip("Show the case dependency tree "
                                  "(what continues from / starts from what)")
        self._tree_btn.clicked.connect(self._show_tree)
        self._run_btn = QPushButton("Run")
        self._run_btn.setIcon(_icon("run"))
        self._run_btn.clicked.connect(self._run)
        for b in (self._add_btn, self._mod_btn, self._dup_btn, self._del_btn,
                  self._up_btn, self._down_btn):
            row.addWidget(b)
        row.addWidget(self._tree_btn)
        row.addStretch(1)
        row.addWidget(self._run_btn)
        root.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)
        style.apply(self)
        self._refresh()

    def _build_add_menu(self) -> QMenu:
        """The Add ▾ menu: the bespoke nonlinear editor, plus one entry per
        registered saveable analysis type (:mod:`case_types`)."""
        menu = QMenu(self)
        menu.addAction("Nonlinear Static…", self._add)
        if case_types._ORDER:
            menu.addSeparator()
            for ct in case_types._ORDER:
                menu.addAction(
                    f"{ct.type_label}…",
                    lambda _=False, t=ct.type_id: self._add_case(t))
        return menu

    # ------------------------------------------------------------- table model
    def _refresh(self) -> None:
        rows: list[dict] = [
            {"kind": "linear", "name": "Linear Static", "type": "Static",
             "detail": "current loads / combinations", "icon": "undeformed",
             "runnable": True},
        ]
        for i, c in enumerate(self._cases):
            proto = c.protocol + (" · staged" if c.continue_from else "")
            rows.append({
                "kind": "nonlinear", "case_index": i, "case_id": c.id,
                "name": c.name, "type": "Nonlinear Static",
                "detail": f"{proto} · node {c.control_node} "
                          f"{_DOF.get(c.control_dof, '?')}"
                          + self._ic_suffix(c),
                "notes": getattr(c, "notes", ""),
                "icon": "run", "runnable": True})
        # saved multi-instance cases for migrated built-in types (Modal,
        # Buckling, …) — full Add/Modify/Delete/Run via their case_types adapter
        for i, c in enumerate(self._acases):
            ct = case_types.get(c.type)
            rows.append({
                "kind": "analysis", "case_index": i, "case_id": c.id,
                "type_id": c.type, "name": c.name,
                "type": (ct.type_label if ct else c.type),
                "detail": (ct.detail(self._project, c.params) if ct
                           else "saved case") + self._ic_suffix(c),
                "notes": getattr(c, "notes", ""),
                "icon": (ct.icon if ct else "run"), "runnable": True})
        rows.append({"kind": "stages", "name": "Construction Stages",
                     "type": "Construction Stages",
                     "detail": "incremental erection · camber",
                     "icon": "run", "runnable": True})
        for name in _PLANNED:
            rows.append({"kind": "planned", "name": name, "type": name,
                         "detail": "planned", "icon": None, "runnable": False})

        self._row_meta = rows
        self.table.setRowCount(len(rows))
        for r, meta in enumerate(rows):
            name_it = QTableWidgetItem(meta["name"])
            if meta["icon"]:
                name_it.setIcon(_icon(meta["icon"]))
            type_it = QTableWidgetItem(meta["type"])
            det_it = QTableWidgetItem(meta["detail"])
            notes = (meta.get("notes") or "").strip()
            if notes:                                    # surface notes on hover
                tip = f"{meta['name']} — {notes}"
                for it in (name_it, type_it, det_it):
                    it.setToolTip(tip)
            if meta["kind"] == "planned":                # greyed + not selectable
                for it in (name_it, type_it, det_it):
                    it.setFlags(Qt.ItemFlag.NoItemFlags)
                    it.setForeground(_muted())
            self.table.setItem(r, 0, name_it)
            self.table.setItem(r, 1, type_it)
            self.table.setItem(r, 2, det_it)
        self._sync_buttons()
        self._apply_filter()

    def _ic_suffix(self, case) -> str:
        """`" · from ‹source›"` when ``case`` starts from another case's committed
        state (E2), else empty. The source is always a nonlinear case; a dangling
        reference is flagged so a broken chain is visible in the list."""
        ic = getattr(case, "initial_condition", ("zero",))
        if not (ic and ic[0] == "state"):
            return ""
        src = next((c for c in self._cases if c.id == ic[1]), None)
        return f" · from {src.name}" if src else " · from (missing case)"

    def _selected(self) -> dict | None:
        r = self.table.currentRow()
        if 0 <= r < len(self._row_meta):
            return self._row_meta[r]
        return None

    def _sync_buttons(self) -> None:
        m = self._selected()
        editable = bool(m and m["kind"] in ("nonlinear", "analysis"))
        self._mod_btn.setEnabled(editable)
        self._dup_btn.setEnabled(editable)
        self._del_btn.setEnabled(editable)
        can_up = can_down = False
        if editable:                                 # reorder within its own list
            lst = self._cases if m["kind"] == "nonlinear" else self._acases
            can_up = m["case_index"] > 0
            can_down = m["case_index"] < len(lst) - 1
        self._up_btn.setEnabled(can_up)
        self._down_btn.setEnabled(can_down)
        self._run_btn.setEnabled(bool(m and m.get("runnable")))

    # ---------------------------------------------------------------- actions
    def _proxy_project(self):
        proxy = copy.copy(self._project)
        proxy.nonlinear_cases = self._cases
        proxy.analysis_cases = self._acases
        return proxy

    def _next_case_id(self) -> int:
        return max((c.id for c in self._acases), default=0) + 1

    def _unique_name(self, name: str, exclude_id=None) -> str:
        """Keep saved-case names distinct (CSiBridge does): append ``(2)``, ``(3)``
        … on a clash. ``exclude_id`` skips the case being modified."""
        name = name or "Case"
        taken = {c.name for c in self._acases if c.id != exclude_id}
        if name not in taken:
            return name
        i = 2
        while f"{name} ({i})" in taken:
            i += 1
        return f"{name} ({i})"

    def _select_row(self, kind: str, case_id) -> None:
        for r, meta in enumerate(self._row_meta):
            if meta["kind"] == kind and meta.get("case_id") == case_id:
                self.table.setCurrentCell(r, 0)
                return

    def _duplicate(self) -> None:
        """Clone the selected saved case (nonlinear or built-in) — the fast way
        to define a variant (an RS-X → RS-Y, a Moving HL-93 → Moving Permit)."""
        m = self._selected()
        if not (m and m["kind"] in ("nonlinear", "analysis")):
            return
        if m["kind"] == "analysis":
            clone = copy.deepcopy(self._acases[m["case_index"]])
            clone.id = self._next_case_id()
            clone.name = self._unique_name(f"{clone.name} (copy)")
            self._acases.append(clone)
        else:
            clone = copy.deepcopy(self._cases[m["case_index"]])
            clone.id = max((c.id for c in self._cases), default=0) + 1
            clone.name = f"{clone.name} (copy)"
            self._cases.append(clone)
        self._refresh()
        self._select_row(m["kind"], clone.id)

    def _move(self, delta: int) -> None:
        """Reorder the selected case within its own list (nonlinear or saved)."""
        m = self._selected()
        if not (m and m["kind"] in ("nonlinear", "analysis")):
            return
        lst = self._cases if m["kind"] == "nonlinear" else self._acases
        i = m["case_index"]
        j = i + delta
        if not (0 <= j < len(lst)):
            return
        lst[i], lst[j] = lst[j], lst[i]
        self._refresh()
        self._select_row(m["kind"], m["case_id"])

    def _add_case(self, type_id: str) -> None:
        ct = case_types.get(type_id)
        if ct is None:
            return
        case = ct.edit(self, self._proxy_project())
        if case is None:
            return
        case.id = self._next_case_id()
        case.name = self._unique_name(case.name)
        self._acases.append(case)
        self._refresh()
        self._select_row("analysis", case.id)

    def _add(self) -> None:
        from nonlinear_cases import NonlinearCaseDialog
        c = NonlinearCaseDialog.edit(self, self._proxy_project())
        if c is None:
            return
        if any(x.id == c.id for x in self._cases):
            QMessageBox.warning(self, "Duplicate", f"Case {c.id} already exists.")
            return
        self._cases.append(c)
        self._refresh()
        self.table.setCurrentCell(len(self._cases), 0)      # +1 for linear row

    def _modify(self) -> None:
        m = self._selected()
        if not m:
            return
        if m["kind"] == "analysis":
            ct = case_types.get(m["type_id"])
            if ct is None:
                return
            idx = m["case_index"]
            case = ct.edit(self, self._proxy_project(), self._acases[idx])
            if case is not None:
                case.id = self._acases[idx].id       # id is not user-editable
                case.name = self._unique_name(case.name, exclude_id=case.id)
                self._acases[idx] = case
                self._refresh()
                self._select_row("analysis", case.id)
            return
        if m["kind"] != "nonlinear":
            return
        from nonlinear_cases import NonlinearCaseDialog
        idx = m["case_index"]
        c = NonlinearCaseDialog.edit(self, self._proxy_project(),
                                     self._cases[idx])
        if c is not None:
            self._cases[idx] = c
            self._refresh()

    def _delete(self) -> None:
        m = self._selected()
        if not m:
            return
        if m["kind"] == "analysis":
            del self._acases[m["case_index"]]
            self._refresh()
            return
        if m["kind"] != "nonlinear":
            return
        cid = m["case_id"]
        # Block deletion while another case depends on this one: a staged
        # continuation (continue_from) or, since E2, a case that starts from this
        # case's committed state via its initial_condition (any nonlinear or
        # saved analysis case). Name the dependents so the fix is obvious.
        blockers = [c.name for c in self._cases if c.continue_from == cid]
        blockers += [c.name for c in (self._cases + self._acases)
                     if tuple(getattr(c, "initial_condition", ("zero",)))
                     == ("state", cid)]
        blockers = sorted(dict.fromkeys(blockers))     # de-dup, keep order-ish
        if blockers:
            QMessageBox.warning(
                self, "In use",
                f"'{m['name']}' is used as the initial condition / continuation "
                f"of: {', '.join(blockers)}.\n\nEdit or delete those cases "
                "first.")
            return
        del self._cases[m["case_index"]]
        self._refresh()

    def _show_tree(self) -> None:
        """Open the read-only Load Case Tree on the *in-progress* edits (so it
        reflects unsaved Add / Modify / Delete), matching CSiBridge's *Show Load
        Case Tree*."""
        from case_tree_dialog import CaseTreeDialog
        CaseTreeDialog.show_tree(self, self._proxy_project())

    def _apply_filter(self) -> None:
        """Hide rows whose name / type / details don't contain the filter text
        (case-insensitive). A blank filter shows everything."""
        text = self._filter.text().strip().lower()
        for r, meta in enumerate(self._row_meta):
            hay = (f"{meta.get('name', '')} {meta.get('type', '')} "
                   f"{meta.get('detail', '')}").lower()
            self.table.setRowHidden(r, bool(text) and text not in hay)

    def _context_menu(self, pos) -> None:
        """Right-click actions for the row under the cursor — the same Run /
        Modify / Duplicate / Delete / Show-tree the buttons offer, gated the
        same way."""
        it = self.table.itemAt(pos)
        if it is not None:
            self.table.setCurrentCell(it.row(), 0)
        self._build_context_menu().exec(
            self.table.viewport().mapToGlobal(pos))

    def _build_context_menu(self) -> QMenu:
        """Build (but don't show) the context menu for the current selection —
        split out from :meth:`_context_menu` so it is testable without the
        blocking modal ``exec``."""
        m = self._selected()
        editable = bool(m and m["kind"] in ("nonlinear", "analysis"))
        menu = QMenu(self)
        menu.addAction("Run", self._run).setEnabled(bool(m and m.get("runnable")))
        menu.addSeparator()
        menu.addAction("Modify…", self._modify).setEnabled(editable)
        menu.addAction("Duplicate", self._duplicate).setEnabled(editable)
        menu.addAction("Delete", self._delete).setEnabled(editable)
        menu.addSeparator()
        menu.addAction("Show tree…", self._show_tree)
        return menu

    def _run(self) -> None:
        m = self._selected()
        if not (m and m.get("runnable")):
            return
        if m["kind"] == "linear":
            self._run_request = ("linear",)
        elif m["kind"] == "nonlinear":
            self._run_request = ("nonlinear", m["case_id"])
        elif m["kind"] == "analysis":
            self._run_request = ("case", m["case_id"])
        elif m["kind"] == "stages":
            self._run_request = ("stages",)
        self.accept()

    @classmethod
    def manage(cls, parent, project):
        dlg = cls(parent, project)
        if not dlg.exec():
            return None
        return dlg._cases, dlg._acases, dlg._run_request


def _muted():
    from PySide6.QtGui import QBrush, QColor
    return QBrush(QColor(style.MUTED))
