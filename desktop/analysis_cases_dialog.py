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
  Vehicle Dynamics, Influence Surface, Cable Tuning — see :mod:`case_types`).
  **Add ▾** creates one (seeding the type's setup dialog), **Modify** /
  **Delete** manage it, and **Run** uses its stored params with no re-prompt
  (analysis-cases-manager plan).
* **Time History** — a built-in launcher for the nonlinear dynamic dialog (a
  runner-dialog + ground-motion record → left a launcher, see the plan).
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
                               QHeaderView, QMenu, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

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

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Case", "Type", "Details"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            self.table.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            self.table.SelectionMode.SingleSelection)
        self.table.setEditTriggers(self.table.EditTrigger.NoEditTriggers)
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
        self._del_btn = QPushButton("Delete")
        self._del_btn.clicked.connect(self._delete)
        self._run_btn = QPushButton("Run")
        self._run_btn.setIcon(_icon("run"))
        self._run_btn.clicked.connect(self._run)
        for b in (self._add_btn, self._mod_btn, self._del_btn):
            row.addWidget(b)
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
                          f"{_DOF.get(c.control_dof, '?')}",
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
                           else "saved case"),
                "notes": getattr(c, "notes", ""),
                "icon": (ct.icon if ct else "run"), "runnable": True})
        rows.append({"kind": "timehistory", "name": "Time History",
                     "type": "Time History",
                     "detail": "ground-motion record", "icon": "run",
                     "runnable": True})
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

    def _selected(self) -> dict | None:
        r = self.table.currentRow()
        if 0 <= r < len(self._row_meta):
            return self._row_meta[r]
        return None

    def _sync_buttons(self) -> None:
        m = self._selected()
        editable = bool(m and m["kind"] in ("nonlinear", "analysis"))
        self._mod_btn.setEnabled(editable)
        self._del_btn.setEnabled(editable)
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

    def _select_case(self, case_id) -> None:
        for r, meta in enumerate(self._row_meta):
            if meta["kind"] == "analysis" and meta["case_id"] == case_id:
                self.table.setCurrentCell(r, 0)
                return

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
        self._select_case(case.id)

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
                self._select_case(case.id)
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
        used = [c.id for c in self._cases if c.continue_from == cid]
        if used:
            QMessageBox.warning(self, "In use", f"Case {cid} is continued-from "
                                f"by case(s) {', '.join(map(str, used))}.")
            return
        del self._cases[m["case_index"]]
        self._refresh()

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
        elif m["kind"] == "timehistory":
            self._run_request = ("timehistory",)
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
