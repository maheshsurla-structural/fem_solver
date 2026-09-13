"""Unified analysis-cases home (plan A1) — the desktop counterpart of
CSiBridge's *Define ▸ Load Cases* list.

One table lists **every** analysis case with a Type column and icon:

* **Linear Static** — the always-available run of the current model under its
  loads / combinations (a built-in launcher, not a stored entity).
* **Nonlinear Static** — each saved :class:`project.NonlinearCase`; add / modify
  / delete these here (delegating to :class:`nonlinear_cases.NonlinearCaseDialog`).
* **Time History** — a built-in launcher for the nonlinear dynamic dialog.
* **Modal** — a built-in launcher for the free-vibration eigen-analysis.
* **Response Spectrum** — a built-in launcher for the modal-superposition
  seismic analysis (design spectrum → SRSS / CQC).
* **Buckling** — a built-in launcher for linear (eigenvalue) buckling
  ``(K + λ·K_g)·φ = 0`` on a member-sub-divided model.
* **Moving Load** — greyed roadmap row so the list signals where the product
  is going (matching the reference tools).

The dialog never runs anything itself: **Run** records a request and closes;
the owning window dispatches it (linear-static run, or opening the pushover /
time-history / modal runner). :meth:`manage` returns
``(nonlinear_cases, run_request)`` or ``None`` if cancelled. Built from the L1
scaffold; headless-constructible.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout,
                               QHeaderView, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

import style
from project import NonlinearCase

_DOF = {0: "Ux", 1: "Uy", 2: "Rz", 3: "Rx", 4: "Ry", 5: "Rz"}

# roadmap placeholders — shown greyed so the list previews the plan
_PLANNED = ["Moving Load"]


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
        self._add_btn = QPushButton("Add nonlinear case…")
        self._add_btn.setIcon(_icon("run"))
        self._add_btn.clicked.connect(self._add)
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
                "icon": "run", "runnable": True})
        rows.append({"kind": "timehistory", "name": "Time History",
                     "type": "Time History",
                     "detail": "ground-motion record", "icon": "run",
                     "runnable": True})
        rows.append({"kind": "modal", "name": "Modal", "type": "Modal",
                     "detail": "eigen · free vibration", "icon": "undeformed",
                     "runnable": True})
        rows.append({"kind": "responsespectrum", "name": "Response Spectrum",
                     "type": "Response Spectrum",
                     "detail": "modal superposition · SRSS/CQC", "icon": "run",
                     "runnable": True})
        rows.append({"kind": "buckling", "name": "Buckling", "type": "Buckling",
                     "detail": "eigenvalue · (K + λ·K_g)", "icon": "run",
                     "runnable": True})
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
        is_nl = bool(m and m["kind"] == "nonlinear")
        self._mod_btn.setEnabled(is_nl)
        self._del_btn.setEnabled(is_nl)
        self._run_btn.setEnabled(bool(m and m.get("runnable")))

    # ---------------------------------------------------------------- actions
    def _proxy_project(self):
        proxy = copy.copy(self._project)
        proxy.nonlinear_cases = self._cases
        return proxy

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
        from nonlinear_cases import NonlinearCaseDialog
        m = self._selected()
        if not (m and m["kind"] == "nonlinear"):
            return
        idx = m["case_index"]
        c = NonlinearCaseDialog.edit(self, self._proxy_project(),
                                     self._cases[idx])
        if c is not None:
            self._cases[idx] = c
            self._refresh()

    def _delete(self) -> None:
        m = self._selected()
        if not (m and m["kind"] == "nonlinear"):
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
        elif m["kind"] == "timehistory":
            self._run_request = ("timehistory",)
        elif m["kind"] == "modal":
            self._run_request = ("modal",)
        elif m["kind"] == "responsespectrum":
            self._run_request = ("responsespectrum",)
        elif m["kind"] == "buckling":
            self._run_request = ("buckling",)
        self.accept()

    @classmethod
    def manage(cls, parent, project):
        dlg = cls(parent, project)
        if not dlg.exec():
            return None
        return dlg._cases, dlg._run_request


def _muted():
    from PySide6.QtGui import QBrush, QColor
    return QBrush(QColor(style.MUTED))
