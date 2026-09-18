"""Run-analysis control (plan A4) — the desktop counterpart of CSiBridge's
*Run Analysis* dialog.

Lists every runnable analysis case (each saved analysis case — Linear Static,
Modal, Buckling, … — and each Nonlinear Static case) with a per-row **Run / Do
not run** action and a live **Status** column, plus a **Run Now** button. Saved
analysis cases run headlessly from their stored params, so a flagged one is
returned to the owning window as a ``("case", id)`` request; nonlinear and
time-history cases need their own interactive worker+plot dialogs, so a flagged
one is *queued* and returned to open after this control closes.

Built from the L1 scaffold; pure Qt, headless-constructible. :meth:`run`
returns the queued deferred requests (nonlinear / time-history) or ``[]``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QHBoxLayout, QHeaderView, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

import case_types
import style
from project import NonlinearCase  # noqa: F401  (type hint / clarity)

_DOF = {0: "Ux", 1: "Uy", 2: "Rz", 3: "Rx", 4: "Ry", 5: "Rz"}


class RunAnalysisDialog(QDialog):
    """Choose which cases to run, run the batchable ones, show status."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Run analysis")
        self._project = project
        self._requests: list = []              # deferred nonlinear / TH requests
        self.resize(620, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Case", "Type", "Action",
                                              "Status"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            self.table.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(self.table.EditTrigger.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.table, 1)

        self._rows: list[dict] = []            # {kind, ..., action: QComboBox}
        self._build_rows()

        row = QHBoxLayout()
        self._run_btn = QPushButton("Run Now")
        self._run_btn.clicked.connect(self._run_now)
        row.addStretch(1)
        row.addWidget(self._run_btn)
        root.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        # the sole Close button reports as "reject"; treat it as accept so the
        # queued requests survive to the caller
        btns.button(QDialogButtonBox.StandardButton.Close).clicked.connect(
            self.accept)
        root.addWidget(btns)
        style.apply(self)

    # ------------------------------------------------------------- table build
    def _entries(self) -> list[dict]:
        # E3b: Linear Static is a saved case now (listed below with the other
        # analysis cases), not a special built-in row.
        rows: list[dict] = []
        for c in self._project.nonlinear_cases:
            rows.append({"kind": "nonlinear", "case_id": c.id, "name": c.name,
                         "type": "Nonlinear Static", "default_run": False})
        # Every saved, multi-instance analysis case (Modal, Buckling, Response
        # Spectrum, Moving Load, …, and Time History) — dispatched through its
        # case_types adapter, so the run control lists *every* analysis, matching
        # the Analysis-cases home (E5a). Time History is one of these now, so the
        # old standalone launcher row is gone.
        for c in getattr(self._project, "analysis_cases", []):
            ct = case_types.get(c.type)
            rows.append({"kind": "analysis", "case_id": c.id, "name": c.name,
                         "type": (ct.type_label if ct else c.type),
                         "default_run": False})
        return rows

    def _build_rows(self) -> None:
        entries = self._entries()
        self.table.setRowCount(len(entries))
        self._rows = []
        for r, meta in enumerate(entries):
            self.table.setItem(r, 0, QTableWidgetItem(meta["name"]))
            self.table.setItem(r, 1, QTableWidgetItem(meta["type"]))
            action = QComboBox()
            action.addItem("Run", True)
            action.addItem("Do not run", False)
            action.setCurrentIndex(0 if meta["default_run"] else 1)
            self.table.setCellWidget(r, 2, action)
            # seed the last-known session status (E5c) instead of a flat "Not run"
            key = self._status_key(meta)
            st = self._project.case_status(key) if key else None
            self.table.setItem(
                r, 3, QTableWidgetItem(st["status"] if st else "Not run"))
            meta["action"] = action
            self._rows.append(meta)

    @staticmethod
    def _status_key(meta):
        kind = meta["kind"]
        if kind in ("nonlinear", "analysis"):
            return (kind, meta["case_id"])
        return None

    def _set_status(self, r: int, text: str) -> None:
        self.table.item(r, 3).setText(text)

    # ------------------------------------------------------------- run
    def _run_now(self) -> None:
        self._requests = []
        for r, meta in enumerate(self._rows):
            if not meta["action"].currentData():          # "Do not run"
                self._set_status(r, "—")
                continue
            if meta["kind"] == "nonlinear":
                self._requests.append(("nonlinear", meta["case_id"]))
                self._set_status(r, "Queued (opens dialog)")
            else:                                          # saved analysis case
                self._requests.append(("case", meta["case_id"]))
                self._set_status(r, "Queued")

    def deferred_requests(self) -> list:
        return list(self._requests)

    @classmethod
    def run(cls, parent, project) -> list:
        dlg = cls(parent, project)
        dlg.exec()
        return dlg.deferred_requests()
