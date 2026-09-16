"""Run-analysis control (plan A4) — the desktop counterpart of CSiBridge's
*Run Analysis* dialog.

Lists every runnable analysis case (Linear Static, each Nonlinear Static case,
Time History) with a per-row **Run / Do not run** action and a live **Status**
column, plus a **Run Now** button. Linear static is batch-runnable, so it runs
inline here (Running → Done / Failed / No model) through an injected callback;
nonlinear and time-history cases need their own interactive worker+plot dialogs,
so a flagged one is *queued* and returned to the owning window to open after
this control closes.

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

    def __init__(self, parent, project, run_linear=None):
        super().__init__(parent)
        self.setWindowTitle("Run analysis")
        self._project = project
        self._run_linear = run_linear          # callable() -> truthy on success
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
        rows = [{"kind": "linear", "name": "Linear Static", "type": "Static",
                 "default_run": True}]
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
            self.table.setItem(r, 3, QTableWidgetItem("Not run"))
            meta["action"] = action
            self._rows.append(meta)

    def _set_status(self, r: int, text: str) -> None:
        self.table.item(r, 3).setText(text)

    # ------------------------------------------------------------- run
    def _run_now(self) -> None:
        from PySide6.QtWidgets import QApplication
        self._requests = []
        for r, meta in enumerate(self._rows):
            if not meta["action"].currentData():          # "Do not run"
                self._set_status(r, "—")
                continue
            if meta["kind"] == "linear":
                self._set_status(r, "Running…")
                QApplication.processEvents()
                try:
                    ok = self._run_linear() if self._run_linear else None
                except Exception as exc:                   # noqa: BLE001
                    self._set_status(r, f"Failed: {exc}")
                else:
                    self._set_status(r, "Done" if ok else "No model")
            elif meta["kind"] == "nonlinear":
                self._requests.append(("nonlinear", meta["case_id"]))
                self._set_status(r, "Queued (opens dialog)")
            else:                                          # saved analysis case
                self._requests.append(("case", meta["case_id"]))
                self._set_status(r, "Queued")

    def deferred_requests(self) -> list:
        return list(self._requests)

    @classmethod
    def run(cls, parent, project, run_linear=None) -> list:
        dlg = cls(parent, project, run_linear)
        dlg.exec()
        return dlg.deferred_requests()
