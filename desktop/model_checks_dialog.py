"""Model-checks dialog (plan §16 G-S5) — presents :func:`model_checks.check_project`
findings grouped by severity, each with a fix hint, before a run.

Pure Qt — headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialogButtonBox, QDialog, QLabel, QListWidget,
                               QListWidgetItem, QVBoxLayout)

import model_checks as MC

_ICON = {"error": "⛔", "warning": "⚠", "info": "ℹ"}
_COLOR = {"error": "#c62828", "warning": "#f9a825", "info": "#1f5f8b"}


class ModelChecksDialog(QDialog):
    def __init__(self, parent, checks):
        super().__init__(parent)
        self.setWindowTitle("Model checks")
        self.resize(560, 380)
        v = QVBoxLayout(self)

        n_err, n_warn, n_info = MC.summarize(checks)
        if not checks or (n_err == 0 and n_warn == 0):
            head = "✓ No problems found — the model looks ready to run."
        else:
            head = (f"{n_err} error(s), {n_warn} warning(s), {n_info} note(s).")
        self.summary = QLabel(head)
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)

        self.list = QListWidget()
        self.list.setWordWrap(True)
        v.addWidget(self.list, 1)
        for c in checks:
            text = f"{_ICON.get(c.level, '·')}  {c.message}"
            if c.hint:
                text += f"\n     → {c.hint}"
            it = QListWidgetItem(text)
            it.setForeground(Qt.GlobalColor.darkGray if c.level == "info"
                             else it.foreground())
            it.setData(Qt.ItemDataRole.UserRole, c.level)
            self.list.addItem(it)
        if not checks:
            self.list.addItem(QListWidgetItem("✓ nothing to report"))

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        v.addWidget(btns)

    @classmethod
    def show_for(cls, parent, project) -> list:
        """Build the checks for ``project``, show the dialog, and return them."""
        checks = MC.check_project(project)
        cls(parent, checks).exec()
        return checks
