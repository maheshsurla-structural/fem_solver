"""Model-checks dialog (plan §16 G-S5) — presents :func:`model_checks.check_project`
findings grouped by severity, each with a fix hint, before a run.

Pure Qt — headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QListWidget,
                               QListWidgetItem, QVBoxLayout)

import model_checks as MC
import style

_ICON = {"error": "⛔", "warning": "⚠", "info": "ℹ"}


def _sev_color(level: str) -> str:
    """Severity → themed ink (error=BAD, warning=WARN, else muted)."""
    return {"error": style.BAD, "warning": style.WARN}.get(level, style.MUTED)


class ModelChecksDialog(QDialog):
    def __init__(self, parent, checks):
        super().__init__(parent)
        self.setWindowTitle("Model checks")
        self.resize(560, 380)
        v = QVBoxLayout(self)
        v.setContentsMargins(style.SP_LG, style.SP_LG, style.SP_LG, style.SP_LG)
        v.setSpacing(style.SP_SM)

        n_err, n_warn, n_info = MC.summarize(checks)
        if not checks or (n_err == 0 and n_warn == 0):
            head = "✓ No problems found — the model looks ready to run."
        else:
            head = (f"{n_err} error(s), {n_warn} warning(s), {n_info} note(s).")
        self.summary = QLabel(head)
        self.summary.setObjectName("h3")
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
            it.setForeground(QColor(_sev_color(c.level)))
            it.setData(Qt.ItemDataRole.UserRole, c.level)
            self.list.addItem(it)
        if not checks:
            self.list.addItem(QListWidgetItem("✓ nothing to report"))

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        v.addWidget(btns)
        style.apply(self)

    @classmethod
    def show_for(cls, parent, project) -> list:
        """Build the checks for ``project``, show the dialog, and return them."""
        checks = MC.check_project(project)
        cls(parent, checks).exec()
        return checks
