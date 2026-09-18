"""Customizable viewport toolbar — the command model plus the "Customize
Toolbar" dialog (a two-list add / remove / reorder editor, in the style of
Midas Civil's *Custom Toolbar*).

A :class:`ToolCommand` is one entry the toolbar can show; the dialog edits the
ordered list of command ids that make up the bar. Kept UI-agnostic of the view
so it is easy to unit-test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QToolButton, QVBoxLayout)

import icons
import style

SEPARATOR_ID = "|"                    # a visual gap between icon groups
# the order groups appear in the dialog's "Choose commands" filter
GROUP_ORDER = ("Tools", "Navigate", "Orient", "Model", "Edit", "Active",
               "Analysis", "Results", "Appearance")


@dataclass
class ToolCommand:
    """One button the viewport toolbar can carry.

    ``kind`` is ``"tool"`` (an exclusive, checkable interaction tool whose id is
    the view mode), ``"toggle"`` (an independent checkable — e.g. the 2-D lock),
    or ``"action"`` (a momentary command). ``activate`` runs on click:
    ``activate()`` for tool/action, ``activate(checked)`` for a toggle.
    """
    id: str
    label: str
    icon: str
    group: str = "Navigate"
    kind: str = "action"
    activate: Optional[Callable] = None


class CustomizeToolbarDialog(QDialog):
    """Edit the ordered command list of the viewport toolbar."""

    def __init__(self, registry: dict, current: list, default: list,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Customize Toolbar")
        self.setMinimumWidth(560)
        self._registry = registry
        self._current = [c for c in current
                         if c == SEPARATOR_ID or c in registry]
        self._default = list(default)

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Choose commands"))
        self._group_combo = QComboBox()
        self._group_combo.addItem("All commands", "")
        for g in GROUP_ORDER:
            if any(c.group == g for c in registry.values()):
                self._group_combo.addItem(g, g)
        self._group_combo.currentIndexChanged.connect(self._refresh_available)
        root.addWidget(self._group_combo)

        cols = QHBoxLayout()
        root.addLayout(cols, 1)

        # left — available commands
        left = QVBoxLayout()
        left.addWidget(QLabel("Commands"))
        self._avail = QListWidget()
        self._avail.itemDoubleClicked.connect(lambda _i: self._add())
        left.addWidget(self._avail)
        cols.addLayout(left, 1)

        # middle — add / remove
        mid = QVBoxLayout()
        mid.addStretch(1)
        self._add_btn = QPushButton("Add  >>")
        self._add_btn.clicked.connect(self._add)
        self._rm_btn = QPushButton("<<  Remove")
        self._rm_btn.clicked.connect(self._remove)
        mid.addWidget(self._add_btn)
        mid.addWidget(self._rm_btn)
        mid.addStretch(1)
        cols.addLayout(mid)

        # right — current toolbar + reorder
        right = QVBoxLayout()
        right.addWidget(QLabel("Toolbar"))
        row = QHBoxLayout()
        self._cur = QListWidget()
        self._cur.itemDoubleClicked.connect(lambda _i: self._remove())
        row.addWidget(self._cur, 1)
        reorder = QVBoxLayout()
        reorder.addStretch(1)
        self._up_btn = QToolButton()
        self._up_btn.setArrowType(Qt.ArrowType.UpArrow)   # native, font-free
        self._up_btn.setToolTip("Move up")
        self._up_btn.clicked.connect(lambda: self._move(-1))
        self._down_btn = QToolButton()
        self._down_btn.setArrowType(Qt.ArrowType.DownArrow)
        self._down_btn.setToolTip("Move down")
        self._down_btn.clicked.connect(lambda: self._move(1))
        reorder.addWidget(self._up_btn)
        reorder.addWidget(self._down_btn)
        reorder.addStretch(1)
        row.addLayout(reorder)
        right.addLayout(row, 1)
        cols.addLayout(right, 1)

        bottom = QHBoxLayout()
        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset)
        bottom.addWidget(reset)
        bottom.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        bottom.addWidget(bb)
        root.addLayout(bottom)

        self._refresh()

    # -- list population -----------------------------------------------------
    def _icon(self, name: str):
        return icons.icon(name, style.ICON)

    def _refresh(self) -> None:
        self._refresh_available()
        self._refresh_current()

    def _refresh_available(self) -> None:
        self._avail.clear()
        sep = QListWidgetItem("<Separator>")
        sep.setData(Qt.ItemDataRole.UserRole, SEPARATOR_ID)
        self._avail.addItem(sep)
        grp = self._group_combo.currentData()
        for cid, cmd in sorted(self._registry.items(),
                               key=lambda kv: kv[1].label.lower()):
            if cid in self._current:
                continue                      # already on the bar (once only)
            if grp and cmd.group != grp:
                continue
            it = QListWidgetItem(self._icon(cmd.icon), cmd.label)
            it.setData(Qt.ItemDataRole.UserRole, cid)
            self._avail.addItem(it)

    def _refresh_current(self) -> None:
        self._cur.clear()
        for cid in self._current:
            if cid == SEPARATOR_ID:
                it = QListWidgetItem("<Separator>")
            else:
                cmd = self._registry.get(cid)
                if cmd is None:
                    continue
                it = QListWidgetItem(self._icon(cmd.icon), cmd.label)
            it.setData(Qt.ItemDataRole.UserRole, cid)
            self._cur.addItem(it)

    # -- edits ---------------------------------------------------------------
    def _add(self) -> None:
        it = self._avail.currentItem()
        if it is None:
            return
        cid = it.data(Qt.ItemDataRole.UserRole)
        row = self._cur.currentRow()
        at = row + 1 if row >= 0 else len(self._current)
        self._current.insert(at, cid)
        self._refresh()
        self._cur.setCurrentRow(at)

    def _remove(self) -> None:
        row = self._cur.currentRow()
        if row < 0:
            return
        del self._current[row]
        self._refresh()
        self._cur.setCurrentRow(min(row, self._cur.count() - 1))

    def _move(self, delta: int) -> None:
        row = self._cur.currentRow()
        new = row + delta
        if row < 0 or not (0 <= new < len(self._current)):
            return
        self._current[row], self._current[new] = (self._current[new],
                                                   self._current[row])
        self._refresh_current()
        self._cur.setCurrentRow(new)

    def _reset(self) -> None:
        self._current = list(self._default)
        self._refresh()

    def result_layout(self) -> list:
        return list(self._current)
