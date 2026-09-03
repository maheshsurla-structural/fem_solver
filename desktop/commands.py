"""Undo/redo commands for project edits.

Each edit is captured as before/after deep-copy snapshots of the ``Project``;
undo/redo restore them through the main window. The Project is a small
dataclass tree, so snapshotting is cheap — this keeps undo simple and robust
(no per-operation inverse logic to get wrong).
"""
from __future__ import annotations

from PySide6.QtGui import QUndoCommand


class EditCommand(QUndoCommand):
    def __init__(self, window, text, before, after, select=None):
        super().__init__(text)
        self._window = window
        self._before = before
        self._after = after
        self._select = select

    def redo(self) -> None:                 # also invoked on the initial push
        self._window._restore(self._after, self._select)

    def undo(self) -> None:
        self._window._restore(self._before, None)
