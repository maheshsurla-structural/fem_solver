"""Reusable themed widgets for the desktop GUI.

``CollapsibleGroup`` — a card with a clickable header that expands/collapses its
body, so a tall stack of option groups (the Section Designer's definition
panel) stays scannable. Drop-in for a QGroupBox: put your layout on ``.body``.

    grp = CollapsibleGroup("Materials")
    form = QFormLayout(grp.body)
    ...
    parent_layout.addWidget(grp)
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget


class CollapsibleGroup(QWidget):
    """A titled card whose body collapses when its header is clicked."""

    def __init__(self, title: str, parent=None, *, collapsed: bool = False):
        super().__init__(parent)
        self.setObjectName("collGroup")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.toggle = QToolButton()
        self.toggle.setObjectName("collHeader")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(not collapsed)
        self.toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(self._arrow(not collapsed))
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.clicked.connect(self._on_toggled)
        outer.addWidget(self.toggle)

        self.body = QWidget()
        self.body.setObjectName("collBody")
        # inset so the caller's layout doesn't touch the card border
        self.body.setContentsMargins(10, 2, 10, 10)
        self.body.setVisible(not collapsed)
        outer.addWidget(self.body)

    @staticmethod
    def _arrow(expanded: bool) -> Qt.ArrowType:
        return Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow

    def _on_toggled(self, checked: bool) -> None:
        self.body.setVisible(checked)
        self.toggle.setArrowType(self._arrow(checked))

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setChecked(expanded)
        self._on_toggled(expanded)
