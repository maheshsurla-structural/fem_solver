"""The File **backstage** (plan ribbon R8).

A full-window overlay that replaces the ribbon's plain File dropdown with a
CSiBridge / Office-style backstage: a left rail of file commands, a Recent-files
list, and an About card — all on the app's card scaffold, themed light/dark and
comfortable/compact. Opened from the ribbon's File button; the back arrow or Esc
closes it.

The widget is deliberately dumb: it renders the ``QAction`` objects it is handed
(so enabled state and behaviour stay owned by the shell) and emits
``openRecentRequested`` / ``closed`` for the shell to act on. The shell supplies
and persists the recent list.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QSizePolicy, QToolButton, QVBoxLayout, QWidget)

import style
from analysis_ui import GroupCard


class Backstage(QWidget):
    """Full-window File backstage overlay (plan ribbon R8)."""

    closed = Signal()
    openRecentRequested = Signal(str)

    def __init__(self, parent, file_actions, *, product="femsolver desktop",
                 monogram=None):
        super().__init__(parent)
        self.setObjectName("backstage")
        # A QWidget subclass only honours a QSS ``background`` when styled-bg is
        # on; without it the overlay is transparent and the view bleeds through.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)          # fully occlude what's behind

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- top bar: a back arrow that closes the backstage ----------------
        top = QWidget()
        top.setObjectName("backstageTop")
        trow = QHBoxLayout(top)
        trow.setContentsMargins(style.SP_MD, style.SP_SM, style.SP_MD, style.SP_SM)
        back = QToolButton()
        back.setObjectName("backstageBack")
        back.setText("←  File")
        back.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.clicked.connect(self.close_panel)
        trow.addWidget(back)
        trow.addStretch(1)
        root.addWidget(top)

        # ---- body: [ command rail | recent + about ] -----------------------
        body = QHBoxLayout()
        body.setContentsMargins(style.SP_XL, style.SP_LG, style.SP_XL, style.SP_XL)
        body.setSpacing(style.SP_XL)

        rail = QVBoxLayout()
        rail.setSpacing(style.SP_XS)
        for act in file_actions:
            btn = QToolButton()
            btn.setObjectName("backstageCmd")
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            btn.setIcon(act.icon())
            btn.setText(act.text().replace("&", ""))    # full label, no mnemonic
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Fixed)
            # close first so any dialog the action opens is unobstructed
            btn.clicked.connect(lambda _=False, a=act: (self.close_panel(),
                                                        a.trigger()))
            rail.addWidget(btn)
        rail.addStretch(1)
        rail_box = QWidget()
        rail_box.setObjectName("backstageRail")
        rail_box.setLayout(rail)
        rail_box.setFixedWidth(240)
        body.addWidget(rail_box)

        right = QVBoxLayout()
        right.setSpacing(style.SP_LG)
        recent_card = GroupCard("Recent", form=False)
        self._recent = QListWidget()
        self._recent.setObjectName("recentList")
        self._recent.itemActivated.connect(self._emit_recent)
        self._recent.itemClicked.connect(self._emit_recent)
        recent_card.body_layout().addWidget(self._recent)
        self._recent_empty = QLabel("No recent projects yet — Open or Save one.")
        self._recent_empty.setObjectName("hintLabel")
        recent_card.body_layout().addWidget(self._recent_empty)
        right.addWidget(recent_card, 1)

        about = GroupCard("About", form=False)
        arow = QHBoxLayout()
        arow.setSpacing(style.SP_MD)
        if monogram is not None:
            mark = QLabel()
            mark.setPixmap(monogram.pixmap(44, 44))
            arow.addWidget(mark, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(2)
        name = QLabel(product)
        name.setObjectName("h2")
        tag = QLabel("Structural finite-element modelling · preview")
        tag.setObjectName("sub")
        col.addWidget(name)
        col.addWidget(tag)
        arow.addLayout(col)
        arow.addStretch(1)
        about.body_layout().addLayout(arow)
        right.addWidget(about, 0)

        body.addLayout(right, 1)
        root.addLayout(body, 1)

    # ---- recent list -------------------------------------------------------
    def set_recent(self, paths) -> None:
        """(Re)populate the recent list; ``paths`` is newest-first."""
        self._recent.clear()
        for p in paths:
            from pathlib import Path
            item = QListWidgetItem(f"{Path(p).name}      {p}")
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            self._recent.addItem(item)
        has = bool(paths)
        self._recent.setVisible(has)
        self._recent_empty.setVisible(not has)

    def _emit_recent(self, item) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.close_panel()
            self.openRecentRequested.emit(path)

    # ---- open / close ------------------------------------------------------
    def close_panel(self) -> None:
        self.hide()
        self.closed.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close_panel()
        else:
            super().keyPressEvent(event)
