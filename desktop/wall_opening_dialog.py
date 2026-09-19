"""Wall openings dialog (wall plan W5).

Add / remove rectangular openings (doors, windows) in a wall panel. Openings are
entered in physical dimensions — distance from the panel's bottom-left corner
plus a width and height — and stored as fractions ``(u0, v0, u1, v1)`` of the
panel (``u`` along the base nodes[0]→nodes[1], ``v`` up nodes[0]→nodes[3]).
Mesh cells inside an opening are dropped at build time.
"""
from __future__ import annotations

import math

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout,
                               QHeaderView, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

import style
from units import Quantity, UnitSystem


def _dist(a, b):
    return math.dist(a, b)


class WallOpeningDialog(QDialog):
    """Edit the rectangular openings of one wall ``Area``."""

    def __init__(self, parent, project, area, units: UnitSystem | None = None):
        super().__init__(parent)
        self.setWindowTitle("Wall openings")
        self.resize(480, 420)
        self._project = project
        self._area = area
        self._us = units or UnitSystem()
        self._lu = self._us.label(Quantity.LENGTH)
        self.result = None

        # panel dimensions from the quad corners: L = |n0→n1|, H = |n0→n3|
        c = {n.id: (n.x, n.y, n.z) for n in project.nodes}
        n = area.nodes
        self._L = _dist(c[n[0]], c[n[1]]) if len(n) >= 2 else 1.0
        self._H = _dist(c[n[0]], c[n[3]]) if len(n) >= 4 else 1.0

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        head = QLabel("Wall openings")
        head.setObjectName("h2")
        root.addWidget(head)
        us = self._us
        sub = QLabel(f"Panel {us.to_display(self._L, Quantity.LENGTH):.3g} × "
                     f"{us.to_display(self._H, Quantity.LENGTH):.3g} {self._lu} "
                     f"(width × height). Offsets from the bottom-left corner.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(
            [f"X ({self._lu})", f"Z ({self._lu})",
             f"Width ({self._lu})", f"Height ({self._lu})"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.tbl, 1)

        row = QHBoxLayout()
        add = QPushButton("＋ Opening")
        add.clicked.connect(self._add)
        rem = QPushButton("Remove")
        rem.clicked.connect(self._remove)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        root.addLayout(row)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)
        style.apply(self)
        self._load()

    def _load(self) -> None:
        us = self._us
        from project import _opening_is_rect
        rects = [o for o in self._area.openings if _opening_is_rect(o)]
        # polygon openings are preserved untouched (this table edits rectangles)
        self._polygons = [o for o in self._area.openings
                          if not _opening_is_rect(o)]
        self.tbl.setRowCount(len(rects))
        for r, (u0, v0, u1, v1) in enumerate(rects):
            vals = [u0 * self._L, v0 * self._H,
                    (u1 - u0) * self._L, (v1 - v0) * self._H]
            for cci, v in enumerate(vals):
                self.tbl.setItem(r, cci, QTableWidgetItem(
                    f"{us.to_display(v, Quantity.LENGTH):.4g}"))

    def _add(self) -> None:
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        us = self._us
        # a default window: quarter-panel, offset in a bit
        defs = [0.25 * self._L, 0.25 * self._H, 0.5 * self._L, 0.5 * self._H]
        for cci, v in enumerate(defs):
            self.tbl.setItem(r, cci, QTableWidgetItem(
                f"{us.to_display(v, Quantity.LENGTH):.4g}"))

    def _remove(self) -> None:
        r = self.tbl.currentRow()
        if r >= 0:
            self.tbl.removeRow(r)

    def _accept(self) -> None:
        us = self._us

        def _si(item):
            try:
                return us.to_si(float(item.text()), Quantity.LENGTH)
            except (AttributeError, ValueError):
                return 0.0

        openings = []
        for r in range(self.tbl.rowCount()):
            x = _si(self.tbl.item(r, 0))
            z = _si(self.tbl.item(r, 1))
            w = _si(self.tbl.item(r, 2))
            h = _si(self.tbl.item(r, 3))
            if w <= 0 or h <= 0 or self._L <= 0 or self._H <= 0:
                continue
            openings.append((x / self._L, z / self._H,
                             (x + w) / self._L, (z + h) / self._H))
        # keep any polygon openings the table doesn't edit
        openings.extend(getattr(self, "_polygons", []))
        self.result = openings
        self.accept()

    @classmethod
    def edit(cls, parent, project, area, units: UnitSystem | None = None):
        dlg = cls(parent, project, area, units)
        return dlg.result if dlg.exec() else None
