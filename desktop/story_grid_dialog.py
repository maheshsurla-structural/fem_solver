"""Story & grid manager (wall plan W4).

Defines the building's stories (named levels at an elevation, with a height) and
grid lines (named X/Y reference lines) — the ETABS modeling context that later
slices pier forces by story, labels the tree and (W4b/c) snaps drawing and
replicates similar stories. Pure Qt; edits two small tables and returns the
resulting ``Story`` / ``GridLine`` lists.

Elevations and coordinates are shown/entered in the project's length unit and
stored SI.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QColorDialog, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

import style
from project import GridLine, Story
from units import Quantity, UnitSystem


class StoryGridDialog(QDialog):
    """Edit the project's stories and grid lines."""

    def __init__(self, parent, project, units: UnitSystem | None = None):
        super().__init__(parent)
        self.setWindowTitle("Stories & grid")
        self.resize(560, 560)
        self._project = project
        self._us = units or UnitSystem()
        self._lu = self._us.label(Quantity.LENGTH)
        self.result = None

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_SM)

        # ---- stories ----
        root.addWidget(self._h2("Stories"))
        self._relinking = False        # re-entrancy guard for height↔elev linking
        self.st_tbl = QTableWidget(0, 5)
        self.st_tbl.setHorizontalHeaderLabels(
            ["Name", f"Elevation ({self._lu})", f"Height ({self._lu})",
             "Similar to", "Color"])
        self.st_tbl.verticalHeader().setVisible(False)
        self.st_tbl.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.st_tbl.itemChanged.connect(self._on_story_item)
        self.st_tbl.cellDoubleClicked.connect(self._on_story_double_click)
        root.addWidget(self.st_tbl)
        row = QHBoxLayout()
        for label, slot in (("＋ Story", self._add_story),
                            ("Remove", self._remove_story),
                            ("Generate…", self._generate)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        root.addLayout(row)

        # ---- grid ----
        root.addWidget(self._h2("Grid lines"))
        self.gr_tbl = QTableWidget(0, 3)
        self.gr_tbl.setHorizontalHeaderLabels(
            ["Name", "Axis", f"Coordinate ({self._lu})"])
        self.gr_tbl.verticalHeader().setVisible(False)
        self.gr_tbl.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self.gr_tbl)
        grow = QHBoxLayout()
        for label, slot in (("＋ Grid line", self._add_grid),
                            ("Remove", self._remove_grid)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            grow.addWidget(b)
        grow.addStretch(1)
        root.addLayout(grow)

        hint = QLabel("Stories are levels at an elevation; a node/area belongs "
                      "to a story by its height. Grid line axis 'x' is a line "
                      "of constant X (runs in Y); 'y' is constant Y.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)
        style.apply(self)

        self._load()

    def _h2(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("h2")
        return lbl

    # ------------------------------------------------------------- load / rows
    def _load(self) -> None:
        us = self._us
        stories = self._project.stories_sorted()         # base → top (row 0 = base)
        self._relinking = True
        self.st_tbl.setRowCount(len(stories))
        for r, s in enumerate(stories):
            self.st_tbl.setItem(r, 0, QTableWidgetItem(s.name))
            self.st_tbl.setItem(r, 1, QTableWidgetItem(
                f"{us.to_display(s.elev, Quantity.LENGTH):.4g}"))
            self.st_tbl.setItem(r, 2, QTableWidgetItem(
                f"{us.to_display(s.height, Quantity.LENGTH):.4g}"))
            self._set_color_cell(r, s.color)
        self._relinking = False
        self._rebuild_similar_combos()
        grids = sorted(self._project.grid_lines, key=lambda g: (g.axis, g.coord))
        self.gr_tbl.setRowCount(len(grids))
        for r, g in enumerate(grids):
            self.gr_tbl.setItem(r, 0, QTableWidgetItem(g.name))
            self.gr_tbl.setCellWidget(r, 1, self._axis_combo(g.axis))
            self.gr_tbl.setItem(r, 2, QTableWidgetItem(
                f"{us.to_display(g.coord, Quantity.LENGTH):.4g}"))

    def _axis_combo(self, axis="x"):
        c = QComboBox()
        c.addItem("x (const X)", "x")
        c.addItem("y (const Y)", "y")
        c.setCurrentIndex(0 if axis == "x" else 1)
        return c

    def _add_story(self) -> None:
        """Add a story on top of the current stack, keeping existing heights —
        its elevation follows from the level below plus a typical height."""
        r = self.st_tbl.rowCount()
        self._relinking = True
        self.st_tbl.insertRow(r)
        below_elev = 0.0
        if r > 0:
            it = self.st_tbl.item(r - 1, 1)
            try:
                below_elev = float(it.text()) if it else 0.0
            except ValueError:
                below_elev = 0.0
        h = 3.0 if r > 0 else 0.0                        # base has no height
        self.st_tbl.setItem(r, 0, QTableWidgetItem(f"Story {r}" if r else "Base"))
        self.st_tbl.setItem(r, 1, QTableWidgetItem(f"{below_elev + h:g}"))
        self.st_tbl.setItem(r, 2, QTableWidgetItem(f"{h:g}"))
        self._set_color_cell(r, None)
        self._relinking = False
        self._rebuild_similar_combos()

    def _remove_story(self) -> None:
        r = self.st_tbl.currentRow()
        if r >= 0:
            self.st_tbl.removeRow(r)
            self._rebuild_similar_combos()

    # ------------------------------------------------ height ↔ elevation linking
    def _story_num(self, r, c):
        it = self.st_tbl.item(r, c)
        try:
            return float(it.text()) if it and it.text() else 0.0
        except ValueError:
            return 0.0

    def _set_num(self, r, c, v):
        self.st_tbl.setItem(r, c, QTableWidgetItem(f"{v:g}"))

    def _on_story_item(self, item) -> None:
        """Keep Height and Elevation consistent (base-anchored). Editing a
        height recomputes the elevations above (keep-heights); editing an
        elevation recomputes the adjacent heights (keep-elevations); moving the
        base elevation shifts the whole stack."""
        if self._relinking:
            return
        col = item.column()
        if col == 0:                                     # a rename → refresh combos
            self._rebuild_similar_combos()
            return
        if col not in (1, 2):
            return
        self._relinking = True
        try:
            n = self.st_tbl.rowCount()
            E = [self._story_num(r, 1) for r in range(n)]
            H = [self._story_num(r, 2) for r in range(n)]
            if n:
                H[0] = 0.0                               # base has no height
            row = item.row()
            if col == 2 and row >= 1:                    # keep heights
                for i in range(1, n):
                    E[i] = E[i - 1] + H[i]
            elif col == 1 and row == 0:                  # move the datum
                for i in range(1, n):
                    E[i] = E[i - 1] + H[i]
            elif col == 1:                               # keep elevations
                H[row] = E[row] - E[row - 1]
                if row + 1 < n:
                    H[row + 1] = E[row + 1] - E[row]
            for i in range(n):
                self._set_num(i, 1, E[i])
                self._set_num(i, 2, H[i])
        finally:
            self._relinking = False

    # ------------------------------------------------------------ similar-to
    def _rebuild_similar_combos(self) -> None:
        """A 'Similar to' dropdown per row: pick another story as this row's
        master (drives the W7 similar-story draw scope). Preserves selections
        by name across a rebuild."""
        names = [self.st_tbl.item(r, 0).text().strip()
                 if self.st_tbl.item(r, 0) else "" for r in range(
                     self.st_tbl.rowCount())]
        for r in range(self.st_tbl.rowCount()):
            prev = None
            w = self.st_tbl.cellWidget(r, 3)
            if isinstance(w, QComboBox):
                prev = w.currentData()
            combo = QComboBox()
            combo.addItem("— none —", None)
            for j, nm in enumerate(names):
                if j != r and nm:
                    combo.addItem(nm, nm)
            i = combo.findData(prev)
            combo.setCurrentIndex(i if i >= 0 else 0)
            self.st_tbl.setCellWidget(r, 3, combo)

    def _set_similar(self, r, master):
        combo = self.st_tbl.cellWidget(r, 3)
        if isinstance(combo, QComboBox):
            i = combo.findData(master)
            combo.setCurrentIndex(i if i >= 0 else 0)

    # ------------------------------------------------------------ color
    def _set_color_cell(self, r, color) -> None:
        it = QTableWidgetItem("")
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if color:
            it.setBackground(QBrush(QColor(color)))
            it.setData(Qt.ItemDataRole.UserRole, color)
        self.st_tbl.setItem(r, 4, it)

    def _on_story_double_click(self, r, c) -> None:
        if c != 4:
            return
        it = self.st_tbl.item(r, 4)
        cur = it.data(Qt.ItemDataRole.UserRole) if it else None
        col = QColorDialog.getColor(QColor(cur) if cur else QColor("#888888"),
                                    self, "Story color")
        if col.isValid():
            self._set_color_cell(r, col.name())

    def _add_grid(self) -> None:
        r = self.gr_tbl.rowCount()
        self.gr_tbl.insertRow(r)
        self.gr_tbl.setItem(r, 0, QTableWidgetItem(chr(ord("A") + r)))
        self.gr_tbl.setCellWidget(r, 1, self._axis_combo("x"))
        self.gr_tbl.setItem(r, 2, QTableWidgetItem("0"))

    def _remove_grid(self) -> None:
        r = self.gr_tbl.currentRow()
        if r >= 0:
            self.gr_tbl.removeRow(r)

    def _generate(self) -> None:
        """Quick stack: base elevation + N stories of a typical height."""
        n, ok = QInputDialog.getInt(self, "Generate stories",
                                    "Number of stories above base:", 3, 1, 200)
        if not ok:
            return
        h, ok = QInputDialog.getDouble(
            self, "Generate stories", f"Typical story height ({self._lu}):",
            3.0, 0.1, 1000.0, 2)
        if not ok:
            return
        h_si = self._us.to_si(h, Quantity.LENGTH)
        self.stories_from_stack(0.0, [h_si] * n)
        self._load()

    def stories_from_stack(self, base_elev: float, heights: list) -> None:
        """Populate the project stories from a base elevation and a list of
        story heights (SI, bottom→top): Base at ``base_elev`` then one level per
        height. Replaces the current stories."""
        stories = [Story(id=1, name="Base", elev=float(base_elev), height=0.0)]
        z = float(base_elev)
        for i, h in enumerate(heights, start=1):
            z += float(h)
            stories.append(Story(id=i + 1, name=f"Story {i}", elev=z,
                                 height=float(h)))
        self._project.stories = stories

    # ------------------------------------------------------------- accept
    def _accept(self) -> None:
        us = self._us

        def _f(item, default=0.0):
            try:
                return us.to_si(float(item.text()), Quantity.LENGTH)
            except (AttributeError, ValueError):
                return default

        stories = []
        for r in range(self.st_tbl.rowCount()):
            name_it = self.st_tbl.item(r, 0)
            name = name_it.text().strip() if name_it else ""
            if not name:
                continue
            combo = self.st_tbl.cellWidget(r, 3)
            master = combo.currentData() if isinstance(combo, QComboBox) else None
            color_it = self.st_tbl.item(r, 4)
            color = color_it.data(Qt.ItemDataRole.UserRole) if color_it else None
            stories.append(Story(id=r + 1, name=name,
                                 elev=_f(self.st_tbl.item(r, 1)),
                                 height=_f(self.st_tbl.item(r, 2)),
                                 master=master, color=color))
        grids = []
        for r in range(self.gr_tbl.rowCount()):
            name_it = self.gr_tbl.item(r, 0)
            name = name_it.text().strip() if name_it else ""
            if not name:
                continue
            combo = self.gr_tbl.cellWidget(r, 1)
            axis = combo.currentData() if combo else "x"
            grids.append(GridLine(id=r + 1, name=name, axis=axis,
                                  coord=_f(self.gr_tbl.item(r, 2))))
        self.result = (stories, grids)
        self.accept()

    @classmethod
    def manage(cls, parent, project, units: UnitSystem | None = None):
        dlg = cls(parent, project, units)
        return dlg.result if dlg.exec() else None
