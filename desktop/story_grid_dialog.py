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

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QHeaderView, QInputDialog, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

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
        self.st_tbl = QTableWidget(0, 3)
        self.st_tbl.setHorizontalHeaderLabels(
            ["Name", f"Elevation ({self._lu})", f"Height ({self._lu})"])
        self.st_tbl.verticalHeader().setVisible(False)
        self.st_tbl.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
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
        stories = self._project.stories_sorted()
        self.st_tbl.setRowCount(len(stories))
        for r, s in enumerate(stories):
            self.st_tbl.setItem(r, 0, QTableWidgetItem(s.name))
            self.st_tbl.setItem(r, 1, QTableWidgetItem(
                f"{us.to_display(s.elev, Quantity.LENGTH):.4g}"))
            self.st_tbl.setItem(r, 2, QTableWidgetItem(
                f"{us.to_display(s.height, Quantity.LENGTH):.4g}"))
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
        r = self.st_tbl.rowCount()
        self.st_tbl.insertRow(r)
        self.st_tbl.setItem(r, 0, QTableWidgetItem(f"S{r + 1}"))
        self.st_tbl.setItem(r, 1, QTableWidgetItem("0"))
        self.st_tbl.setItem(r, 2, QTableWidgetItem("3"))

    def _remove_story(self) -> None:
        r = self.st_tbl.currentRow()
        if r >= 0:
            self.st_tbl.removeRow(r)

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
            stories.append(Story(id=r + 1, name=name,
                                 elev=_f(self.st_tbl.item(r, 1)),
                                 height=_f(self.st_tbl.item(r, 2))))
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
