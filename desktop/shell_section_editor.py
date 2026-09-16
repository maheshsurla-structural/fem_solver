"""Thickness / shell-section editor (slab plan S4).

The surface-element analogue of the beam ``SectionDialog``: define the
through-thickness *thickness properties* an ``Area`` (slab / wall / shell)
references. ``ShellSectionDialog`` edits one :class:`project.ShellSection`
(id / name / formulation kind / thickness + SAP-style stiffness modifiers);
``ShellSectionManagerDialog`` is the list / add / edit / delete manager, wired
into the Home ribbon next to Sections and Materials.

Pure Qt (no solver / OpenGL) so it is headless-constructible under
``QT_QPA_PLATFORM=offscreen``. The ``.data()`` / ``.edit()`` / ``.manage()``
contracts mirror the other property managers.
"""
from __future__ import annotations

import copy

from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

import style
from analysis_ui import dialog_buttons
from editing import _id_spin, _next_id
from project import (SHELL_KIND_LABELS, SHELL_KINDS, ShellSection)
from unit_widgets import UnitSpin, labeled
from units import Quantity, UnitSystem

# Stiffness modifiers exposed in the editor, grouped for the UI. Any factor left
# at 1.0 is not stored (``project._has_modifiers`` / ``_ModifiedShellSection``
# treat a missing key as 1.0), keeping saved files tidy.
_MODIFIER_GROUPS = (
    ("Membrane (f)", ("f11", "f22", "f12")),
    ("Bending (m)", ("m11", "m22", "m12")),
    ("Shear (v)", ("v13", "v23")),
)


def _modifier_spin(value: float) -> QDoubleSpinBox:
    sp = QDoubleSpinBox()
    sp.setRange(0.0, 1000.0)
    sp.setDecimals(3)
    sp.setSingleStep(0.05)
    sp.setValue(float(value))
    return sp


class ShellSectionDialog(QDialog):
    """Edit one :class:`project.ShellSection`. Pass an existing section to edit,
    or ``None`` to add (id pre-filled with the next free value)."""

    def __init__(self, parent, project, section=None):
        super().__init__(parent)
        self.setWindowTitle("Edit thickness" if section else "Add thickness")
        self._us = UnitSystem.from_project(project)
        root = QVBoxLayout(self)

        form = QFormLayout()
        self.id_spin = _id_spin(
            section.id if section
            else _next_id([s.id for s in project.shell_sections]),
            editing=section is not None)
        form.addRow("Section id", self.id_spin)

        self.name = QLineEdit(section.name if section else "")
        form.addRow("Name", self.name)

        self.kind = QComboBox()
        for k in SHELL_KINDS:
            self.kind.addItem(SHELL_KIND_LABELS.get(k, k), k)
        if section:
            idx = self.kind.findData(section.kind)
            if idx >= 0:
                self.kind.setCurrentIndex(idx)
        form.addRow("Type", self.kind)

        self.thickness = UnitSpin(Quantity.LENGTH, self._us,
                                  si=section.thickness if section else 0.20,
                                  decimals=4, step=0.01, rng=(0.0, 1.0e6))
        form.addRow(labeled("Thickness", self.thickness), self.thickness)
        root.addLayout(form)

        # ---- stiffness modifiers (SAP-style) ------------------------------
        box = QGroupBox("Stiffness modifiers")
        grid = QGridLayout(box)
        mods = dict(getattr(section, "modifiers", {}) or {}) if section else {}
        self._mod_spins: dict[str, QDoubleSpinBox] = {}
        row = 0
        for title, keys in _MODIFIER_GROUPS:
            grid.addWidget(QLabel(title), row, 0)
            for col, key in enumerate(keys, start=1):
                sp = _modifier_spin(mods.get(key, 1.0))
                self._mod_spins[key] = sp
                cell = QWidget()
                h = QHBoxLayout(cell)
                h.setContentsMargins(0, 0, 0, 0)
                h.addWidget(QLabel(key))
                h.addWidget(sp)
                grid.addWidget(cell, row, col)
            row += 1
        root.addWidget(box)

        root.addWidget(dialog_buttons(self))
        style.apply(self)

    def data(self) -> ShellSection:
        name = (self.name.text().strip()
                or f"Thickness {self.id_spin.value()}")
        mods = {k: sp.value() for k, sp in self._mod_spins.items()
                if abs(sp.value() - 1.0) > 1e-9}
        return ShellSection(
            id=self.id_spin.value(), name=name,
            thickness=self.thickness.si_value(),
            kind=self.kind.currentData(), modifiers=mods)

    @classmethod
    def edit(cls, parent, project, section=None):
        dlg = cls(parent, project, section)
        return dlg.data() if dlg.exec() else None


class ShellSectionManagerDialog(QDialog):
    """List / add / edit / delete the project's thickness (shell) sections.
    Returns the edited list via :meth:`result_sections` on a successful close."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Thickness sections")
        self._project = project
        self._sections = copy.deepcopy(project.shell_sections)
        self._us = UnitSystem.from_project(project)

        v = QVBoxLayout(self)
        v.setContentsMargins(style.SP_LG, style.SP_LG, style.SP_LG, style.SP_LG)
        v.setSpacing(style.SP_SM)

        head = QLabel("Thickness / shell sections")
        head.setObjectName("h2")
        v.addWidget(head)
        sub = QLabel("Through-thickness properties for slabs, walls and shells "
                     "(assigned to area objects).")
        sub.setObjectName("sub")
        v.addWidget(sub)

        body = QHBoxLayout()
        body.setSpacing(style.SP_MD)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["id", "name", "type", "thickness"])
        self.table.setMinimumSize(480, 240)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(lambda *_: self._edit())
        body.addWidget(self.table, 1)

        col = QVBoxLayout()
        col.setSpacing(style.SP_SM)
        for label, cb in (("Add…", self._add), ("Edit…", self._edit),
                          ("Delete", self._delete)):
            b = QPushButton(label)
            b.clicked.connect(cb)
            col.addWidget(b)
        col.addStretch(1)
        body.addLayout(col)
        v.addLayout(body)

        v.addWidget(dialog_buttons(self))
        style.apply(self)
        self._refresh()

    def _refresh(self) -> None:
        self.table.setRowCount(len(self._sections))
        for r, s in enumerate(self._sections):
            self.table.setItem(r, 0, QTableWidgetItem(str(s.id)))
            self.table.setItem(r, 1, QTableWidgetItem(s.name))
            self.table.setItem(r, 2, QTableWidgetItem(
                SHELL_KIND_LABELS.get(s.kind, s.kind)))
            t_disp = self._us.to_display(s.thickness, Quantity.LENGTH)
            self.table.setItem(r, 3, QTableWidgetItem(
                f"{t_disp:g} {self._us.label(Quantity.LENGTH)}"))

    def _selected_row(self) -> int:
        return self.table.currentRow()

    def _proxy_project(self):
        # ShellSectionDialog only reads .shell_sections (for id defaults) and
        # units — hand it a shallow proxy carrying our working list.
        proxy = copy.copy(self._project)
        proxy.shell_sections = self._sections
        return proxy

    def _add(self) -> None:
        sec = ShellSectionDialog.edit(self, self._proxy_project())
        if sec is None:
            return
        if any(s.id == sec.id for s in self._sections):
            QMessageBox.warning(self, "Duplicate",
                                f"Thickness {sec.id} already exists.")
            return
        self._sections.append(sec)
        self._refresh()

    def _edit(self) -> None:
        r = self._selected_row()
        if r < 0:
            return
        sec = ShellSectionDialog.edit(self, self._proxy_project(),
                                      self._sections[r])
        if sec is not None:
            self._sections[r] = sec
            self._refresh()

    def _delete(self) -> None:
        r = self._selected_row()
        if r < 0:
            return
        sid = self._sections[r].id
        used = [a.id for a in getattr(self._project, "areas", [])
                if a.shell_section == sid]
        if used:
            QMessageBox.warning(self, "In use",
                                f"Thickness {sid} is used by area(s) "
                                f"{', '.join(map(str, used))}.")
            return
        del self._sections[r]
        self._refresh()

    def result_sections(self) -> list:
        return self._sections

    @classmethod
    def manage(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result_sections() if dlg.exec() else None
