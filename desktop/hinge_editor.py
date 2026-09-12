"""Fiber plastic-hinge editor + member assignment (GUI-3, plan §14 stage 3).

Three dialogs, mirroring the material editor:

* :class:`HingeDialog` edits one :class:`project.Hinge` property — name, whether
  its length is relative (a fraction of the member length) or absolute, and the
  end-I / end-J plastic-hinge lengths.
* :class:`HingeManagerDialog` is the list / add / edit / delete manager (with a
  delete-in-use guard).
* :class:`HingeAssignmentDialog` assigns a defined hinge to the project's fiber
  (Section-Designer) members — the only members a hinge affects.

A member with a hinge assigned compiles to a finite-length
``FiberHingeBeamColumn2D`` (elastic member + fiber plastic hinge at each end)
instead of the distributed fiber element (see ``nonlinear.build_nonlinear_model``).

Pure Qt — headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import copy

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from project import Hinge


def _next_id(ids) -> int:
    return (max(ids) + 1) if ids else 1


def _has_fiber(section) -> bool:
    return bool(getattr(section, "gsd_spec", None))


def _length_text(hinge, unit: str) -> str:
    """Human-readable length spec for the manager table."""
    lp_j = hinge.lp if hinge.lp_j is None else hinge.lp_j
    if hinge.relative:
        body = (f"{hinge.lp:g}·L" if hinge.lp_j is None
                else f"I {hinge.lp:g}·L / J {lp_j:g}·L")
        return f"relative · {body}"
    body = (f"{hinge.lp:g} {unit}" if hinge.lp_j is None
            else f"I {hinge.lp:g} / J {lp_j:g} {unit}")
    return f"absolute · {body}"


class HingeDialog(QDialog):
    def __init__(self, parent, project, hinge=None):
        super().__init__(parent)
        self.setWindowTitle("Edit hinge" if hinge else "Add hinge")
        self._project = project
        form = QFormLayout(self)

        self.id_spin = QSpinBox()
        self.id_spin.setRange(1, 10_000_000)
        self.id_spin.setValue(hinge.id if hinge
                              else _next_id([h.id for h in project.hinges]))
        self.id_spin.setEnabled(hinge is None)
        form.addRow("Hinge id", self.id_spin)

        self.name = QLineEdit(hinge.name if hinge else "Fiber hinge")
        form.addRow("Name", self.name)

        self.relative = QCheckBox("length is a fraction of the member length")
        self.relative.setChecked(hinge.relative if hinge else True)
        self.relative.toggled.connect(self._on_relative)
        form.addRow(self.relative)

        self.lp_i = self._len_spin(hinge.lp if hinge else 0.1)
        form.addRow("Hinge length I", self.lp_i)

        self.symmetric = QCheckBox("end J same as end I")
        self.symmetric.setChecked(hinge.lp_j is None if hinge else True)
        self.symmetric.toggled.connect(self._on_symmetric)
        form.addRow(self.symmetric)

        lp_j0 = (hinge.lp_j if (hinge and hinge.lp_j is not None)
                 else (hinge.lp if hinge else 0.1))
        self.lp_j = self._len_spin(lp_j0)
        self.lp_j_row = QLabel("Hinge length J")
        form.addRow(self.lp_j_row, self.lp_j)

        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray;")
        form.addRow(self.hint)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

        self._on_relative(self.relative.isChecked())
        self._on_symmetric(self.symmetric.isChecked())

    @staticmethod
    def _len_spin(value: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setDecimals(4)
        s.setRange(0.0, 1.0e6)
        s.setValue(value)
        return s

    def _on_relative(self, rel: bool) -> None:
        unit = self._project.length_unit
        for s in (self.lp_i, self.lp_j):
            s.setSuffix("" if rel else f" {unit}")
            s.setSingleStep(0.05 if rel else 0.1)
        self.hint.setText(
            "Relative: each end hinge spans this fraction of a member's length "
            "(e.g. 0.1 → 10%). Typical 0.05–0.15."
            if rel else
            f"Absolute: each end hinge is this many {unit}. Must satisfy "
            f"lp(I) + lp(J) < the member length.")

    def _on_symmetric(self, sym: bool) -> None:
        self.lp_j.setEnabled(not sym)
        self.lp_j_row.setEnabled(not sym)
        if sym:
            self.lp_j.setValue(self.lp_i.value())

    def accept(self) -> None:
        if self.symmetric.isChecked():
            self.lp_j.setValue(self.lp_i.value())
        super().accept()

    def data(self) -> Hinge:
        return Hinge(
            id=self.id_spin.value(),
            name=self.name.text().strip() or f"Hinge {self.id_spin.value()}",
            lp=float(self.lp_i.value()),
            lp_j=(None if self.symmetric.isChecked()
                  else float(self.lp_j.value())),
            relative=self.relative.isChecked())

    @classmethod
    def edit(cls, parent, project, hinge=None):
        dlg = cls(parent, project, hinge)
        return dlg.data() if dlg.exec() else None


class HingeManagerDialog(QDialog):
    """List / add / edit / delete hinge properties. Returns the edited list via
    :meth:`result_hinges` after a successful close."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Hinges")
        self._project = project
        self._hinges = copy.deepcopy(project.hinges)
        v = QVBoxLayout(self)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["id", "name", "length"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumSize(440, 220)
        v.addWidget(self.table)

        row = QHBoxLayout()
        for label, cb in (("Add…", self._add), ("Edit…", self._edit),
                          ("Delete", self._delete)):
            b = QPushButton(label)
            b.clicked.connect(cb)
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)
        self._refresh()

    def _refresh(self) -> None:
        unit = self._project.length_unit
        self.table.setRowCount(len(self._hinges))
        for r, h in enumerate(self._hinges):
            self.table.setItem(r, 0, QTableWidgetItem(str(h.id)))
            self.table.setItem(r, 1, QTableWidgetItem(h.name))
            self.table.setItem(r, 2, QTableWidgetItem(_length_text(h, unit)))

    def _proxy_project(self):
        proxy = copy.copy(self._project)
        proxy.hinges = self._hinges
        return proxy

    def _add(self) -> None:
        h = HingeDialog.edit(self, self._proxy_project())
        if h is None:
            return
        if any(x.id == h.id for x in self._hinges):
            QMessageBox.warning(self, "Duplicate",
                                f"Hinge {h.id} already exists.")
            return
        self._hinges.append(h)
        self._refresh()

    def _edit(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return
        h = HingeDialog.edit(self, self._proxy_project(), self._hinges[r])
        if h is not None:
            self._hinges[r] = h
            self._refresh()

    def _delete(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return
        hid = self._hinges[r].id
        used = [mb.id for mb in self._project.members
                if getattr(mb, "hinge", None) == hid]
        if used:
            QMessageBox.warning(self, "In use",
                                f"Hinge {hid} is assigned to member(s) "
                                f"{', '.join(map(str, used))}.")
            return
        del self._hinges[r]
        self._refresh()

    def result_hinges(self) -> list:
        return self._hinges

    @classmethod
    def manage(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result_hinges() if dlg.exec() else None


class HingeAssignmentDialog(QDialog):
    """Assign a defined hinge to each fiber (Section-Designer) member. Returns
    ``{member_id: hinge_id | None}`` via :meth:`result_assignments`."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Assign hinges")
        self._project = project
        secs = {s.id: s for s in project.sections}
        self._members = [mb for mb in project.members
                         if _has_fiber(secs.get(mb.section))]
        v = QVBoxLayout(self)

        if not project.hinges:
            v.addWidget(QLabel("No hinge properties defined yet — "
                               "use Hinges… to add one first."))
        elif not self._members:
            v.addWidget(QLabel("No fiber (Section Designer) members to assign "
                               "a hinge to."))

        self.table = QTableWidget(len(self._members), 3)
        self.table.setHorizontalHeaderLabels(["member", "section", "hinge"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumSize(460, 240)
        self._combos: dict[int, QComboBox] = {}
        for r, mb in enumerate(self._members):
            sec = secs.get(mb.section)
            self.table.setItem(r, 0, QTableWidgetItem(str(mb.id)))
            self.table.setItem(r, 1, QTableWidgetItem(
                f"{mb.section}: {getattr(sec, 'name', '')}"))
            combo = QComboBox()
            combo.addItem("— none —", None)
            for h in project.hinges:
                combo.addItem(f"{h.id}: {h.name}", h.id)
            cur = combo.findData(getattr(mb, "hinge", None))
            combo.setCurrentIndex(cur if cur >= 0 else 0)
            self.table.setCellWidget(r, 2, combo)
            self._combos[mb.id] = combo
        v.addWidget(self.table)

        # quick "set all" helper
        if project.hinges and self._members:
            row = QHBoxLayout()
            self._all_combo = QComboBox()
            self._all_combo.addItem("— none —", None)
            for h in project.hinges:
                self._all_combo.addItem(f"{h.id}: {h.name}", h.id)
            apply_all = QPushButton("Set all to")
            apply_all.clicked.connect(self._set_all)
            row.addStretch(1)
            row.addWidget(apply_all)
            row.addWidget(self._all_combo)
            v.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        v.addWidget(btns)

    def _set_all(self) -> None:
        hid = self._all_combo.currentData()
        for combo in self._combos.values():
            i = combo.findData(hid)
            combo.setCurrentIndex(i if i >= 0 else 0)

    def result_assignments(self) -> dict:
        return {mid: combo.currentData()
                for mid, combo in self._combos.items()}

    @classmethod
    def assign(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result_assignments() if dlg.exec() else None
