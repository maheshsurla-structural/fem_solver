"""Member (line / UDL) load dialog (plan L5).

A :class:`project.MemberLoad` is a uniform transverse line load on a member, in
the member's **local** axes (N/m) — ``wy`` (local-y) always, ``wz`` (local-z)
in 3-D. The data model has carried this for a while, but there was no way to
add or edit one in the GUI; this fills that gap in the grouped scaffold style
(:mod:`analysis_ui`), matching the nodal :class:`editing.LoadDialog`.

Pure Qt, headless-constructible under ``QT_QPA_PLATFORM=offscreen``; ``.edit()``
runs it modally and returns the ``MemberLoad`` (or ``None`` if cancelled).
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout,
                               QLabel, QVBoxLayout, QWidget)

import analysis_ui as ui
import style
from project import MemberLoad


def _combo(entries) -> QComboBox:
    c = QComboBox()
    for label, value in entries:
        c.addItem(label, value)
    return c


def _select(combo: QComboBox, value) -> None:
    i = combo.findData(value)
    if i >= 0:
        combo.setCurrentIndex(i)


def _udl_spin(value: float = 0.0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(-1.0e12, 1.0e12)
    s.setDecimals(3)
    s.setSingleStep(1000.0)
    s.setValue(value)
    return s


def _with_hint(spin: QDoubleSpinBox, hint: str) -> QWidget:
    host = QWidget()
    h = QHBoxLayout(host)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(style.SP_SM)
    h.addWidget(spin, 1)
    lbl = QLabel(hint)
    lbl.setObjectName("hintLabel")
    h.addWidget(lbl)
    return host


class MemberLoadDialog(QDialog):
    """Add / edit a uniform line load on a member."""

    def __init__(self, parent, project, mload=None):
        super().__init__(parent)
        self.setWindowTitle("Edit line load" if mload else "Add line load")
        self._three_d = project.ndm == 3
        unit = f"{project.force_unit}/{project.length_unit}"
        outer = QVBoxLayout(self)
        outer.setContentsMargins(style.SP_LG, style.SP_LG,
                                 style.SP_LG, style.SP_LG)
        outer.setSpacing(style.SP_MD)

        self.member = _combo([(f"{m.id}:  {m.n1} → {m.n2}", m.id)
                              for m in project.members])
        if mload is not None:
            _select(self.member, mload.member)
        self.case = _combo([(c.name, c.id) for c in project.load_cases])
        if mload is not None:
            _select(self.case, getattr(mload, "case", project.default_case_id()))

        applied = ui.GroupCard("Applied to")
        applied.add_row("Member", self.member)
        applied.add_row("Load case", self.case)

        loads = ui.GroupCard(f"Uniform line load  [{unit}]")
        self.wy = _udl_spin(mload.wy if mload else 0.0)
        loads.add_row("w_y  (local y)",
                      _with_hint(self.wy, "+ along local +y"))
        self.wz = None
        if self._three_d:
            self.wz = _udl_spin(mload.wz if mload else 0.0)
            loads.add_row("w_z  (local z)",
                          _with_hint(self.wz, "+ along local +z"))
        note = QLabel("Uniform load in the member's local axes, per unit length.")
        note.setObjectName("hintLabel")
        note.setWordWrap(True)
        loads.add_full_row(note)

        outer.addWidget(applied)
        outer.addWidget(loads)
        outer.addWidget(ui.dialog_buttons(self))
        style.apply(self)

    def data(self) -> MemberLoad:
        return MemberLoad(member=self.member.currentData(),
                          wy=float(self.wy.value()),
                          wz=float(self.wz.value()) if self.wz is not None
                          else 0.0,
                          case=self.case.currentData())

    @classmethod
    def edit(cls, parent, project, mload=None):
        dlg = cls(parent, project, mload)
        return dlg.data() if dlg.exec() else None
