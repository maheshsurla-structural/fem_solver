"""Construction-stage manager (bridge GUI plan G3).

``StageManagerDialog`` defines the construction sequence: an ordered list of
stages, each naming the members that become active ("born") in it. Members not
assigned to any stage are treated as built before the sequence (initially
active — e.g. piers). An **auto-sequence** button partitions the members into
N stages left-to-right for quick set-up.

The owning window feeds the sequence to an incremental staged analysis
(:class:`femsolver.bridges.IncrementalStagedAnalysis`) under member self-weight
and reports the camber (:func:`femsolver.bridges.staged_camber`).

Headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

import copy

from PySide6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

import style
from analysis_ui import GroupCard, dialog_buttons
from project import Stage

_ROLE = 0x0100                                          # Qt.UserRole


class StageManagerDialog(QDialog):
    """List / add / delete / order construction stages + assign members."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Construction stages")
        self._project = project
        self._stages = copy.deepcopy(project.stages)
        self.resize(640, 460)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)
        head = QLabel("Construction sequence")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel("Each stage names the members built in it, in order. "
                     "Unassigned members are built first. Self-weight is applied "
                     "as each stage is erected.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        body = QHBoxLayout()
        body.setSpacing(style.SP_MD)

        # ---- left: stage list + controls ----
        left = QVBoxLayout()
        self.stage_list = QListWidget()
        self.stage_list.currentRowChanged.connect(self._on_stage_selected)
        left.addWidget(self.stage_list, 1)
        row = QHBoxLayout()
        for label, cb in (("Add", self._add), ("Delete", self._delete),
                          ("↑", self._up), ("↓", self._down)):
            b = QPushButton(label)
            b.clicked.connect(cb)
            row.addWidget(b)
        left.addLayout(row)
        auto = QHBoxLayout()
        auto.addWidget(QLabel("Auto:"))
        self.nseg = QSpinBox()
        self.nseg.setRange(1, 200)
        self.nseg.setValue(min(4, max(1, len(project.members))))
        auto.addWidget(self.nseg)
        ab = QPushButton("Sequence L→R")
        ab.clicked.connect(self._auto_sequence)
        auto.addWidget(ab)
        auto.addStretch(1)
        left.addLayout(auto)
        lw = QWidget()
        lw.setLayout(left)
        body.addWidget(lw, 1)

        # ---- right: selected-stage editor ----
        editor = GroupCard("Selected stage", form=False)
        self.name = QLineEdit()
        self.name.setPlaceholderText("stage name")
        self.name.textEdited.connect(self._on_name_edited)
        editor.body_layout().addWidget(QLabel("Name:"))
        editor.body_layout().addWidget(self.name)
        editor.body_layout().addWidget(QLabel("Members built in this stage:"))
        self.members = QListWidget()
        self.members.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        for mb in project.members:
            it = QListWidgetItem(f"member {mb.id}  ({mb.n1}→{mb.n2})")
            it.setData(_ROLE, mb.id)
            self.members.addItem(it)
        self.members.itemSelectionChanged.connect(self._on_members_changed)
        editor.body_layout().addWidget(self.members, 1)
        body.addWidget(editor, 1)
        root.addLayout(body, 1)

        root.addWidget(dialog_buttons(self))
        style.apply(self)
        self._refresh_stage_list()
        if self._stages:
            self.stage_list.setCurrentRow(0)
        else:
            self._set_editor_enabled(False)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _midx(project, member) -> float:
        nodes = {n.id: n for n in project.nodes}
        a, b = nodes.get(member.n1), nodes.get(member.n2)
        if a is None or b is None:
            return 0.0
        return 0.5 * (a.x + b.x)

    def _next_id(self) -> int:
        return (max((s.id for s in self._stages), default=0) + 1)

    def _current(self):
        r = self.stage_list.currentRow()
        return self._stages[r] if 0 <= r < len(self._stages) else None

    def _set_editor_enabled(self, on: bool) -> None:
        self.name.setEnabled(on)
        self.members.setEnabled(on)

    def _refresh_stage_list(self) -> None:
        self.stage_list.blockSignals(True)
        self.stage_list.clear()
        for i, s in enumerate(self._stages):
            self.stage_list.addItem(
                f"{i + 1}. {s.name}  ({len(s.add_members)} members)")
        self.stage_list.blockSignals(True)
        self.stage_list.blockSignals(False)

    # ------------------------------------------------------------- selection
    def _on_stage_selected(self, row: int) -> None:
        s = self._current()
        self._set_editor_enabled(s is not None)
        self.name.blockSignals(True)
        self.members.blockSignals(True)
        self.name.setText(s.name if s else "")
        ids = set(s.add_members) if s else set()
        for i in range(self.members.count()):
            it = self.members.item(i)
            it.setSelected(it.data(_ROLE) in ids)
        self.name.blockSignals(False)
        self.members.blockSignals(False)

    def _on_name_edited(self, text: str) -> None:
        s = self._current()
        if s:
            s.name = text
            self._refresh_keep_row()

    def _on_members_changed(self) -> None:
        s = self._current()
        if s is None:
            return
        chosen = [self.members.item(i).data(_ROLE)
                  for i in range(self.members.count())
                  if self.members.item(i).isSelected()]
        s.add_members = chosen
        # exclusivity: a member belongs to at most one stage
        for other in self._stages:
            if other is not s:
                other.add_members = [m for m in other.add_members
                                     if m not in chosen]
        self._refresh_keep_row()

    def _refresh_keep_row(self) -> None:
        r = self.stage_list.currentRow()
        self._refresh_stage_list()
        self.stage_list.blockSignals(True)
        self.stage_list.setCurrentRow(r)
        self.stage_list.blockSignals(False)

    # --------------------------------------------------------------- actions
    def _add(self) -> None:
        self._stages.append(Stage(id=self._next_id(),
                                  name=f"Stage {len(self._stages) + 1}"))
        self._refresh_stage_list()
        self.stage_list.setCurrentRow(len(self._stages) - 1)

    def _delete(self) -> None:
        r = self.stage_list.currentRow()
        if 0 <= r < len(self._stages):
            del self._stages[r]
            self._refresh_stage_list()
            self.stage_list.setCurrentRow(min(r, len(self._stages) - 1))
            if not self._stages:
                self._set_editor_enabled(False)

    def _move(self, delta: int) -> None:
        r = self.stage_list.currentRow()
        j = r + delta
        if 0 <= r < len(self._stages) and 0 <= j < len(self._stages):
            self._stages[r], self._stages[j] = self._stages[j], self._stages[r]
            self._refresh_stage_list()
            self.stage_list.setCurrentRow(j)

    def _up(self) -> None:
        self._move(-1)

    def _down(self) -> None:
        self._move(+1)

    def _auto_sequence(self) -> None:
        n = int(self.nseg.value())
        members = sorted(self._project.members,
                         key=lambda m: self._midx(self._project, m))
        if not members:
            return
        self._stages = []
        # contiguous partition of members into n groups (left → right)
        k = len(members)
        for i in range(n):
            lo = (i * k) // n
            hi = ((i + 1) * k) // n
            grp = [m.id for m in members[lo:hi]]
            if grp:
                self._stages.append(Stage(id=i + 1, name=f"Stage {i + 1}",
                                          add_members=grp))
        self._refresh_stage_list()
        self.stage_list.setCurrentRow(0)

    # ---------------------------------------------------------------- result
    def result(self) -> list:
        return self._stages

    @classmethod
    def manage(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result() if dlg.exec() else None
