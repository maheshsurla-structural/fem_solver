"""Cable-stayed tuning setup — the *Analysis-cases ▸ Cable Tuning* counterpart
of MIDAS "Unknown Load Factor" / CSiBridge target-force.

The stay pretensions are solved so that, under dead load, chosen deck nodes
reach a target (usually zero) vertical deflection. ``CableTuningDialog`` picks
the **stay members** (the tunable cables) and the **target nodes**; the owning
window runs :func:`femsolver.bridges.unknown_load_factors`.

Note: stays are designated among the model's existing members (the ULF solution
is exact for the model as built). A dedicated pin-ended cable/truss member type
is a future refinement. Headless-constructible under
``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QAbstractItemView, QDialog, QLabel, QListWidget,
                               QListWidgetItem, QVBoxLayout)

import style
from analysis_ui import CaseHeader, GroupCard, dialog_buttons

_ROLE = 0x0100                                          # Qt.UserRole


class CableTuningDialog(QDialog):
    """Pick the stay members to tune and the deck nodes to target."""

    def __init__(self, parent, project, *, initial: dict | None = None,
                 name: str = "Cable Tuning", notes: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Cable tuning")
        self._project = project

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)
        self.header = CaseHeader(name=name, type_label="Cable Tuning",
                                 notes=notes)
        root.addWidget(self.header)
        head = QLabel("Cable-stayed tuning (unknown load factor)")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel("Solves the stay pretensions so the target deck nodes "
                     "reach zero vertical deflection under dead load. Define "
                     "the dead load first.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        cab = GroupCard("Stay cables (members to tune)", form=False)
        self.cables = QListWidget()
        self.cables.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self.cables.setMinimumHeight(90)
        for mb in project.members:
            it = QListWidgetItem(f"member {mb.id}  ({mb.n1}→{mb.n2})")
            it.setData(_ROLE, mb.id)
            self.cables.addItem(it)
        cab.body_layout().addWidget(self.cables)
        root.addWidget(cab)

        tgt = GroupCard("Target nodes (zero vertical deflection)", form=False)
        self.targets = QListWidget()
        self.targets.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self.targets.setMinimumHeight(90)
        for nd in project.nodes:
            it = QListWidgetItem(
                f"node {nd.id}  ({nd.x:g}, {getattr(nd, 'y', 0):g})")
            it.setData(_ROLE, nd.id)
            self.targets.addItem(it)
        tgt.body_layout().addWidget(self.targets)
        root.addWidget(tgt)

        if initial:
            self._seed(initial)
        root.addStretch(1)
        root.addWidget(dialog_buttons(self))
        style.apply(self)

    def _seed(self, p: dict) -> None:
        """Seed the stay + target selections from a saved case's params."""
        cables = set(p.get("cables") or [])
        for i in range(self.cables.count()):
            it = self.cables.item(i)
            it.setSelected(it.data(_ROLE) in cables)
        targets = set(p.get("targets") or [])
        for i in range(self.targets.count()):
            it = self.targets.item(i)
            it.setSelected(it.data(_ROLE) in targets)

    def cable_members(self) -> list:
        return [it.data(_ROLE) for it in self.cables.selectedItems()]

    def target_nodes(self) -> list:
        return [it.data(_ROLE) for it in self.targets.selectedItems()]

    def result(self):
        return {"cables": self.cable_members(),
                "targets": self.target_nodes()}

    @classmethod
    def configure(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result() if dlg.exec() else None
