"""Influence-surface / multi-lane setup — the *Analysis-cases ▸ Influence
Surface* counterpart of CSiBridge / MIDAS moving-load on a 2-D deck.

Where the Moving Load case (G1) is a 1-D lane on a girder line, this places the
load anywhere on a **2-D deck / grillage** (a 3-D model): a unit load traverses
every deck node to build the influence *surface* for a chosen response, then
vehicles are placed in the AASHTO design lanes with multiple-presence factors to
find the governing effect.

``InfluenceSurfaceDialog`` picks the deck nodes, the response (vertical
displacement or reaction), the vehicle, and whether to apply multiple presence.
:meth:`configure` returns a config dict; the owning window runs
:meth:`femsolver.bridges.InfluenceLineEngine.influence_surface` +
:func:`femsolver.bridges.multi_lane_envelope`.

Headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QLabel, QListWidget, QListWidgetItem,
                               QVBoxLayout)

import style
from analysis_ui import CaseHeader, GroupCard, dialog_buttons
from pick import PickDialog

_ROLE = 0x0100                                          # Qt.UserRole
_VEHICLES = [
    ("HL-93 design truck", "hl93_truck"),
    ("HL-93 design tandem", "hl93_tandem"),
    ("IRC Class A", "irc_class_a"),
    ("IRC Class 70R", "irc_70r"),
]
_RESPONSES = [
    ("Vertical displacement", "disp"),
    ("Vertical reaction", "reaction"),
]


class InfluenceSurfaceDialog(PickDialog):
    """Configure an influence-surface / multi-lane analysis (3-D deck).

    Deck nodes and the response node can be picked by clicking in the model
    while the dialog is open (see :mod:`pick`)."""

    def __init__(self, parent, project, *, initial: dict | None = None,
                 name: str = "Influence Surface", notes: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Influence surface")
        self._project = project

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)
        self.header = CaseHeader(name=name, type_label="Influence Surface",
                                 notes=notes)
        root.addWidget(self.header)
        head = QLabel("Influence surface / multi-lane moving load")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel("A unit load traverses the deck; the influence surface "
                     "drives the AASHTO multi-lane vehicle envelope.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        deck_card = GroupCard("Deck surface", form=False)
        self.deck_list = QListWidget()
        self.deck_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self.deck_list.setMinimumHeight(120)
        for nd in project.nodes:
            it = QListWidgetItem(
                f"{nd.id}:  ({nd.x:g}, {getattr(nd, 'y', 0):g}, "
                f"{getattr(nd, 'z', 0):g})")
            it.setData(_ROLE, nd.id)
            self.deck_list.addItem(it)
            it.setSelected(True)
        deck_card.body_layout().addWidget(
            QLabel("Deck nodes the load can stand on (plan = X, Y):"))
        deck_card.body_layout().addWidget(self.deck_list)
        root.addWidget(deck_card)

        cfg = GroupCard("Vehicle & response")
        self.resp = QComboBox()
        for label, key in _RESPONSES:
            self.resp.addItem(label, key)
        cfg.add_row("Response", self.resp)
        self.node = QComboBox()
        for nd in project.nodes:
            self.node.addItem(f"node {nd.id}", nd.id)
        if project.nodes:
            self.node.setCurrentIndex(len(project.nodes) // 2)
        cfg.add_row("At node", self.node)
        # Pick the response node (combo) or toggle deck nodes (list) by clicking
        # in the model; focus a field to aim picks at it.
        self.register_pick_field("node", self.node)
        self.register_pick_field("node", self.deck_list)
        self.vehicle = QComboBox()
        for label, key in _VEHICLES:
            self.vehicle.addItem(label, key)
        cfg.add_row("Vehicle", self.vehicle)
        self.multi = QCheckBox("Apply AASHTO multiple-presence factors")
        self.multi.setChecked(True)
        cfg.add_full_row(self.multi)
        root.addWidget(cfg)

        if initial:
            self._seed(initial)
        root.addStretch(1)
        root.addWidget(dialog_buttons(self))
        style.apply(self)

    def _seed(self, p: dict) -> None:
        """Seed from a saved case's params."""
        deck = set(p.get("deck") or [])
        for i in range(self.deck_list.count()):
            it = self.deck_list.item(i)
            it.setSelected(it.data(_ROLE) in deck)
        resp = p.get("response") or ("disp", None)
        ri = self.resp.findData(resp[0])
        if ri >= 0:
            self.resp.setCurrentIndex(ri)
        ni = self.node.findData(resp[1] if len(resp) > 1 else None)
        if ni >= 0:
            self.node.setCurrentIndex(ni)
        vi = self.vehicle.findData(p.get("vehicle"))
        if vi >= 0:
            self.vehicle.setCurrentIndex(vi)
        self.multi.setChecked(bool(p.get("multi_presence", True)))

    def deck_nodes(self) -> list:
        return [it.data(_ROLE) for it in self.deck_list.selectedItems()]

    def result(self):
        return {
            "deck": self.deck_nodes(),
            "response": (self.resp.currentData(), self.node.currentData()),
            "vehicle": self.vehicle.currentData(),
            "multi_presence": self.multi.isChecked(),
        }

    @classmethod
    def configure(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result() if dlg.exec() else None
