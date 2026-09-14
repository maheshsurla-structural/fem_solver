"""Moving-load / influence-line setup — the *Analysis-cases ▸ Moving Load*
counterpart of CSiBridge / MIDAS.

``MovingLoadDialog`` collects a **lane** (the path of girder nodes the load
travels over), a **vehicle** (AASHTO HL-93 or IRC presets), and the **response**
to build the influence line for (a member force, a nodal displacement, or a
reaction). It runs nothing: :meth:`configure` returns a config dict, and the
owning window builds the influence line + vehicle envelope via the engine's
:class:`femsolver.bridges.InfluenceLineEngine`.

Built on the shared card scaffold (:mod:`analysis_ui`); headless-constructible
under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QLabel,
                               QListWidget, QListWidgetItem, QVBoxLayout,
                               QWidget)

import style
from analysis_ui import GroupCard, dialog_buttons

# label -> vehicle key ("hl93" = full HL-93 envelope, else a MovingLoad preset)
_VEHICLES = [
    ("AASHTO HL-93 (full envelope)", "hl93"),
    ("HL-93 design truck", "hl93_truck"),
    ("HL-93 design tandem", "hl93_tandem"),
    ("IRC Class A", "irc_class_a"),
    ("IRC Class 70R", "irc_70r"),
]
# label -> (kind, needs_member)
_RESPONSES = [
    ("Bending moment", ("M", True)),
    ("Shear", ("V", True)),
    ("Vertical displacement", ("disp", False)),
    ("Vertical reaction", ("reaction", False)),
]


class MovingLoadDialog(QDialog):
    """Configure a moving-load / influence-line analysis."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Moving load")
        self._project = project

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)

        head = QLabel("Moving load / influence line")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel("A unit load travels the lane; the influence line drives "
                     "the vehicle envelope (max / min effect).")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # ---- lane: multi-select node path (sorted by X) ----
        lane_card = GroupCard("Lane", form=False)
        self.lane_list = QListWidget()
        self.lane_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self.lane_list.setMinimumHeight(120)
        for nd in sorted(project.nodes, key=lambda n: (n.x, getattr(n, "y", 0))):
            it = QListWidgetItem(f"{nd.id}:  ({nd.x:g}, {getattr(nd, 'y', 0):g})")
            it.setData(0x0100, nd.id)                # Qt.UserRole
            self.lane_list.addItem(it)
            it.setSelected(True)                     # default: whole model
        lane_card.body_layout().addWidget(
            QLabel("Nodes the load travels over (ordered by X):"))
        lane_card.body_layout().addWidget(self.lane_list)
        root.addWidget(lane_card)

        # ---- vehicle + response ----
        cfg = GroupCard("Vehicle & response")
        self.vehicle = QComboBox()
        for label, key in _VEHICLES:
            self.vehicle.addItem(label, key)
        cfg.add_row("Vehicle", self.vehicle)

        self.resp = QComboBox()
        for label, spec in _RESPONSES:
            self.resp.addItem(label, spec)
        self.resp.currentIndexChanged.connect(self._sync_target)
        cfg.add_row("Response", self.resp)

        self.member = QComboBox()
        for mb in project.members:
            self.member.addItem(f"member {mb.id}  ({mb.n1}→{mb.n2})", mb.id)
        cfg.add_row("Member", self.member)
        self.end = QComboBox()
        self.end.addItem("end i", "i")
        self.end.addItem("end j", "j")
        cfg.add_row("At end", self.end)

        self.node = QComboBox()
        for nd in project.nodes:
            self.node.addItem(f"node {nd.id}", nd.id)
        cfg.add_row("Node", self.node)
        root.addWidget(cfg)

        self._member_rows = (self.member, self.end)
        self._node_rows = (self.node,)
        self._sync_target()

        root.addStretch(1)
        root.addWidget(dialog_buttons(self))
        style.apply(self)

    def _sync_target(self) -> None:
        _kind, needs_member = self.resp.currentData()
        for w in self._member_rows:
            w.setEnabled(needs_member)
        for w in self._node_rows:
            w.setEnabled(not needs_member)

    def lane_nodes(self) -> list:
        ids = []
        for it in self.lane_list.selectedItems():
            ids.append(it.data(0x0100))
        # order along the lane by X
        by_id = {n.id: n for n in self._project.nodes}
        ids.sort(key=lambda i: (by_id[i].x, getattr(by_id[i], "y", 0)))
        return ids

    def result(self):
        """``dict(lane, vehicle, response)`` where ``response`` is
        ``(kind, target_id, end)`` — ``end`` only used for M / V."""
        kind, needs_member = self.resp.currentData()
        target = (self.member.currentData() if needs_member
                  else self.node.currentData())
        return {
            "lane": self.lane_nodes(),
            "vehicle": self.vehicle.currentData(),
            "response": (kind, target, self.end.currentData()),
        }

    @classmethod
    def configure(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result() if dlg.exec() else None
