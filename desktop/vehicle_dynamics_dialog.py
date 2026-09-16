"""Vehicle-dynamics setup — the *Analysis-cases ▸ Vehicle Dynamics* counterpart
of the "moving load — dynamic" case in CSiBridge / MIDAS.

Where the static Moving Load case (G1) gives the worst-position effect, this
runs a vehicle **crossing at speed** and integrates the dynamics — the dynamic
amplification factor (DAF) and, for a sprung-mass vehicle, the wheel–deck
contact force. ``VehicleDynamicsDialog`` picks the lane, the analysis kind
(constant moving force, or a coupled sprung-mass interaction), the vehicle, the
speed, the bridge damping, and the response node. :meth:`configure` returns a
config dict; the owning window runs
:class:`femsolver.bridges.MovingForceAnalysis` /
:class:`femsolver.bridges.VBIAnalysis`.

Built on the shared card scaffold; headless-constructible under
``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDoubleSpinBox, QLabel, QListWidget,
                               QListWidgetItem, QStackedWidget, QVBoxLayout,
                               QWidget)

import style
from analysis_ui import CaseHeader, GroupCard, dialog_buttons
from pick import PickDialog

_ROLE = 0x0100                                          # Qt.UserRole
# label -> MovingLoad preset key (axle trains for the moving-force model)
_VEHICLES = [
    ("HL-93 design truck", "hl93_truck"),
    ("HL-93 design tandem", "hl93_tandem"),
    ("IRC Class A", "irc_class_a"),
    ("IRC Class 70R", "irc_70r"),
]


class VehicleDynamicsDialog(PickDialog):
    """Configure a vehicle-dynamics (moving-load time-history) analysis. The
    response node and lane nodes can be picked in the model (see :mod:`pick`)."""

    def __init__(self, parent, project, *, initial: dict | None = None,
                 name: str = "Vehicle Dynamics", notes: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Vehicle dynamics")
        self._project = project

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)
        self.header = CaseHeader(name=name, type_label="Vehicle Dynamics",
                                 notes=notes)
        root.addWidget(self.header)
        head = QLabel("Vehicle dynamics / moving-load time-history")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel("A vehicle crosses the lane at speed; the transient solve "
                     "gives the dynamic amplification (DAF) and response history.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        lane_card = GroupCard("Lane", form=False)
        self.lane_list = QListWidget()
        self.lane_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self.lane_list.setMinimumHeight(100)
        for nd in sorted(project.nodes, key=lambda n: (n.x, getattr(n, "y", 0))):
            it = QListWidgetItem(f"{nd.id}:  ({nd.x:g}, {getattr(nd, 'y', 0):g})")
            it.setData(_ROLE, nd.id)
            self.lane_list.addItem(it)
            it.setSelected(True)
        lane_card.body_layout().addWidget(
            QLabel("Nodes the vehicle travels over (ordered by X):"))
        lane_card.body_layout().addWidget(self.lane_list)
        root.addWidget(lane_card)

        cfg = GroupCard("Vehicle & speed")
        self.kind = QComboBox()
        self.kind.addItem("Moving force (constant axles)", "force")
        self.kind.addItem("Sprung-mass (interaction)", "vbi")
        self.kind.currentIndexChanged.connect(self._on_kind)
        cfg.add_row("Analysis", self.kind)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._force_page())
        self.stack.addWidget(self._vbi_page())
        cfg.add_full_row(self.stack)

        self.speed = QDoubleSpinBox()
        self.speed.setRange(1.0, 500.0)
        self.speed.setValue(60.0)
        cfg.add_row("Speed [km/h]", self.speed)
        self.zeta = QDoubleSpinBox()
        self.zeta.setRange(0.0, 20.0)
        self.zeta.setValue(2.0)
        cfg.add_row("Bridge damping ζ [%]", self.zeta)
        self.node = QComboBox()
        for nd in project.nodes:
            self.node.addItem(f"node {nd.id}", nd.id)
        if project.nodes:                              # default: a mid node
            self.node.setCurrentIndex(len(project.nodes) // 2)
        cfg.add_row("Response node", self.node)
        # Pick the response node (combo) or toggle lane nodes (list) in the model.
        self.register_pick_field("node", self.node)
        self.register_pick_field("node", self.lane_list)
        root.addWidget(cfg)

        if initial:
            self._seed(initial)
        root.addStretch(1)
        root.addWidget(dialog_buttons(self))
        style.apply(self)

    def _seed(self, p: dict) -> None:
        """Seed from a saved case's params (stored SI → display units)."""
        lane = set(p.get("lane") or [])
        for i in range(self.lane_list.count()):
            it = self.lane_list.item(i)
            it.setSelected(it.data(_ROLE) in lane)
        ki = self.kind.findData(p.get("kind", "force"))
        if ki >= 0:
            self.kind.setCurrentIndex(ki)              # swaps the stack page
        vi = self.vehicle.findData(p.get("vehicle"))
        if vi >= 0:
            self.vehicle.setCurrentIndex(vi)
        self.mass.setValue(float(p.get("mass", 20000.0)) / 1.0e3)   # kg → t
        self.bounce.setValue(float(p.get("bounce", 2.0)))
        self.susp.setValue(float(p.get("susp_damp", 0.10)) * 100.0)  # frac → %
        self.speed.setValue(float(p.get("speed", 16.667)) * 3.6)     # m/s → km/h
        self.zeta.setValue(float(p.get("zeta", 0.02)) * 100.0)       # frac → %
        ni = self.node.findData(p.get("node"))
        if ni >= 0:
            self.node.setCurrentIndex(ni)

    # ---------------------------------------------------------- vehicle pages
    def _force_page(self) -> QWidget:
        card = GroupCard("Axle train")
        self.vehicle = QComboBox()
        for label, key in _VEHICLES:
            self.vehicle.addItem(label, key)
        card.add_row("Vehicle", self.vehicle)
        return card

    def _vbi_page(self) -> QWidget:
        card = GroupCard("Sprung-mass vehicle")
        self.mass = QDoubleSpinBox()
        self.mass.setRange(0.1, 1000.0)
        self.mass.setValue(20.0)
        card.add_row("Sprung mass [t]", self.mass)
        self.bounce = QDoubleSpinBox()
        self.bounce.setRange(0.1, 20.0)
        self.bounce.setValue(2.0)
        card.add_row("Bounce frequency [Hz]", self.bounce)
        self.susp = QDoubleSpinBox()
        self.susp.setRange(0.0, 80.0)
        self.susp.setValue(10.0)
        card.add_row("Suspension damping [%]", self.susp)
        return card

    def _on_kind(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)

    # ---------------------------------------------------------------- result
    def lane_nodes(self) -> list:
        by_id = {n.id: n for n in self._project.nodes}
        ids = [it.data(_ROLE) for it in self.lane_list.selectedItems()]
        ids.sort(key=lambda i: (by_id[i].x, getattr(by_id[i], "y", 0)))
        return ids

    def result(self):
        return {
            "lane": self.lane_nodes(),
            "kind": self.kind.currentData(),
            "vehicle": self.vehicle.currentData(),
            "mass": float(self.mass.value()) * 1.0e3,   # t → kg
            "bounce": float(self.bounce.value()),
            "susp_damp": float(self.susp.value()) / 100.0,
            "speed": float(self.speed.value()) / 3.6,   # km/h → m/s
            "zeta": float(self.zeta.value()) / 100.0,
            "node": self.node.currentData(),
        }

    @classmethod
    def configure(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result() if dlg.exec() else None
