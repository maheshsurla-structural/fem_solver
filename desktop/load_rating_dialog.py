"""Load-rating (AASHTO LRFR) setup — the *Analysis-cases ▸ Load Rating*
counterpart of MIDAS Civil / CSiBridge rating.

``LoadRatingDialog`` collects everything the LRFR rating factor needs for one
member effect (MBE §6A):

* a **lane** (the girder-node path the live load travels), and the **rated
  effect** — a bending moment or shear at a member end;
* the member **capacity** ``Rn`` and the unfactored **dead-load effects**
  ``DC`` (components) / ``DW`` (wearing surface) / ``P`` (other permanent),
  entered in the project's display units;
* the LRFD **resistance factor** ``phi`` plus the **condition** (phi_c) and
  **system** (phi_s) factors, and the dynamic load allowance ``IM``;
* which **rating levels** to compute — design inventory + operating always,
  plus optional legal (by ADTT) and permit (by live-load factor).

It runs nothing: :meth:`configure` returns a config dict in SI, and the owning
window builds the HL-93 influence-line live-load effect and rates it via
:func:`femsolver.bridges.rate_member`.  Built on the shared card scaffold
(:mod:`analysis_ui`); headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QDialog, QDoubleSpinBox, QLabel, QListWidget,
                               QListWidgetItem, QSpinBox, QVBoxLayout)

import style
from analysis_ui import GroupCard, dialog_buttons
from units import Quantity, UnitSystem

# label -> (component, quantity) — LRFR rates a force effect (moment or shear)
_EFFECTS = [
    ("Bending moment", ("M", Quantity.MOMENT)),
    ("Shear", ("V", Quantity.FORCE)),
]
# condition factor phi_c (MBE Table 6A.4.2.3-1)
_CONDITION = [
    ("Good / Satisfactory  (phi_c = 1.00)", 1.00),
    ("Fair  (0.95)", 0.95),
    ("Poor  (0.85)", 0.85),
]
# system factor phi_s (MBE Table 6A.4.2.4-1)
_SYSTEM = [
    ("Multi-girder / slab  (phi_s = 1.00)", 1.00),
    ("Four-girder, spacing <= 4 ft  (0.95)", 0.95),
    ("Riveted two-girder / truss  (0.90)", 0.90),
    ("Welded two-girder / truss  (0.85)", 0.85),
    ("Three-girder  (0.85)", 0.85),
]


def _spin(value, lo=0.0, hi=1.0e12, step=1.0, decimals=3):
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(value)
    return sb


class LoadRatingDialog(QDialog):
    """Configure an AASHTO LRFR load-rating case."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Load rating (LRFR)")
        self._project = project
        self._us = UnitSystem.from_project(project)

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)

        head = QLabel("Load rating — AASHTO LRFR")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel("Rating factor RF = (C − γDC·DC − γDW·DW − γP·P) / "
                     "(γLL·(LL+IM)); the HL-93 live-load effect comes from the "
                     "influence line of the rated effect.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # ---- lane ----
        lane_card = GroupCard("Lane", form=False)
        self.lane_list = QListWidget()
        self.lane_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self.lane_list.setMinimumHeight(96)
        for nd in sorted(project.nodes, key=lambda n: (n.x, getattr(n, "y", 0))):
            it = QListWidgetItem(f"{nd.id}:  ({nd.x:g}, {getattr(nd, 'y', 0):g})")
            it.setData(0x0100, nd.id)
            self.lane_list.addItem(it)
            it.setSelected(True)
        lane_card.body_layout().addWidget(
            QLabel("Nodes the live load travels over (ordered by X):"))
        lane_card.body_layout().addWidget(self.lane_list)
        root.addWidget(lane_card)

        # ---- rated effect ----
        eff = GroupCard("Rated effect")
        self.effect = QComboBox()
        for label, spec in _EFFECTS:
            self.effect.addItem(label, spec)
        self.effect.currentIndexChanged.connect(self._sync_units)
        eff.add_row("Effect", self.effect)
        self.member = QComboBox()
        for mb in project.members:
            self.member.addItem(f"member {mb.id}  ({mb.n1}→{mb.n2})", mb.id)
        eff.add_row("Member", self.member)
        self.end = QComboBox()
        self.end.addItem("end i", "i")
        self.end.addItem("end j", "j")
        eff.add_row("At end", self.end)
        root.addWidget(eff)

        # ---- capacity & dead load (display units) ----
        cap = GroupCard("Capacity & dead load")
        self.Rn = _spin(1000.0)
        self.DC = _spin(200.0)
        self.DW = _spin(50.0)
        self.P = _spin(0.0, lo=-1.0e12)
        cap.add_row("Nominal resistance Rn", self.Rn)
        cap.add_row("Dead load — components DC", self.DC)
        cap.add_row("Dead load — wearing surface DW", self.DW)
        cap.add_row("Other permanent P (signed)", self.P)
        root.addWidget(cap)

        # ---- factors ----
        fac = GroupCard("Resistance & evaluation factors")
        self.phi = _spin(1.00, lo=0.0, hi=1.5, step=0.05, decimals=2)
        fac.add_row("Resistance factor φ", self.phi)
        self.phi_c = QComboBox()
        for label, v in _CONDITION:
            self.phi_c.addItem(label, v)
        fac.add_row("Condition factor φc", self.phi_c)
        self.phi_s = QComboBox()
        for label, v in _SYSTEM:
            self.phi_s.addItem(label, v)
        fac.add_row("System factor φs", self.phi_s)
        self.im = _spin(0.33, lo=0.0, hi=1.0, step=0.01, decimals=2)
        fac.add_row("Dynamic allowance IM", self.im)
        root.addWidget(fac)

        # ---- rating levels ----
        lvl = GroupCard("Rating levels")
        lvl.body_layout().addWidget(
            QLabel("Design Inventory (γLL 1.75) and Operating (1.35) are "
                   "always computed."))
        self.legal = QCheckBox("Legal load — γLL from ADTT")
        self.legal.setChecked(True)
        self.legal.toggled.connect(lambda on: self.adtt.setEnabled(on))
        lvl.body_layout().addWidget(self.legal)
        self.adtt = QSpinBox()
        self.adtt.setRange(0, 500000)
        self.adtt.setValue(5000)
        self.adtt.setPrefix("ADTT = ")
        lvl.add_row("  ADTT (one direction)", self.adtt)
        self.permit = QCheckBox("Permit load — specify γLL")
        self.permit.toggled.connect(lambda on: self.permit_gamma.setEnabled(on))
        lvl.body_layout().addWidget(self.permit)
        self.permit_gamma = _spin(1.15, lo=0.5, hi=2.5, step=0.05, decimals=2)
        self.permit_gamma.setEnabled(False)
        lvl.add_row("  Permit γLL", self.permit_gamma)
        root.addWidget(lvl)

        root.addStretch(1)
        root.addWidget(dialog_buttons(self))
        style.apply(self)
        self._sync_units()

    # ------------------------------------------------------------------ helpers
    def _quantity(self) -> Quantity:
        return self.effect.currentData()[1]

    def _sync_units(self) -> None:
        """Show the capacity/dead-load spin-box units for the rated effect."""
        unit = " " + self._us.label(self._quantity())
        for sb in (self.Rn, self.DC, self.DW, self.P):
            sb.setSuffix(unit)

    def lane_nodes(self) -> list:
        ids = [it.data(0x0100) for it in self.lane_list.selectedItems()]
        by_id = {n.id: n for n in self._project.nodes}
        ids.sort(key=lambda i: (by_id[i].x, getattr(by_id[i], "y", 0)))
        return ids

    def result(self) -> dict:
        """Config dict in **SI** — capacity / dead loads converted from the
        displayed units via the project's :class:`UnitSystem`."""
        comp, qty = self.effect.currentData()
        to_si = self._us.to_si
        return {
            "lane": self.lane_nodes(),
            "response": (comp, self.member.currentData(),
                         self.end.currentData()),
            "Rn": to_si(self.Rn.value(), qty),
            "DC": to_si(self.DC.value(), qty),
            "DW": to_si(self.DW.value(), qty),
            "P": to_si(self.P.value(), qty),
            "phi": self.phi.value(),
            "phi_c": self.phi_c.currentData(),
            "phi_s": self.phi_s.currentData(),
            "im": self.im.value(),
            "adtt": self.adtt.value() if self.legal.isChecked() else None,
            "permit_gamma_LL": (self.permit_gamma.value()
                                if self.permit.isChecked() else None),
        }

    @classmethod
    def configure(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result() if dlg.exec() else None
