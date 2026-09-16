"""Temperature-gradient load setup — the *Analysis-cases ▸ Temperature
Gradient* counterpart of CSiBridge / MIDAS.

A bridge deck heated by the sun has a nonlinear temperature distribution
through its depth. ``TemperatureGradientDialog`` picks the gradient (AASHTO
§3.12.3 positive vertical gradient by solar zone, or a simple linear
top→bottom profile), the coefficient of thermal expansion, and the members it
applies to. It runs nothing: :meth:`configure` returns a config dict; the
owning window reduces the gradient on each member's section
(:func:`femsolver.bridges.equivalent_thermal_actions`), applies the equivalent
beam actions, and reports the self-equilibrated stress + deflection.

Section depth/width are derived from each member's ``A`` / ``Iz`` as a
rectangular equivalent, so no extra geometry input is needed. Built on the
shared card scaffold; headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDoubleSpinBox, QLabel, QListWidget,
                               QListWidgetItem, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)

import style
from analysis_ui import CaseHeader, GroupCard, dialog_buttons
from pick import PickDialog


class TemperatureGradientDialog(PickDialog):
    """Configure a temperature-gradient load case. Apply-to members can be
    toggled by clicking them in the model while open (see :mod:`pick`)."""

    def __init__(self, parent, project, *, initial: dict | None = None,
                 name: str = "Temperature Gradient", notes: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Temperature gradient")
        self._project = project

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)

        self.header = CaseHeader(name=name, type_label="Temperature Gradient",
                                 notes=notes)
        root.addWidget(self.header)

        head = QLabel("Temperature-gradient load")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel("A vertical gradient through the deck depth → self-"
                     "equilibrated stress (determinate) and continuity moments "
                     "(continuous).")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        card = GroupCard("Gradient")
        self.source = QComboBox()
        self.source.addItem("AASHTO positive vertical", "aashto")
        self.source.addItem("Linear (top → bottom)", "linear")
        card.add_row("Type", self.source)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._aashto_page())
        self.stack.addWidget(self._linear_page())
        card.add_full_row(self.stack)

        self.alpha = QDoubleSpinBox()
        self.alpha.setDecimals(2)
        self.alpha.setRange(0.01, 100.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(1.00)                       # ×1e-5 /°C
        card.add_row("Expansion α [×10⁻⁵ /°C]", self.alpha)
        root.addWidget(card)

        apply_card = GroupCard("Apply to members", form=False)
        self.members = QListWidget()
        self.members.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection)
        self.members.setMinimumHeight(100)
        for mb in project.members:
            it = QListWidgetItem(f"member {mb.id}  ({mb.n1}→{mb.n2})")
            it.setData(0x0100, mb.id)                   # Qt.UserRole
            self.members.addItem(it)
            it.setSelected(True)                        # default: all
        apply_card.body_layout().addWidget(self.members)
        self.register_pick_field("member", self.members)  # click members in model
        root.addWidget(apply_card)

        self.source.currentIndexChanged.connect(self.stack.setCurrentIndex)
        if initial:
            self._seed(initial)
        root.addStretch(1)
        root.addWidget(dialog_buttons(self))
        style.apply(self)

    def _seed(self, p: dict) -> None:
        """Seed the widgets from a saved case's params (α stored in SI)."""
        si = self.source.findData(p.get("source", "aashto"))
        if si >= 0:
            self.source.setCurrentIndex(si)
            self.stack.setCurrentIndex(si)
        zi = self.zone.findData(p.get("zone"))
        if zi >= 0:
            self.zone.setCurrentIndex(zi)
        self.dt_top.setValue(float(p.get("dt_top", 20.0)))
        self.dt_bot.setValue(float(p.get("dt_bot", 0.0)))
        self.alpha.setValue(float(p.get("alpha", 1.0e-5)) / 1.0e-5)
        mem = set(p.get("members") or [])
        for i in range(self.members.count()):
            it = self.members.item(i)
            it.setSelected(it.data(0x0100) in mem)

    def _aashto_page(self) -> QWidget:
        card = GroupCard("AASHTO parameters")
        self.zone = QComboBox()
        for z in (1, 2, 3, 4):
            self.zone.addItem(f"Solar zone {z}", z)
        self.zone.setCurrentIndex(2)                    # zone 3
        card.add_row("Zone", self.zone)
        return card

    def _linear_page(self) -> QWidget:
        card = GroupCard("Linear parameters")
        self.dt_top = QDoubleSpinBox()
        self.dt_top.setRange(-100.0, 100.0)
        self.dt_top.setValue(20.0)
        self.dt_bot = QDoubleSpinBox()
        self.dt_bot.setRange(-100.0, 100.0)
        self.dt_bot.setValue(0.0)
        card.add_row("ΔT top [°C]", self.dt_top)
        card.add_row("ΔT bottom [°C]", self.dt_bot)
        return card

    def member_ids(self) -> list:
        return [it.data(0x0100) for it in self.members.selectedItems()]

    def result(self):
        return {
            "source": self.source.currentData(),
            "zone": self.zone.currentData(),
            "dt_top": float(self.dt_top.value()),
            "dt_bot": float(self.dt_bot.value()),
            "alpha": float(self.alpha.value()) * 1.0e-5,
            "members": self.member_ids(),
        }

    @classmethod
    def configure(cls, parent, project):
        dlg = cls(parent, project)
        return dlg.result() if dlg.exec() else None
