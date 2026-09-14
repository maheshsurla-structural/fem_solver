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
from analysis_ui import GroupCard, dialog_buttons


class TemperatureGradientDialog(QDialog):
    """Configure a temperature-gradient load case."""

    def __init__(self, parent, project):
        super().__init__(parent)
        self.setWindowTitle("Temperature gradient")
        self._project = project

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)

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
        root.addWidget(apply_card)

        self.source.currentIndexChanged.connect(self.stack.setCurrentIndex)
        root.addStretch(1)
        root.addWidget(dialog_buttons(self))
        style.apply(self)

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
