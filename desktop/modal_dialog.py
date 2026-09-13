"""Modal (free-vibration) analysis setup — the *Load Case Data ▸ Modal*
counterpart of CSiBridge / MIDAS.

``ModalDialog`` collects the two inputs the engine's
:class:`femsolver.analysis.eigen.EigenAnalysis` needs: how many modes to
extract and which mass formulation (consistent or lumped) to use. It runs
nothing — :meth:`configure` returns ``(num_modes, lumped)`` or ``None`` if
cancelled, and the owning window builds + runs the eigen analysis.

Built on the shared card scaffold (:mod:`analysis_ui`); pure Qt, so it is
headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QLabel, QSpinBox,
                               QVBoxLayout)

import style
from analysis_ui import GroupCard, dialog_buttons


class ModalDialog(QDialog):
    """Configure a free-vibration eigen-analysis."""

    def __init__(self, parent, *, max_modes: int = 20, default_modes: int = 6):
        super().__init__(parent)
        self.setWindowTitle("Modal analysis")
        max_modes = max(1, int(max_modes))
        default_modes = max(1, min(int(default_modes), max_modes))

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)

        head = QLabel("Free-vibration modes")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel("Solves K·φ = ω²·M·φ for the lowest modes. "
                     "Mass comes from each material's density.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        card = GroupCard("Parameters")
        self.modes = QSpinBox()
        self.modes.setRange(1, max_modes)
        self.modes.setValue(default_modes)
        card.add_row("Number of modes", self.modes)

        self.mass = QComboBox()
        self.mass.addItem("Consistent", False)
        self.mass.addItem("Lumped", True)
        card.add_row("Mass matrix", self.mass)
        root.addWidget(card)

        root.addStretch(1)
        root.addWidget(dialog_buttons(self))
        style.apply(self)

    def result(self) -> tuple[int, bool]:
        """``(num_modes, lumped)``."""
        return int(self.modes.value()), bool(self.mass.currentData())

    @classmethod
    def configure(cls, parent, *, max_modes: int = 20,
                  default_modes: int = 6):
        dlg = cls(parent, max_modes=max_modes, default_modes=default_modes)
        return dlg.result() if dlg.exec() else None
