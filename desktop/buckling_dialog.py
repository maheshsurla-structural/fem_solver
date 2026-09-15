"""Linear (eigenvalue) buckling setup — the *Load Case Data ▸ Buckling*
counterpart of CSiBridge / MIDAS / STAAD.

``BucklingDialog`` collects the three inputs the analysis needs: the
**reference load** whose axial forces build the geometric stiffness (the
buckling factor λ multiplies this load), the number of modes, and how finely
to **sub-divide** each member — the commercial-grade way to capture member
(Euler) buckling between joints, not just global sway. It runs nothing:
:meth:`configure` returns ``(selection, num_modes, subdivisions)`` or ``None``.

Built on the shared card scaffold (:mod:`analysis_ui`); pure Qt, so it is
headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QLabel, QSpinBox,
                               QVBoxLayout)

import style
from analysis_ui import CaseHeader, GroupCard, dialog_buttons


class BucklingDialog(QDialog):
    """Configure a linear-buckling analysis.

    Saveable as a :class:`project.AnalysisCase` (analysis-cases-manager plan):
    ``initial`` seeds the widgets from a saved case's ``params``, and the
    :class:`~analysis_ui.CaseHeader` collects its Name / Notes. Run directly
    (via :meth:`configure`) the header is ignored.
    """

    def __init__(self, parent, project, *, max_modes: int = 20,
                 default_modes: int = 4, initial: dict | None = None,
                 name: str = "Buckling", notes: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Buckling analysis")
        max_modes = max(1, int(max_modes))
        default_modes = max(1, min(int(default_modes), max_modes))

        root = QVBoxLayout(self)
        root.setContentsMargins(style.SP_LG, style.SP_LG,
                                style.SP_LG, style.SP_LG)
        root.setSpacing(style.SP_MD)

        self.header = CaseHeader(name=name, type_label="Buckling", notes=notes)
        root.addWidget(self.header)

        head = QLabel("Linear buckling")
        head.setObjectName("h2")
        root.addWidget(head)
        sub = QLabel("Solves (K + λ·K_g)·φ = 0. The critical factor λ "
                     "multiplies the reference load below.")
        sub.setObjectName("sub")
        sub.setWordWrap(True)
        root.addWidget(sub)

        card = GroupCard("Parameters")
        self.reference = QComboBox()
        self.reference.addItem("All load cases", ("all", None))
        for c in project.load_cases:
            self.reference.addItem(f"Case: {c.name}", ("case", c.id))
        for combo in getattr(project, "combinations", []):
            self.reference.addItem(f"Combination: {combo.name}",
                                   ("combination", combo.id))
        card.add_row("Reference load", self.reference)

        self.modes = QSpinBox()
        self.modes.setRange(1, max_modes)
        self.modes.setValue(default_modes)
        card.add_row("Number of modes", self.modes)

        self.subdivisions = QSpinBox()
        self.subdivisions.setRange(1, 20)
        self.subdivisions.setValue(6)
        card.add_row("Sub-divisions / member", self.subdivisions)
        root.addWidget(card)

        if initial:                                    # seed from a saved case
            sel = tuple(initial.get("selection") or ("all", None))
            for i in range(self.reference.count()):
                if tuple(self.reference.itemData(i) or ()) == sel:
                    self.reference.setCurrentIndex(i)
                    break
            self.modes.setValue(max(1, min(int(initial.get("num_modes",
                                                           default_modes)),
                                          max_modes)))
            self.subdivisions.setValue(int(initial.get("subdivisions", 6)))

        hint = QLabel("More sub-divisions resolve member buckling between "
                      "joints; 4–8 is usually plenty.")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        root.addStretch(1)
        root.addWidget(dialog_buttons(self))
        style.apply(self)

    def result(self):
        """``(selection, num_modes, subdivisions)``."""
        return (self.reference.currentData(), int(self.modes.value()),
                int(self.subdivisions.value()))

    @classmethod
    def configure(cls, parent, project, *, max_modes: int = 20,
                  default_modes: int = 4):
        dlg = cls(parent, project, max_modes=max_modes,
                  default_modes=default_modes)
        return dlg.result() if dlg.exec() else None
