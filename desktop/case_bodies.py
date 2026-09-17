"""Embeddable analysis-case *bodies* (E1) — the type-specific parameter widgets,
without the surrounding Name / Notes / Type / Stiffness / buttons chrome.

The unified :class:`case_editor.CaseEditorDialog` hosts one of these in a
``QStackedWidget`` under a Type ▾ dropdown (CSiBridge's *Load Case Data*), so the
shared chrome stays put while the body swaps with the type. Each body exposes a
uniform contract:

* ``TITLE`` / ``SUB`` — a heading + one-line description for the shell;
* ``case_params()`` — the JSON-friendly ``params`` dict stored on the
  :class:`project.AnalysisCase` (identical to what the type's adapter stores);
* ``validate()`` — an error string to block OK, or ``None``.

These are deliberately small; the standalone per-type dialogs still exist for the
direct-run path and are migrated onto these bodies over the strangler. Pure Qt,
headless-constructible.
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QSpinBox, QVBoxLayout, QWidget

import style
from analysis_ui import GroupCard


class _Body(QWidget):
    TITLE = ""
    SUB = ""

    def _root(self) -> QVBoxLayout:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(style.SP_MD)
        return lay

    def validate(self) -> str | None:      # noqa: D401 - overridden where needed
        return None

    def case_params(self) -> dict:
        raise NotImplementedError


class ModalBody(_Body):
    """Modal: number of modes + mass formulation."""

    TITLE = "Free-vibration modes"
    SUB = ("Solves K·φ = ω²·M·φ for the lowest modes. Mass comes from each "
           "material's density.")

    def __init__(self, project, *, initial: dict | None = None,
                 max_modes: int = 20, default_modes: int = 6):
        super().__init__()
        max_modes = max(1, int(max_modes))
        default_modes = max(1, min(int(default_modes), max_modes))
        lay = self._root()
        card = GroupCard("Parameters")
        self.modes = QSpinBox()
        self.modes.setRange(1, max_modes)
        self.modes.setValue(default_modes)
        card.add_row("Number of modes", self.modes)
        self.mass = QComboBox()
        self.mass.addItem("Consistent", False)
        self.mass.addItem("Lumped", True)
        card.add_row("Mass matrix", self.mass)
        lay.addWidget(card)
        if initial:
            self.modes.setValue(max(1, min(int(initial.get("num_modes",
                                                            default_modes)),
                                           max_modes)))
            self.mass.setCurrentIndex(1 if initial.get("lumped") else 0)

    def case_params(self) -> dict:
        return {"num_modes": int(self.modes.value()),
                "lumped": bool(self.mass.currentData())}


class BucklingBody(_Body):
    """Linear buckling: reference load + modes + sub-divisions."""

    TITLE = "Linear buckling"
    SUB = ("Solves (K + λ·K_g)·φ = 0; the critical factor λ multiplies the "
           "reference load (or, from an initial state, that state's load).")

    def __init__(self, project, *, initial: dict | None = None,
                 max_modes: int = 20, default_modes: int = 4):
        super().__init__()
        max_modes = max(1, int(max_modes))
        default_modes = max(1, min(int(default_modes), max_modes))
        lay = self._root()
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
        lay.addWidget(card)
        if initial:
            sel = tuple(initial.get("selection") or ("all", None))
            for i in range(self.reference.count()):
                if tuple(self.reference.itemData(i) or ()) == sel:
                    self.reference.setCurrentIndex(i)
                    break
            self.modes.setValue(max(1, min(int(initial.get("num_modes",
                                                           default_modes)),
                                           max_modes)))
            self.subdivisions.setValue(int(initial.get("subdivisions", 6)))

    def case_params(self) -> dict:
        return {"selection": list(self.reference.currentData() or ("all", None)),
                "num_modes": int(self.modes.value()),
                "subdivisions": int(self.subdivisions.value())}


# type_id -> (menu label, body class). The unified editor's Type ▾ order.
REGISTRY: list[tuple[str, str, type]] = [
    ("modal", "Modal", ModalBody),
    ("buckling", "Buckling", BucklingBody),
]
