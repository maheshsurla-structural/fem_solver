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

from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox,
                               QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QStackedWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

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


class ResponseSpectrumBody(_Body):
    """Response spectrum: code / custom spectrum source + modal combination.

    A compact, preview-free sibling of :class:`response_spectrum_dialog.
    ResponseSpectrumDialog` (the standalone dialog keeps its live Sa(T) plot for
    the direct-run path); it stores the identical ``params`` so a case round-trips
    through either. ``validate`` reuses the shared ``spectrum_from_params``."""

    TITLE = "Response spectrum"
    SUB = ("Modal superposition against a design spectrum — ASCE 7, Eurocode 8, "
           "IS 1893, or a custom Sa(T) table.")

    def __init__(self, project, *, initial: dict | None = None,
                 max_modes: int = 20, default_modes: int = 6):
        super().__init__()
        self._max_modes = max_modes = max(1, int(max_modes))
        default_modes = max(1, min(int(default_modes), max_modes))
        lay = self._root()

        src = GroupCard("Spectrum")
        self.source = QComboBox()
        self.source.addItem("ASCE 7", "asce7")
        self.source.addItem("Eurocode 8", "ec8")
        self.source.addItem("IS 1893", "is1893")
        self.source.addItem("Custom table", "custom")
        src.add_row("Source", self.source)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._asce7_page())
        self.stack.addWidget(self._ec8_page())
        self.stack.addWidget(self._is1893_page())
        self.stack.addWidget(self._custom_page())
        src.add_full_row(self.stack)
        lay.addWidget(src)
        self.source.currentIndexChanged.connect(self.stack.setCurrentIndex)

        an = GroupCard("Analysis")
        self.modes = QSpinBox()
        self.modes.setRange(1, max_modes)
        self.modes.setValue(default_modes)
        an.add_row("Number of modes", self.modes)
        self.direction = QComboBox()
        self.direction.addItem("X", "x")
        self.direction.addItem("Y", "y")
        if project.ndm >= 3:
            self.direction.addItem("Z", "z")
        an.add_row("Direction", self.direction)
        self.combination = QComboBox()
        self.combination.addItem("CQC", "cqc")
        self.combination.addItem("SRSS", "srss")
        an.add_row("Modal combination", self.combination)
        self.damping = QDoubleSpinBox()
        self.damping.setRange(0.001, 0.99)
        self.damping.setDecimals(3)
        self.damping.setSingleStep(0.01)
        self.damping.setValue(0.05)
        an.add_row("Damping ratio ζ", self.damping)
        lay.addWidget(an)

        if initial:
            self._seed(initial)

    @staticmethod
    def _spin(value, *, decimals=3, lo=0.0, hi=1.0e6, step=0.1):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(value)
        return s

    def _asce7_page(self):
        card = GroupCard("ASCE 7 parameters")
        self.a_SDS = self._spin(1.0, step=0.05)
        self.a_SD1 = self._spin(0.6, step=0.05)
        self.a_TL = self._spin(8.0, step=1.0, decimals=1)
        card.add_row("S_DS [g]", self.a_SDS)
        card.add_row("S_D1 [g]", self.a_SD1)
        card.add_row("T_L [s]", self.a_TL)
        return card

    def _ec8_page(self):
        card = GroupCard("Eurocode 8 parameters")
        self.e_ag = self._spin(2.5, step=0.1)
        self.e_ground = QComboBox()
        for g in ("A", "B", "C", "D", "E"):
            self.e_ground.addItem(g, g)
        self.e_ground.setCurrentText("C")
        self.e_q = self._spin(1.5, step=0.5, lo=1.0)
        self.e_type = QComboBox()
        self.e_type.addItem("Type 1 (M ≥ 5.5)", 1)
        self.e_type.addItem("Type 2 (M < 5.5)", 2)
        card.add_row("a_g [m/s²]", self.e_ag)
        card.add_row("Ground type", self.e_ground)
        card.add_row("Behaviour q", self.e_q)
        card.add_row("Spectrum", self.e_type)
        return card

    def _is1893_page(self):
        card = GroupCard("IS 1893 parameters")
        self.i_zone = QComboBox()
        for z, name in ((2, "II (0.10)"), (3, "III (0.16)"),
                        (4, "IV (0.24)"), (5, "V (0.36)")):
            self.i_zone.addItem(f"Zone {name}", z)
        self.i_zone.setCurrentIndex(2)
        self.i_I = self._spin(1.0, step=0.1, lo=0.1)
        self.i_R = self._spin(5.0, step=0.5, lo=1.0)
        self.i_soil = QComboBox()
        for s, name in ((1, "I — Rock/Hard"), (2, "II — Medium"),
                        (3, "III — Soft")):
            self.i_soil.addItem(name, s)
        card.add_row("Seismic zone", self.i_zone)
        card.add_row("Importance I", self.i_I)
        card.add_row("Response R", self.i_R)
        card.add_row("Soil type", self.i_soil)
        return card

    def _custom_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(style.SP_SM)
        v.addWidget(QLabel("Period T [s] vs pseudo-acceleration Sa [m/s²]"))
        self.custom = QTableWidget(0, 2)
        self.custom.setHorizontalHeaderLabels(["T [s]", "Sa [m/s²]"])
        self.custom.verticalHeader().setVisible(False)
        self.custom.horizontalHeader().setStretchLastSection(True)
        self.custom.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        for T, Sa in ((0.1, 9.8), (0.5, 9.8), (1.0, 4.9), (2.0, 2.45),
                      (4.0, 1.2)):
            self._add_custom_row(T, Sa)
        v.addWidget(self.custom)
        row = QHBoxLayout()
        add = QPushButton("Add point")
        add.clicked.connect(lambda: self._add_custom_row(0.0, 0.0))
        rem = QPushButton("Remove point")
        rem.clicked.connect(self._remove_custom_row)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        v.addLayout(row)
        return page

    def _add_custom_row(self, T, Sa):
        r = self.custom.rowCount()
        self.custom.insertRow(r)
        self.custom.setItem(r, 0, QTableWidgetItem(f"{T:g}"))
        self.custom.setItem(r, 1, QTableWidgetItem(f"{Sa:g}"))

    def _remove_custom_row(self):
        r = self.custom.currentRow()
        if r < 0:
            r = self.custom.rowCount() - 1
        if r >= 0:
            self.custom.removeRow(r)

    def _custom_points(self):
        pts = []
        for r in range(self.custom.rowCount()):
            try:
                pts.append((float(self.custom.item(r, 0).text()),
                            float(self.custom.item(r, 1).text())))
            except (AttributeError, ValueError):
                continue
        pts.sort(key=lambda p: p[0])
        out, seen = [], set()
        for T, Sa in pts:
            if T not in seen:
                seen.add(T)
                out.append((T, Sa))
        return out

    def case_params(self) -> dict:
        return {
            "source": self.source.currentData(),
            "damping": float(self.damping.value()),
            "num_modes": int(self.modes.value()),
            "direction": self.direction.currentData(),
            "combination": self.combination.currentData(),
            "asce7": {"SDS": self.a_SDS.value(), "SD1": self.a_SD1.value(),
                      "TL": self.a_TL.value()},
            "ec8": {"ag": self.e_ag.value(), "ground": self.e_ground.currentData(),
                    "q": self.e_q.value(), "type": self.e_type.currentData()},
            "is1893": {"zone": self.i_zone.currentData(), "I": self.i_I.value(),
                       "R": self.i_R.value(), "soil": self.i_soil.currentData()},
            "custom": [[T, Sa] for T, Sa in self._custom_points()],
        }

    def validate(self) -> str | None:
        from response_spectrum_dialog import spectrum_from_params
        try:
            spectrum_from_params(self.case_params())
        except ValueError as exc:
            return str(exc)
        return None

    def _seed(self, p: dict) -> None:
        self.damping.setValue(float(p.get("damping", 0.05)))
        self.modes.setValue(max(1, min(int(p.get("num_modes", 6)),
                                       self._max_modes)))
        for combo, key, default in ((self.direction, "direction", "x"),
                                    (self.combination, "combination", "cqc")):
            i = combo.findData(p.get(key, default))
            if i >= 0:
                combo.setCurrentIndex(i)
        a = p.get("asce7", {})
        self.a_SDS.setValue(float(a.get("SDS", 1.0)))
        self.a_SD1.setValue(float(a.get("SD1", 0.6)))
        self.a_TL.setValue(float(a.get("TL", 8.0)))
        e = p.get("ec8", {})
        self.e_ag.setValue(float(e.get("ag", 2.5)))
        gi = self.e_ground.findData(e.get("ground", "C"))
        if gi >= 0:
            self.e_ground.setCurrentIndex(gi)
        self.e_q.setValue(float(e.get("q", 1.5)))
        ti = self.e_type.findData(e.get("type", 1))
        if ti >= 0:
            self.e_type.setCurrentIndex(ti)
        i = p.get("is1893", {})
        zi = self.i_zone.findData(i.get("zone", 4))
        if zi >= 0:
            self.i_zone.setCurrentIndex(zi)
        self.i_I.setValue(float(i.get("I", 1.0)))
        self.i_R.setValue(float(i.get("R", 5.0)))
        soi = self.i_soil.findData(i.get("soil", 2))
        if soi >= 0:
            self.i_soil.setCurrentIndex(soi)
        custom = p.get("custom")
        if custom:
            self.custom.setRowCount(0)
            for pair in custom:
                try:
                    self._add_custom_row(float(pair[0]), float(pair[1]))
                except (TypeError, ValueError, IndexError):
                    continue
        si = self.source.findData(p.get("source", "asce7"))
        if si >= 0:
            self.source.setCurrentIndex(si)


# type_id -> (menu label, body class). The unified editor's Type ▾ order.
REGISTRY: list[tuple[str, str, type]] = [
    ("modal", "Modal", ModalBody),
    ("buckling", "Buckling", BucklingBody),
    ("responsespectrum", "Response Spectrum", ResponseSpectrumBody),
]
