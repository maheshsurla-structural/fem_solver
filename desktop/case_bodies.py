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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QDoubleSpinBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QSpinBox,
                               QStackedWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

import style
from analysis_ui import GroupCard
# Reuse the option lists / constants / helpers from the standalone dialogs
# (single source of truth) so a case built here stores identical params.
from influence_surface_dialog import _RESPONSES as _IS_RESPONSES
from influence_surface_dialog import _VEHICLES as _IS_VEHICLES
from load_rating_dialog import _CONDITION, _EFFECTS, _SYSTEM
from load_rating_dialog import _spin as _lr_spin
from moving_load_dialog import _RESPONSES as _ML_RESPONSES
from moving_load_dialog import _VEHICLES as _ML_VEHICLES
from timehistory_dialog import _DIRS as _TH_DIRS
from units import UnitSystem
from vehicle_dynamics_dialog import _VEHICLES as _VD_VEHICLES

_ROLE = 0x0100                                          # Qt.UserRole


def _dspin(value, lo=0.0, hi=1.0e6, *, decimals=3, step=0.1):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    return s


class _Body(QWidget):
    TITLE = ""
    SUB = ""
    # Whether this analysis can start from a nonlinear case's committed state
    # (E2). The editor shows the Stiffness-to-use card only for such types; the
    # rest (static / influence analyses) run against the current model.
    SUPPORTS_IC = True

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
        self.reference.addItem("All load patterns", ("all", None))
        for c in project.load_cases:
            self.reference.addItem(f"Pattern: {c.name}", ("case", c.id))
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


class CableTuningBody(_Body):
    """Cable-stayed tuning: pick the stay members + the target deck nodes."""

    TITLE = "Cable-stayed tuning (unknown load factor)"
    SUB = ("Solves the stay pretensions so the target deck nodes reach zero "
           "vertical deflection under dead load. Define the dead load first.")
    SUPPORTS_IC = False

    def __init__(self, project, *, initial: dict | None = None,
                 max_modes: int = 20, default_modes: int = 6):
        super().__init__()
        lay = self._root()
        cab = GroupCard("Stay cables (members to tune)", form=False)
        self.cables = _multi_list(90)
        for mb in project.members:
            _add_id_row(self.cables, f"member {mb.id}  ({mb.n1}→{mb.n2})", mb.id)
        cab.body_layout().addWidget(self.cables)
        lay.addWidget(cab)
        tgt = GroupCard("Target nodes (zero vertical deflection)", form=False)
        self.targets = _multi_list(90)
        for nd in project.nodes:
            _add_id_row(self.targets,
                        f"node {nd.id}  ({nd.x:g}, {getattr(nd, 'y', 0):g})",
                        nd.id)
        tgt.body_layout().addWidget(self.targets)
        lay.addWidget(tgt)
        if initial:
            _select_ids(self.cables, initial.get("cables"))
            _select_ids(self.targets, initial.get("targets"))

    def case_params(self) -> dict:
        return {"cables": _selected_ids(self.cables),
                "targets": _selected_ids(self.targets)}


class TemperatureGradientBody(_Body):
    """Temperature gradient: AASHTO / linear profile + members to apply it to."""

    TITLE = "Temperature-gradient load"
    SUB = ("A vertical gradient through the deck depth → self-equilibrated "
           "stress (determinate) and continuity moments (continuous).")
    SUPPORTS_IC = False

    def __init__(self, project, *, initial: dict | None = None,
                 max_modes: int = 20, default_modes: int = 6):
        super().__init__()
        lay = self._root()
        card = GroupCard("Gradient")
        self.source = QComboBox()
        self.source.addItem("AASHTO positive vertical", "aashto")
        self.source.addItem("Linear (top → bottom)", "linear")
        card.add_row("Type", self.source)
        self.stack = QStackedWidget()
        self.zone = QComboBox()
        for z in (1, 2, 3, 4):
            self.zone.addItem(f"Solar zone {z}", z)
        self.zone.setCurrentIndex(2)
        aashto = GroupCard("AASHTO parameters")
        aashto.add_row("Zone", self.zone)
        self.dt_top = self._t(20.0)
        self.dt_bot = self._t(0.0)
        linear = GroupCard("Linear parameters")
        linear.add_row("ΔT top [°C]", self.dt_top)
        linear.add_row("ΔT bottom [°C]", self.dt_bot)
        self.stack.addWidget(aashto)
        self.stack.addWidget(linear)
        card.add_full_row(self.stack)
        self.alpha = QDoubleSpinBox()
        self.alpha.setDecimals(2)
        self.alpha.setRange(0.01, 100.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(1.00)
        card.add_row("Expansion α [×10⁻⁵ /°C]", self.alpha)
        lay.addWidget(card)
        apply_card = GroupCard("Apply to members", form=False)
        self.members = _multi_list(100)
        for mb in project.members:
            it = _add_id_row(self.members, f"member {mb.id}  ({mb.n1}→{mb.n2})",
                             mb.id)
            it.setSelected(True)                    # default: all
        apply_card.body_layout().addWidget(self.members)
        lay.addWidget(apply_card)
        self.source.currentIndexChanged.connect(self.stack.setCurrentIndex)
        if initial:
            self._seed(initial)

    @staticmethod
    def _t(value):
        s = QDoubleSpinBox()
        s.setRange(-100.0, 100.0)
        s.setValue(value)
        return s

    def _seed(self, p: dict) -> None:
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
        _select_ids(self.members, p.get("members"))

    def case_params(self) -> dict:
        return {"source": self.source.currentData(),
                "zone": self.zone.currentData(),
                "dt_top": float(self.dt_top.value()),
                "dt_bot": float(self.dt_bot.value()),
                "alpha": float(self.alpha.value()) * 1.0e-5,
                "members": _selected_ids(self.members)}


class MovingLoadBody(_Body):
    """Moving load / influence line: a lane node path + vehicle + response."""

    TITLE = "Moving load / influence line"
    SUB = ("A unit load travels the lane; the influence line drives the vehicle "
           "envelope (max / min effect).")
    SUPPORTS_IC = False

    def __init__(self, project, *, initial: dict | None = None,
                 max_modes: int = 20, default_modes: int = 6):
        super().__init__()
        self._project = project
        lay = self._root()
        lane = GroupCard("Lane", form=False)
        lane.body_layout().addWidget(
            QLabel("Nodes the load travels over (ordered by X):"))
        self.lane_list = _multi_list(120)
        for nd in sorted(project.nodes, key=lambda n: (n.x, getattr(n, "y", 0))):
            it = _add_id_row(self.lane_list,
                             f"{nd.id}:  ({nd.x:g}, {getattr(nd, 'y', 0):g})",
                             nd.id)
            it.setSelected(True)
        lane.body_layout().addWidget(self.lane_list)
        lay.addWidget(lane)
        cfg = GroupCard("Vehicle & response")
        self.vehicle = QComboBox()
        for label, key in _ML_VEHICLES:
            self.vehicle.addItem(label, key)
        cfg.add_row("Vehicle", self.vehicle)
        self.resp = QComboBox()
        for label, spec in _ML_RESPONSES:
            self.resp.addItem(label, spec)
        self.resp.currentIndexChanged.connect(lambda *_: self._sync_target())
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
        lay.addWidget(cfg)
        if initial:
            self._seed(initial)
        self._sync_target()

    def _sync_target(self) -> None:
        _kind, needs_member = self.resp.currentData()
        self.member.setEnabled(needs_member)
        self.end.setEnabled(needs_member)
        self.node.setEnabled(not needs_member)

    def _seed(self, p: dict) -> None:
        _select_ids(self.lane_list, p.get("lane"))
        vi = self.vehicle.findData(p.get("vehicle"))
        if vi >= 0:
            self.vehicle.setCurrentIndex(vi)
        resp = p.get("response") or ()
        if resp:
            for i in range(self.resp.count()):
                if self.resp.itemData(i)[0] == resp[0]:
                    self.resp.setCurrentIndex(i)
                    break
            target = resp[1] if len(resp) > 1 else None
            mi = self.member.findData(target)
            if mi >= 0:
                self.member.setCurrentIndex(mi)
            ni = self.node.findData(target)
            if ni >= 0:
                self.node.setCurrentIndex(ni)
            ei = self.end.findData(resp[2] if len(resp) > 2 else "i")
            if ei >= 0:
                self.end.setCurrentIndex(ei)

    def _lane_nodes(self) -> list:
        by_id = {n.id: n for n in self._project.nodes}
        ids = _selected_ids(self.lane_list)
        ids.sort(key=lambda i: (by_id[i].x, getattr(by_id[i], "y", 0)))
        return ids

    def case_params(self) -> dict:
        kind, needs_member = self.resp.currentData()
        target = (self.member.currentData() if needs_member
                  else self.node.currentData())
        return {"lane": self._lane_nodes(),
                "vehicle": self.vehicle.currentData(),
                "response": (kind, target, self.end.currentData())}


class InfluenceSurfaceBody(_Body):
    """Influence surface / multi-lane (3-D deck): deck nodes + response + vehicle."""

    TITLE = "Influence surface / multi-lane moving load"
    SUB = ("A unit load traverses the deck; the influence surface drives the "
           "AASHTO multi-lane vehicle envelope.")
    SUPPORTS_IC = False

    def __init__(self, project, *, initial: dict | None = None,
                 max_modes: int = 20, default_modes: int = 6):
        super().__init__()
        lay = self._root()
        deck = GroupCard("Deck surface", form=False)
        deck.body_layout().addWidget(
            QLabel("Deck nodes the load can stand on (plan = X, Y):"))
        self.deck_list = _multi_list(120)
        for nd in project.nodes:
            it = _add_id_row(
                self.deck_list,
                f"{nd.id}:  ({nd.x:g}, {getattr(nd, 'y', 0):g}, "
                f"{getattr(nd, 'z', 0):g})", nd.id)
            it.setSelected(True)
        deck.body_layout().addWidget(self.deck_list)
        lay.addWidget(deck)
        cfg = GroupCard("Vehicle & response")
        self.resp = QComboBox()
        for label, key in _IS_RESPONSES:
            self.resp.addItem(label, key)
        cfg.add_row("Response", self.resp)
        self.node = QComboBox()
        for nd in project.nodes:
            self.node.addItem(f"node {nd.id}", nd.id)
        if project.nodes:
            self.node.setCurrentIndex(len(project.nodes) // 2)
        cfg.add_row("At node", self.node)
        self.vehicle = QComboBox()
        for label, key in _IS_VEHICLES:
            self.vehicle.addItem(label, key)
        cfg.add_row("Vehicle", self.vehicle)
        self.multi = QCheckBox("Apply AASHTO multiple-presence factors")
        self.multi.setChecked(True)
        cfg.add_full_row(self.multi)
        lay.addWidget(cfg)
        if initial:
            self._seed(initial)

    def _seed(self, p: dict) -> None:
        _select_ids(self.deck_list, p.get("deck"))
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

    def case_params(self) -> dict:
        return {"deck": _selected_ids(self.deck_list),
                "response": (self.resp.currentData(), self.node.currentData()),
                "vehicle": self.vehicle.currentData(),
                "multi_presence": self.multi.isChecked()}


class VehicleDynamicsBody(_Body):
    """Vehicle dynamics: lane + moving-force / sprung-mass vehicle at speed.

    Fixed-factor display↔SI conversions (t↔kg, %↔fraction, km/h↔m/s) match the
    standalone dialog so a case round-trips."""

    TITLE = "Vehicle dynamics / moving-load time-history"
    SUB = ("A vehicle crosses the lane at speed; the transient solve gives the "
           "dynamic amplification (DAF) and response history.")
    SUPPORTS_IC = False

    def __init__(self, project, *, initial: dict | None = None,
                 max_modes: int = 20, default_modes: int = 6):
        super().__init__()
        self._project = project
        lay = self._root()
        lane = GroupCard("Lane", form=False)
        lane.body_layout().addWidget(
            QLabel("Nodes the vehicle travels over (ordered by X):"))
        self.lane_list = _multi_list(100)
        for nd in sorted(project.nodes, key=lambda n: (n.x, getattr(n, "y", 0))):
            it = _add_id_row(self.lane_list,
                             f"{nd.id}:  ({nd.x:g}, {getattr(nd, 'y', 0):g})",
                             nd.id)
            it.setSelected(True)
        lane.body_layout().addWidget(self.lane_list)
        lay.addWidget(lane)
        cfg = GroupCard("Vehicle & speed")
        self.kind = QComboBox()
        self.kind.addItem("Moving force (constant axles)", "force")
        self.kind.addItem("Sprung-mass (interaction)", "vbi")
        self.kind.currentIndexChanged.connect(
            lambda i: self.stack.setCurrentIndex(i))
        cfg.add_row("Analysis", self.kind)
        self.stack = QStackedWidget()
        force = GroupCard("Axle train")
        self.vehicle = QComboBox()
        for label, key in _VD_VEHICLES:
            self.vehicle.addItem(label, key)
        force.add_row("Vehicle", self.vehicle)
        vbi = GroupCard("Sprung-mass vehicle")
        self.mass = _dspin(20.0, 0.1, 1000.0)
        self.bounce = _dspin(2.0, 0.1, 20.0)
        self.susp = _dspin(10.0, 0.0, 80.0)
        vbi.add_row("Sprung mass [t]", self.mass)
        vbi.add_row("Bounce frequency [Hz]", self.bounce)
        vbi.add_row("Suspension damping [%]", self.susp)
        self.stack.addWidget(force)
        self.stack.addWidget(vbi)
        cfg.add_full_row(self.stack)
        self.speed = _dspin(60.0, 1.0, 500.0)
        cfg.add_row("Speed [km/h]", self.speed)
        self.zeta = _dspin(2.0, 0.0, 20.0)
        cfg.add_row("Bridge damping ζ [%]", self.zeta)
        self.node = QComboBox()
        for nd in project.nodes:
            self.node.addItem(f"node {nd.id}", nd.id)
        if project.nodes:
            self.node.setCurrentIndex(len(project.nodes) // 2)
        cfg.add_row("Response node", self.node)
        lay.addWidget(cfg)
        if initial:
            self._seed(initial)

    def _seed(self, p: dict) -> None:
        _select_ids(self.lane_list, p.get("lane"))
        ki = self.kind.findData(p.get("kind", "force"))
        if ki >= 0:
            self.kind.setCurrentIndex(ki)
        vi = self.vehicle.findData(p.get("vehicle"))
        if vi >= 0:
            self.vehicle.setCurrentIndex(vi)
        self.mass.setValue(float(p.get("mass", 20000.0)) / 1.0e3)
        self.bounce.setValue(float(p.get("bounce", 2.0)))
        self.susp.setValue(float(p.get("susp_damp", 0.10)) * 100.0)
        self.speed.setValue(float(p.get("speed", 16.667)) * 3.6)
        self.zeta.setValue(float(p.get("zeta", 0.02)) * 100.0)
        ni = self.node.findData(p.get("node"))
        if ni >= 0:
            self.node.setCurrentIndex(ni)

    def _lane(self) -> list:
        by_id = {n.id: n for n in self._project.nodes}
        ids = _selected_ids(self.lane_list)
        ids.sort(key=lambda i: (by_id[i].x, getattr(by_id[i], "y", 0)))
        return ids

    def case_params(self) -> dict:
        return {"lane": self._lane(), "kind": self.kind.currentData(),
                "vehicle": self.vehicle.currentData(),
                "mass": float(self.mass.value()) * 1.0e3,
                "bounce": float(self.bounce.value()),
                "susp_damp": float(self.susp.value()) / 100.0,
                "speed": float(self.speed.value()) / 3.6,
                "zeta": float(self.zeta.value()) / 100.0,
                "node": self.node.currentData()}


class LoadRatingBody(_Body):
    """AASHTO LRFR load rating: rated effect + capacity/dead loads + factors.

    Capacity / dead loads are entered in display units and stored SI via the
    project's :class:`UnitSystem` (reused, so no duplicated conversion math)."""

    TITLE = "Load rating — AASHTO LRFR"
    SUB = ("RF = (C − γDC·DC − γDW·DW − γP·P) / (γLL·(LL+IM)); the HL-93 "
           "live-load effect comes from the influence line of the rated effect.")
    SUPPORTS_IC = False

    def __init__(self, project, *, initial: dict | None = None,
                 max_modes: int = 20, default_modes: int = 6):
        super().__init__()
        self._project = project
        self._us = UnitSystem.from_project(project)
        lay = self._root()
        lane = GroupCard("Lane", form=False)
        lane.body_layout().addWidget(
            QLabel("Nodes the live load travels over (ordered by X):"))
        self.lane_list = _multi_list(96)
        for nd in sorted(project.nodes, key=lambda n: (n.x, getattr(n, "y", 0))):
            it = _add_id_row(self.lane_list,
                             f"{nd.id}:  ({nd.x:g}, {getattr(nd, 'y', 0):g})",
                             nd.id)
            it.setSelected(True)
        lane.body_layout().addWidget(self.lane_list)
        lay.addWidget(lane)
        eff = GroupCard("Rated effect")
        self.effect = QComboBox()
        for label, spec in _EFFECTS:
            self.effect.addItem(label, spec)
        self.effect.currentIndexChanged.connect(lambda *_: self._sync_units())
        eff.add_row("Effect", self.effect)
        self.member = QComboBox()
        for mb in project.members:
            self.member.addItem(f"member {mb.id}  ({mb.n1}→{mb.n2})", mb.id)
        eff.add_row("Member", self.member)
        self.end = QComboBox()
        self.end.addItem("end i", "i")
        self.end.addItem("end j", "j")
        eff.add_row("At end", self.end)
        lay.addWidget(eff)
        cap = GroupCard("Capacity & dead load")
        self.Rn = _lr_spin(1000.0)
        self.DC = _lr_spin(200.0)
        self.DW = _lr_spin(50.0)
        self.P = _lr_spin(0.0, lo=-1.0e12)
        cap.add_row("Nominal resistance Rn", self.Rn)
        cap.add_row("Dead load — components DC", self.DC)
        cap.add_row("Dead load — wearing surface DW", self.DW)
        cap.add_row("Other permanent P (signed)", self.P)
        lay.addWidget(cap)
        fac = GroupCard("Resistance & evaluation factors")
        self.phi = _lr_spin(1.00, lo=0.0, hi=1.5, step=0.05, decimals=2)
        fac.add_row("Resistance factor φ", self.phi)
        self.phi_c = QComboBox()
        for label, v in _CONDITION:
            self.phi_c.addItem(label, v)
        fac.add_row("Condition factor φc", self.phi_c)
        self.phi_s = QComboBox()
        for label, v in _SYSTEM:
            self.phi_s.addItem(label, v)
        fac.add_row("System factor φs", self.phi_s)
        self.im = _lr_spin(0.33, lo=0.0, hi=1.0, step=0.01, decimals=2)
        fac.add_row("Dynamic allowance IM", self.im)
        lay.addWidget(fac)
        lvl = GroupCard("Rating levels")
        lvl.body_layout().addWidget(
            QLabel("Design Inventory (γLL 1.75) and Operating (1.35) are always "
                   "computed."))
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
        self.permit_gamma = _lr_spin(1.15, lo=0.5, hi=2.5, step=0.05, decimals=2)
        self.permit_gamma.setEnabled(False)
        lvl.add_row("  Permit γLL", self.permit_gamma)
        lay.addWidget(lvl)
        if initial:
            self._seed(initial)
        self._sync_units()

    def _quantity(self):
        return self.effect.currentData()[1]

    def _sync_units(self) -> None:
        unit = " " + self._us.label(self._quantity())
        for sb in (self.Rn, self.DC, self.DW, self.P):
            sb.setSuffix(unit)

    def _seed(self, p: dict) -> None:
        resp = p.get("response") or ("M", None, "i")
        for i in range(self.effect.count()):
            if self.effect.itemData(i)[0] == resp[0]:
                self.effect.setCurrentIndex(i)
                break
        qty = self._quantity()
        to_disp = self._us.to_display
        self.Rn.setValue(to_disp(float(p.get("Rn", 0.0)), qty))
        self.DC.setValue(to_disp(float(p.get("DC", 0.0)), qty))
        self.DW.setValue(to_disp(float(p.get("DW", 0.0)), qty))
        self.P.setValue(to_disp(float(p.get("P", 0.0)), qty))
        _select_ids(self.lane_list, p.get("lane"))
        mi = self.member.findData(resp[1] if len(resp) > 1 else None)
        if mi >= 0:
            self.member.setCurrentIndex(mi)
        ei = self.end.findData(resp[2] if len(resp) > 2 else "i")
        if ei >= 0:
            self.end.setCurrentIndex(ei)
        self.phi.setValue(float(p.get("phi", 1.0)))
        ci = self.phi_c.findData(float(p.get("phi_c", 1.0)))
        if ci >= 0:
            self.phi_c.setCurrentIndex(ci)
        psi = self.phi_s.findData(float(p.get("phi_s", 1.0)))
        if psi >= 0:
            self.phi_s.setCurrentIndex(psi)
        self.im.setValue(float(p.get("im", 0.33)))
        adtt = p.get("adtt")
        self.legal.setChecked(adtt is not None)
        if adtt is not None:
            self.adtt.setValue(int(adtt))
        pg = p.get("permit_gamma_LL")
        self.permit.setChecked(pg is not None)
        if pg is not None:
            self.permit_gamma.setValue(float(pg))

    def _lane(self) -> list:
        by_id = {n.id: n for n in self._project.nodes}
        ids = _selected_ids(self.lane_list)
        ids.sort(key=lambda i: (by_id[i].x, getattr(by_id[i], "y", 0)))
        return ids

    def case_params(self) -> dict:
        comp, qty = self.effect.currentData()
        to_si = self._us.to_si
        return {"lane": self._lane(),
                "response": (comp, self.member.currentData(),
                             self.end.currentData()),
                "Rn": to_si(self.Rn.value(), qty), "DC": to_si(self.DC.value(), qty),
                "DW": to_si(self.DW.value(), qty), "P": to_si(self.P.value(), qty),
                "phi": self.phi.value(), "phi_c": self.phi_c.currentData(),
                "phi_s": self.phi_s.currentData(), "im": self.im.value(),
                "adtt": self.adtt.value() if self.legal.isChecked() else None,
                "permit_gamma_LL": (self.permit_gamma.value()
                                    if self.permit.isChecked() else None)}


class TimeHistoryBody(_Body):
    """Nonlinear time history: monitor + saved ground-motion function + damping.

    IC-capable (can continue from a nonlinear case's committed state); the shell
    shows the *hold source loads* checkbox and injects ``hold_source_loads`` into
    the params (see :attr:`HOLD_IN_PARAMS`)."""

    TITLE = "Nonlinear time history"
    SUB = ("Base excitation from a saved ground-motion function (Analysis ▸ "
           "Functions); nonlinear (fiber) transient solve.")
    SUPPORTS_IC = True
    SHOW_HOLD = True
    HOLD_IN_PARAMS = True

    def __init__(self, project, *, initial: dict | None = None,
                 max_modes: int = 20, default_modes: int = 6):
        super().__init__()
        lay = self._root()
        node_ids = [n.id for n in project.nodes]
        free = [n.id for n in project.nodes
                if not (n.supports and any(n.supports))]
        default_node = (free[-1] if free else
                        (node_ids[-1] if node_ids else None))
        monitor = GroupCard("Monitor")
        self.node = QComboBox()
        for i in node_ids:
            self.node.addItem(str(i), i)
        if default_node is not None:
            di = self.node.findData(default_node)
            if di >= 0:
                self.node.setCurrentIndex(di)
        self.direction = QComboBox()
        for label, data in _TH_DIRS:
            self.direction.addItem(label, data)
        di = self.direction.findData(("y", 1))
        if di >= 0:
            self.direction.setCurrentIndex(di)
        monitor.add_row("Monitor node", self.node)
        monitor.add_row("Direction", self.direction)
        lay.addWidget(monitor)
        gm = GroupCard("Ground motion")
        self.function = QComboBox()
        for f in project.th_functions:
            self.function.addItem(
                f"{f.name}  ({f.npts} pts · {f.duration:.3g}s"
                f"{' · g' if f.in_g else ''})", f.id)
        gm.add_row("Function", self.function)
        self._no_fn = QLabel("No time-history functions yet — define them in "
                             "Analysis ▸ Functions.")
        self._no_fn.setObjectName("hintLabel")
        self._no_fn.setWordWrap(True)
        self._no_fn.setVisible(self.function.count() == 0)
        gm.add_full_row(self._no_fn)
        self.scale = _dspin(1.0, decimals=4, step=0.1)
        gm.add_row("Scale factor", self.scale)
        lay.addWidget(gm)
        model = GroupCard("Damping & mass")
        self.zeta = _dspin(0.05, decimals=3, step=0.01)
        self.density = _dspin(2400.0, 0.0, 1.0e15, decimals=1, step=100.0)
        model.add_row("Damping ζ", self.zeta)
        model.add_row(f"Density [kg/{project.length_unit}³]", self.density)
        lay.addWidget(model)
        if initial:
            self._seed(initial)

    def validate(self) -> str | None:
        if self.function.count() == 0 or self.function.currentData() is None:
            return ("Define a time-history function (Analysis ▸ Functions) and "
                    "select it first.")
        return None

    def _seed(self, p: dict) -> None:
        ni = self.node.findData(p.get("control_node"))
        if ni >= 0:
            self.node.setCurrentIndex(ni)
        want = p.get("direction", "y")
        for i in range(self.direction.count()):
            if self.direction.itemData(i)[0] == want:
                self.direction.setCurrentIndex(i)
                break
        fi = self.function.findData(p.get("function_id"))
        if fi >= 0:
            self.function.setCurrentIndex(fi)
        self.scale.setValue(float(p.get("scale", 1.0)))
        self.zeta.setValue(float(p.get("zeta", 0.05)))
        self.density.setValue(float(p.get("density", 2400.0)))

    def case_params(self) -> dict:
        return {"function_id": self.function.currentData(),
                "control_node": self.node.currentData(),
                "direction": self.direction.currentData()[0],
                "scale": float(self.scale.value()),
                "zeta": float(self.zeta.value()),
                "density": float(self.density.value())}


class LinearStaticBody(_Body):
    """Linear static: a *Loads Applied* grid — one scale per load pattern.

    Solves K·u = F where F is the listed patterns summed at their scales
    (E3b). A pattern left at 0 is not applied, so this is how you save
    "1.0 Dead + 0.5 Live" as a named case without building a combination."""

    TITLE = "Linear static"
    SUB = ("Applies each load pattern at the scale below (summed) and solves "
           "K·u = F. Leave a pattern at 0 to omit it.")
    SUPPORTS_IC = False

    _READONLY = (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

    def __init__(self, project, *, initial: dict | None = None, **_ignored):
        super().__init__()
        from project import NATURE_LABELS, normalize_loads_applied
        lay = self._root()
        self._patterns = list(project.load_cases)
        card = GroupCard("Loads applied", form=False)
        self.grid = QTableWidget(len(self._patterns), 3)
        self.grid.setHorizontalHeaderLabels(["Load pattern", "Nature",
                                             "Scale factor"])
        self.grid.verticalHeader().setVisible(False)
        self.grid.horizontalHeader().setStretchLastSection(True)
        self._spins: list[QDoubleSpinBox] = []
        for r, c in enumerate(self._patterns):
            name_it = QTableWidgetItem(c.name)
            name_it.setFlags(self._READONLY)
            self.grid.setItem(r, 0, name_it)
            nat_it = QTableWidgetItem(NATURE_LABELS.get(c.nature, c.nature))
            nat_it.setFlags(self._READONLY)
            self.grid.setItem(r, 1, nat_it)
            spin = _dspin(0.0, lo=-1000.0, hi=1000.0, decimals=3, step=0.25)
            self.grid.setCellWidget(r, 2, spin)
            self._spins.append(spin)
        card.body_layout().addWidget(self.grid)
        lay.addWidget(card)
        # Seed the spins: an existing case's saved rows, else a friendly default
        # of the first pattern ×1 (a runnable "1.0 <first>" out of the box).
        seed = {pid: s for pid, s in
                normalize_loads_applied((initial or {}).get("loads_applied"))}
        if not initial and self._patterns:
            seed = {self._patterns[0].id: 1.0}
        for c, spin in zip(self._patterns, self._spins):
            spin.setValue(seed.get(c.id, 0.0))

    def _rows(self) -> list:
        """The non-zero (pattern_id, scale) rows the user set."""
        return [[c.id, float(sp.value())]
                for c, sp in zip(self._patterns, self._spins)
                if sp.value()]

    def case_params(self) -> dict:
        return {"loads_applied": self._rows()}

    def validate(self) -> str | None:
        if not self._patterns:
            return "Define a load pattern first (Loads ▸ Patterns)."
        if not self._rows():
            return "Set a non-zero scale on at least one load pattern."
        return None


def _multi_list(min_h: int) -> QListWidget:
    lst = QListWidget()
    lst.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
    lst.setMinimumHeight(min_h)
    return lst


def _add_id_row(lst: QListWidget, label: str, ident) -> QListWidgetItem:
    it = QListWidgetItem(label)
    it.setData(_ROLE, ident)
    lst.addItem(it)
    return it


def _selected_ids(lst: QListWidget) -> list:
    return [it.data(_ROLE) for it in lst.selectedItems()]


def _select_ids(lst: QListWidget, ids) -> None:
    want = set(ids or [])
    for i in range(lst.count()):
        it = lst.item(i)
        it.setSelected(it.data(_ROLE) in want)


# type_id -> (menu label, body class). The unified editor's Type ▾ order.
REGISTRY: list[tuple[str, str, type]] = [
    ("linstatic", "Linear Static", LinearStaticBody),
    ("modal", "Modal", ModalBody),
    ("buckling", "Buckling", BucklingBody),
    ("responsespectrum", "Response Spectrum", ResponseSpectrumBody),
    ("tempgradient", "Temperature Gradient", TemperatureGradientBody),
    ("cabletuning", "Cable Tuning", CableTuningBody),
    ("movingload", "Moving Load", MovingLoadBody),
    ("influencesurface", "Influence Surface", InfluenceSurfaceBody),
    ("loadrating", "Load Rating", LoadRatingBody),
    ("vehicledynamics", "Vehicle Dynamics", VehicleDynamicsBody),
    ("timehistory", "Time History", TimeHistoryBody),
]
