"""Response-spectrum analysis setup — the *Load Case Data ▸ Response Spectrum*
counterpart of CSiBridge / MIDAS.

``ResponseSpectrumDialog`` defines the pseudo-acceleration spectrum and the
modal-combination parameters that
:class:`femsolver.analysis.response_spectrum.ResponseSpectrumAnalysis` needs.
The spectrum can come from a design code — **ASCE 7**, **Eurocode 8**, or
**IS 1893** — or a **custom table** of ``(T, Sa)`` points; a live plot redraws
as the inputs change, so what you see is exactly the ``Sa(T)`` the engine will
sample. It runs nothing: :meth:`configure` returns
``(spectrum, num_modes, direction, combination)`` or ``None`` if cancelled.

Built on the shared card scaffold (:mod:`analysis_ui`) + a matplotlib Agg
canvas — headless-constructible under ``QT_QPA_PLATFORM=offscreen``.
"""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog,
                               QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

import style
from analysis_ui import CaseHeader, GroupCard, dialog_buttons

_G = 9.80665                    # gravity (m/s²) — converts Sa in g → m/s²


# ------------------------------------------------------------ code Sa(T) forms
def asce7_Sa(T: float, *, SDS: float, SD1: float, TL: float) -> float:
    """ASCE 7 design spectrum (Sec. 11.4.6), returned in m/s²."""
    if SDS <= 0.0:
        return 0.0
    T0 = 0.2 * SD1 / SDS
    TS = SD1 / SDS
    if T < T0:
        Sa_g = SDS * (0.4 + 0.6 * T / T0) if T0 > 0 else SDS
    elif T <= TS:
        Sa_g = SDS
    elif T <= TL:
        Sa_g = SD1 / T
    else:
        Sa_g = SD1 * TL / (T * T)
    return _G * Sa_g


def _clean_custom_points(raw):
    """Sorted, distinct-period ``(T, Sa)`` points from a raw list of pairs."""
    pts = []
    for pair in raw or []:
        try:
            pts.append((float(pair[0]), float(pair[1])))
        except (TypeError, ValueError, IndexError):
            continue
    pts.sort(key=lambda p: p[0])
    out, seen = [], set()
    for T, Sa in pts:
        if T in seen:
            continue
        seen.add(T)
        out.append((T, Sa))
    return out


def spectrum_from_params(params: dict):
    """Build a :class:`femsolver.ResponseSpectrum` from a saved case's
    JSON-friendly **inputs** (source + per-code parameters + damping) — the
    headless counterpart of :meth:`ResponseSpectrumDialog.build_spectrum`, used
    both by the live preview and by :mod:`case_types` at Run time. Raises
    ``ValueError`` for an under-defined custom table."""
    from femsolver import ResponseSpectrum

    zeta = float(params.get("damping", 0.05))
    src = params.get("source", "asce7")
    if src == "custom":
        pts = _clean_custom_points(params.get("custom"))
        if len(pts) < 2:
            raise ValueError("a custom spectrum needs at least two points with "
                             "distinct periods")
        return ResponseSpectrum([p[0] for p in pts], [p[1] for p in pts],
                                damping_ratio=zeta)
    if src == "asce7":
        a = params.get("asce7", {})
        SDS, SD1, TL = (a.get("SDS", 1.0), a.get("SD1", 0.6), a.get("TL", 8.0))
        fn = lambda T: asce7_Sa(T, SDS=SDS, SD1=SD1, TL=TL)  # noqa: E731
        t_max = max(10.0, TL)
    elif src == "ec8":
        from femsolver.design import ec8
        e = params.get("ec8", {})
        fn = lambda T: ec8.design_spectrum_Sd(  # noqa: E731
            T, a_g=e.get("ag", 2.5), ground_type=e.get("ground", "C"),
            q=e.get("q", 1.5), spectrum_type=e.get("type", 1))
        t_max = 10.0
    else:                                                    # is1893
        from femsolver.design import is1893
        i = params.get("is1893", {})
        fn = lambda T: is1893.Ah_coefficient(  # noqa: E731
            T=T, zone=i.get("zone", 4), importance=i.get("I", 1.0),
            R=i.get("R", 5.0), soil_type=i.get("soil", 2))["A_h"] * _G
        t_max = 6.0
    return ResponseSpectrum.from_function(
        fn, T_min=0.02, T_max=t_max, n_points=200, damping_ratio=zeta)


class ResponseSpectrumDialog(QDialog):
    """Define a response spectrum + modal-combination parameters."""

    def __init__(self, parent, *, ndm: int = 2, max_modes: int = 20,
                 default_modes: int = 6, initial: dict | None = None,
                 name: str = "Response Spectrum", notes: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Response spectrum")
        self._max_modes = max_modes = max(1, int(max_modes))
        default_modes = max(1, min(int(default_modes), max_modes))

        outer = QHBoxLayout(self)
        outer.setContentsMargins(style.SP_LG, style.SP_LG,
                                 style.SP_LG, style.SP_LG)
        outer.setSpacing(style.SP_MD)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(style.SP_MD)

        self.header = CaseHeader(name=name, type_label="Response Spectrum",
                                 notes=notes)
        lv.addWidget(self.header)

        # ---- spectrum source + swappable parameter pages ----
        src_card = GroupCard("Spectrum")
        self.source = QComboBox()
        self.source.addItem("ASCE 7", "asce7")
        self.source.addItem("Eurocode 8", "ec8")
        self.source.addItem("IS 1893", "is1893")
        self.source.addItem("Custom table", "custom")
        src_card.add_row("Source", self.source)
        self.stack = QStackedWidget()
        self.stack.addWidget(self._asce7_page())
        self.stack.addWidget(self._ec8_page())
        self.stack.addWidget(self._is1893_page())
        self.stack.addWidget(self._custom_page())
        src_card.add_full_row(self.stack)
        lv.addWidget(src_card)

        # ---- analysis parameters ----
        an = GroupCard("Analysis")
        self.modes = QSpinBox()
        self.modes.setRange(1, max_modes)
        self.modes.setValue(default_modes)
        an.add_row("Number of modes", self.modes)
        self.direction = QComboBox()
        self.direction.addItem("X", "x")
        self.direction.addItem("Y", "y")
        if ndm >= 3:
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
        lv.addWidget(an)

        lv.addStretch(1)
        lv.addWidget(dialog_buttons(self))
        outer.addWidget(left, 0)

        # ---- live Sa(T) preview ----
        self._fig = Figure(figsize=(3.8, 3.0), layout="constrained")
        self._ax = self._fig.add_subplot(111)
        self._canvas = Canvas(self._fig)
        self._canvas.setMinimumWidth(360)
        outer.addWidget(self._canvas, 1)

        self.source.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.source.currentIndexChanged.connect(self._redraw)
        self.damping.valueChanged.connect(self._redraw)
        if initial:
            self._seed(initial)
        style.apply(self)
        self._redraw()

    # ------------------------------------------------------------- param pages
    def _spin(self, value, *, decimals=3, lo=0.0, hi=1.0e6, step=0.1):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(value)
        s.valueChanged.connect(self._redraw)
        return s

    def _asce7_page(self) -> QWidget:
        card = GroupCard("ASCE 7 parameters")
        self.a_SDS = self._spin(1.0, step=0.05)
        self.a_SD1 = self._spin(0.6, step=0.05)
        self.a_TL = self._spin(8.0, step=1.0, decimals=1)
        card.add_row("S_DS [g]", self.a_SDS)
        card.add_row("S_D1 [g]", self.a_SD1)
        card.add_row("T_L [s]", self.a_TL)
        return card

    def _ec8_page(self) -> QWidget:
        card = GroupCard("Eurocode 8 parameters")
        self.e_ag = self._spin(2.5, step=0.1)          # m/s²
        self.e_ground = QComboBox()
        for g in ("A", "B", "C", "D", "E"):
            self.e_ground.addItem(g, g)
        self.e_ground.setCurrentText("C")
        self.e_ground.currentIndexChanged.connect(self._redraw)
        self.e_q = self._spin(1.5, step=0.5, lo=1.0)
        self.e_type = QComboBox()
        self.e_type.addItem("Type 1 (M ≥ 5.5)", 1)
        self.e_type.addItem("Type 2 (M < 5.5)", 2)
        self.e_type.currentIndexChanged.connect(self._redraw)
        card.add_row("a_g [m/s²]", self.e_ag)
        card.add_row("Ground type", self.e_ground)
        card.add_row("Behaviour q", self.e_q)
        card.add_row("Spectrum", self.e_type)
        return card

    def _is1893_page(self) -> QWidget:
        card = GroupCard("IS 1893 parameters")
        self.i_zone = QComboBox()
        for z, name in ((2, "II (0.10)"), (3, "III (0.16)"),
                        (4, "IV (0.24)"), (5, "V (0.36)")):
            self.i_zone.addItem(f"Zone {name}", z)
        self.i_zone.setCurrentIndex(2)                  # zone IV
        self.i_zone.currentIndexChanged.connect(self._redraw)
        self.i_I = self._spin(1.0, step=0.1, lo=0.1)
        self.i_R = self._spin(5.0, step=0.5, lo=1.0)
        self.i_soil = QComboBox()
        for s, name in ((1, "I — Rock/Hard"), (2, "II — Medium"),
                        (3, "III — Soft")):
            self.i_soil.addItem(name, s)
        self.i_soil.currentIndexChanged.connect(self._redraw)
        card.add_row("Seismic zone", self.i_zone)
        card.add_row("Importance I", self.i_I)
        card.add_row("Response R", self.i_R)
        card.add_row("Soil type", self.i_soil)
        return card

    def _custom_page(self) -> QWidget:
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
        self.custom.itemChanged.connect(lambda *_: self._redraw())
        v.addWidget(self.custom)
        row = QHBoxLayout()
        add = QPushButton("Add point")
        add.clicked.connect(lambda: (self._add_custom_row(0.0, 0.0),
                                     self._redraw()))
        rem = QPushButton("Remove point")
        rem.clicked.connect(self._remove_custom_row)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        v.addLayout(row)
        return page

    def _add_custom_row(self, T: float, Sa: float) -> None:
        r = self.custom.rowCount()
        self.custom.insertRow(r)
        self.custom.setItem(r, 0, QTableWidgetItem(f"{T:g}"))
        self.custom.setItem(r, 1, QTableWidgetItem(f"{Sa:g}"))

    def _remove_custom_row(self) -> None:
        r = self.custom.currentRow()
        if r < 0:
            r = self.custom.rowCount() - 1
        if r >= 0:
            self.custom.removeRow(r)
            self._redraw()

    # ---------------------------------------------------------- build spectrum
    def _current_source(self) -> str:
        return self.source.currentData()

    def _custom_points(self):
        pts = []
        for r in range(self.custom.rowCount()):
            try:
                T = float(self.custom.item(r, 0).text())
                Sa = float(self.custom.item(r, 1).text())
            except (AttributeError, ValueError):
                continue
            pts.append((T, Sa))
        pts.sort(key=lambda p: p[0])
        # drop duplicate periods (keep first) so the table is strictly increasing
        out, seen = [], set()
        for T, Sa in pts:
            if T in seen:
                continue
            seen.add(T)
            out.append((T, Sa))
        return out

    def params(self) -> dict:
        """The spectrum **inputs** as a JSON-friendly dict (all source pages, so
        switching source in a later Modify keeps the others). This is what a
        saved :class:`project.AnalysisCase` stores; :func:`spectrum_from_params`
        rebuilds the :class:`ResponseSpectrum` from it."""
        return {
            "source": self._current_source(),
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

    def _seed(self, p: dict) -> None:
        """Seed every widget from a saved case's params."""
        self.damping.setValue(float(p.get("damping", 0.05)))
        self.modes.setValue(max(1, min(int(p.get("num_modes", 6)),
                                       self._max_modes)))
        di = self.direction.findData(p.get("direction", "x"))
        if di >= 0:
            self.direction.setCurrentIndex(di)
        ci = self.combination.findData(p.get("combination", "cqc"))
        if ci >= 0:
            self.combination.setCurrentIndex(ci)
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
            for pair in _clean_custom_points(custom):
                self._add_custom_row(pair[0], pair[1])
        # source last: its signal swaps the visible page + redraws
        si = self.source.findData(p.get("source", "asce7"))
        if si >= 0:
            self.source.setCurrentIndex(si)

    def build_spectrum(self):
        """Construct a :class:`ResponseSpectrum` from the current inputs, or
        raise ``ValueError`` if a custom table is under-defined."""
        return spectrum_from_params(self.params())

    def _redraw(self) -> None:
        self._ax.clear()
        try:
            spec = self.build_spectrum()
            import numpy as np
            Tg = np.linspace(float(spec.periods[0]), float(spec.periods[-1]),
                             250)
            Sa = [spec.Sa(t) for t in Tg]
            self._ax.plot(Tg, Sa, "-", lw=1.8, color=style.C_PRIMARY)
            self._ax.set_xlabel("period T (s)")
            self._ax.set_ylabel("Sa (m/s²)")
            self._ax.set_title("Design spectrum")
            self._ax.set_ylim(bottom=0.0)
            style.beautify_axes(self._ax)
        except Exception as exc:                          # noqa: BLE001
            self._ax.text(0.5, 0.5, f"({exc})", ha="center", va="center",
                          transform=self._ax.transAxes, fontsize=8,
                          color=style.MUTED, wrap=True)
        self._canvas.draw_idle()

    # ---------------------------------------------------------------- result
    def result(self):
        """``(spectrum, num_modes, direction, combination)``. Raises
        ``ValueError`` for an invalid custom table."""
        return (self.build_spectrum(), int(self.modes.value()),
                self.direction.currentData(), self.combination.currentData())

    @classmethod
    def configure(cls, parent, *, ndm: int = 2, max_modes: int = 20,
                  default_modes: int = 6):
        from PySide6.QtWidgets import QMessageBox
        dlg = cls(parent, ndm=ndm, max_modes=max_modes,
                  default_modes=default_modes)
        while dlg.exec():
            try:
                return dlg.result()
            except ValueError as exc:
                QMessageBox.warning(dlg, "Response spectrum", str(exc))
        return None
