"""General Section Designer — a desktop (Qt) front-end over ``section_gui_core``.

Desktop counterpart of the Streamlit Section Designer (``streamlit_app.py``).
Both drive the *same* GUI-agnostic engine, ``section_gui_core`` — none of the
section maths lives here. Only the presentation layer differs:

    Streamlit                     Desktop (this module)
    ----------------------------  --------------------------------------
    @st.cache_data                functools.lru_cache (on the Spec)
    st.* widgets                  Qt widgets
    svg_of() -> st.image          svg_of() -> QSvgWidget
    Plotly / Altair               matplotlib (FigureCanvasQTAgg)
    st.dataframe                  QTableWidget
    st.html(report_html)          QTextBrowser(report_html)

Opens as a standalone window from the FEM app (Tools ▸ Section Designer); the
Streamlit app is left untouched.

Parity implemented here
-----------------------
* Parametric kinds: Rectangular, Circular, L, T, Hollow box, **PSC girder**
  (incl. strand parameters).
* Display **units** (force / length / stress) that drive the analysis charts,
  metrics and the report — via ``core.Units`` (``core.FORCE_N`` / ``LENGTH_M``
  / ``STRESS_PA``).
* **Constitutive models** — concrete model + steel model and their strain
  parameters (``core.CONC_MODELS`` / ``STEEL_MODELS``). These are ``Spec``
  fields and are now passed through ``core.mphi_props`` to the M-φ and
  verification, so the section's material laws actually take effect.
* P-M interaction (nominal + design curve, landmarks), with a **demand check**
  (``core.demand_check``) reporting the D/C ratio and governing action.
* Moment-curvature with **neutral-axis angle** and milestone metrics /table.
* Verification table (``core.items_data`` with ``mphi_props``) and the full
  Report (``core.report_html`` with mphi + demand + meta).
* Export: section JSON, verification CSV, report HTML.

Follow-ups (engine already supports these; tracked as T2.08 on the roadmap):
Custom polygon + Composite editors, rebar/tendon *arrangement* generators, the
shared Materials library, multi-section projects, and the 3-D P-M-M surface.
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import replace
from functools import lru_cache

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-d proj.)
from PySide6.QtCore import Qt, QByteArray, QTimer
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow, QMenu,
                               QMessageBox, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

# ``section_gui_core`` / ``streamlit_app`` live at the repo root, one level
# above this ``desktop/`` package. Running ``python desktop/app.py`` only puts
# ``desktop/`` on sys.path, so make the repo root importable too.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import section_gui_core as core
import style
from widgets import CollapsibleGroup

# section kinds this desktop front-end exposes (the engine also knows "Custom"
# and "Composite" — deferred; they need dedicated geometry editors).
_KINDS = ["Rectangular", "Circular", "L-shape", "T-shape", "Hollow box",
          "PSC girder", "Custom", "Composite"]
# non-parametric kinds have their own geometry editors instead of dim fields.
_PARAMETRIC = {"Rectangular", "Circular", "L-shape", "T-shape", "Hollow box",
               "PSC girder"}

# dimension fields per kind (stored on Spec in metres; edited here in mm).
_DIMS_FOR_KIND = {
    "Rectangular": ["b", "h"],
    "Circular":    ["D"],
    "L-shape":     ["leg", "thick"],
    "T-shape":     ["h", "b", "t_f", "t_w"],
    "Hollow box":  ["b", "h", "wall_t"],
    "PSC girder":  ["b", "h"],
}
_DIM_LABEL = {
    "b": "Width b", "h": "Depth h", "D": "Diameter D",
    "leg": "Leg", "thick": "Leg thickness",
    "t_f": "Flange t_f", "t_w": "Web t_w", "wall_t": "Wall t",
}
_REBAR_FOR_KIND = {
    "Rectangular": ["n_top", "n_bot", "n_side"],
    "T-shape":     ["n_top", "n_bot", "n_side"],
    "L-shape":     ["n_top", "n_bot", "n_side"],
    "Hollow box":  ["n_top", "n_bot", "n_side"],
    "PSC girder":  ["n_top"],
    "Circular":    ["n_perim", "spiral"],
}
_REBAR_LABEL = {
    "n_top": "Top bars", "n_bot": "Bottom bars", "n_side": "Side bars/face",
    "n_perim": "Perimeter bars", "spiral": "Spiral (φ cap 0.85)",
}

# rebar / tendon layout-generator arrangements (added to the parametric bars).
# Stored on Spec.rebar_arr / .tendon_arr as nested tuples; z/y/r/w/h in metres,
# angles in degrees. Mirrors streamlit_app._ARR_CODE.
_ARR_CODE = {"Point": "point", "Line": "line", "Arc": "arc",
             "Rectangle": "rect", "Perimeter": "perim"}
_ARR_MESH = {"Coarse": (16, 17), "GSD default": (24, 29), "Fine": (36, 41)}


def _rebar_arr_label(arr) -> str:
    typ, dia, mat, p = arr
    d = f"⌀{dia * 1e3:.0f}"
    if typ == "point":
        s = f"Point {d} @ ({p[0] * 1e3:.0f}, {p[1] * 1e3:.0f}) mm"
    elif typ == "line":
        s = f"Line {int(p[0])}·{d}"
    elif typ == "arc":
        s = f"Arc {int(p[0])}·{d} (r={p[3] * 1e3:.0f} mm)"
    elif typ == "rect":
        s = f"Rect {d} ({int(p[0])}×{int(p[1])})"
    else:
        s = f"Perimeter {int(p[0])}·{d}"
    return s + (f" · {mat}" if mat else "")


def _tendon_arr_label(arr) -> str:
    typ, area, f_pe, _mat, p = arr
    n = 1 if typ == "point" else int(p[0])
    return (f"{typ.title()} Aₚ={area * 1e6:.0f}mm² f_pe={f_pe / 1e6:.0f}MPa "
            f"({n} tendon{'s' if n != 1 else ''})")


@lru_cache(maxsize=128)
def _case(spec: core.Spec):
    """Cache SectionCase by Spec (mirrors the Streamlit @st.cache_data)."""
    return core.build_case(spec)


class SectionDesignerWindow(QMainWindow):
    """Standalone General Section Designer over ``section_gui_core``."""

    def __init__(self, parent=None, *, spec: core.Spec | None = None,
                 code: str | None = None, fem_window=None,
                 section_name: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("General Section Designer")
        self.resize(1240, 860)

        self._fem = fem_window           # FEM MainWindow for the model bridge
        self._spec = spec or core.Spec()
        self._code0 = code if (code and code in core.CODES) else core.CODES[0]
        name0 = section_name or "Section 1"
        # shared material library {name: matd} — constitutive laws live here;
        # sections reference materials by name (confinement stays on the spec).
        self._materials: dict = {"C30": _conc_default(30.0),
                                 "S500": _steel_default(500.0),
                                 "Y1860": _prestress_default(1860.0)}
        # multi-section project: {name: {"spec", "code", "conc_mat", "steel_mat"}}
        self._sections: dict = {name0: {
            "spec": self._spec, "code": self._code0,
            "conc_mat": "C30", "steel_mat": "S500"}}
        self._active = name0
        self._nav_rows: dict = {}        # name -> (item, (name_lbl, sub_lbl))
        self._units = core.Units()
        self._loading = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._recompute_analysis)

        self._build_menu()
        self._build_toolbar(code)

        # workspace tabs: [Section] then the analyses. _build_analysis_panel
        # creates self.tabs (P-M … Report); the Section tab (inputs + drawing)
        # is inserted in front so it's the first tab.
        self._build_analysis_panel()
        self.tabs.insertTab(0, self._build_section_tab(), "Section")
        self.tabs.setCurrentIndex(0)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_section_nav())     # left: section navigator
        split.addWidget(self.tabs)                      # right: workspace tabs
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([210, 1030])
        split.setContentsMargins(10, 10, 10, 10)
        split.setHandleWidth(10)
        self.setCentralWidget(split)
        style.apply(self)

        self._reload_section_nav()
        self._load_form_from_spec()
        self._refresh_comp_mat_combo()
        # bring the seeded materials' laws onto the initial spec
        self._spec = self._apply_material_params(self._spec)
        self._sections[self._active]["spec"] = self._spec
        self._refresh_geometry()
        self._recompute_analysis()

    # ------------------------------------------------------------- chrome
    def _build_menu(self) -> None:
        fm = self.menuBar().addMenu("&File")
        fm.addAction("&New project", self._new_project)
        fm.addAction("&Open project…", self._open_project)
        fm.addAction("&Save project…", self._save_project)
        if self._fem is not None:
            fm.addSeparator()
            fm.addAction("&Apply active section to FEM model",
                         self._apply_to_fem)
        sm = self.menuBar().addMenu("&Sections")
        sm.addAction("&New section", self._new_section)
        sm.addAction("&Duplicate section", self._dup_section)
        sm.addAction("&Rename section…", self._rename_section)
        sm.addAction("De&lete section", self._del_section)
        mm = self.menuBar().addMenu("&Materials")
        mm.addAction("&Manage library…", self._open_materials)
        m = self.menuBar().addMenu("&Export")
        m.addAction("Section as &JSON…", self._export_section_json)
        m.addAction("&Verification as CSV…", self._export_verify_csv)
        m.addAction("&Report as HTML…", self._export_report_html)

    def _build_toolbar(self, code) -> None:
        tb = self.addToolBar("Section")
        tb.setMovable(False)
        tb.addWidget(QLabel("  Design code:  "))
        self.code_combo = QComboBox()
        self.code_combo.addItems(core.CODES)
        if code and code in core.CODES:
            self.code_combo.setCurrentText(code)
        self.code_combo.currentTextChanged.connect(lambda *_: self._queue())
        tb.addWidget(self.code_combo)
        tb.addSeparator()
        tb.addWidget(QLabel("  Units — force:"))
        self.force_combo = QComboBox()
        self.force_combo.addItems(list(core.FORCE_N))
        self.force_combo.setCurrentText("kN")
        tb.addWidget(self.force_combo)
        tb.addWidget(QLabel(" length:"))
        self.length_combo = QComboBox()
        self.length_combo.addItems(list(core.LENGTH_M))
        self.length_combo.setCurrentText("m")
        tb.addWidget(self.length_combo)
        tb.addWidget(QLabel(" stress:"))
        self.stress_combo = QComboBox()
        self.stress_combo.addItems(list(core.STRESS_PA))
        self.stress_combo.setCurrentText("MPa")
        tb.addWidget(self.stress_combo)
        for cb in (self.force_combo, self.length_combo, self.stress_combo):
            cb.currentTextChanged.connect(lambda *_: self._on_units_changed())

    # ------------------------------------------------------ navigator / tab
    def _build_section_nav(self) -> QWidget:
        """Left sidebar: navigate the sections in the model (not inputs)."""
        w = QWidget()
        w.setMinimumWidth(170)
        w.setMaximumWidth(280)
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 8, 6, 8)
        v.setSpacing(6)
        lbl = QLabel("SECTIONS")
        lbl.setObjectName("sub")
        v.addWidget(lbl)
        self.nav_list = QListWidget()
        self.nav_list.setSpacing(2)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        self.nav_list.itemDoubleClicked.connect(
            lambda _it: self._rename_section())
        self.nav_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.nav_list.customContextMenuRequested.connect(self._nav_menu)
        self._nav_rows: dict = {}
        v.addWidget(self.nav_list, 1)
        row = QHBoxLayout()
        for txt, fn in (("＋ New", self._new_section),
                        ("Dup", self._dup_section),
                        ("Del", self._del_section)):
            b = QPushButton(txt)
            b.clicked.connect(lambda _=False, f=fn: f())
            row.addWidget(b)
        v.addLayout(row)
        return w

    def _nav_menu(self, pos) -> None:
        item = self.nav_list.itemAt(pos)
        menu = QMenu(self)
        if item is not None:
            name = item.data(Qt.ItemDataRole.UserRole)

            def _sel(fn):
                self._switch_section(name)
                fn()
            menu.addAction("Rename…", lambda: _sel(self._rename_section))
            menu.addAction("Duplicate", lambda: _sel(self._dup_section))
            menu.addAction("Delete", lambda: _sel(self._del_section))
            menu.addSeparator()
        menu.addAction("New section", self._new_section)
        menu.exec(self.nav_list.mapToGlobal(pos))

    # ---- navigator rows (name + summary + status) ----
    def _spec_summary(self, spec) -> str:
        kind = spec.kind
        if kind in ("Rectangular", "T-shape", "Hollow box", "PSC girder"):
            size = f"{spec.b * 1e3:.0f}×{spec.h * 1e3:.0f} mm"
        elif kind == "Circular":
            size = f"⌀{spec.D * 1e3:.0f} mm"
        elif kind == "L-shape":
            size = f"leg {spec.leg * 1e3:.0f} mm"
        else:
            size = ""
        return f"{kind} · {size}" if size else kind

    def _section_has_reinf(self, spec) -> bool:
        if spec.kind == "Composite":
            return bool(spec.shapes)
        try:
            sec = _case(spec).section
            return bool((sec.reinforcement and sec.reinforcement.bars)
                        or (getattr(sec, "prestress", None)
                            and sec.prestress.tendons))
        except Exception:                              # noqa: BLE001
            return True

    def _make_nav_widget(self, name, rec):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(1)
        nl = QLabel()
        nl.setStyleSheet("font-weight:600; background:transparent;")
        sl = QLabel()
        sl.setObjectName("sub")
        sl.setStyleSheet("background:transparent;")
        lay.addWidget(nl)
        lay.addWidget(sl)
        self._fill_nav_labels(nl, sl, name, rec["spec"])
        return w, (nl, sl)

    def _fill_nav_labels(self, nl, sl, name, spec) -> None:
        warn = "" if self._section_has_reinf(spec) else "⚠ "
        nl.setText(f"{warn}{name}")
        sl.setText(self._spec_summary(spec))

    def _build_section_tab(self) -> QWidget:
        """The 'Section' workspace tab: inputs (left) + cross-section drawing
        and properties (right)."""
        inner = QSplitter(Qt.Orientation.Horizontal)
        inner.addWidget(self._build_definition_panel())
        inner.addWidget(self._build_preview_panel())
        inner.setStretchFactor(0, 0)
        inner.setStretchFactor(1, 1)
        inner.setSizes([340, 480])
        inner.setHandleWidth(8)
        return inner

    # ---------------------------------------------------------- definition
    def _build_definition_panel(self) -> QWidget:
        host = QScrollArea()
        host.setWidgetResizable(True)
        host.setMinimumWidth(320)
        host.setMaximumWidth(460)
        inner = QWidget()
        inner.setObjectName("sd_inner")
        v = QVBoxLayout(inner)
        v.setContentsMargins(4, 4, 10, 4)
        v.setSpacing(10)

        kbox = QGroupBox("Section")
        kf = QFormLayout(kbox)
        self.kind_combo = QComboBox()
        self.kind_combo.addItems(_KINDS)
        self.kind_combo.currentTextChanged.connect(self._on_kind_changed)
        kf.addRow("Kind", self.kind_combo)
        v.addWidget(kbox)

        self.dim_box = CollapsibleGroup("Dimensions [mm]")
        self.dim_form = QFormLayout(self.dim_box.body)
        self._dim_spins: dict[str, QDoubleSpinBox] = {}
        v.addWidget(self.dim_box)

        self.custom_box = self._build_custom_box()
        self.composite_box = self._build_composite_box()
        v.addWidget(self.custom_box)
        v.addWidget(self.composite_box)

        # PSC strands (shown only for PSC girder)
        self.psc_box = CollapsibleGroup("Prestressing strands")
        pf = QFormLayout(self.psc_box.body)
        self.nstr_spin = self._ispin(1, 40)
        self.strand_area_spin = self._dspin(50, 400, 5, " mm²", 0)
        self.fpe_spin = self._dspin(500, 1600, 25, " MPa", 0)
        self.strand_y_spin = self._dspin(-2000, 2000, 10, " mm", 0)
        for w in (self.nstr_spin, self.strand_area_spin, self.fpe_spin,
                  self.strand_y_spin):
            self._connect(w)
        pf.addRow("Strands", self.nstr_spin)
        pf.addRow("Aₚ / strand", self.strand_area_spin)
        pf.addRow("f_pe", self.fpe_spin)
        pf.addRow("Strand y", self.strand_y_spin)
        v.addWidget(self.psc_box)

        # Materials: chosen from the shared library (their constitutive laws
        # live there); the section keeps only confinement (below).
        mbox = CollapsibleGroup("Materials")
        self.mat_box = mbox
        mf = QFormLayout(mbox.body)
        self.conc_mat_combo = QComboBox()
        self.conc_mat_combo.currentTextChanged.connect(
            lambda *_: self._on_material_choice())
        self.steel_mat_combo = QComboBox()
        self.steel_mat_combo.currentTextChanged.connect(
            lambda *_: self._on_material_choice())
        mf.addRow("Concrete", self.conc_mat_combo)
        mf.addRow("Steel", self.steel_mat_combo)
        manage = QPushButton("Manage library…")
        manage.clicked.connect(self._open_materials)
        mf.addRow("", manage)
        v.addWidget(mbox)

        rbox = CollapsibleGroup("Reinforcement")
        self.rebar_box = rbox
        self.rebar_form = QFormLayout(rbox.body)
        self.bardia_combo = QComboBox()
        self.bardia_combo.addItems(list(core.BAR_SIZES.keys()))
        self.bardia_combo.currentTextChanged.connect(
            lambda *_: self._on_value_changed())
        self.cover_spin = self._dspin(10, 150, 5, " mm", 0)
        self._connect(self.cover_spin)
        self.rebar_form.addRow("Bar size", self.bardia_combo)
        self.rebar_form.addRow("Cover", self.cover_spin)
        self._rebar_widgets: dict[str, QWidget] = {}
        v.addWidget(rbox)

        # Rebar / tendon arrangements (layout generators added to the bars)
        self.rebar_arr_list = QListWidget()
        self.rebar_arr_list.setMaximumHeight(90)
        v.addWidget(self._arr_group(
            "Rebar arrangements", self.rebar_arr_list,
            lambda: self._add_arrangement(tendon=False),
            lambda: self._remove_arrangement(tendon=False)))
        self.tendon_arr_list = QListWidget()
        self.tendon_arr_list.setMaximumHeight(90)
        v.addWidget(self._arr_group(
            "Tendon arrangements", self.tendon_arr_list,
            lambda: self._add_arrangement(tendon=True),
            lambda: self._remove_arrangement(tendon=True)))

        # Confinement — section-dependent (transverse reinforcement), so it
        # stays here rather than on the material. Drives the confined concrete
        # response in moment-curvature.
        cmbox = CollapsibleGroup("Confinement & M-φ", collapsed=True)
        cmf = QFormLayout(cmbox.body)
        self.eps_c0_spin = self._dspin(0.001, 0.02, 0.0002, "", 4)
        self.eps_cu_spin = self._dspin(0.002, 0.05, 0.0005, "", 4)
        self.fcu_ratio_spin = self._dspin(0.0, 1.0, 0.05, "", 2)
        self.kappa_max_spin = self._dspin(0.005, 0.5, 0.01, " 1/m", 3)
        for w in (self.eps_c0_spin, self.eps_cu_spin, self.fcu_ratio_spin,
                  self.kappa_max_spin):
            self._connect(w)
        cmf.addRow("ε_c0 (peak, confined)", self.eps_c0_spin)
        cmf.addRow("ε_cu (crush, confined)", self.eps_cu_spin)
        cmf.addRow("f_cu / f'c (residual)", self.fcu_ratio_spin)
        cmf.addRow("κ_max sweep", self.kappa_max_spin)
        v.addWidget(cmbox)

        v.addStretch(1)
        host.setWidget(inner)
        return host

    # ---- Custom (polygon) editor -------------------------------------
    def _build_custom_box(self) -> QGroupBox:
        box = CollapsibleGroup("Custom section — coordinates [mm]")
        v = QVBoxLayout(box.body)
        v.addWidget(QLabel("Vertices (ordered z, y)"))
        self.cust_verts = QTableWidget(0, 2)
        self.cust_verts.setHorizontalHeaderLabels(["z", "y"])
        self.cust_verts.horizontalHeader().setStretchLastSection(True)
        self.cust_verts.setMaximumHeight(130)
        self.cust_verts.itemChanged.connect(lambda *_: self._custom_changed())
        v.addWidget(self.cust_verts)
        vr = QHBoxLayout()
        for txt, fn in (("+ Vertex", lambda: self._add_row(self.cust_verts, 2)),
                        ("- Vertex", lambda: self._del_row(self.cust_verts))):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            vr.addWidget(b)
        v.addLayout(vr)
        v.addWidget(QLabel("Bars (z, y, ⌀mm)"))
        self.cust_bars = QTableWidget(0, 3)
        self.cust_bars.setHorizontalHeaderLabels(["z", "y", "⌀"])
        self.cust_bars.horizontalHeader().setStretchLastSection(True)
        self.cust_bars.setMaximumHeight(120)
        self.cust_bars.itemChanged.connect(lambda *_: self._custom_changed())
        v.addWidget(self.cust_bars)
        br = QHBoxLayout()
        for txt, fn in (("+ Bar", lambda: self._add_row(self.cust_bars, 3)),
                        ("- Bar", lambda: self._del_row(self.cust_bars))):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            br.addWidget(b)
        v.addLayout(br)
        fr = QHBoxLayout()
        fr.addWidget(QLabel("Fill perimeter n:"))
        self.fill_n = self._ispin(3, 100)
        self.fill_n.setValue(8)
        fr.addWidget(self.fill_n)
        self.fill_dia = QComboBox()
        self.fill_dia.addItems(list(core.BAR_SIZES))
        self.fill_dia.setCurrentText("25 mm")
        fr.addWidget(self.fill_dia)
        fb = QPushButton("Fill")
        fb.clicked.connect(self._fill_perimeter)
        fr.addWidget(fb)
        v.addLayout(fr)
        return box

    def _add_row(self, tbl, ncols) -> None:
        r = tbl.rowCount()
        tbl.insertRow(r)
        for c in range(ncols):
            tbl.setItem(r, c, QTableWidgetItem("0"))
        self._custom_changed()

    def _del_row(self, tbl) -> None:
        r = tbl.currentRow()
        if r < 0:
            r = tbl.rowCount() - 1
        if r >= 0:
            tbl.removeRow(r)
            self._custom_changed()

    def _custom_changed(self) -> None:
        if self._loading:
            return
        try:
            outline = tuple((self._cell(self.cust_verts, r, 0) / 1000.0,
                             self._cell(self.cust_verts, r, 1) / 1000.0)
                            for r in range(self.cust_verts.rowCount()))
            bars = tuple((self._cell(self.cust_bars, r, 0) / 1000.0,
                          self._cell(self.cust_bars, r, 1) / 1000.0,
                          self._cell(self.cust_bars, r, 2) / 1000.0)
                         for r in range(self.cust_bars.rowCount()))
        except Exception:                              # noqa: BLE001
            return
        self._spec = replace(self._spec, custom_outline=outline,
                             custom_bars=bars)
        self._refresh_geometry()
        self._queue()

    @staticmethod
    def _cell(tbl, r, c) -> float:
        it = tbl.item(r, c)
        return float(it.text()) if it and it.text() else 0.0

    def _fill_perimeter(self) -> None:
        outline = tuple((self._cell(self.cust_verts, r, 0) / 1000.0,
                         self._cell(self.cust_verts, r, 1) / 1000.0)
                        for r in range(self.cust_verts.rowCount()))
        if len(outline) < 3:
            QMessageBox.information(self, "Fill perimeter",
                                    "Define at least 3 vertices first.")
            return
        dia = core.BAR_SIZES[self.fill_dia.currentText()]
        bars = core.perimeter_bars(outline, self.fill_n.value(),
                                   self._spec.cover, dia)
        self._spec = replace(self._spec, custom_outline=outline,
                             custom_bars=tuple(bars))
        self._load_custom_tables()
        self._refresh_geometry()
        self._queue()

    def _load_custom_tables(self) -> None:
        was = self._loading
        self._loading = True
        outline = self._spec.custom_outline or (
            (-0.20, -0.30), (0.20, -0.30), (0.20, 0.30), (-0.20, 0.30))
        self.cust_verts.setRowCount(len(outline))
        for r, (z, y) in enumerate(outline):
            self.cust_verts.setItem(r, 0, QTableWidgetItem(f"{z * 1e3:.0f}"))
            self.cust_verts.setItem(r, 1, QTableWidgetItem(f"{y * 1e3:.0f}"))
        bars = self._spec.custom_bars
        self.cust_bars.setRowCount(len(bars))
        for r, (z, y, d) in enumerate(bars):
            self.cust_bars.setItem(r, 0, QTableWidgetItem(f"{z * 1e3:.0f}"))
            self.cust_bars.setItem(r, 1, QTableWidgetItem(f"{y * 1e3:.0f}"))
            self.cust_bars.setItem(r, 2, QTableWidgetItem(f"{d * 1e3:.0f}"))
        self._loading = was

    # ---- Composite (material-shapes) editor --------------------------
    def _build_composite_box(self) -> QGroupBox:
        box = CollapsibleGroup("Composite — material shapes")
        v = QVBoxLayout(box.body)
        self.comp_list = QListWidget()
        self.comp_list.setMaximumHeight(90)
        v.addWidget(self.comp_list)
        row = QHBoxLayout()
        for txt, fn in (("Remove", self._comp_remove), ("▲", self._comp_up),
                        ("▼", self._comp_down)):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        f = QFormLayout()
        self.comp_typ = QComboBox()
        self.comp_typ.addItems(["Rectangle", "Circle"])
        self.comp_typ.currentTextChanged.connect(self._comp_typ_changed)
        f.addRow("Shape", self.comp_typ)
        self.comp_mat = QComboBox()
        self.comp_mat.addItems(["concrete", "steel"])
        f.addRow("Material", self.comp_mat)
        self.comp_str = self._dspin(5, 700, 5, " MPa", 0)
        self.comp_str.setValue(30)
        f.addRow("Strength", self.comp_str)
        self.comp_cz = self._dspin(-5000, 5000, 10, " mm", 0)
        self.comp_cy = self._dspin(-5000, 5000, 10, " mm", 0)
        self.comp_w = self._dspin(1, 5000, 10, " mm", 0)
        self.comp_w.setValue(400)
        self.comp_h = self._dspin(1, 5000, 10, " mm", 0)
        self.comp_h.setValue(600)
        self.comp_D = self._dspin(1, 5000, 10, " mm", 0)
        self.comp_D.setValue(400)
        f.addRow("Centre z", self.comp_cz)
        f.addRow("Centre y", self.comp_cy)
        f.addRow("Width", self.comp_w)
        f.addRow("Height", self.comp_h)
        f.addRow("Diameter", self.comp_D)
        v.addLayout(f)
        addb = QPushButton("+ Add shape")
        addb.clicked.connect(self._comp_add)
        v.addWidget(addb)
        self._comp_typ_changed(self.comp_typ.currentText())
        return box

    def _comp_typ_changed(self, typ) -> None:
        rect = typ == "Rectangle"
        self.comp_w.setVisible(rect)
        self.comp_h.setVisible(rect)
        self.comp_D.setVisible(not rect)

    def _refresh_comp_mat_combo(self) -> None:
        cur = self.comp_mat.currentText()
        self.comp_mat.clear()
        self.comp_mat.addItems(["concrete", "steel"] + list(self._materials))
        idx = self.comp_mat.findText(cur)
        self.comp_mat.setCurrentIndex(idx if idx >= 0 else 0)

    def _comp_matd(self) -> dict:
        chosen = self.comp_mat.currentText()
        if chosen in self._materials:            # a shared library material
            return dict(self._materials[chosen])
        if chosen == "steel":
            return dict(kind="steel", fy=self.comp_str.value() * 1e6,
                        steel_model="Bilinear", Es=200e9, steel_b=0.01,
                        steel_fu_ratio=1.5, steel_eps_sh=0.008,
                        steel_eps_su=0.10)
        return dict(kind="concrete", fc=self.comp_str.value() * 1e6,
                    conc_model="Kent-Park", eps_c0=0.002, eps_cu=0.0035,
                    fcu_ratio=0.4, fr_model="sqrt", fr_coeff=0.62,
                    eps_decay=1e-3, conc_f1_ratio=0.4)

    def _comp_outline(self):
        cz, cy = self.comp_cz.value() / 1e3, self.comp_cy.value() / 1e3
        if self.comp_typ.currentText() == "Rectangle":
            w, h = self.comp_w.value() / 1e3, self.comp_h.value() / 1e3
            return ((cz - w / 2, cy - h / 2), (cz + w / 2, cy - h / 2),
                    (cz + w / 2, cy + h / 2), (cz - w / 2, cy + h / 2))
        r = self.comp_D.value() / 2e3
        return tuple((float(cz + r * np.cos(a)), float(cy + r * np.sin(a)))
                     for a in np.linspace(0, 2 * np.pi, 48, endpoint=False))

    def _comp_add(self) -> None:
        shape = (self._comp_outline(), tuple(sorted(self._comp_matd().items())))
        self._spec = replace(self._spec, shapes=self._spec.shapes + (shape,))
        self._refresh_comp_list()
        self._refresh_geometry()
        self._queue()

    def _comp_reorder(self, delta) -> None:
        i = self.comp_list.currentRow()
        sh = list(self._spec.shapes)
        j = i + delta
        if 0 <= i < len(sh) and 0 <= j < len(sh):
            sh[i], sh[j] = sh[j], sh[i]
            self._spec = replace(self._spec, shapes=tuple(sh))
            self._refresh_comp_list()
            self.comp_list.setCurrentRow(j)
            self._refresh_geometry()
            self._queue()

    def _comp_up(self) -> None:
        self._comp_reorder(-1)

    def _comp_down(self) -> None:
        self._comp_reorder(1)

    def _comp_remove(self) -> None:
        i = self.comp_list.currentRow()
        sh = list(self._spec.shapes)
        if 0 <= i < len(sh):
            del sh[i]
            self._spec = replace(self._spec, shapes=tuple(sh))
            self._refresh_comp_list()
            self._refresh_geometry()
            self._queue()

    def _refresh_comp_list(self) -> None:
        self.comp_list.clear()
        for i, (outline, mat_kv) in enumerate(self._spec.shapes):
            md = dict(mat_kv)
            kd = md.get("kind", "concrete")
            strg = (md.get("fc", 0) if kd == "concrete" else md.get("fy", 0))
            self.comp_list.addItem(
                f"Shape {i + 1}: {kd} {strg / 1e6:.0f} MPa · {len(outline)} pts")

    def _build_preview_panel(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(300)
        v = QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(8)

        self.head_name = QLabel("Section")
        self.head_name.setObjectName("h1")
        v.addWidget(self.head_name)
        self.head_sub = QLabel("")
        self.head_sub.setObjectName("sub")
        v.addWidget(self.head_sub)

        # cross-section drawing on a neutral canvas, aspect ratio preserved
        canvas = QGroupBox("Cross-section")
        cv = QVBoxLayout(canvas)
        self.svg = QSvgWidget()
        self.svg.setMinimumHeight(300)
        cv.addWidget(self.svg)
        v.addWidget(canvas, 3)

        props_box = QGroupBox("Section properties")
        pv = QVBoxLayout(props_box)
        self.props = QTableWidget(0, 2)
        self.props.setHorizontalHeaderLabels(["Quantity", "Value"])
        self.props.horizontalHeader().setStretchLastSection(True)
        self.props.verticalHeader().setVisible(False)
        self.props.setAlternatingRowColors(True)
        self.props.setShowGrid(False)
        self.props.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        pv.addWidget(self.props)
        v.addWidget(props_box, 2)
        return w

    def _build_analysis_panel(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(420)
        self.tabs.currentChanged.connect(lambda *_: self._queue())

        # ---- P-M interaction + demand check ----
        pm = QWidget()
        pmv = QVBoxLayout(pm)
        self.pm_fig = Figure(figsize=(4.4, 4.0), layout="constrained")
        self.pm_canvas = Canvas(self.pm_fig)
        pmv.addWidget(self.pm_canvas)
        d1 = QHBoxLayout()
        d1.addWidget(QLabel("Demand P:"))
        self.dem_P = self._dspin(-1e6, 1e6, 50, "", 1)
        d1.addWidget(self.dem_P)
        d1.addWidget(QLabel("Mz:"))
        self.dem_Mz = self._dspin(-1e6, 1e6, 25, "", 1)
        d1.addWidget(self.dem_Mz)
        d1.addWidget(QLabel("My:"))
        self.dem_My = self._dspin(-1e6, 1e6, 25, "", 1)
        d1.addWidget(self.dem_My)
        self.design_chk = QCheckBox("Design (φ)")
        self.design_chk.setChecked(True)
        d1.addWidget(self.design_chk)
        for w in (self.dem_P, self.dem_Mz, self.dem_My):
            w.valueChanged.connect(lambda *_: self._queue())
        self.design_chk.stateChanged.connect(lambda *_: self._queue())
        pmv.addLayout(d1)
        self.dem_lbl = QLabel("—")
        self.dem_lbl.setTextFormat(Qt.TextFormat.RichText)
        pmv.addWidget(self.dem_lbl)
        self.tabs.addTab(pm, "P-M interaction")

        # ---- Moment-curvature ----
        mp = QWidget()
        mpv = QVBoxLayout(mp)
        self.mp_fig = Figure(figsize=(4.4, 4.0), layout="constrained")
        self.mp_canvas = Canvas(self.mp_fig)
        mpv.addWidget(self.mp_canvas)
        pr = QHBoxLayout()
        pr.addWidget(QLabel("Axial P (+comp):"))
        self.mphi_P = self._dspin(-1e6, 1e6, 50, "", 1)
        self.mphi_P.valueChanged.connect(lambda *_: self._queue())
        pr.addWidget(self.mphi_P)
        pr.addWidget(QLabel("N-axis angle θ [deg]:"))
        self.mphi_ang = self._dspin(-180, 180, 5, "", 1)
        self.mphi_ang.valueChanged.connect(lambda *_: self._queue())
        pr.addWidget(self.mphi_ang)
        pr.addStretch(1)
        mpv.addLayout(pr)
        self.mphi_metrics = QLabel("—")
        self.mphi_metrics.setTextFormat(Qt.TextFormat.RichText)
        mpv.addWidget(self.mphi_metrics)
        self.mphi_tbl = QTableWidget(0, 4)
        self.mphi_tbl.setHorizontalHeaderLabels(
            ["Point", "State", "Curvature", "Moment"])
        self.mphi_tbl.horizontalHeader().setStretchLastSection(True)
        self.mphi_tbl.verticalHeader().setVisible(False)
        self.mphi_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.mphi_tbl.setAlternatingRowColors(True)
        self.mphi_tbl.setShowGrid(False)
        self.mphi_tbl.setMaximumHeight(150)
        mpv.addWidget(self.mphi_tbl)
        self.tabs.addTab(mp, "Moment-curvature")

        # ---- Verification ----
        vt = QWidget()
        vtv = QVBoxLayout(vt)
        self.verify_tbl = QTableWidget(0, 4)
        self.verify_tbl.setHorizontalHeaderLabels(
            ["Quantity", "Units", "Computed", "Note"])
        self.verify_tbl.horizontalHeader().setStretchLastSection(True)
        self.verify_tbl.verticalHeader().setVisible(False)
        self.verify_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.verify_tbl.setAlternatingRowColors(True)
        self.verify_tbl.setShowGrid(False)
        vtv.addWidget(self.verify_tbl)
        self.tabs.addTab(vt, "Verification")

        # ---- 3-D P-M-M surface ----
        s3 = QWidget()
        s3v = QVBoxLayout(s3)
        self.s3_fig = Figure(figsize=(4.6, 4.2), layout="constrained")
        self.s3_canvas = Canvas(self.s3_fig)
        s3v.addWidget(self.s3_canvas)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Mesh:"))
        self.mesh_combo = QComboBox()
        self.mesh_combo.addItems(list(_ARR_MESH))
        self.mesh_combo.setCurrentText("Coarse")
        self.mesh_combo.currentTextChanged.connect(lambda *_: self._queue())
        srow.addWidget(self.mesh_combo)
        srow.addStretch(1)
        s3v.addLayout(srow)
        self.tabs.addTab(s3, "3-D P-M-M surface")

        # ---- Report ----
        self.report = QTextBrowser()
        self.tabs.addTab(self.report, "Report")
        return self.tabs

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _dspin(lo, hi, step, suffix, decimals) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        if suffix:
            s.setSuffix(suffix)
        return s

    @staticmethod
    def _ispin(lo, hi) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        return s

    def _connect(self, w) -> None:
        if isinstance(w, QSpinBox):
            w.valueChanged.connect(lambda *_: self._on_value_changed())
        elif isinstance(w, QDoubleSpinBox):
            w.valueChanged.connect(lambda *_: self._on_value_changed())
        elif isinstance(w, QCheckBox):
            w.stateChanged.connect(lambda *_: self._on_value_changed())

    # ------------------------------------------------------ dynamic fields
    def _on_kind_changed(self, kind: str) -> None:
        self._rebuild_dim_fields(kind)
        self._rebuild_rebar_fields(kind)
        self._apply_kind_visibility(kind)
        if kind == "Custom":
            self._load_custom_tables()
        elif kind == "Composite":
            self._refresh_comp_list()
        self._on_value_changed()

    def _apply_kind_visibility(self, kind: str) -> None:
        parametric = kind in _PARAMETRIC
        self.dim_box.setVisible(parametric)
        self.psc_box.setVisible(kind == "PSC girder")
        self.custom_box.setVisible(kind == "Custom")
        self.composite_box.setVisible(kind == "Composite")
        # Composite draws material from its shapes; single-material groups hide.
        self.mat_box.setVisible(kind != "Composite")
        self.rebar_box.setVisible(kind not in ("Custom", "Composite"))

    def _rebuild_dim_fields(self, kind: str) -> None:
        while self.dim_form.rowCount():
            self.dim_form.removeRow(0)
        self._dim_spins.clear()
        for key in _DIMS_FOR_KIND.get(kind, []):
            spin = self._dspin(20, 5000, 10, " mm", 0)
            spin.setValue(getattr(self._spec, key) * 1000.0)
            spin.valueChanged.connect(lambda *_: self._on_value_changed())
            self.dim_form.addRow(_DIM_LABEL[key], spin)
            self._dim_spins[key] = spin

    def _rebuild_rebar_fields(self, kind: str) -> None:
        for w in self._rebar_widgets.values():
            self.rebar_form.removeRow(w)
        self._rebar_widgets.clear()
        for key in _REBAR_FOR_KIND.get(kind, []):
            if key == "spiral":
                w = QCheckBox()
                w.setChecked(bool(getattr(self._spec, key)))
                w.stateChanged.connect(lambda *_: self._on_value_changed())
            else:
                w = self._ispin(0, 64)
                w.setValue(int(getattr(self._spec, key)))
                w.valueChanged.connect(lambda *_: self._on_value_changed())
            self.rebar_form.addRow(_REBAR_LABEL[key], w)
            self._rebar_widgets[key] = w

    # ----------------------------------------------------- arrangements
    @staticmethod
    def _arr_group(title, listw, on_add, on_remove) -> CollapsibleGroup:
        box = CollapsibleGroup(title, collapsed=True)
        gv = QVBoxLayout(box.body)
        gv.addWidget(listw)
        row = QHBoxLayout()
        add = QPushButton("Add…")
        add.clicked.connect(lambda: on_add())
        rem = QPushButton("Remove")
        rem.clicked.connect(lambda: on_remove())
        row.addWidget(add)
        row.addWidget(rem)
        gv.addLayout(row)
        return box

    def _refresh_arr_lists(self) -> None:
        self.rebar_arr_list.clear()
        for arr in self._spec.rebar_arr:
            self.rebar_arr_list.addItem(_rebar_arr_label(arr))
        self.tendon_arr_list.clear()
        for arr in self._spec.tendon_arr:
            self.tendon_arr_list.addItem(_tendon_arr_label(arr))

    def _add_arrangement(self, *, tendon: bool) -> None:
        want = "prestress" if tendon else "steel"
        mats = [n for n, m in self._materials.items()
                if m.get("kind") == want]
        dlg = ArrangementDialog(self, tendon=tendon, steels=mats)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.result_arr is None:
            return
        if tendon:
            self._spec = replace(
                self._spec, tendon_arr=self._spec.tendon_arr + (dlg.result_arr,))
        else:
            self._spec = replace(
                self._spec, rebar_arr=self._spec.rebar_arr + (dlg.result_arr,))
        self._refresh_arr_lists()
        self._refresh_geometry()
        self._queue()

    def _remove_arrangement(self, *, tendon: bool) -> None:
        listw = self.tendon_arr_list if tendon else self.rebar_arr_list
        i = listw.currentRow()
        arrs = list(self._spec.tendon_arr if tendon else self._spec.rebar_arr)
        if not (0 <= i < len(arrs)):
            return
        del arrs[i]
        if tendon:
            self._spec = replace(self._spec, tendon_arr=tuple(arrs))
        else:
            self._spec = replace(self._spec, rebar_arr=tuple(arrs))
        self._refresh_arr_lists()
        self._refresh_geometry()
        self._queue()

    # --------------------------------------------------------- spec <-> form
    def _load_form_from_spec(self) -> None:
        self._loading = True
        s = self._spec
        self.kind_combo.setCurrentText(s.kind if s.kind in _KINDS else _KINDS[0])
        self._rebuild_dim_fields(self.kind_combo.currentText())
        self._rebuild_rebar_fields(self.kind_combo.currentText())
        self.psc_box.setVisible(self.kind_combo.currentText() == "PSC girder")
        self.cover_spin.setValue(s.cover * 1000.0)
        want = f"{int(round(s.bar_dia * 1000))} mm"
        if want in core.BAR_SIZES:
            self.bardia_combo.setCurrentText(want)
        self.nstr_spin.setValue(int(s.n_strand))
        self.strand_area_spin.setValue(s.strand_area * 1e6)
        self.fpe_spin.setValue(s.f_pe / 1e6)
        self.strand_y_spin.setValue(s.strand_y * 1000.0)
        # confinement (section) + M-φ sweep
        self.eps_c0_spin.setValue(s.eps_c0)
        self.eps_cu_spin.setValue(s.eps_cu)
        self.fcu_ratio_spin.setValue(s.fcu_ratio)
        self.kappa_max_spin.setValue(s.kappa_max)
        self._refresh_material_combos()
        self._refresh_arr_lists()
        self._load_custom_tables()
        self._refresh_comp_list()
        self._apply_kind_visibility(self.kind_combo.currentText())
        self._loading = False

    def _spec_from_form(self) -> core.Spec:
        kind = self.kind_combo.currentText()
        ch: dict = {"kind": kind}
        for key, spin in self._dim_spins.items():
            ch[key] = spin.value() / 1000.0
        ch["cover"] = self.cover_spin.value() / 1000.0
        ch["bar_dia"] = core.BAR_SIZES[self.bardia_combo.currentText()]
        for key, w in self._rebar_widgets.items():
            ch[key] = w.isChecked() if key == "spiral" else w.value()
        if kind == "PSC girder":
            ch["n_strand"] = self.nstr_spin.value()
            ch["strand_area"] = self.strand_area_spin.value() * 1e-6
            ch["f_pe"] = self.fpe_spin.value() * 1e6
            ch["strand_y"] = self.strand_y_spin.value() / 1000.0
        # confinement (section-dependent) + the M-φ sweep limit stay here
        ch["eps_c0"] = self.eps_c0_spin.value()
        ch["eps_cu"] = self.eps_cu_spin.value()
        ch["fcu_ratio"] = self.fcu_ratio_spin.value()
        ch["kappa_max"] = self.kappa_max_spin.value()
        spec = replace(self._spec, **ch)
        # material constitutive laws come from the chosen library materials
        return self._apply_material_params(spec)

    # ---- material library <-> section ---------------------------------
    _CONC_KEYS = ("fc", "conc_model", "conc_f1_ratio", "fr_model", "fr_coeff",
                  "eps_decay")
    _STEEL_KEYS = ("fy", "Es", "steel_model", "steel_b", "steel_fu_ratio",
                   "steel_eps_sh", "steel_eps_su")

    def _apply_material_params(self, spec: core.Spec) -> core.Spec:
        """Overlay the chosen concrete/steel materials' constitutive laws onto
        ``spec`` (confinement fields left untouched)."""
        ch: dict = {}
        conc = self._materials.get(self.conc_mat_combo.currentText())
        if conc:
            ch.update({k: conc[k] for k in self._CONC_KEYS if k in conc})
        steel = self._materials.get(self.steel_mat_combo.currentText())
        if steel:
            ch.update({k: steel[k] for k in self._STEEL_KEYS if k in steel})
        return replace(spec, **ch) if ch else spec

    def _concrete_names(self) -> list:
        return [n for n, m in self._materials.items()
                if m.get("kind") == "concrete"]

    def _steel_names(self) -> list:
        return [n for n, m in self._materials.items()
                if m.get("kind") == "steel"]

    def _refresh_material_combos(self) -> None:
        """Repopulate the section's concrete/steel pickers from the library,
        preserving the active section's stored choice."""
        was = self._loading
        self._loading = True
        rec = self._sections.get(self._active, {})
        for combo, names, key in (
                (self.conc_mat_combo, self._concrete_names(), "conc_mat"),
                (self.steel_mat_combo, self._steel_names(), "steel_mat")):
            combo.clear()
            combo.addItems(names)
            want = rec.get(key)
            i = combo.findText(want) if want else -1
            combo.setCurrentIndex(i if i >= 0 else 0)
        self._loading = was

    def _on_material_choice(self) -> None:
        if self._loading:
            return
        rec = self._sections.get(self._active)
        if rec is not None:
            rec["conc_mat"] = self.conc_mat_combo.currentText() or None
            rec["steel_mat"] = self.steel_mat_combo.currentText() or None
        self._on_value_changed()

    # ----------------------------------------------------------- recompute
    def _on_units_changed(self) -> None:
        if self._loading:
            return
        self._units = core.Units(force=self.force_combo.currentText(),
                                 length=self.length_combo.currentText(),
                                 stress=self.stress_combo.currentText())
        self._queue()

    def _on_value_changed(self) -> None:
        if self._loading:
            return
        try:
            self._spec = self._spec_from_form()
        except Exception:                              # noqa: BLE001
            return
        self._refresh_geometry()
        self._queue()

    def _queue(self) -> None:
        if not self._loading:
            # persist the working spec + code into the active section (single
            # choke point — every change funnels through here) before the
            # debounced analysis recompute.
            if self._active in self._sections:
                self._sections[self._active]["spec"] = self._spec
                self._sections[self._active]["code"] = \
                    self.code_combo.currentText()
                self._update_nav_item(self._active)
            self._timer.start()

    def _refresh_geometry(self) -> None:
        try:
            case = _case(self._spec)
            self.svg.load(QByteArray(core.svg_of(case).encode("utf-8")))
            # keep the section undistorted (letterbox to the widget)
            r = self.svg.renderer()
            if r is not None:
                r.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
            self._update_header(case)
            props = core.props_of(case)
            self.props.setRowCount(len(props))
            for row, (k, val) in enumerate(props.items()):
                self.props.setItem(row, 0, QTableWidgetItem(str(k)))
                txt = f"{val:,.0f}" if isinstance(val, (int, float)) else str(val)
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                    | Qt.AlignmentFlag.AlignVCenter)
                self.props.setItem(row, 1, it)
            self.statusBar().clearMessage()
        except Exception as exc:                       # noqa: BLE001
            self.statusBar().showMessage(f"Geometry error: {exc}")

    def _update_header(self, case) -> None:
        """Preview header: section name + a one-line dimensional summary."""
        s = self._spec
        self.head_name.setText(self._active)
        bits = [s.kind]
        if s.kind in ("Rectangular", "T-shape", "Hollow box", "PSC girder"):
            bits.append(f"{s.b * 1e3:.0f} × {s.h * 1e3:.0f} mm")
        elif s.kind == "Circular":
            bits.append(f"⌀ {s.D * 1e3:.0f} mm")
        elif s.kind == "L-shape":
            bits.append(f"leg {s.leg * 1e3:.0f} mm")
        try:
            A = case.section.area
            bits.append(f"A = {A * 1e4:,.0f} cm²")
        except Exception:                              # noqa: BLE001
            pass
        self.head_sub.setText("  ·  ".join(bits))

    def _recompute_analysis(self) -> None:
        code = self.code_combo.currentText()
        try:
            case = _case(self._spec)
        except Exception as exc:                       # noqa: BLE001
            self.statusBar().showMessage(f"Build error: {exc}")
            return
        # tab 0 is the Section (inputs + drawing) — geometry is already live,
        # no analysis to run.
        idx = self.tabs.currentIndex()
        if idx == 0:
            self.statusBar().clearMessage()
            return
        if not self._has_reinforcement(case):
            self.statusBar().showMessage(
                "This section has no reinforcement yet — add bars/strands to "
                "compute interaction, moment-curvature and verification.")
            return
        try:
            if idx == 1:
                self._draw_pm(case, code)
            elif idx == 2:
                self._draw_mphi(case)
            elif idx == 3:
                self._fill_verify(case, code)
            elif idx == 4:
                self._draw_surface(case, code)
            elif idx == 5:
                self._fill_report(case, code)
            self.statusBar().clearMessage()
        except Exception as exc:                       # noqa: BLE001
            self.statusBar().showMessage(f"Analysis error: {exc}")

    def _has_reinforcement(self, case) -> bool:
        """Mirror the Streamlit _ok_to_analyze/_rebar_guard: Composite is valid
        once it has shapes; every other kind needs bars or tendons."""
        if self._spec.kind == "Composite":
            return bool(self._spec.shapes)
        sec = case.section
        return bool((sec.reinforcement and sec.reinforcement.bars)
                    or (getattr(sec, "prestress", None)
                        and sec.prestress.tendons))

    # -------------------------------------------------------------- P-M tab
    def _demands(self):
        P, Mz, My = self.dem_P.value(), self.dem_Mz.value(), self.dem_My.value()
        if not (P or Mz or My):
            return None
        return [{"name": "Demand", "P": P, "Mz": Mz, "My": My}]

    def _draw_pm(self, case, code) -> None:
        u = self._units
        if self._spec.kind == "Composite":
            self.pm_fig.clear()
            ax = self.pm_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Composite section — see the\n"
                    "3-D P-M-M surface tab.", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.pm_canvas.draw_idle()
            self.dem_lbl.setText("Demand check is not available for composite "
                                 "sections in this build.")
            return
        curve, landmarks = core.pmm_slice(case, code)
        self.pm_fig.clear()
        ax = self.pm_fig.add_subplot(111)
        M = [u.M_disp(v) for v in curve["M_nom"]]
        P = [u.P_disp(v) for v in curve["P_nom"]]
        ax.plot(M, P, "-", color="#1f6feb", lw=1.8, label="Nominal P-M")
        if curve.get("has_design"):
            ax.plot([u.M_disp(v) for v in curve["M_des"]],
                    [u.P_disp(v) for v in curve["P_des"]], "--",
                    color="#d1462f", lw=1.5, label="Design φ")
        for _name, val, kind in (landmarks or []):
            if kind == "P":
                ax.axhline(u.P_disp(val), color="#bbb", lw=0.6, ls=":")
        # demand check
        dem = self._demands()
        if dem:
            res = core.demand_check(case, code, dem,
                                    design=self.design_chk.isChecked(),
                                    spec=self._spec)[0]
            ax.plot([u.M_disp(res["M_res"])], [u.P_disp(res["P"])], "o",
                    color="#e3a008", ms=9, label="Demand", zorder=5)
            col = "#1a7f37" if res["status"] == "OK" else "#cf222e"
            self.dem_lbl.setText(
                f"D/C = <b style='color:{col}'>{res['util']:.3f}</b> "
                f"({res['status']}) · governs {res['govern']} · "
                f"M_cap {u.M_disp(res['M_cap']):.4g} {u.Ml}, "
                f"β {res['beta_deg']:.1f}°")
        else:
            self.dem_lbl.setText("Enter a demand (P, Mz, My) to run a check.")
        ax.axhline(0, color="#000", lw=0.5)
        ax.axvline(0, color="#000", lw=0.5)
        ax.set_xlabel(f"M  [{u.Ml}]")
        ax.set_ylabel(f"P  [{u.Fl}]  (+ compression)")
        ax.set_title(f"P-M interaction — {code}")
        style.beautify_axes(ax)
        ax.legend(fontsize=8, loc="best", frameon=False)
        self.pm_canvas.draw_idle()

    # ----------------------------------------------------------- M-φ tab
    def _draw_mphi(self, case) -> None:
        u = self._units
        P = self.mphi_P.value()          # in the current force unit
        P_kN = P * u.fN / 1e3            # -> kN (engine base)
        ang = self.mphi_ang.value()
        if self._spec.kind == "Composite":
            data = core.composite_mphi(self._spec, P_kN, na_angle=ang,
                                       kappa_max=self._spec.kappa_max,
                                       materials=self._materials)
        else:
            data = core.mphi_data(case, P_kN, na_angle=ang,
                                  **core.mphi_props(self._spec))
        self.mp_fig.clear()
        ax = self.mp_fig.add_subplot(111)
        ax.plot([u.curv_disp(k) for k in data["kappa"]],
                [u.M_disp(v) for v in data["M"]], "-", color="#2ca25f", lw=1.8)
        marks = list(data.get("milestones", []))
        if data.get("ideal"):
            marks.append({**data["ideal"], "label": "f",
                          "state": "Idealized yield"})
        # guard: core.composite_mphi returns the 'd' milestone M in N·m while
        # the M array / M_u are kN·m — drop milestones whose M is wildly out of
        # the curve's range so a unit glitch can't blow up the axes.
        m_span = max((abs(v) for v in data["M"]), default=0.0) * 5 or 1e9
        rows = []
        for ms in marks:
            if abs(ms["M"]) > m_span:
                continue
            kx, my = u.curv_disp(ms["kappa"]), u.M_disp(ms["M"])
            ax.plot([kx], [my], "o", ms=6, color="crimson")
            ax.annotate(ms.get("label", ""), (kx, my), fontsize=8,
                        fontweight="bold", color="crimson",
                        textcoords="offset points", xytext=(4, 4))
            rows.append((ms.get("label", ""), ms.get("state", ""), kx, my))
        ax.set_xlabel(f"curvature κ  [{u.Kl}]")
        ax.set_ylabel(f"moment M  [{u.Ml}]")
        ax.set_title(f"Moment-curvature at P = {P:.4g} {u.Fl}")
        style.beautify_axes(ax)
        self.mp_canvas.draw_idle()

        def _m(v):
            return f"{u.M_disp(v):.4g} {u.Ml}" if v else "—"
        mu = data.get("mu_phi")
        mu_txt = f"<b>{mu:.2f}</b>" if mu else "—"
        self.mphi_metrics.setText(
            f"M_cr {_m(data['M_cr'])} &nbsp;·&nbsp; M_y {_m(data['M_y'])} "
            f"&nbsp;·&nbsp; M_u {_m(data['M_u'])} &nbsp;·&nbsp; μ_φ {mu_txt}")
        self.mphi_tbl.setRowCount(len(rows))
        for r, (lab, state, kx, my) in enumerate(rows):
            for col, val in enumerate((lab, state, f"{kx:.4g}", f"{my:.4g}")):
                self.mphi_tbl.setItem(r, col, QTableWidgetItem(str(val)))

    # ------------------------------------------------------ 3-D surface
    def _draw_surface(self, case, code) -> None:
        u = self._units
        na, npl = _ARR_MESH[self.mesh_combo.currentText()]
        if self._spec.kind == "Composite":
            mesh = core.composite_pmm_mesh(self._spec, na, npl,
                                           materials=self._materials)
        else:
            mesh = core.pmm_surface_mesh(case, code, na, npl)
        Mz, My, Pl = mesh["Mz"], mesh["My"], mesh["Plevels"]
        # structured grid: rows = P-levels, cols = angles; close each ring
        X, Y, Z = [], [], []
        for i in range(len(Mz)):
            row_mz = list(Mz[i]) + [Mz[i][0]]
            row_my = list(My[i]) + [My[i][0]]
            X.append([u.M_disp(v) for v in row_mz])
            Y.append([u.M_disp(v) for v in row_my])
            Z.append([u.P_disp(Pl[i])] * len(row_mz))
        self.s3_fig.clear()
        self.s3_fig.set_facecolor(style.PANEL)
        ax = self.s3_fig.add_subplot(111, projection="3d")
        ax.set_facecolor(style.PANEL)
        ax.plot_surface(np.array(X), np.array(Y), np.array(Z),
                        cmap="viridis", alpha=0.9, linewidth=0,
                        rstride=1, cstride=1)
        ax.tick_params(colors=style.AX_TEXT, labelsize=8)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.label.set_color(style.AX_TEXT)
            a.label.set_fontsize(9)
        ax.set_xlabel(f"Mz [{u.Ml}]")
        ax.set_ylabel(f"My [{u.Ml}]")
        ax.set_zlabel(f"P [{u.Fl}]")
        ax.set_title(f"P-Mz-My interaction surface — {code}", color=style.TEXT,
                     fontsize=11, fontweight="bold")
        self.s3_canvas.draw_idle()

    # ------------------------------------------------------ verify / report
    def _verify_rows(self, case, code):
        if self._spec.kind == "Composite":
            return []                    # single-material verification n/a
        return core.items_data(case, code, core.mphi_props(self._spec))

    def _fill_verify(self, case, code) -> None:
        rows = self._verify_rows(case, code)
        self.verify_tbl.setRowCount(len(rows))
        for r, it in enumerate(rows):
            comp = it.get("computed")
            comp = f"{comp:,.4g}" if isinstance(comp, (int, float)) else str(comp)
            for col, val in enumerate((it.get("quantity", ""),
                                       it.get("units", ""), comp,
                                       it.get("note", ""))):
                self.verify_tbl.setItem(r, col, QTableWidgetItem(str(val)))

    def _fill_report(self, case, code) -> None:
        try:
            self.report.setHtml(self._report_html(case, code))
        except Exception:                              # noqa: BLE001
            self.report.setHtml(self._fallback_report(case, code))

    def _fallback_report(self, case, code) -> str:
        """Minimal report for kinds the full report_html can't render (e.g.
        composite): section properties + moment-curvature milestones."""
        u = self._units
        rows = "".join(
            f"<tr><td>{k}</td><td style='text-align:right'>{v:,.0f}</td></tr>"
            if isinstance(v, (int, float)) else
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in core.props_of(case).items())
        mphi = ""
        try:
            m = core.composite_mphi(self._spec, 0.0,
                                    kappa_max=self._spec.kappa_max, materials=self._materials)
            mphi = (f"<p><b>M-φ</b> (P=0): M_cr {u.M_disp(m['M_cr']):.4g}, "
                    f"M_u {u.M_disp(m['M_u']):.4g} {u.Ml}, "
                    f"μ_φ {m.get('mu_phi') or '—'}</p>")
        except Exception:                              # noqa: BLE001
            pass
        return (f"<h2>{self._spec.kind} section — {code}</h2>"
                f"<table border=1 cellspacing=0 cellpadding=4>{rows}</table>"
                f"{mphi}<p><i>Fibre-based composite: P-M-M via the 3-D surface "
                "tab; single-material verification not applicable.</i></p>")

    def _report_html(self, case, code) -> str:
        try:
            if self._spec.kind == "Composite":
                mphi = core.composite_mphi(self._spec, 0.0,
                                           kappa_max=self._spec.kappa_max,
                                           materials=self._materials)
            else:
                mphi = core.mphi_data(case, 0.0, **core.mphi_props(self._spec))
        except Exception:                              # noqa: BLE001
            mphi = None
        dem = None if self._spec.kind == "Composite" else self._demands()
        dres = None
        if dem:
            try:
                dres = core.demand_check(case, code, dem,
                                         design=self.design_chk.isChecked(),
                                         spec=self._spec)
            except Exception:                          # noqa: BLE001
                dres = None
        return core.report_html(case, code, self._units, mphi=mphi,
                                demand_results=dres,
                                meta={"Kind": self._spec.kind, "Code": code})

    # -------------------------------------------------- sections / project
    def _load_active(self) -> None:
        """Point the working spec at the active section and refresh everything."""
        rec = self._sections[self._active]
        self._spec = rec["spec"]
        self._loading = True
        self.code_combo.setCurrentText(rec.get("code", core.CODES[0]))
        self._loading = False
        self._load_form_from_spec()
        self._refresh_geometry()
        self._recompute_analysis()

    def _reload_section_nav(self) -> None:
        self._loading = True
        self.nav_list.clear()
        self._nav_rows = {}
        for name, rec in self._sections.items():
            it = QListWidgetItem()
            it.setData(Qt.ItemDataRole.UserRole, name)
            self.nav_list.addItem(it)
            wdg, labels = self._make_nav_widget(name, rec)
            it.setSizeHint(wdg.sizeHint())
            self.nav_list.setItemWidget(it, wdg)
            self._nav_rows[name] = (it, labels)
        names = list(self._sections)
        if self._active in names:
            self.nav_list.setCurrentRow(names.index(self._active))
        self._loading = False

    def _update_nav_item(self, name: str) -> None:
        """Refresh one navigator row's summary + status (live, on edits)."""
        row = self._nav_rows.get(name)
        rec = self._sections.get(name)
        if row and rec:
            nl, sl = row[1]
            self._fill_nav_labels(nl, sl, name, rec["spec"])

    def _on_nav_changed(self, row: int) -> None:
        if self._loading or not (0 <= row < self.nav_list.count()):
            return
        name = self.nav_list.item(row).data(Qt.ItemDataRole.UserRole)
        if name:
            self._switch_section(name)

    def _switch_section(self, name: str) -> None:
        if self._loading or name not in self._sections or name == self._active:
            return
        self._active = name
        self._load_active()

    def _unique_name(self, base: str) -> str:
        name, i = base, 2
        while name in self._sections:
            name = f"{base} ({i})"
            i += 1
        return name

    def _new_section(self) -> None:
        name = self._unique_name(f"Section {len(self._sections) + 1}")
        self._sections[name] = {"spec": core.Spec(),
                                "code": self.code_combo.currentText(),
                                "conc_mat": None, "steel_mat": None}
        self._active = name
        self._reload_section_nav()
        self._load_active()

    def _dup_section(self) -> None:
        name = self._unique_name(f"{self._active} copy")
        rec = self._sections[self._active]
        self._sections[name] = {"spec": rec["spec"], "code": rec["code"],
                                "conc_mat": rec.get("conc_mat"),
                                "steel_mat": rec.get("steel_mat")}
        self._active = name
        self._reload_section_nav()
        self._load_active()

    def _rename_section(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        new, ok = QInputDialog.getText(self, "Rename section",
                                       "New name:", text=self._active)
        new = new.strip()
        if not ok or not new or new == self._active:
            return
        if new in self._sections:
            QMessageBox.information(self, "Rename", "A section with that name "
                                    "already exists.")
            return
        # preserve order
        self._sections = {new if k == self._active else k: v
                          for k, v in self._sections.items()}
        self._active = new
        self._reload_section_nav()

    def _del_section(self) -> None:
        if len(self._sections) <= 1:
            QMessageBox.information(self, "Delete section",
                                    "A project needs at least one section.")
            return
        del self._sections[self._active]
        self._active = next(iter(self._sections))
        self._reload_section_nav()
        self._load_active()

    def _new_project(self) -> None:
        self._sections = {"Section 1": {"spec": core.Spec(),
                                        "code": self.code_combo.currentText(),
                                        "conc_mat": None, "steel_mat": None}}
        self._active = "Section 1"
        self._materials = {}
        self._reload_section_nav()
        self._load_active()

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "",
                                              "Section project (*.json)")
        if not path:
            return
        try:
            text = open(path, encoding="utf-8").read()
            sections, active, _demands, materials = core.project_from_json(text)
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self._sections = sections
        self._active = active
        self._materials = materials
        self._reload_section_nav()
        self._load_active()
        self.statusBar().showMessage(f"Opened {path}")

    def _save_project(self) -> None:
        # make sure the active section is current before serialising
        self._sections[self._active]["spec"] = self._spec
        self._sections[self._active]["code"] = self.code_combo.currentText()
        path, _ = QFileDialog.getSaveFileName(self, "Save project",
                                              "sections.json",
                                              "Section project (*.json)")
        if not path:
            return
        try:
            text = core.project_to_json(self._sections, self._active,
                                        {}, self._materials)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            self.statusBar().showMessage(f"Saved {path}")
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))

    def _open_materials(self) -> None:
        dlg = MaterialsDialog(self, materials=self._materials)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._materials = dlg.materials
            self._refresh_material_combos()   # section pickers
            self._refresh_comp_mat_combo()    # composite shape picker
            self._on_value_changed()          # re-resolve laws onto the spec

    def _apply_to_fem(self) -> None:
        """Push the active section into the bridged FEM model as a Section whose
        A / Iz / Iy / J come from this GSD section (and carry the gsd_spec)."""
        if self._fem is None:
            return
        from dataclasses import asdict
        try:
            self._fem.apply_gsd_section(self._active, asdict(self._spec),
                                        self.code_combo.currentText())
            self.statusBar().showMessage(
                f"Applied '{self._active}' to the FEM model.")
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Apply failed", str(exc))

    # -------------------------------------------------------------- export
    def _export_section_json(self) -> None:
        try:
            text = core.section_json_of(_case(self._spec))
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        if not text:
            QMessageBox.information(self, "Export", "No section JSON available "
                                    "for this kind.")
            return
        self._write("section.json", "JSON (*.json)", text)

    def _export_verify_csv(self) -> None:
        code = self.code_combo.currentText()
        rows = self._verify_rows(_case(self._spec), code)
        path, _ = QFileDialog.getSaveFileName(self, "Export verification",
                                              "verification.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["Quantity", "Units", "Computed", "Tol %", "Note"])
                for it in rows:
                    w.writerow([it.get("quantity", ""), it.get("units", ""),
                                it.get("computed", ""), it.get("tol_pct", ""),
                                it.get("note", "")])
            self.statusBar().showMessage(f"Saved {path}")
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))

    def _export_report_html(self) -> None:
        code = self.code_combo.currentText()
        html = self._report_html(_case(self._spec), code)
        self._write("report.html", "HTML (*.html)", html)

    def _write(self, default_name, flt, text) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export", default_name, flt)
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            self.statusBar().showMessage(f"Saved {path}")
        except Exception as exc:                       # noqa: BLE001
            QMessageBox.critical(self, "Export failed", str(exc))


class ArrangementDialog(QDialog):
    """Add a rebar or tendon *arrangement* (layout generator). Returns the
    engine tuple via ``result_arr`` — ``(code, dia, mat, params)`` for rebar or
    ``(code, area, f_pe, mat, params)`` for tendon. Coordinates entered in mm,
    stored in metres; angles in degrees. Mirrors the Streamlit add-forms."""

    def __init__(self, parent=None, *, tendon: bool = False, steels=None):
        super().__init__(parent)
        self._tendon = tendon
        self.result_arr = None
        self.setWindowTitle("Add tendon arrangement" if tendon
                            else "Add rebar arrangement")
        v = QVBoxLayout(self)

        top = QFormLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(core.TENDON_ARR_TYPES if tendon
                                 else core.REBAR_ARR_TYPES)
        self.type_combo.currentTextChanged.connect(self._rebuild)
        top.addRow("Type", self.type_combo)
        if tendon:
            self.area_spin = self._d(50, 5000, 10, " mm²", 0, 140)
            self.fpe_spin = self._d(500, 1600, 25, " MPa", 0, 1100)
            top.addRow("Aₚ / tendon", self.area_spin)
            top.addRow("f_pe", self.fpe_spin)
        else:
            self.dia_combo = QComboBox()
            self.dia_combo.addItems(list(core.BAR_SIZES))
            self.dia_combo.setCurrentText("25 mm")
            top.addRow("Bar size", self.dia_combo)
        self.mat_combo = QComboBox()
        self.mat_combo.addItems(["(section steel)"] + list(steels or []))
        top.addRow("Material", self.mat_combo)
        v.addLayout(top)

        self.param_box = QGroupBox("Parameters")
        self.param_form = QFormLayout(self.param_box)
        self._spins: dict[str, QDoubleSpinBox] = {}
        v.addWidget(self.param_box)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self._rebuild()

    @staticmethod
    def _d(lo, hi, step, suffix, dec, val) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(dec)
        if suffix:
            s.setSuffix(suffix)
        s.setValue(val)
        return s

    def _rebuild(self, *_a) -> None:
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self._spins.clear()
        typ = self.type_combo.currentText()
        # (label, key, suffix, decimals, default, is_int)
        specs: list[tuple] = []
        mm = (" mm", 0)
        if typ == "Point":
            specs = [("z", "z", *mm, 0, 0), ("y", "y", *mm, (-350 if self._tendon
                                                             else 0), 0)]
        elif typ == "Line":
            specs = [("Count n", "n", "", 0, 4 if self._tendon else 3, 1),
                     ("z1", "z1", *mm, -200 if self._tendon else -150, 0),
                     ("y1", "y1", *mm, -350 if self._tendon else -250, 0),
                     ("z2", "z2", *mm, 200 if self._tendon else 150, 0),
                     ("y2", "y2", *mm, -350 if self._tendon else -250, 0)]
        elif typ == "Arc":
            specs = [("Count n", "n", "", 0, 4, 1),
                     ("centre z", "cz", *mm, 0, 0),
                     ("centre y", "cy", *mm, 0, 0),
                     ("radius", "r", *mm, 300 if self._tendon else 200, 0),
                     ("start °", "a1", " °", 0, 200 if self._tendon else 0, 0),
                     ("end °", "a2", " °", 0, 340 if self._tendon else 360, 0)]
        elif typ == "Rectangle":
            specs = [("n horiz.", "nz", "", 0, 3, 1), ("n vert.", "ny", "", 0, 3, 1),
                     ("centre z", "cz", *mm, 0, 0), ("centre y", "cy", *mm, 0, 0),
                     ("width", "w", *mm, 300, 0), ("height", "h", *mm, 500, 0)]
        else:                                        # Perimeter
            specs = [("Count n", "n", "", 0, 8, 1), ("cover", "cov", *mm, 50, 0)]
        for label, key, suffix, dec, default, is_int in specs:
            lo, hi = (1, 200) if is_int else (-5000, 5000)
            sp = self._d(lo, hi, 1 if is_int else 10, suffix,
                         0 if is_int else dec, default)
            self.param_form.addRow(label, sp)
            self._spins[key] = sp

    def _params(self) -> tuple:
        typ = self.type_combo.currentText()
        g = {k: sp.value() for k, sp in self._spins.items()}

        def m(k):                                    # mm -> metres
            return g[k] / 1000.0
        if typ == "Point":
            return (m("z"), m("y"))
        if typ == "Line":
            return (int(g["n"]), m("z1"), m("y1"), m("z2"), m("y2"))
        if typ == "Arc":
            return (int(g["n"]), m("cz"), m("cy"), m("r"), g["a1"], g["a2"])
        if typ == "Rectangle":
            return (int(g["nz"]), int(g["ny"]), m("cz"), m("cy"), m("w"), m("h"))
        return (int(g["n"]), m("cov"))               # Perimeter

    def _accept(self) -> None:
        code = _ARR_CODE[self.type_combo.currentText()]
        params = self._params()
        mat = self.mat_combo.currentText()
        mat = "" if mat == "(section steel)" else mat
        if self._tendon:
            self.result_arr = (code, self.area_spin.value() * 1e-6,
                               self.fpe_spin.value() * 1e6, mat, params)
        else:
            self.result_arr = (code, core.BAR_SIZES[self.dia_combo.currentText()],
                               mat, params)
        self.accept()


def _conc_default(fc_mpa=30.0) -> dict:
    return dict(kind="concrete", fc=fc_mpa * 1e6, conc_model="Kent-Park",
                eps_c0=0.002, eps_cu=0.0035, fcu_ratio=0.4, fr_model="sqrt",
                fr_coeff=0.62, eps_decay=1e-3, conc_f1_ratio=0.4)


def _steel_default(fy_mpa=500.0) -> dict:
    return dict(kind="steel", fy=fy_mpa * 1e6, steel_model="Bilinear",
                Es=200e9, steel_b=0.01, steel_fu_ratio=1.5,
                steel_eps_sh=0.008, steel_eps_su=0.10)


def _prestress_default(fpu_mpa=1860.0) -> dict:
    """Prestressing (tendon) steel — the second 'rebar steel' kind. f_py ~ 0.9
    f_pu; strand law is a bilinear per the engine (E_p 195 GPa)."""
    return dict(kind="prestress", fpu=fpu_mpa * 1e6, fpy=0.9 * fpu_mpa * 1e6,
                Ep=195e9, ps_b=0.005)


# material parameter editors: (label, key, spec) where spec is
# ("spin", lo, hi, step, decimals, store_scale) — display value ×scale = stored
# SI value (f'c MPa→Pa, E_s GPa→Pa) — or ("combo", options).
_CONC_FIELDS = [
    ("f'c [MPa]", "fc", ("spin", 5, 150, 1, 0, 1e6)),
    ("Concrete model", "conc_model", ("combo", list(core.CONC_MODELS))),
    ("Trilinear knee f1/f'c", "conc_f1_ratio", ("spin", 0.1, 0.9, 0.05, 2, 1)),
    ("Rupture model", "fr_model", ("combo", ["sqrt", "ec2"])),
    ("Rupture coeff", "fr_coeff", ("spin", 0.1, 1.0, 0.01, 2, 1)),
    ("Tension-stiff decay ε", "eps_decay", ("spin", 0.0, 0.01, 0.0005, 4, 1)),
]
_STEEL_FIELDS = [
    ("f_y [MPa]", "fy", ("spin", 200, 700, 10, 0, 1e6)),
    ("E_s [GPa]", "Es", ("spin", 150, 230, 5, 0, 1e9)),
    ("Steel model", "steel_model", ("combo", list(core.STEEL_MODELS))),
    ("Hardening b", "steel_b", ("spin", 0.0, 0.1, 0.005, 3, 1)),
    ("f_su / f_y", "steel_fu_ratio", ("spin", 1.0, 2.0, 0.05, 2, 1)),
    ("ε_sh onset", "steel_eps_sh", ("spin", 0.0, 0.05, 0.001, 3, 1)),
    ("ε_su ultimate", "steel_eps_su", ("spin", 0.0, 0.3, 0.005, 3, 1)),
]
_PRESTRESS_FIELDS = [
    ("f_pu [MPa]", "fpu", ("spin", 1500, 2100, 10, 0, 1e6)),
    ("f_py [MPa]", "fpy", ("spin", 1300, 1900, 10, 0, 1e6)),
    ("E_p [GPa]", "Ep", ("spin", 180, 210, 5, 0, 1e9)),
    ("Hardening b", "ps_b", ("spin", 0.0, 0.05, 0.005, 3, 1)),
]


class MaterialsDialog(QDialog):
    """Manage the shared materials library ({name: matd}) — each material's full
    constitutive law (concrete model + rupture/tension-stiffening, or steel
    model + hardening) is edited here and reused by every section that
    references it. Confinement is NOT here — it lives on the section."""

    def __init__(self, parent=None, *, materials: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Materials library")
        self.resize(660, 560)
        self.materials = dict(materials or {})
        self._names = list(self.materials.keys())
        self._mats = [dict(m) for m in self.materials.values()]
        self._cur = -1
        self._widgets: dict = {}
        self._loading = False

        row = QHBoxLayout(self)
        # left: material list + add/remove
        left = QVBoxLayout()
        self.listw = QListWidget()
        self.listw.currentRowChanged.connect(self._select)
        left.addWidget(self.listw)
        brow = QHBoxLayout()
        for txt, fn in (("+ Concrete", lambda: self._add(_conc_default())),
                        ("+ Steel", lambda: self._add(_steel_default())),
                        ("+ Prestress", lambda: self._add(_prestress_default())),
                        ("Remove", self._remove)):
            b = QPushButton(txt)
            b.clicked.connect(fn)
            brow.addWidget(b)
        left.addLayout(brow)
        row.addLayout(left, 1)

        # right: the selected material's editable parameters
        right = QVBoxLayout()
        self.editor = QGroupBox("Material")
        self.form = QFormLayout(self.editor)
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._name_changed)
        self.form.addRow("Name", self.name_edit)
        right.addWidget(self.editor)
        # stress-strain reference diagram for the selected material
        chart_box = QGroupBox("Stress-strain diagram")
        cbl = QVBoxLayout(chart_box)
        self.mat_fig = Figure(figsize=(4.0, 2.4), layout="constrained")
        self.mat_canvas = Canvas(self.mat_fig)
        self.mat_canvas.setMinimumHeight(200)
        cbl.addWidget(self.mat_canvas)
        right.addWidget(chart_box, 1)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        right.addWidget(bb)
        row.addLayout(right, 2)

        self._reload_list()
        if self._names:
            self.listw.setCurrentRow(0)

    # ---- list ----
    @staticmethod
    def _strength(md: dict) -> float:
        kind = md.get("kind", "concrete")
        key = {"concrete": "fc", "steel": "fy", "prestress": "fpu"}.get(kind,
                                                                        "fc")
        return md.get(key, 0.0) / 1e6

    def _item_text(self, nm: str, md: dict) -> str:
        return f"{nm}   ·  {md.get('kind', 'concrete')} {self._strength(md):.0f} MPa"

    def _reload_list(self) -> None:
        was = self._loading
        self._loading = True
        self.listw.clear()
        for nm, md in zip(self._names, self._mats):
            self.listw.addItem(self._item_text(nm, md))
        self._loading = was

    def _add(self, md: dict) -> None:
        prefix = {"concrete": "C", "steel": "S", "prestress": "Y"}.get(
            md["kind"], "M")
        base = f"{prefix}{self._strength(md):.0f}"
        name, i = base, 2
        while name in self._names:
            name = f"{base} ({i})"
            i += 1
        self._names.append(name)
        self._mats.append(md)
        self._reload_list()
        self.listw.setCurrentRow(len(self._names) - 1)

    def _remove(self) -> None:
        r = self.listw.currentRow()
        if 0 <= r < len(self._mats):
            del self._mats[r]
            del self._names[r]
            self._cur = -1
            self._reload_list()
            self.listw.setCurrentRow(min(r, len(self._names) - 1))

    # ---- editor form ----
    def _select(self, r: int) -> None:
        if self._loading:
            return
        if not (0 <= r < len(self._mats)):
            self._cur = -1
            return
        self._cur = r
        self._build_form(self._mats[r])

    def _clear_form_rows(self) -> None:
        # keep row 0 (Name); drop the rest
        while self.form.rowCount() > 1:
            self.form.removeRow(1)
        self._widgets.clear()

    def _build_form(self, md: dict) -> None:
        self._loading = True
        self._clear_form_rows()
        self.name_edit.setText(self._names[self._cur])
        fields = {"concrete": _CONC_FIELDS, "steel": _STEEL_FIELDS,
                  "prestress": _PRESTRESS_FIELDS}.get(md.get("kind"),
                                                      _CONC_FIELDS)
        title = {"prestress": "Prestressing steel"}.get(
            md.get("kind"), md.get("kind", "material").title())
        self.editor.setTitle(title)
        for label, key, spec in fields:
            if spec[0] == "spin":
                _, lo, hi, step, dec, scale = spec
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setSingleStep(step)
                w.setDecimals(dec)
                w.setValue(md.get(key, 0.0) / scale)
                w.valueChanged.connect(self._field_changed)
            else:
                w = QComboBox()
                w.addItems(spec[1])
                w.setCurrentText(str(md.get(key, spec[1][0])))
                w.currentTextChanged.connect(self._field_changed)
            self.form.addRow(label, w)
            self._widgets[key] = (w, spec)
        self._loading = False
        self._draw_material_chart(md)

    def _draw_material_chart(self, md: dict) -> None:
        """AdSec-style stress-strain reference diagram for the material (the
        engine's own uniaxial law). Concrete uses a nominal unconfined
        ε_c0/ε_cu baseline — confinement is applied per section."""
        self.mat_fig.clear()
        ax = self.mat_fig.add_subplot(111)
        try:
            kind = md.get("kind")
            if kind == "concrete":
                m = {"eps_c0": 0.002, "eps_cu": 0.0035, "fcu_ratio": 0.4, **md}
                law = core.concrete_uniaxial_from(m)
                eps = np.linspace(-m["eps_cu"] * 1.05, 0.0015, 240)
            elif kind == "prestress":
                from femsolver.materials.uniaxial import UniaxialBilinear
                law = UniaxialBilinear(E=md.get("Ep", 195e9),
                                       sigma_y=md.get("fpy", 1675e6),
                                       b=md.get("ps_b", 0.005))
                eps = np.linspace(-0.005, 0.025, 240)
            else:
                law = core.steel_uniaxial_from(md)
                esu = (md.get("steel_eps_su", 0.05)
                       if md.get("steel_model") == "Park strain-hardening"
                       else 0.02)
                eps = np.linspace(-esu, esu, 240)
            sig = [law.get_response(float(e))[0] / 1e6 for e in eps]
            ax.plot(eps * 1e3, sig, "-", color=style.C_PRIMARY, lw=2.0)
            ax.axhline(0, color=style.AX_SPINE, lw=0.6)
            ax.axvline(0, color=style.AX_SPINE, lw=0.6)
            ax.set_xlabel("strain ε  [‰]")
            ax.set_ylabel("stress σ  [MPa]")
            style.beautify_axes(ax)
        except Exception as exc:                       # noqa: BLE001
            ax.text(0.5, 0.5, f"—\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
        self.mat_canvas.draw_idle()

    def _field_changed(self, *_a) -> None:
        if self._loading or not (0 <= self._cur < len(self._mats)):
            return
        md = self._mats[self._cur]
        for key, (w, spec) in self._widgets.items():
            if spec[0] == "spin":
                md[key] = w.value() * spec[5]
            else:
                md[key] = w.currentText()
        it = self.listw.item(self._cur)     # refresh just this row's label
        if it:
            it.setText(self._item_text(self._names[self._cur], md))
        self._draw_material_chart(md)

    def _name_changed(self) -> None:
        if self._loading or not (0 <= self._cur < len(self._names)):
            return
        nm = self.name_edit.text().strip()
        if nm:
            self._names[self._cur] = nm
            it = self.listw.item(self._cur)
            if it:
                it.setText(self._item_text(nm, self._mats[self._cur]))

    def _accept(self) -> None:
        out: dict = {}
        for nm, md in zip(self._names, self._mats):
            nm = (nm or "Material").strip()
            while nm in out:
                nm += "*"
            out[nm] = dict(md)
        self.materials = out
        self.accept()
