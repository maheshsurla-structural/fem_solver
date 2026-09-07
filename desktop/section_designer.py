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
import re
import sys
from dataclasses import replace
from functools import lru_cache

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-d proj.)
from PySide6.QtCore import Qt, QByteArray, QTimer
from PySide6.QtCore import QSize
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QToolButton
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow, QMenu,
                               QMessageBox, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QStackedWidget, QStyledItemDelegate,
                               QTableWidget,
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


class ComboBoxDelegate(QStyledItemDelegate):
    """Table-cell delegate that shows the value as plain text but edits it with
    a drop-down — so a column of choices reads as a table, not a strip of
    embedded combo widgets. ``options`` is a zero-arg callable returning the
    current choice list (evaluated each time the cell is edited)."""

    def __init__(self, options, parent=None):
        super().__init__(parent)
        self._options = options

    def createEditor(self, parent, option, index):
        cb = QComboBox(parent)
        cb.addItems([str(o) for o in self._options()])
        # commit + close as soon as the user picks, so one click edits the cell
        cb.activated.connect(lambda *_: (self.commitData.emit(cb),
                                         self.closeEditor.emit(cb)))
        return cb

    def setEditorData(self, editor, index):
        val = str(index.data(Qt.ItemDataRole.EditRole) or "")
        i = editor.findText(val)
        editor.setCurrentIndex(i if i >= 0 else 0)
        editor.showPopup()

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class SectionDesignerWindow(QMainWindow):
    """Standalone General Section Designer over ``section_gui_core``."""

    def __init__(self, parent=None, *, spec: core.Spec | None = None,
                 code: str | None = None, fem_window=None,
                 section_name: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("General Section Designer")
        self.resize(1240, 860)

        self._fem = fem_window           # FEM MainWindow for the model bridge
        # default section: reinforcement comes from the AdSec GROUPS table, so
        # start with zeroed parametric counts + a couple of starter groups.
        self._spec = spec or replace(
            core.Spec(), n_top=0, n_bot=0, n_side=0, n_perim=0,
            rebar_groups=(("Top", "3B25", "", "S500"),
                          ("Bottom", "3B25", "", "S500")))
        self._code0 = code if (code and code in core.CODES) else core.CODES[0]
        name0 = section_name or "Section 1"
        # shared material library {name: matd} — constitutive laws live here.
        self._materials: dict = {"C30": _conc_default(30.0),
                                 "S500": _steel_default(500.0),
                                 "Y1860": _prestress_default(1860.0)}
        # multi-section project: {name: {"spec", "code", "conc_mat"}}
        self._sections: dict = {name0: {
            "spec": self._spec, "code": self._code0, "conc_mat": "C30"}}
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
        self.code_combo.currentTextChanged.connect(
            lambda *_: self._on_code_changed())
        tb.addWidget(self.code_combo)
        tb.addWidget(QLabel("  Rebar:  "))
        # Reinforcement standard, decoupled from the design code — drives the
        # Pattern notation (US #-sizes for ASTM, metric ⌀mm otherwise).
        self.rebar_std_combo = QComboBox()
        for label, data in (("Follow design code", "auto"),
                            ("ASTM (US #)", "ACI"), ("EN (⌀ mm)", "EC2"),
                            ("IS (⌀ mm)", "IS")):
            self.rebar_std_combo.addItem(label, data)
        # default to metric ⌀mm — the presets are metric, and the rebar standard
        # is deliberately decoupled from the (US-default) design code.
        self.rebar_std_combo.setCurrentIndex(
            self.rebar_std_combo.findData("EC2"))
        self.rebar_std_combo.currentIndexChanged.connect(
            lambda *_: self._on_rebar_std_changed())
        tb.addWidget(self.rebar_std_combo)
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
        # New: a menu of starter presets (plus a blank section)
        new_btn = QToolButton()
        new_btn.setText("＋ New")
        new_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(new_btn)
        menu.addAction("Blank section", self._new_section)
        menu.addSeparator()
        for name in _section_presets():
            menu.addAction(name,
                           lambda _=False, n=name: self._new_from_preset(n))
        new_btn.setMenu(menu)
        row.addWidget(new_btn)
        for txt, fn in (("Dup", self._dup_section), ("Del", self._del_section)):
            b = QPushButton(txt)
            b.clicked.connect(lambda _=False, f=fn: f())
            row.addWidget(b)
        v.addLayout(row)
        return w

    def _new_from_preset(self, preset: str) -> None:
        spec = _section_presets().get(preset)
        if spec is None:
            return
        name = self._unique_name(preset)
        self._sections[name] = {"spec": spec,
                                "code": self.code_combo.currentText(),
                                "conc_mat": "C30"}
        self._active = name
        self._reload_section_nav()
        self._load_active()

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
        lay = QHBoxLayout(w)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(8)
        thumb = QSvgWidget()
        thumb.setFixedSize(QSize(44, 44))
        lay.addWidget(thumb)
        txt = QVBoxLayout()
        txt.setSpacing(1)
        nl = QLabel()
        nl.setStyleSheet("font-weight:600; background:transparent;")
        sl = QLabel()
        sl.setObjectName("sub")
        sl.setStyleSheet("background:transparent;")
        txt.addWidget(nl)
        txt.addWidget(sl)
        lay.addLayout(txt, 1)
        self._fill_nav_labels(nl, sl, name, rec["spec"])
        self._fill_nav_thumb(thumb, rec["spec"])
        return w, (nl, sl, thumb)

    def _fill_nav_labels(self, nl, sl, name, spec) -> None:
        warn = "" if self._section_has_reinf(spec) else "⚠ "
        nl.setText(f"{warn}{name}")
        sl.setText(self._spec_summary(spec))

    def _fill_nav_thumb(self, thumb, spec) -> None:
        """Mini cross-section drawing for a navigator row (aspect preserved)."""
        try:
            thumb.load(QByteArray(core.svg_of(_case(spec)).encode("utf-8")))
            r = thumb.renderer()
            if r is not None:
                r.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        except Exception:                              # noqa: BLE001
            thumb.load(QByteArray(b""))

    def _build_section_tab(self) -> QWidget:
        """The 'Section' workspace tab: inputs (left) + cross-section drawing
        and properties (right)."""
        inner = QSplitter(Qt.Orientation.Horizontal)
        inner.addWidget(self._build_definition_panel())
        inner.addWidget(self._build_preview_panel())
        inner.setStretchFactor(0, 1)
        inner.setStretchFactor(1, 1)
        inner.setSizes([520, 520])
        inner.setHandleWidth(8)
        return inner

    # ---------------------------------------------------------- definition
    def _build_definition_panel(self) -> QWidget:
        host = QScrollArea()
        host.setWidgetResizable(True)
        host.setMinimumWidth(360)
        host.setMaximumWidth(720)
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

        # Materials: only the section concrete is chosen here (one per section);
        # steel & prestressing steel are chosen per bar/tendon in the tables
        # below, so different rods can use different steels.
        mbox = CollapsibleGroup("Materials")
        self.mat_box = mbox
        mf = QFormLayout(mbox.body)
        self.conc_mat_combo = QComboBox()
        self.conc_mat_combo.currentTextChanged.connect(
            lambda *_: self._on_material_choice())
        mf.addRow("Concrete", self.conc_mat_combo)
        manage = QPushButton("Manage library…")
        manage.clicked.connect(self._open_materials)
        mf.addRow("", manage)
        v.addWidget(mbox)

        # Reinforcement — Streamlit-style sub-tabs: Rebars (an AdSec GROUPS
        # table), Tendons (arrangement table), Confinement (M-φ / Mander).
        v.addWidget(self._build_reinforcement_group())

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
        # milestone table (left) + strain-profile diagram (right)
        bottom = QHBoxLayout()
        self.mphi_tbl = QTableWidget(0, 4)
        self.mphi_tbl.setHorizontalHeaderLabels(
            ["Point", "State", "Curvature", "Moment"])
        self.mphi_tbl.horizontalHeader().setStretchLastSection(True)
        self.mphi_tbl.verticalHeader().setVisible(False)
        self.mphi_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.mphi_tbl.setAlternatingRowColors(True)
        self.mphi_tbl.setShowGrid(False)
        self.mphi_tbl.setMaximumHeight(170)
        bottom.addWidget(self.mphi_tbl, 1)

        strain_box = QVBoxLayout()
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Strain diagram at:"))
        self.strain_combo = QComboBox()
        self.strain_combo.currentIndexChanged.connect(
            lambda *_: self._draw_strain_profile())
        srow.addWidget(self.strain_combo, 1)
        strain_box.addLayout(srow)
        self.strain_fig = Figure(figsize=(3.2, 2.2), layout="constrained")
        self.strain_canvas = Canvas(self.strain_fig)
        self.strain_canvas.setMaximumHeight(190)
        strain_box.addWidget(self.strain_canvas)
        self.strain_metrics = QLabel("—")
        self.strain_metrics.setObjectName("caption")
        self.strain_metrics.setTextFormat(Qt.TextFormat.RichText)
        strain_box.addWidget(self.strain_metrics)
        bottom.addLayout(strain_box, 1)
        mpv.addLayout(bottom)

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

        # ---- M-M contour (biaxial slice at a chosen P) ----
        mm = QWidget()
        mmv = QVBoxLayout(mm)
        self.mm_fig = Figure(figsize=(4.2, 4.0), layout="constrained")
        self.mm_canvas = Canvas(self.mm_fig)
        mmv.addWidget(self.mm_canvas)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Axial P (+comp):"))
        self.mm_P = self._dspin(-1e6, 1e6, 50, "", 1)
        self.mm_P.valueChanged.connect(lambda *_: self._queue())
        mrow.addWidget(self.mm_P)
        mrow.addWidget(QLabel("Demand Mz:"))
        self.mm_Mz = self._dspin(-1e6, 1e6, 25, "", 1)
        self.mm_Mz.valueChanged.connect(lambda *_: self._queue())
        mrow.addWidget(self.mm_Mz)
        mrow.addWidget(QLabel("My:"))
        self.mm_My = self._dspin(-1e6, 1e6, 25, "", 1)
        self.mm_My.valueChanged.connect(lambda *_: self._queue())
        mrow.addWidget(self.mm_My)
        mrow.addStretch(1)
        mmv.addLayout(mrow)

        # ---- Stress field (fibre stresses under a plane-sections strain) ----
        sf = QWidget()
        sfv = QVBoxLayout(sf)
        self.sf_fig = Figure(figsize=(4.2, 4.2), layout="constrained")
        self.sf_canvas = Canvas(self.sf_fig)
        sfv.addWidget(self.sf_canvas)
        sfr = QHBoxLayout()
        sfr.addWidget(QLabel("ε top-fibre [‰]:"))
        self.sf_etop = self._dspin(-20, 20, 0.1, "", 2)
        self.sf_etop.setValue(-1.5)
        self.sf_etop.valueChanged.connect(lambda *_: self._queue())
        sfr.addWidget(self.sf_etop)
        sfr.addWidget(QLabel("ε bottom-fibre [‰]:"))
        self.sf_ebot = self._dspin(-20, 20, 0.1, "", 2)
        self.sf_ebot.setValue(1.0)
        self.sf_ebot.valueChanged.connect(lambda *_: self._queue())
        sfr.addWidget(self.sf_ebot)
        sfr.addStretch(1)
        sfv.addLayout(sfr)

        # ---- combined P-M-M interaction tab (P-M curve | 3-D surface | M-M) --
        inter = QWidget()
        iv = QVBoxLayout(inter)
        selrow = QHBoxLayout()
        selrow.addWidget(QLabel("View:"))
        self.inter_view = QComboBox()
        self.inter_view.addItems(["P-M curve", "3-D P-M-M surface",
                                  "M-M contour"])
        self.inter_view.currentIndexChanged.connect(self._on_inter_view)
        selrow.addWidget(self.inter_view)
        selrow.addStretch(1)
        iv.addLayout(selrow)
        self.inter_stack = QStackedWidget()
        self.inter_stack.addWidget(pm)          # 0: P-M curve
        self.inter_stack.addWidget(s3)          # 1: 3-D surface
        self.inter_stack.addWidget(mm)          # 2: M-M contour
        iv.addWidget(self.inter_stack, 1)

        # ---- Report ----
        self.report = QTextBrowser()

        # assemble the workspace tabs (Section is inserted at 0 later)
        self.tabs.addTab(inter, "P-M-M interaction")
        self.tabs.addTab(mp, "Moment-curvature")
        self.tabs.addTab(vt, "Verification")
        self.tabs.addTab(sf, "Stress field")
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
        self._apply_kind_visibility(kind)
        if kind == "Custom":
            self._load_custom_tables()
        elif kind == "Composite":
            self._refresh_comp_list()
        # group Type options track the shape (rect family vs round); rebuild the
        # table so the combos reflect the new kind, then sync any snapped types.
        if hasattr(self, "groups_tbl"):
            self._refresh_groups_table()
            self._spec = replace(self._spec,
                                 rebar_groups=self._read_groups_table())
        self._on_value_changed()

    def _apply_kind_visibility(self, kind: str) -> None:
        parametric = kind in _PARAMETRIC
        self.dim_box.setVisible(parametric)
        self.psc_box.setVisible(kind == "PSC girder")
        self.custom_box.setVisible(kind == "Custom")
        self.composite_box.setVisible(kind == "Composite")
        # Composite draws material from its shapes; single-material groups hide.
        self.mat_box.setVisible(kind != "Composite")
        self._apply_cover_visibility()

    def _apply_cover_visibility(self) -> None:
        """Variable (per-face) cover is only meaningful for the rectangular
        family (Top/Bottom/Sides). Show the mode picker there; elsewhere fall
        back to the single uniform-cover spin."""
        if not hasattr(self, "cover_mode_combo"):
            return
        kind = self.kind_combo.currentText()
        faced = kind in ("Rectangular", "Hollow box")
        self.cover_mode_combo.setVisible(faced)
        variable = faced and self.cover_mode_combo.currentText() == "Variable"
        self.cover_face_row.setVisible(variable)
        self.cover_spin.setVisible(not variable)

    def _on_cover_mode_changed(self) -> None:
        self._apply_cover_visibility()
        if not self._loading:
            self._on_value_changed()

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

    # ------------------------------------------------ reinforcement (groups)
    def _build_reinforcement_group(self) -> CollapsibleGroup:
        """Reinforcement input as Streamlit-style sub-tabs: Rebars (an AdSec
        GROUPS table), Tendons (arrangement table), Confinement (M-φ / Mander)."""
        box = CollapsibleGroup("Reinforcement")
        outer = QVBoxLayout(box.body)
        outer.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        tabs.addTab(self._build_rebars_tab(), "Rebars")
        tabs.addTab(self._build_tendons_tab(), "Tendons")
        tabs.addTab(self._build_confinement_tab(), "Confinement")
        outer.addWidget(tabs)
        return box

    def _build_rebars_tab(self) -> QWidget:
        """Uniform cover + spiral + an AdSec-style GROUPS table (each row a
        reinforcement group: Type / Material / Pattern / Position(s)). The
        group types track the shape (Top/Bottom/Sides/Link for the rectangular
        family, Perimeter/Line/Arc/Single otherwise)."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 6, 2, 2)
        # Cover: Uniform (all faces) or Variable (Top/Bottom/Sides), AdSec-style.
        cov = QHBoxLayout()
        cov.addWidget(QLabel("Cover"))
        self.cover_mode_combo = QComboBox()
        self.cover_mode_combo.addItems(["Uniform", "Variable"])
        self.cover_mode_combo.currentIndexChanged.connect(
            lambda *_: self._on_cover_mode_changed())
        cov.addWidget(self.cover_mode_combo)
        self.cover_spin = self._dspin(5, 150, 5, " mm", 0)
        self.cover_spin.valueChanged.connect(lambda *_: self._on_value_changed())
        cov.addWidget(self.cover_spin)
        cov.addStretch(1)
        self.spiral_chk = QCheckBox("Spiral (φ cap 0.85)")
        self.spiral_chk.stateChanged.connect(lambda *_: self._on_value_changed())
        cov.addWidget(self.spiral_chk)
        v.addLayout(cov)
        # Per-face covers (shown only for the rectangular family + Variable mode)
        self.cover_face_row = QWidget()
        fcov = QHBoxLayout(self.cover_face_row)
        fcov.setContentsMargins(0, 0, 0, 0)
        self.cover_top_spin = self._dspin(5, 150, 5, "", 0)
        self.cover_bot_spin = self._dspin(5, 150, 5, "", 0)
        self.cover_side_spin = self._dspin(5, 150, 5, "", 0)
        for lab, sp in (("Top", self.cover_top_spin), ("Bottom",
                        self.cover_bot_spin), ("Sides", self.cover_side_spin)):
            fcov.addWidget(QLabel(f"{lab} [mm]"))
            sp.valueChanged.connect(lambda *_: self._on_value_changed())
            fcov.addWidget(sp)
        fcov.addStretch(1)
        self.cover_face_row.setVisible(False)
        v.addWidget(self.cover_face_row)

        self.groups_tbl = QTableWidget(0, 4)
        self.groups_tbl.setHorizontalHeaderLabels(
            ["Type", "Material", "Pattern", "Position(s) [mm, deg]"])
        hh = self.groups_tbl.horizontalHeader()
        hh.setStretchLastSection(True)
        self.groups_tbl.verticalHeader().setVisible(False)
        self.groups_tbl.setAlternatingRowColors(True)
        self.groups_tbl.setMinimumHeight(150)
        # Type + Material edit via a drop-down but render as plain text cells.
        self.groups_tbl.setItemDelegateForColumn(
            0, ComboBoxDelegate(self._group_type_options, self.groups_tbl))
        self.groups_tbl.setItemDelegateForColumn(
            1, ComboBoxDelegate(self._steel_names, self.groups_tbl))
        self.groups_tbl.itemChanged.connect(self._on_group_item_changed)
        v.addWidget(self.groups_tbl)

        row = QHBoxLayout()
        add = QPushButton("＋ Add group…")
        add.clicked.connect(self._add_group_via_dialog)
        rem = QPushButton("Remove selected")
        rem.clicked.connect(self._remove_group_row)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        v.addLayout(row)

        self.group_warn = QLabel("")
        self.group_warn.setWordWrap(True)
        self.group_warn.setStyleSheet("color:#c0392b;")
        self.group_warn.setVisible(False)
        v.addWidget(self.group_warn)
        self.groups_cap = QLabel("")
        self.groups_cap.setWordWrap(True)
        self.groups_cap.setObjectName("hintLabel")
        self.groups_cap.setStyleSheet("color:#5a6b7b; font-size:11px;")
        v.addWidget(self.groups_cap)
        self._refresh_groups_caption()
        return w

    def _build_tendons_tab(self) -> QWidget:
        """Tendons as an editable GROUPS table (Type / Material / Count / Aₚ /
        f_pe / Position) — Point places 1 tendon, Line/Arc place Count along
        the geometry — mirroring the Rebars table. Writes spec.tendon_arr."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 6, 2, 2)
        self.tendon_tbl = QTableWidget(0, 6)
        self.tendon_tbl.setHorizontalHeaderLabels(
            ["Type", "Material", "Count", "Aₚ [mm²]", "f_pe [MPa]",
             "Position(s) [mm, deg]"])
        self.tendon_tbl.horizontalHeader().setStretchLastSection(True)
        self.tendon_tbl.verticalHeader().setVisible(False)
        self.tendon_tbl.setAlternatingRowColors(True)
        self.tendon_tbl.setMinimumHeight(150)
        self.tendon_tbl.setItemDelegateForColumn(
            0, ComboBoxDelegate(lambda: core.TENDON_ARR_TYPES, self.tendon_tbl))
        self.tendon_tbl.setItemDelegateForColumn(
            1, ComboBoxDelegate(self._tendon_mat_names, self.tendon_tbl))
        self.tendon_tbl.itemChanged.connect(self._on_tendon_item_changed)
        v.addWidget(self.tendon_tbl)
        row = QHBoxLayout()
        add = QPushButton("＋ Add tendon")
        add.clicked.connect(lambda: self._add_tendon_row())
        rem = QPushButton("Remove selected")
        rem.clicked.connect(self._remove_tendon_row)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        v.addLayout(row)
        cap = QLabel(
            "Each row is a tendon group.  <b>Point</b> = 1 tendon at z,y · "
            "<b>Line</b> = z1,y1; z2,y2 · <b>Arc</b> = cz,cy,r,a1,a2 place "
            "<i>Count</i> tendons.  Aₚ = area per tendon, f_pe = effective "
            "prestress.")
        cap.setWordWrap(True)
        cap.setObjectName("hintLabel")
        cap.setStyleSheet("color:#5a6b7b; font-size:11px;")
        v.addWidget(cap)
        return w

    def _build_confinement_tab(self) -> QWidget:
        """Mander confinement — read from the section's **Link** tie group +
        geometry (hoop ⌀/spacing/grade, tie legs, core dims, ρ_cc). Shows the
        computed confined-concrete quantities and drives the two-zone M-φ. Below
        it, the base (unconfined) M-φ / sweep parameters."""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(2, 6, 2, 2)
        self.conf_status = QLabel("")
        self.conf_status.setWordWrap(True)
        self.conf_status.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(self.conf_status)
        self.conf_metrics = QLabel("")
        self.conf_metrics.setWordWrap(True)
        self.conf_metrics.setTextFormat(Qt.TextFormat.RichText)
        self.conf_metrics.setVisible(False)
        v.addWidget(self.conf_metrics)

        pbox = CollapsibleGroup("Base concrete (M-φ) parameters", collapsed=True)
        f = QFormLayout(pbox.body)
        self.eps_c0_spin = self._dspin(0.001, 0.02, 0.0002, "", 4)
        self.eps_cu_spin = self._dspin(0.002, 0.05, 0.0005, "", 4)
        self.fcu_ratio_spin = self._dspin(0.0, 1.0, 0.05, "", 2)
        self.kappa_max_spin = self._dspin(0.005, 0.5, 0.01, " 1/m", 3)
        for wgt in (self.eps_c0_spin, self.eps_cu_spin, self.fcu_ratio_spin,
                    self.kappa_max_spin):
            wgt.valueChanged.connect(lambda *_: self._on_value_changed())
        f.addRow("ε_c0 (peak, unconfined)", self.eps_c0_spin)
        f.addRow("ε_cu (crush, unconfined)", self.eps_cu_spin)
        f.addRow("f_cu / f'c (residual)", self.fcu_ratio_spin)
        f.addRow("κ_max sweep", self.kappa_max_spin)
        v.addWidget(pbox)
        v.addStretch(1)
        return w

    _CONF_REASONS = {
        "shape": "Mander confinement applies to Rectangular / Circular "
                 "sections only.",
        "no-link": "Add a tie (<b>Link</b>) group on the Rebars tab — e.g. "
                   "<code>B10-150</code> with legs <code>2x2</code> — to apply "
                   "Mander confinement.",
        "no-spacing": "The tie (<b>Link</b>) group needs a spacing, e.g. "
                      "<code>B10-150</code>.",
        "bad-pattern": "Couldn't read the tie (<b>Link</b>) pattern.",
        "no-cage": "Need a longitudinal cage (≥ 4 bars) for the confined core.",
    }

    def _refresh_confinement_echo(self) -> None:
        """Recompute the section-derived Mander confinement and echo it (or the
        reason it isn't applied) into the Confinement tab."""
        if not hasattr(self, "conf_status"):
            return
        conf, info = _section_confinement(self._spec, self._materials)
        if conf is None:
            self.conf_metrics.setVisible(False)
            self.conf_status.setText(
                self._CONF_REASONS.get(info, "Confinement not applied."))
            return
        try:
            r = core.mander_confinement(dict(
                fc=self._spec.fc, eps_c0=self._spec.eps_c0, **conf))
        except Exception as exc:                       # noqa: BLE001
            self.conf_metrics.setVisible(False)
            self.conf_status.setText(f"Confinement error: {exc}")
            return
        i = info if isinstance(info, dict) else {}
        tie = (f"⌀{i.get('dia_h', 0) * 1e3:.0f} @ {i.get('s', 0) * 1e3:.0f} mm"
               if i else "tie")
        legs = f"{int(i.get('ny', 2))}×{int(i.get('nz', 2))} legs" if i else ""
        self.conf_status.setText(
            f"<b>Mander confinement applied</b> from the Link tie "
            f"({tie}, {legs}) → the M-φ uses a confined core + unconfined "
            f"cover.")
        self.conf_metrics.setVisible(True)
        kcc = r.get("kcc", 1.0)
        self.conf_metrics.setText(
            f"f′cc <b>{r['fcc'] / 1e6:.1f}</b> MPa "
            f"(k<sub>cc</sub> {kcc:.2f}) &nbsp;·&nbsp; "
            f"ε_cc <b>{r['eps_cc']:.4f}</b> &nbsp;·&nbsp; "
            f"ε_cu <b>{r['eps_cu']:.4f}</b> &nbsp;·&nbsp; "
            f"k_e <b>{r.get('ke', 0):.3f}</b> &nbsp;·&nbsp; "
            f"f_l <b>{r.get('fl', 0) / 1e6:.2f}</b> MPa &nbsp;·&nbsp; "
            f"ρ_cc <b>{r.get('rho_cc', 0) * 100:.2f}%</b>")

    def _group_type_options(self) -> list:
        kind = (self.kind_combo.currentText()
                if hasattr(self, "kind_combo") else self._spec.kind)
        faced = kind in ("Rectangular", "Hollow box")
        return list(core.REBAR_GROUP_TYPES_RECT if faced
                    else core.REBAR_GROUP_TYPES_ROUND)

    def _add_group_row(self, group=None) -> None:
        """Append a group row (Type / Material / Pattern / Position). Type and
        Material edit via the drop-down delegate but store as plain text. With
        ``group`` given it seeds from a stored tuple (no change fired); called
        bare (the ＋ button) it seeds a default and commits."""
        types = self._group_type_options()
        steels = self._steel_names()
        t, pat, pos, mat = group or (
            types[0], self._default_pattern(), "",
            steels[0] if steels else "")
        tbl = self.groups_tbl
        was = getattr(self, "_loading_groups", False)
        self._loading_groups = True
        r = tbl.rowCount()
        tbl.insertRow(r)
        for c, val in enumerate((t, mat, pat, pos)):
            tbl.setItem(r, c, QTableWidgetItem(str(val)))
        self._loading_groups = was
        if not was and group is None:
            self._groups_changed()

    def _remove_group_row(self) -> None:
        r = self.groups_tbl.currentRow()
        if r < 0:
            return
        self.groups_tbl.removeRow(r)
        self._groups_changed()

    def _add_group_via_dialog(self) -> None:
        """Open the guided Add-group dialog: the user fills structured inputs
        (bar size, count or spacing, position) and the dialog writes the AdSec
        pattern. Single accepts many coordinates (folding in individual bars).
        Inline cell editing stays available for those who know the notation."""
        dlg = AddGroupDialog(self, types=self._group_type_options(),
                             steels=self._steel_names(),
                             notation=self._rebar_notation())
        if dlg.exec() != QDialog.DialogCode.Accepted or not dlg.result_groups:
            return
        for g in dlg.result_groups:
            self._add_group_row(g)
        self._groups_changed()

    # -------------------------------------------------- tendon groups table
    def _tendon_mat_names(self) -> list:
        return ["(default strand)"] + self._names_of_kind("prestress")

    @staticmethod
    def _tendon_row_fields(arr):
        """Unpack a stored tendon_arr tuple into display fields (mm / MPa), or
        return the default new-row seed when ``arr`` is None."""
        if arr is None:
            return ("Line", "(default strand)", 4, 140.0, 1100.0,
                    "-200,-350; 200,-350")
        typ, area, f_pe, mat, p = (arr[0], arr[1], arr[2], arr[3], list(arr[4]))
        mat_name = mat or "(default strand)"
        area_mm, fpe_mpa = area * 1e6, f_pe / 1e6
        if typ == "point":
            count, pos = 1, f"{p[0] * 1e3:g},{p[1] * 1e3:g}"
        elif typ == "line":
            count = int(p[0])
            pos = (f"{p[1] * 1e3:g},{p[2] * 1e3:g}; "
                   f"{p[3] * 1e3:g},{p[4] * 1e3:g}")
        else:                                            # arc
            count = int(p[0])
            pos = (f"{p[1] * 1e3:g},{p[2] * 1e3:g},{p[3] * 1e3:g},"
                   f"{p[4]:g},{p[5]:g}")
        return (typ.title(), mat_name, count, area_mm, fpe_mpa, pos)

    def _add_tendon_row(self, arr=None) -> None:
        typ, mat_name, count, area_mm, fpe_mpa, pos = self._tendon_row_fields(arr)
        tbl = self.tendon_tbl
        was = getattr(self, "_loading_tendon", False)
        self._loading_tendon = True
        r = tbl.rowCount()
        tbl.insertRow(r)
        vals = (typ, mat_name, str(count), f"{area_mm:g}", f"{fpe_mpa:g}", pos)
        for c, val in enumerate(vals):
            tbl.setItem(r, c, QTableWidgetItem(str(val)))
        self._loading_tendon = was
        if not was and arr is None:
            self._tendons_changed()

    def _remove_tendon_row(self) -> None:
        r = self.tendon_tbl.currentRow()
        if r < 0:
            return
        self.tendon_tbl.removeRow(r)
        self._tendons_changed()

    def _on_tendon_item_changed(self, *_) -> None:
        if getattr(self, "_loading_tendon", False):
            return
        self._tendons_changed()

    def _read_tendon_table(self) -> tuple:
        tbl = self.tendon_tbl
        out = []
        for r in range(tbl.rowCount()):
            t_it, m_it = tbl.item(r, 0), tbl.item(r, 1)
            typ = (t_it.text() if t_it else "Line").strip().lower() or "line"
            name = (m_it.text() if m_it else "(default strand)").strip()
            mat = "" if (not name or name.startswith("(")) else name

            def _num(c, d=0.0):
                it = tbl.item(r, c)
                try:
                    return float((it.text() if it else "").strip())
                except ValueError:
                    return d
            count = max(int(_num(2, 1) or 1), 1)
            area = _num(3, 0.0) / 1e6
            f_pe = _num(4, 0.0) * 1e6
            pos_it = tbl.item(r, 5)
            p = core._parse_positions(pos_it.text() if pos_it else "")
            if typ == "point" and len(p) >= 2:
                params = (p[0] / 1e3, p[1] / 1e3)
            elif typ == "line" and len(p) >= 4:
                params = (count, p[0] / 1e3, p[1] / 1e3, p[2] / 1e3, p[3] / 1e3)
            elif typ == "arc" and len(p) >= 5:
                params = (count, p[0] / 1e3, p[1] / 1e3, p[2] / 1e3, p[3], p[4])
            else:
                continue
            out.append((typ, area, f_pe, mat, params))
        return tuple(out)

    def _tendons_changed(self) -> None:
        if getattr(self, "_loading_tendon", False) or self._loading:
            return
        self._spec = replace(self._spec, tendon_arr=self._read_tendon_table())
        self._on_value_changed()

    def _refresh_tendon_table(self) -> None:
        if not hasattr(self, "tendon_tbl"):
            return
        was = getattr(self, "_loading_tendon", False)
        self._loading_tendon = True
        self.tendon_tbl.setRowCount(0)
        for arr in self._spec.tendon_arr:
            self._add_tendon_row(arr)
        self._loading_tendon = was

    def _on_group_item_changed(self, *_) -> None:
        if getattr(self, "_loading_groups", False):
            return
        self._groups_changed()

    def _read_groups_table(self) -> tuple:
        tbl = self.groups_tbl
        out = []
        for r in range(tbl.rowCount()):
            def _t(c):
                it = tbl.item(r, c)
                return (it.text() if it else "").strip()
            typ, mat, pat, pos = _t(0), _t(1), _t(2), _t(3)
            if not pat:
                continue
            out.append((typ, pat, pos, mat))
        return tuple(out)

    def _groups_changed(self) -> None:
        if getattr(self, "_loading_groups", False) or self._loading:
            return
        groups = self._read_groups_table()
        self._spec = replace(self._spec, rebar_groups=groups)
        self._validate_groups(groups)
        self._on_value_changed()

    def _validate_groups(self, groups) -> None:
        notn = self._rebar_notation()
        bad = []
        for g in groups:
            try:
                core.parse_bar_desc(g[1], notation=notn)   # strict: off-code -> bad
            except ValueError:
                bad.append(g[1])
        if bad:
            self.group_warn.setText(
                f"Not valid for the {self._rebar_fam()} rebar standard: "
                + ", ".join(dict.fromkeys(bad))
                + f".  Use e.g. {self._rebar_examples()}.")
        self.group_warn.setVisible(bool(bad))

    def _refresh_groups_table(self) -> None:
        """Repopulate the groups table from the active spec (on section switch,
        kind change, or a library edit). Type combos follow the shape; each
        stored group's Material is preserved when the grade still exists."""
        if not hasattr(self, "groups_tbl"):
            return
        was = getattr(self, "_loading_groups", False)
        self._loading_groups = True
        self.groups_tbl.setRowCount(0)
        types = self._group_type_options()
        default_steel = (self._steel_names() or [""])[0]
        for g in self._spec.rebar_groups:
            t = g[0] if len(g) > 0 else types[0]
            pat = g[1] if len(g) > 1 else ""
            pos = g[2] if len(g) > 2 else ""
            mat = g[3] if len(g) > 3 and g[3] else default_steel
            self._add_group_row((t, pat, pos, mat))
        self._loading_groups = was
        self._validate_groups(self._spec.rebar_groups)

    # --------------------------------------------------------- spec <-> form
    def _load_form_from_spec(self) -> None:
        self._loading = True
        s = self._spec
        self.kind_combo.setCurrentText(s.kind if s.kind in _KINDS else _KINDS[0])
        self._rebuild_dim_fields(self.kind_combo.currentText())
        self.psc_box.setVisible(self.kind_combo.currentText() == "PSC girder")
        self.nstr_spin.setValue(int(s.n_strand))
        self.strand_area_spin.setValue(s.strand_area * 1e6)
        self.fpe_spin.setValue(s.f_pe / 1e6)
        self.strand_y_spin.setValue(s.strand_y * 1000.0)
        # cover + spiral (Rebars tab), confinement (M-φ sweep)
        self.cover_spin.setValue(s.cover * 1e3)
        self.cover_mode_combo.setCurrentText(
            "Variable" if s.cover_variable else "Uniform")
        self.cover_top_spin.setValue(s.cover_top * 1e3)
        self.cover_bot_spin.setValue(s.cover_bot * 1e3)
        self.cover_side_spin.setValue(s.cover_side * 1e3)
        self.spiral_chk.setChecked(bool(s.spiral))
        self.eps_c0_spin.setValue(s.eps_c0)
        self.eps_cu_spin.setValue(s.eps_cu)
        self.fcu_ratio_spin.setValue(s.fcu_ratio)
        self.kappa_max_spin.setValue(s.kappa_max)
        self._refresh_material_combos()
        self._refresh_tendon_table()
        self._refresh_groups_table()
        self._load_custom_tables()
        self._refresh_comp_list()
        self._apply_kind_visibility(self.kind_combo.currentText())
        self._loading = False

    def _spec_from_form(self) -> core.Spec:
        kind = self.kind_combo.currentText()
        ch: dict = {"kind": kind}
        for key, spin in self._dim_spins.items():
            ch[key] = spin.value() / 1000.0
        # reinforcement now comes entirely from the arrangement tables, so the
        # parametric bar counts are always zero (no section-level bars).
        ch.update(n_top=0, n_bot=0, n_side=0, n_perim=0)
        ch["cover"] = self.cover_spin.value() / 1e3
        # variable per-face cover (rectangular family only)
        faced = kind in ("Rectangular", "Hollow box")
        variable = faced and self.cover_mode_combo.currentText() == "Variable"
        ch["cover_variable"] = variable
        if variable:
            ch["cover_top"] = self.cover_top_spin.value() / 1e3
            ch["cover_bot"] = self.cover_bot_spin.value() / 1e3
            ch["cover_side"] = self.cover_side_spin.value() / 1e3
        ch["spiral"] = self.spiral_chk.isChecked()
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
    _PS_KEYS = ("Ep", "fpy", "ps_b")

    def _apply_material_params(self, spec: core.Spec) -> core.Spec:
        """Overlay constitutive laws onto ``spec``: the chosen section concrete,
        plus a *nominal* section steel and prestress derived from the
        reinforcement tables (the first rebar/tendon group's material, else a
        library default). M-φ / stress-field use each bar's own material; the
        P-M-M and verification paths use this nominal steel."""
        ch: dict = {}
        conc = self._materials.get(self.conc_mat_combo.currentText())
        if conc:
            ch.update({k: conc[k] for k in self._CONC_KEYS if k in conc})
        steel = self._nominal_group_steel(spec.rebar_groups)
        if steel:
            ch.update({k: steel[k] for k in self._STEEL_KEYS if k in steel})
        ps = self._nominal_material(spec.tendon_arr, 3, "prestress")
        if ps:
            ch.update({k: ps[k] for k in self._PS_KEYS if k in ps})
        return replace(spec, **ch) if ch else spec

    def _nominal_group_steel(self, groups):
        """The section's nominal steel: the first rebar group naming a library
        steel, else the first library steel (so the single-material P-M-M /
        verification paths always have a steel). M-φ / stress-field still honour
        each bar's own group material via ``_analysis_spec``."""
        for g in groups:
            name = g[3] if len(g) > 3 else None
            md = self._materials.get(name) if isinstance(name, str) else None
            if md and md.get("kind") == "steel":
                return md
        names = self._steel_names()
        return self._materials.get(names[0]) if names else None

    def _nominal_material(self, arrs, mat_idx, kind):
        """The section's nominal steel/prestress material: the first arrangement
        that names a library material of ``kind``, else the first such library
        material (so the P-M-M / verification paths always have a steel)."""
        for arr in arrs:
            name = arr[mat_idx] if mat_idx < len(arr) else None
            md = self._materials.get(name) if isinstance(name, str) else None
            if md and md.get("kind") == kind:
                return md
        names = self._names_of_kind(kind)
        return self._materials.get(names[0]) if names else None

    def _analysis_spec(self) -> core.Spec:
        """The working spec with each reinforcement group's / arrangement's
        chosen material name resolved to its steel properties (a sorted-items
        tuple), so the engine gives those bars their own law (mixed-material
        reinforcement). Bars with no chosen material keep the section steel. The
        stored spec keeps names (for the table + library sync); only this
        analysis copy embeds props."""
        def _resolve_arr(arr):
            mat = arr[2]
            md = self._materials.get(mat) if isinstance(mat, str) else None
            if md and md.get("kind") == "steel":
                return (arr[0], arr[1], tuple(sorted(md.items())), arr[3])
            return arr

        def _resolve_grp(g):
            pos = g[2] if len(g) > 2 else ""
            mat = g[3] if len(g) > 3 else ""
            md = self._materials.get(mat) if isinstance(mat, str) else None
            slot = (tuple(sorted(md.items()))
                    if md and md.get("kind") == "steel" else "")
            return (g[0], g[1], pos, slot)

        spec = self._spec
        if spec.rebar_arr:
            spec = replace(spec, rebar_arr=tuple(
                _resolve_arr(a) for a in spec.rebar_arr))
        if spec.rebar_groups:
            spec = replace(spec, rebar_groups=tuple(
                _resolve_grp(g) for g in spec.rebar_groups))
        return spec

    def _names_of_kind(self, kind) -> list:
        return [n for n, m in self._materials.items() if m.get("kind") == kind]

    def _concrete_names(self) -> list:
        return self._names_of_kind("concrete")

    def _steel_names(self) -> list:
        return self._names_of_kind("steel")

    def _refresh_material_combos(self) -> None:
        """Repopulate the section's concrete/steel/prestress pickers from the
        library, preserving the active section's stored choice."""
        was = self._loading
        self._loading = True
        rec = self._sections.get(self._active, {})
        self.conc_mat_combo.clear()
        self.conc_mat_combo.addItems(self._concrete_names())
        want = rec.get("conc_mat")
        i = self.conc_mat_combo.findText(want) if want else -1
        self.conc_mat_combo.setCurrentIndex(i if i >= 0 else 0)
        self._refresh_tendon_table()
        self._refresh_groups_table()
        self._loading = was

    def _on_material_choice(self) -> None:
        if self._loading:
            return
        rec = self._sections.get(self._active)
        if rec is not None:
            rec["conc_mat"] = self.conc_mat_combo.currentText() or None
        self._on_value_changed()

    # ------------------------------------------------ rebar standard / notation
    def _rebar_fam(self) -> str:
        """Material family of the reinforcement standard — an explicit
        ACI/EC2/IS choice, or (auto) the current design code's family."""
        rc = self.rebar_std_combo.currentData()
        if rc in ("ACI", "EC2", "IS"):
            return rc
        return _CODE_FAMILY.get(self.code_combo.currentText(), "EC2")

    def _rebar_notation(self) -> str:
        """Bar-size notation for the current rebar standard: US ``#``-sizes for
        ASTM/ACI, metric ⌀mm otherwise (passed to ``parse_bar_desc``)."""
        return "us" if self._rebar_fam() == "ACI" else "metric"

    def _rebar_examples(self) -> str:
        return ("4#8, #5-150 (n#8 / #5-s)" if self._rebar_notation() == "us"
                else "4B25, B16-200 (nBd / Bd-s)")

    def _default_pattern(self) -> str:
        return "4#8" if self._rebar_notation() == "us" else "3B20"

    def _on_rebar_std_changed(self) -> None:
        if self._loading:
            return
        self._refresh_groups_caption()
        self._groups_changed()

    def _on_code_changed(self) -> None:
        # design code drives the "Follow design code" notation, so refresh the
        # caption + re-validate the patterns when it changes too.
        self._refresh_groups_caption()
        if hasattr(self, "groups_tbl"):
            self._validate_groups(self._spec.rebar_groups)
        self._queue()

    def _refresh_groups_caption(self) -> None:
        if not hasattr(self, "groups_cap"):
            return
        us = self._rebar_notation() == "us"
        head = ("<b>n#N</b> = n bars #N · <b>#N-s</b> = #N at s mm spacing"
                if us else
                "<b>nBd</b> = n bars ⌀d mm · <b>Bd-s</b> = ⌀d mm at s mm spacing")
        tie = "#3-150" if us else "B10-150"
        self.groups_cap.setText(
            f"{head}.  <b>Link</b> = shear tie (Pattern gives hoop ⌀ &amp; "
            f"spacing, e.g. <code>{tie}</code>; put tie legs n_y×n_z in its "
            "Position, e.g. <code>2x2</code>, for Mander confinement). "
            "Top/Bottom/Sides/Perimeter take a blank Position; "
            "<b>Single</b> = z,y · <b>Line</b> = z1,y1; z2,y2 · "
            "<b>Arc</b> = cz,cy,r,a1,a2.")

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

    def _on_inter_view(self, i: int) -> None:
        """Switch the P-M-M interaction sub-view (P-M / 3-D / M-M) and recompute."""
        self.inter_stack.setCurrentIndex(i)
        self._queue()

    def _recompute_analysis(self) -> None:
        code = self.code_combo.currentText()
        try:
            case = _case(self._analysis_spec())   # per-arrangement materials
        except Exception as exc:                       # noqa: BLE001
            self.statusBar().showMessage(f"Build error: {exc}")
            return
        # the confinement echo (Section tab) is an input readout — refresh it
        # every recompute regardless of the active analysis tab.
        self._refresh_confinement_echo()
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
            if idx == 1:                    # P-M-M interaction (sub-view)
                sub = self.inter_view.currentIndex()
                if sub == 0:
                    self._draw_pm(case, code)
                elif sub == 1:
                    self._draw_surface(case, code)
                else:
                    self._draw_mm_contour(case, code)
            elif idx == 2:
                self._draw_mphi(case)
            elif idx == 3:
                self._fill_verify(case, code)
            elif idx == 4:
                self._draw_stress_field(case)
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
        confined = False
        if self._spec.kind == "Composite":
            data = core.composite_mphi(self._spec, P_kN, na_angle=ang,
                                       kappa_max=self._spec.kappa_max,
                                       materials=self._materials)
        else:
            conf, _info = _section_confinement(self._spec, self._materials)
            if conf is not None and self._spec.kind in ("Rectangular",
                                                        "Circular"):
                # Mander two-zone (confined core + unconfined cover) from the
                # section's Link tie group.
                data = core.confined_mphi(self._spec, conf, P_kN, na_angle=ang,
                                          kappa_max=self._spec.kappa_max)
                confined = True
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
        suffix = "  ·  Mander confined core" if confined else ""
        ax.set_title(f"Moment-curvature at P = {P:.4g} {u.Fl}{suffix}")
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

        # strain-profile milestones (real ones carry eps0/eps_top/eps_steel)
        self._mphi_data = data
        real = [ms for ms in data.get("milestones", []) if "eps0" in ms]
        self._strain_marks = real
        self._loading = True
        self.strain_combo.clear()
        self.strain_combo.addItems(
            [f"{ms.get('label', '')} · {ms.get('state', '')}" for ms in real])
        if real:
            self.strain_combo.setCurrentIndex(len(real) - 1)
        self._loading = False
        self._draw_strain_profile()

    def _draw_strain_profile(self) -> None:
        """Strain profile ε(y) = ε0 − y·κ at the selected M-φ milestone, with
        the rebar strains and the neutral-axis depth (mirrors the Streamlit)."""
        self.strain_fig.clear()
        ax = self.strain_fig.add_subplot(111)
        marks = getattr(self, "_strain_marks", [])
        data = getattr(self, "_mphi_data", None)
        i = self.strain_combo.currentIndex()
        if (not marks or data is None or "y_top" not in data
                or not (0 <= i < len(marks))):
            ax.set_axis_off()
            self.strain_canvas.draw_idle()
            self.strain_metrics.setText("—")
            return
        ms = marks[i]
        eps0, kap = ms["eps0"], ms["kappa"]
        y_top, y_bot = data["y_top"], data["y_bot"]
        # concrete strain profile (linear), y in mm
        ys = [y_bot, y_top]
        ax.plot([eps0 - y * kap for y in ys], [y * 1e3 for y in ys], "-",
                color=style.C_SECONDARY, marker="o", lw=1.8)
        # rebar strains
        rys = data.get("rebar_ys", [])
        if rys:
            ax.plot([eps0 - ry * kap for ry in rys], [ry * 1e3 for ry in rys],
                    "o", color="#1f3b6f", ms=5)
        ax.axvline(0, color=style.AX_SPINE, lw=0.8, ls="--")
        ax.set_xlabel("strain ε  (tension +)")
        ax.set_ylabel("y from centroid [mm]")
        style.beautify_axes(ax)
        self.strain_canvas.draw_idle()
        na = (y_top - eps0 / kap) if abs(kap) > 1e-9 else None
        na_txt = f"{na * 1e3:.0f} mm from top" if na is not None else "—"
        self.strain_metrics.setText(
            f"ε_c(top) <b>{ms.get('eps_top', 0):+.4f}</b> &nbsp;·&nbsp; "
            f"ε_s(max) <b>{ms.get('eps_steel', 0):+.4f}</b> &nbsp;·&nbsp; "
            f"NA {na_txt}")

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

    # ------------------------------------------------------ M-M contour
    def _draw_mm_contour(self, case, code) -> None:
        u = self._units
        self.mm_fig.clear()
        ax = self.mm_fig.add_subplot(111)
        if self._spec.kind == "Composite":
            ax.text(0.5, 0.5, "Composite section —\nsee the 3-D surface tab.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.mm_canvas.draw_idle()
            return
        P = self.mm_P.value()
        P_kN = P * u.fN / 1e3
        grid = core.pmm_surface_grid(case, code)
        Mz, My = core.mm_contour(grid, P_kN)
        # close the ring
        mz = [u.M_disp(v) for v in list(Mz) + [Mz[0]]]
        my = [u.M_disp(v) for v in list(My) + [My[0]]]
        ax.plot(mz, my, "-", color=style.C_PRIMARY, lw=1.8,
                label=f"Capacity @ P={P:.0f} {u.Fl}")
        ax.fill(mz, my, color=style.ACCENT_SOFT, alpha=0.5)
        dMz, dMy = self.mm_Mz.value(), self.mm_My.value()
        if dMz or dMy:
            ax.plot([dMz], [dMy], "o", color=style.C_DEMAND, ms=9,
                    label="Demand", zorder=5)
        ax.axhline(0, color=style.AX_SPINE, lw=0.6)
        ax.axvline(0, color=style.AX_SPINE, lw=0.6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(f"Mz  [{u.Ml}]")
        ax.set_ylabel(f"My  [{u.Ml}]")
        ax.set_title(f"M-M interaction @ P = {P:.4g} {u.Fl} — {code}")
        style.beautify_axes(ax)
        ax.legend(fontsize=8, loc="best", frameon=False)
        self.mm_canvas.draw_idle()

    # ------------------------------------------------------ stress field
    def _draw_stress_field(self, case) -> None:
        """Fibre-stress field under a plane-sections strain state (ε linear in
        y, tension +): concrete fibres + rebar coloured by the section's own
        material laws, with the neutral axis. Mirrors the Streamlit view."""
        self.sf_fig.clear()
        ax = self.sf_fig.add_subplot(111)
        s = self._spec
        if s.kind == "Composite":
            ax.text(0.5, 0.5, "Stress field is single-material —\n"
                    "not available for composite.", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            self.sf_canvas.draw_idle()
            return
        e_top = self.sf_etop.value() / 1000.0
        e_bot = self.sf_ebot.value() / 1000.0
        from shapely.geometry import Point
        conc = core.concrete_uniaxial_from(dict(
            fc=s.fc, conc_model=s.conc_model, eps_c0=s.eps_c0, eps_cu=s.eps_cu,
            fcu_ratio=s.fcu_ratio, fr_model=s.fr_model, fr_coeff=s.fr_coeff,
            eps_decay=s.eps_decay, conc_f1_ratio=s.conc_f1_ratio))
        steel = core.steel_uniaxial_from(dict(
            fy=s.fy, Es=s.Es, steel_model=s.steel_model, steel_b=s.steel_b,
            steel_fu_ratio=s.steel_fu_ratio, steel_eps_sh=s.steel_eps_sh,
            steel_eps_su=s.steel_eps_su))
        poly = case.section.geometry.polygon
        minz, miny, maxz, maxy = poly.bounds
        span = (maxy - miny) or 1.0

        def eps(y):
            return e_bot + (e_top - e_bot) * (y - miny) / span

        Z, Y, S = [], [], []
        for yy in np.linspace(miny, maxy, 48):
            sig = conc.get_response(float(eps(yy)))[0] / 1e6
            for zz in np.linspace(minz, maxz, 32):
                if poly.contains(Point(float(zz), float(yy))):
                    Z.append(zz * 1e3)
                    Y.append(yy * 1e3)
                    S.append(sig)
        bars = (case.section.reinforcement.bars
                if case.section.reinforcement else [])
        bsig = [steel.get_response(float(eps(b.y)))[0] / 1e6 for b in bars]
        # scale colours to the concrete field so its gradient stays readable;
        # steel (σ several× higher) saturates the ends of the same scale.
        clim = max([abs(v) for v in S] + [1.0])
        bsig = [max(-clim, min(clim, v)) for v in bsig]
        # section outline
        ex, ey = poly.exterior.xy
        ax.plot([z * 1e3 for z in ex], [y * 1e3 for y in ey], color="#5b7aa8",
                lw=1.5, zorder=1)
        sc = ax.scatter(Z, Y, c=S, cmap="RdBu_r", vmin=-clim, vmax=clim,
                        marker="s", s=16, linewidths=0, zorder=2)
        if bars:
            ax.scatter([b.z * 1e3 for b in bars], [b.y * 1e3 for b in bars],
                       c=bsig, cmap="RdBu_r", vmin=-clim, vmax=clim, s=60,
                       edgecolors="#222", linewidths=1, zorder=3)
        # neutral axis (eps = 0)
        if (e_top > 0) != (e_bot > 0) and e_top != e_bot:
            y0 = (miny + (0.0 - e_bot) * span / (e_top - e_bot)) * 1e3
            if miny * 1e3 <= y0 <= maxy * 1e3:
                ax.axhline(y0, color="#e5484d", lw=1.4, ls="--")
                ax.annotate("N.A.", (maxz * 1e3, y0), color="#e5484d",
                            fontsize=8, va="bottom", ha="right")
        self.sf_fig.colorbar(sc, ax=ax, label="σ  [MPa]", shrink=0.85)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title("Fibre-stress field (tension +)")
        style.beautify_axes(ax)
        self.sf_canvas.draw_idle()

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
        """Refresh one navigator row's summary + status + thumbnail (live)."""
        row = self._nav_rows.get(name)
        rec = self._sections.get(name)
        if row and rec:
            nl, sl, thumb = row[1]
            self._fill_nav_labels(nl, sl, name, rec["spec"])
            self._fill_nav_thumb(thumb, rec["spec"])

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
        self._sections[name] = {"spec": _rc_spec(kind="Rectangular", b=0.40, h=0.60),
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
        self._sections = {"Section 1": {"spec": _rc_spec(kind="Rectangular", b=0.40, h=0.60),
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
        dlg = MaterialsDialog(self, materials=self._materials,
                              code=self.code_combo.currentText())
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


class AddGroupDialog(QDialog):
    """Guided add-a-reinforcement-group dialog: the user picks Type, bar size,
    quantity (by count or spacing) and position from structured fields, and the
    dialog builds the AdSec Pattern string — so nobody needs to know the ``nBd``
    / ``Bd-s`` notation. ``Single`` shows a coordinate table (many bars at
    once), folding in the old 'Individual bars' path. Result is a list of
    ``(type, pattern, position, material)`` in ``result_groups``.

    Metric sizes are ⌀mm (pattern ``B25``); US sizes are #-numbers (``#8``)."""

    _MM_SIZES = ["8", "10", "12", "16", "20", "25", "32", "40"]
    _US_SIZES = ["3", "4", "5", "6", "7", "8", "9", "10", "11", "14", "18"]

    def __init__(self, parent=None, *, types=None, steels=None,
                 notation="metric"):
        super().__init__(parent)
        self.result_groups = []
        self._notation = notation
        self.setWindowTitle("Add reinforcement group")
        self.setMinimumWidth(380)
        root = QVBoxLayout(self)

        form = QFormLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(list(types or core.REBAR_GROUP_TYPES_RECT))
        self.type_combo.currentTextChanged.connect(lambda *_: self._rebuild())
        form.addRow("Type", self.type_combo)
        self.mat_combo = QComboBox()
        self.mat_combo.addItems(list(steels or []) or ["(section steel)"])
        form.addRow("Material", self.mat_combo)
        self.size_combo = QComboBox()
        self.size_combo.setEditable(True)
        if notation == "us":
            self.size_combo.addItems([f"#{s}" for s in self._US_SIZES])
            self.size_combo.setCurrentText("#8")
            form.addRow("Bar size (US #)", self.size_combo)
        else:
            self.size_combo.addItems(self._MM_SIZES)
            self.size_combo.setCurrentText("20")
            form.addRow("Bar ⌀ [mm]", self.size_combo)
        root.addLayout(form)

        self._dyn = QWidget()
        self._dyn_form = QFormLayout(self._dyn)
        self._dyn_form.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._dyn)

        self.preview = QLabel("")
        self.preview.setStyleSheet("color:#5a6b7b; font-size:11px;")
        root.addWidget(self.preview)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        self.size_combo.currentTextChanged.connect(lambda *_: self._update_preview())
        root.addWidget(bb)
        self._w = {}
        self._rebuild()

    # ---- little spin factories ----
    @staticmethod
    def _ispin(lo, hi, val):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    @staticmethod
    def _dspin(lo, hi, val, suffix=" mm"):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(0)
        s.setSingleStep(5)
        s.setSuffix(suffix)
        s.setValue(val)
        return s

    def _connect_preview(self, *widgets):
        for wdg in widgets:
            if isinstance(wdg, (QSpinBox, QDoubleSpinBox)):
                wdg.valueChanged.connect(lambda *_: self._update_preview())
            elif isinstance(wdg, QComboBox):
                wdg.currentTextChanged.connect(lambda *_: self._update_preview())

    def _rebuild(self) -> None:
        while self._dyn_form.rowCount():
            self._dyn_form.removeRow(0)
        self._w = {}
        typ = self.type_combo.currentText()
        if typ in ("Top", "Bottom", "Sides", "Perimeter"):
            mode = QComboBox()
            mode.addItems(["By count", "By spacing"])
            mode.currentTextChanged.connect(lambda *_: self._on_mode())
            self._w["mode"] = mode
            self._dyn_form.addRow("Quantity", mode)
            self._w["qty_label"] = QLabel("Number of bars")
            self._w["qty"] = self._ispin(1, 50, 3)
            self._dyn_form.addRow(self._w["qty_label"], self._w["qty"])
            self._connect_preview(mode, self._w["qty"])
        elif typ == "Line":
            self._w["n"] = self._ispin(1, 50, 4)
            self._dyn_form.addRow("Number of bars", self._w["n"])
            for k, lab, val in (("z1", "z₁ [mm]", -150),
                                ("y1", "y₁ [mm]", -250),
                                ("z2", "z₂ [mm]", 150),
                                ("y2", "y₂ [mm]", -250)):
                self._w[k] = self._dspin(-5000, 5000, val)
                self._dyn_form.addRow(lab, self._w[k])
            self._connect_preview(self._w["n"])
        elif typ == "Arc":
            self._w["n"] = self._ispin(1, 50, 6)
            self._dyn_form.addRow("Number of bars", self._w["n"])
            for k, lab, val, sfx in (("cz", "centre z [mm]", 0, " mm"),
                                     ("cy", "centre y [mm]", 0, " mm"),
                                     ("r", "radius [mm]", 200, " mm"),
                                     ("a1", "start angle", 0, "°"),
                                     ("a2", "end angle", 180, "°")):
                self._w[k] = self._dspin(-5000, 5000, val, sfx)
                self._dyn_form.addRow(lab, self._w[k])
            self._connect_preview(self._w["n"])
        elif typ == "Single":
            self._w["tbl"] = tbl = QTableWidget(0, 2)
            tbl.setHorizontalHeaderLabels(["z [mm]", "y [mm]"])
            tbl.horizontalHeader().setStretchLastSection(True)
            tbl.verticalHeader().setVisible(False)
            tbl.setMinimumHeight(140)
            self._dyn_form.addRow(tbl)
            self._single_add_row(0.0, 0.0)
            btns = QHBoxLayout()
            addb = QPushButton("＋ Bar")
            addb.clicked.connect(lambda: self._single_add_row(0.0, 0.0))
            remb = QPushButton("Remove")
            remb.clicked.connect(self._single_remove_row)
            btns.addWidget(addb)
            btns.addWidget(remb)
            btns.addStretch(1)
            holder = QWidget()
            holder.setLayout(btns)
            self._dyn_form.addRow(holder)
        elif typ == "Link":
            self._w["s"] = self._ispin(25, 600, 150)
            self._dyn_form.addRow("Tie spacing [mm]", self._w["s"])
            self._w["ny"] = self._ispin(2, 12, 2)
            self._dyn_form.addRow("Legs across (n_y)", self._w["ny"])
            self._w["nz"] = self._ispin(2, 12, 2)
            self._dyn_form.addRow("Legs along (n_z)", self._w["nz"])
            self._connect_preview(self._w["s"], self._w["ny"], self._w["nz"])
        self._update_preview()

    def _on_mode(self) -> None:
        spacing = self._w["mode"].currentText() == "By spacing"
        self._w["qty_label"].setText("Spacing [mm]" if spacing
                                     else "Number of bars")
        self._w["qty"].setRange(*((25, 600) if spacing else (1, 50)))
        self._w["qty"].setValue(200 if spacing else 3)
        self._update_preview()

    def _single_add_row(self, z, y) -> None:
        tbl = self._w["tbl"]
        r = tbl.rowCount()
        tbl.insertRow(r)
        tbl.setItem(r, 0, QTableWidgetItem(f"{z:g}"))
        tbl.setItem(r, 1, QTableWidgetItem(f"{y:g}"))

    def _single_remove_row(self) -> None:
        tbl = self._w["tbl"]
        r = tbl.currentRow()
        if r >= 0:
            tbl.removeRow(r)

    def _size_token(self) -> str:
        txt = self.size_combo.currentText().strip()
        if self._notation == "us":
            return "#" + txt.lstrip("#").strip()
        return "B" + txt.lstrip("Bb").strip()

    def _pattern(self, typ) -> str:
        tok = self._size_token()
        if typ in ("Top", "Bottom", "Sides", "Perimeter"):
            if self._w["mode"].currentText() == "By spacing":
                return f"{tok}-{self._w['qty'].value()}"
            return f"{self._w['qty'].value()}{tok}"
        if typ in ("Line", "Arc"):
            return f"{self._w['n'].value()}{tok}"
        if typ == "Link":
            return f"{tok}-{self._w['s'].value()}"
        return f"1{tok}"                       # Single

    def _update_preview(self) -> None:
        if not self._w:
            self.preview.setText("")
            return
        try:
            typ = self.type_combo.currentText()
            self.preview.setText(f"Pattern → <b>{self._pattern(typ)}</b>")
        except Exception:                              # noqa: BLE001
            self.preview.setText("")

    def _accept(self) -> None:
        typ = self.type_combo.currentText()
        name = self.mat_combo.currentText()
        mat = "" if name.startswith("(") else name
        pat = self._pattern(typ)
        groups = []
        if typ in ("Top", "Bottom", "Sides", "Perimeter"):
            groups.append((typ, pat, "", mat))
        elif typ == "Line":
            pos = (f"{self._w['z1'].value():g},{self._w['y1'].value():g}; "
                   f"{self._w['z2'].value():g},{self._w['y2'].value():g}")
            groups.append((typ, pat, pos, mat))
        elif typ == "Arc":
            pos = (f"{self._w['cz'].value():g},{self._w['cy'].value():g},"
                   f"{self._w['r'].value():g},{self._w['a1'].value():g},"
                   f"{self._w['a2'].value():g}")
            groups.append((typ, pat, pos, mat))
        elif typ == "Link":
            pos = f"{self._w['ny'].value()}x{self._w['nz'].value()}"
            groups.append((typ, pat, pos, mat))
        else:                                          # Single — many points
            tbl = self._w["tbl"]
            for r in range(tbl.rowCount()):
                zi, yi = tbl.item(r, 0), tbl.item(r, 1)
                try:
                    z = float((zi.text() if zi else "").strip())
                    y = float((yi.text() if yi else "").strip())
                except ValueError:
                    continue
                groups.append(("Single", pat, f"{z:g},{y:g}", mat))
            if not groups:
                QMessageBox.warning(self, "No bars",
                                    "Add at least one bar coordinate.")
                return
        try:
            core.parse_bar_desc(pat, notation=self._notation)
        except ValueError:
            QMessageBox.warning(
                self, "Invalid size",
                f"Could not build a valid bar pattern ('{pat}') from these "
                "inputs for this rebar standard.")
            return
        self.result_groups = groups
        self.accept()



def _conc_default(fc_mpa=30.0) -> dict:
    return dict(kind="concrete", fc=fc_mpa * 1e6, conc_model="Kent-Park",
                eps_c0=0.002, eps_cu=0.0035, fcu_ratio=0.4, fr_model="sqrt",
                fr_coeff=0.62, eps_decay=1e-3, conc_f1_ratio=0.4)


def _steel_default(fy_mpa=500.0) -> dict:
    return dict(kind="steel", fy=fy_mpa * 1e6, steel_model="Bilinear",
                Es=200e9, steel_b=0.01, steel_fu_ratio=1.5,
                steel_eps_sh=0.008, steel_eps_su=0.10)


# Section-Designer design code -> CODE_GRADES family (standard grade catalog).
_CODE_FAMILY = {"AASHTO LRFD 2024": "ACI", "Eurocode 2": "EC2",
                "IS 456:2000": "IS"}


def _rc_spec(**kw) -> core.Spec:
    """A Spec with the parametric bar counts zeroed — reinforcement comes from
    the AdSec groups table, not section-level counts."""
    return core.Spec(n_top=0, n_bot=0, n_side=0, n_perim=0, **kw)


def _section_presets() -> dict:
    """Named starter sections {label: fresh Spec} for the New menu. Bars are
    seeded as AdSec reinforcement groups (the groups table is the source)."""
    S = "S500"
    return {
        "Blank rectangular": _rc_spec(kind="Rectangular", b=0.40, h=0.60),
        "Rectangular RC beam": _rc_spec(
            kind="Rectangular", b=0.30, h=0.60, cover=0.04, rebar_groups=(
                ("Top", "2B20", "", S), ("Bottom", "3B20", "", S))),
        "Square RC column": _rc_spec(
            kind="Rectangular", b=0.40, h=0.40, cover=0.04,
            rebar_groups=(("Perimeter", "8B25", "", S),)),
        "Circular RC column": _rc_spec(
            kind="Circular", D=0.50, spiral=True, cover=0.04,
            rebar_groups=(("Perimeter", "8B25", "", S),)),
        # T-shape / PSC are not "faced" (only Rectangular / Hollow box are), so
        # their bars are placed as Line groups at explicit z,y positions [mm].
        "T-beam": _rc_spec(
            kind="T-shape", b=1.0, h=0.70, t_f=0.15, t_w=0.30, cover=0.04,
            rebar_groups=(("Line", "2B20", "-100,300; 100,300", S),
                          ("Line", "4B20", "-100,-300; 100,-300", S))),
        "PSC girder": _rc_spec(
            kind="PSC girder", b=0.50, h=1.20, n_strand=10, strand_area=140e-6,
            f_pe=1200e6, strand_y=-0.50, cover=0.04,
            rebar_groups=(("Line", "2B20", "-180,550; 180,550", S),)),
    }


def _parse_legs(txt):
    """Parse a tie group's leg count 'n_y×n_z' (accepts 2x3, 2×3, 2,3, or a
    single 3→3×3). Defaults to a perimeter hoop 2×2."""
    nums = re.findall(r"\d+", str(txt or ""))
    if len(nums) >= 2:
        return max(int(nums[0]), 1), max(int(nums[1]), 1)
    if len(nums) == 1:
        return max(int(nums[0]), 1), max(int(nums[0]), 1)
    return 2, 2


def _auto_section_confinement(spec, materials):
    """Read Mander confinement parameters straight off a Rectangular/Circular
    section: core dims from the geometry minus cover, the hoop A_sp/s/s'/f_yh/
    ε_su from the section's **Link** (tie) group and its steel grade, tie legs
    from that group's Position ('n_y×n_z'), and ρ_cc / n_long from the
    longitudinal bars. Returns ``(conf_dict, info)`` — or ``(None, reason)`` if
    the shape is unsupported or no usable tie group is defined."""
    if spec.kind not in ("Rectangular", "Circular"):
        return None, "shape"
    link = next((g for g in spec.rebar_groups if g and g[0] == "Link"), None)
    if link is None:
        return None, "no-link"
    try:
        _n, dia_h, s = core.parse_bar_desc(link[1] if len(link) > 1 else "")
    except Exception:                              # noqa: BLE001
        return None, "bad-pattern"
    if not s or s <= 0:
        return None, "no-spacing"           # a tie needs a spacing (e.g. B10-150)
    asp = core.bar_area(dia_h)
    sp = max(s - dia_h, 1e-3)               # clear spacing s' = s − hoop dia
    hoopmat = materials.get(link[3]) if len(link) > 3 else None
    fyh = float((hoopmat or {}).get("fy", 400e6))
    esu = float((hoopmat or {}).get("steel_eps_su", 0.10))
    es_h = float((hoopmat or {}).get("Es", 200e9))
    ny, nz = _parse_legs(link[2] if len(link) > 2 else "")
    cover = float(spec.cover)
    try:
        sec = core.build_case(spec).section
        bars = sec.reinforcement.bars if sec.reinforcement else []
    except Exception:                              # noqa: BLE001
        bars = []
    nlong = len(bars)
    if nlong < 4:            # no real longitudinal cage → confinement undefined
        return None, "no-cage"
    as_long = sum(b.area for b in bars)

    conf = dict(conf_shape=spec.kind, conf_fyh=fyh, conf_Asp=asp, conf_s=s,
                conf_sp=sp, conf_eps_su_h=esu, conf_Es_h=es_h,
                conf_ny=ny, conf_nz=nz, conf_nlong=nlong)
    if spec.kind == "Circular":
        ds = max(spec.D - 2.0 * cover, 1e-3)
        a_core = np.pi / 4.0 * ds * ds
        conf["conf_ds"] = ds
        conf["conf_hooptype"] = "Spiral" if spec.spiral else "Hoop"
    else:
        bc = max(spec.b - 2.0 * cover, 1e-3)
        dc = max(spec.h - 2.0 * cover, 1e-3)
        a_core = bc * dc
        conf["conf_bc"], conf["conf_dc"] = bc, dc
    conf["conf_rho_cc"] = (min(max(as_long / a_core, 0.005), 0.08)
                           if a_core > 0 else 0.02)
    info = dict(dia_h=dia_h, s=s, sp=sp, fyh=fyh, esu=esu, ny=ny, nz=nz,
                nlong=nlong, cover=cover, as_long=as_long, a_core=a_core,
                hoop_grade=(link[3] if len(link) > 3 else None))
    return conf, info


def _section_confinement(spec, materials):
    """Effective Mander confinement for a section: user-override inputs
    (``spec.conf_manual`` when ``spec.conf_override``) else auto-read from the
    tie group + geometry. Per-output overrides (``spec.conf_out``) and the ε_cu
    method layer on top as ``ov_*`` / ``conf_ecu_method`` keys, so every
    override flows straight into the moment-curvature. Returns ``(conf, info)``."""
    if getattr(spec, "conf_override", False) and spec.conf_manual:
        conf, info = {k: v for (k, v) in spec.conf_manual}, {"override": True}
    else:
        conf, info = _auto_section_confinement(spec, materials)
    if conf is not None:
        ov = {f"ov_{k}": v for (k, v) in getattr(spec, "conf_out", ())}
        ov["conf_ecu_method"] = getattr(spec, "conf_ecu_method", "energy")
        conf = dict(conf, **ov)
    return conf, info


def _material_from_grade(fam: str, kind: str, grade: str) -> dict:
    """Build a library material dict from a standard code grade
    (``core.CODE_GRADES``): concrete carries the family's model."""
    g = core.CODE_GRADES[fam]
    if kind == "concrete":
        md = _conc_default(g["concrete"][grade] / 1e6)
        md["conc_model"] = g.get("conc_model", md["conc_model"])
        return md
    return _steel_default(g["steel"][grade] / 1e6)


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

    def __init__(self, parent=None, *, materials: dict | None = None,
                 code: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Materials library")
        self.resize(660, 560)
        self._family0 = _CODE_FAMILY.get(code or "", "ACI")
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
        # standard code-grade catalog (ACI / EC2 / IS)
        grow = QHBoxLayout()
        grow.addWidget(QLabel("Std:"))
        self.grade_fam = QComboBox()
        self.grade_fam.addItems(list(core.CODE_GRADES))
        self.grade_fam.setCurrentText(self._family0)
        self.grade_fam.currentTextChanged.connect(self._reload_grades)
        grow.addWidget(self.grade_fam)
        self.grade_combo = QComboBox()
        grow.addWidget(self.grade_combo, 1)
        gadd = QPushButton("Add grade")
        gadd.clicked.connect(self._add_grade)
        grow.addWidget(gadd)
        left.addLayout(grow)
        self._reload_grades()
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
        self._add_named(f"{prefix}{self._strength(md):.0f}", md)

    def _add_named(self, base: str, md: dict) -> None:
        name, i = base, 2
        while name in self._names:
            name = f"{base} ({i})"
            i += 1
        self._names.append(name)
        self._mats.append(md)
        self._reload_list()
        self.listw.setCurrentRow(len(self._names) - 1)

    # ---- standard code-grade catalog ----
    def _reload_grades(self, *_a) -> None:
        fam = self.grade_fam.currentText()
        g = core.CODE_GRADES.get(fam, {})
        self.grade_combo.clear()
        for kind in ("concrete", "steel"):
            for grade in g.get(kind, {}):
                self.grade_combo.addItem(f"{kind[0].upper()}· {grade}",
                                         (kind, grade))

    def _add_grade(self) -> None:
        data = self.grade_combo.currentData()
        if not data:
            return
        kind, grade = data
        md = _material_from_grade(self.grade_fam.currentText(), kind, grade)
        self._add_named(grade, md)

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
