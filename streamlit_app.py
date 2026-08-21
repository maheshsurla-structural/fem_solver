"""Section Designer -- interactive GUI (Streamlit).

Build an arbitrary RC / PSC section, see its properties, biaxial P-M-M
interaction (AASHTO LRFD 2024, Eurocode 2, IS 456) and moment-curvature
live, then verify against Midas GSD in the same screen and export the
standard files.

Run:  streamlit run streamlit_app.py

Runs the real femsolver engine (see ``section_gui_core``), so results
match the CLI / benchmark suite exactly.
"""
from __future__ import annotations

import base64
import copy
import datetime as _dt
import os
import re
import sys
from dataclasses import astuple, replace
from pathlib import Path

# femsolver is a src-layout package that is not pip-installed; put it on
# the path so `streamlit run streamlit_app.py` works from a clean checkout.
_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import section_gui_core as core
from femsolver.benchmarks.section_designer import (
    VerificationItem,
    compare_items,
    to_markdown,
)

st.set_page_config(
    page_title="Section Designer",
    page_icon=":material/architecture:",
    layout="wide",
    initial_sidebar_state="expanded",
)

_LOGO = (
    '<svg width="34" height="34" viewBox="0 0 34 34" fill="none">'
    '<rect x="4" y="4" width="26" height="26" rx="3" fill="none" '
    'stroke="#fff" stroke-width="2"/>'
    '<circle cx="10" cy="10" r="2" fill="#fff"/>'
    '<circle cx="24" cy="10" r="2" fill="#fff"/>'
    '<circle cx="10" cy="24" r="2" fill="#fff"/>'
    '<circle cx="24" cy="24" r="2" fill="#fff"/>'
    '<circle cx="17" cy="17" r="2.4" fill="#fbbf24"/></svg>')

_APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --sd-accent:#2563eb; --sd-teal:#0e97ab; --sd-steel:#e08a00;
  --sd-comp:#1d4ed8; --sd-tens:#e5484d; --sd-good:#12a150;
  --sd-line:rgba(120,142,176,.30); --sd-line-2:rgba(120,142,176,.16);
  --sd-panel:rgba(128,150,184,.07); --sd-panel-2:rgba(128,150,184,.13);
  --sd-sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --sd-mono:"IBM Plex Mono","SF Mono",ui-monospace,SFMono-Regular,monospace;
}

/* ---- base type: technical pairing (IBM Plex Sans + Mono) ---- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
.stMarkdown, .stMarkdown p, button, input, select, textarea, label{
  font-family:var(--sd-sans);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .app-title,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{ font-family:var(--sd-sans) !important; }
.block-container{ padding-top:1.5rem; padding-bottom:4rem; max-width:1440px; }
[data-testid="stHeader"]{ background:transparent; }

/* ---- toolbar (app header) ---- */
.app-header{ display:flex; align-items:center; gap:16px;
  background:linear-gradient(100deg,#12234d,#1e3a8a 44%,#0e7d94);
  color:#fff; padding:11px 24px; padding-right:180px; border-radius:0;
  margin:0; box-shadow:0 4px 16px rgba(20,40,90,.22); }
/* Full-width fixed application title bar (AdSec-style): spans the entire
   viewport width across the very top — over the sidebar and the content both —
   and stays put on every page. Fixed is applied to the header's
   ELEMENT-CONTAINER wrapper (st.markdown's own wrapper), which sits directly
   under the page's vertical block with no transformed ancestor, so `fixed`
   resolves against the viewport. z-index sits just BELOW Streamlit's ~60-px top
   toolbar (z 999990) so Deploy / ⋮ stay on top and clickable; the header's
   right padding keeps the context chips clear of those buttons. */
[data-testid="stElementContainer"]:has(> .stMarkdown .app-header){
  position:fixed; top:0; left:0; right:0; width:100%;
  z-index:999989; margin:0; padding:0; }
/* push the main content below the fixed banner (Streamlit's own top toolbar
   stays at the very top so Deploy / ⋮ remain reachable). */
[data-testid="stMain"]{ padding-top:72px; }
/* The sidebar paints ABOVE the banner (higher z-index), so it can't simply be
   padded — its background would still cover the banner's left edge. Offset the
   whole sidebar DOWN by the banner height so it sits cleanly beneath the bar,
   letting the title bar run unbroken across the full width. */
[data-testid="stSidebar"]{ margin-top:66px; }
/* Keep the sidebar permanently docked: remove the collapse (hide) control and
   the drag-to-resize handle, and reclaim the now-empty header strip so the nav
   sits just under the title bar. */
[data-testid="stSidebarCollapseButton"]{ display:none !important; }
[data-testid="stSidebarResizeHandle"]{ display:none !important; }
[data-testid="stSidebarHeader"]{ display:none; }
[data-testid="stSidebarUserContent"]{ padding-top:1.1rem; }
.app-header .logo{ background:rgba(255,255,255,.15); border-radius:9px;
  padding:6px 6px 3px; display:flex; flex:none; }
.app-title{ font-size:1.3rem; font-weight:700; line-height:1.05; letter-spacing:-.3px; }
.app-sub{ font-family:var(--sd-mono); font-size:.75rem; opacity:.92; margin-top:3px; }
.app-sub b{ font-weight:600; }
.hdr-right{ margin-left:auto; display:flex; align-items:center; gap:8px; }
.hchip{ display:inline-flex; align-items:center; gap:7px; height:30px; padding:0 12px;
  border-radius:8px; background:rgba(255,255,255,.13);
  border:1px solid rgba(255,255,255,.20); font-size:.78rem; font-weight:600;
  white-space:nowrap; }
.hchip .k{ opacity:.72; font-weight:500; text-transform:uppercase;
  letter-spacing:.5px; font-size:.64rem; }
.hchip.units{ font-family:var(--sd-mono); }
@media (max-width:900px){ .hchip.tag{ display:none; } }

/* ---- sidebar as the left navigator/inputs panel ---- */
[data-testid="stSidebar"]{ border-right:1px solid var(--sd-line); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3{
  text-transform:uppercase; letter-spacing:.6px; font-size:.72rem;
  color:var(--sd-accent); font-weight:700; }
[data-testid="stSidebar"] [data-testid="stExpander"]{
  border:1px solid var(--sd-line); border-radius:10px; overflow:hidden;
  background:var(--sd-panel); }
[data-testid="stSidebar"] [data-testid="stExpander"] summary{ font-weight:600; }

/* ---- metric / property cards ---- */
[data-testid="stMetric"]{ background:var(--sd-panel); border:1px solid var(--sd-line);
  padding:12px 14px; border-radius:11px; }
[data-testid="stMetricValue"]{ font-weight:700; font-family:var(--sd-mono); }

/* ---- tabs (canvas tab strip) ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"]{ gap:2px;
  border-bottom:1px solid var(--sd-line); }
[data-testid="stTabs"] [data-baseweb="tab"]{ font-weight:600; }
[data-testid="stTabs"] button[aria-selected="true"]{ color:var(--sd-accent); }
[data-testid="stTabs"] button[aria-selected="true"] p{ font-weight:700; }

/* ---- data grids / tables ---- */
[data-testid="stDataFrame"], [data-testid="stDataEditor"]{
  border:1px solid var(--sd-line); border-radius:10px; }
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataEditor"] [role="columnheader"]{
  text-transform:uppercase; letter-spacing:.4px; font-size:.68rem; font-weight:700; }

/* ---- buttons ---- */
.stButton button, .stDownloadButton button{ border-radius:8px; font-weight:600; }
.stButton button[kind="primary"],
.stDownloadButton button[kind="primary"]{ font-weight:700; }

/* ---- code / inline numbers ---- */
code, kbd{ font-family:var(--sd-mono); }

/* ---- status bar ---- */
.sd-status{ display:flex; align-items:center; gap:18px; flex-wrap:wrap;
  margin:10px 0 0; padding:8px 15px; border-radius:10px;
  border:1px solid var(--sd-line); background:var(--sd-panel);
  font-family:var(--sd-mono); font-size:.73rem; opacity:.92; }
.sd-status .s{ display:inline-flex; align-items:center; gap:7px; }
.sd-status b{ font-weight:600; }
.sd-status .sp{ flex:1; }
.sd-led{ width:8px; height:8px; border-radius:50%; background:var(--sd-good);
  box-shadow:0 0 0 3px rgba(18,161,80,.22); flex:none; }

/* ---- section list (AdSec-style gallery) ---- */
.sd-thumb{ width:100%; height:150px; display:flex; align-items:center;
  justify-content:center; overflow:hidden; border:1px solid var(--sd-line-2);
  border-radius:10px; background:var(--sd-panel); }
.sd-thumb svg{ max-width:100%; max-height:100%; width:auto; height:auto; }
.sd-listhdr{ font-size:1.15rem; font-weight:700; color:var(--sd-accent);
  margin:2px 0 2px; letter-spacing:-.2px; }
.sd-addlbl{ text-transform:uppercase; letter-spacing:.6px; font-size:.68rem;
  font-weight:700; opacity:.7; margin:2px 0 6px; }
.sd-colhdr{ text-transform:uppercase; letter-spacing:.4px; font-size:.66rem;
  font-weight:700; opacity:.65; }
.sd-secname{ font-size:1.05rem; font-weight:700; color:var(--sd-accent);
  line-height:1.1; }
.sd-seckind{ font-size:.82rem; opacity:.75; margin-top:2px; }
.sd-rowmeta{ font-family:var(--sd-mono); font-size:.74rem; opacity:.85; }
</style>
"""


def _inject_css():
    st.markdown(_APP_CSS, unsafe_allow_html=True)


def _header(spec, code, u):
    active = st.session_state.active_section
    units = f"{u.force} · {u.length} · {u.stress}"
    st.markdown(
        f'<div class="app-header"><div class="logo">{_LOGO}</div>'
        f'<div><div class="app-title">Section Designer</div>'
        f'<div class="app-sub">Model / <b>{active}</b> &middot; {spec.kind} '
        f'&middot; {code}</div></div>'
        '<div class="hdr-right">'
        f'<span class="hchip"><span class="k">Code</span>{code}</span>'
        f'<span class="hchip units">{units}</span>'
        '<span class="hchip tag">Biaxial P&ndash;M&ndash;M &middot; '
        'M&ndash;&phi; &middot; GSD verification</span>'
        '</div></div>',
        unsafe_allow_html=True)


def _statusbar(spec, case, code, u):
    """CAD-style status strip below the tabs: engine, code, model counts."""
    n_bar = len(case.section.reinforcement.bars) \
        if case.section.reinforcement else 0
    n_ten = len(case.section.prestress.tendons) \
        if case.section.prestress else 0
    st.markdown(
        '<div class="sd-status">'
        '<span class="s"><span class="sd-led"></span> <b>femsolver</b> engine '
        '&middot; fibre-based</span>'
        f'<span class="s">Code&nbsp;<b>{code}</b></span>'
        f'<span class="s">{spec.kind}</span>'
        f'<span class="s">{n_bar} bars &middot; {n_ten} tendons</span>'
        '<span class="sp"></span>'
        f'<span class="s">Units&nbsp;<b>{u.force}&middot;{u.length}&middot;'
        f'{u.stress}</b></span>'
        '</div>', unsafe_allow_html=True)


_inject_css()


# ------------------------------------------------------------ cached compute

@st.cache_data(show_spinner=False)
def compute_landmarks(spec_tuple, code):
    """Strong-axis interaction landmarks (P_o, P_n,max, M_n@P=0, balanced,
    ...) for the section's code -- list of (name, value, kind) or None.
    Landmarks are single-material code-block quantities -> None for composite."""
    spec = core.Spec(*spec_tuple)
    if spec.kind == "Composite":
        return None
    _curve, lm = core.pmm_slice(core.build_case(spec), code)
    return lm


@st.cache_data(show_spinner="Computing moment-curvature …")
def compute_mphi(spec_tuple, p_kn, na_angle=0.0, mats_frozen=()):
    spec = core.Spec(*spec_tuple)
    if spec.kind == "Composite":
        return core.composite_mphi(spec, p_kn, na_angle=na_angle,
                                   kappa_max=spec.kappa_max,
                                   materials=_mats_from_frozen(mats_frozen))
    return core.mphi_data(core.build_case(spec), p_kn,
                          na_angle=na_angle, **core.mphi_props(spec))


@st.cache_data(show_spinner="Building 3-D surface …")
def compute_surface3d(spec_tuple, code, na, npl, mats_frozen=()):
    spec = core.Spec(*spec_tuple)
    if spec.kind == "Composite":
        return core.composite_pmm_mesh(spec, na, npl,
                                       materials=_mats_from_frozen(mats_frozen))
    return core.pmm_surface_mesh(core.build_case(spec), code, na, npl)


@st.cache_data(show_spinner="Checking load combinations …")
def compute_demand(spec_tuple, code, demands_tuple, design, na, npl,
                   method="P", mats_frozen=()):
    spec = core.Spec(*spec_tuple)
    demands = [{"name": n, "P": p, "Mz": mz, "My": my}
               for (n, p, mz, my) in demands_tuple]
    return core.demand_check(core.build_case(spec), code, demands,
                             design=design, na=na, nd=npl, method=method,
                             spec=spec, materials=_mats_from_frozen(mats_frozen))


@st.cache_data(show_spinner="Building verification set …")
def compute_items(spec_tuple, code):
    spec = core.Spec(*spec_tuple)
    if spec.kind == "Composite":
        return []                         # single-material verification n/a
    return core.items_data(core.build_case(spec), code,
                           core.mphi_props(spec))


# --------------------------------------------------- calculate / results gate

MESH_GRID = {"Coarse": (16, 17), "GSD default": (24, 29), "Fine": (36, 41)}


def _dsurf():
    return st.session_state.get("dsurf") or "Design (φ)"


DC_METHODS = {"Keep P constant": "P", "Keep M constant": "M",
              "Keep M/P constant": "MP"}


def _dc_method():
    return DC_METHODS.get(st.session_state.get("dc_method_lbl"), "P")


def _mats_frozen():
    """Hashable snapshot of the materials library (for caching / signatures)."""
    mats = st.session_state.get("materials", {})
    return tuple(sorted((str(n), tuple(sorted(m.items())))
                        for n, m in mats.items()))


def _mats_from_frozen(frozen):
    return {n: dict(kv) for n, kv in frozen}


def _calc_sig():
    """Everything the analysis depends on, for the active section — a change
    marks results stale until the user re-clicks Calculate. Display-only
    settings (unit system, angle/axial view, shaded toggle) are excluded. The
    materials library enters the signature only for composite sections (whose
    fibre analysis resolves per-shape / per-bar materials from it)."""
    active = st.session_state.active_section
    rec = st.session_state.sections[active]
    demands = tuple(st.session_state.get("demands", {}).get(active, []))
    mats = _mats_frozen() if rec["spec"].kind == "Composite" else ()
    return (astuple(rec["spec"]), rec["code"], demands,
            st.session_state.get("mesh_density", "GSD default"), _dsurf(),
            _dc_method(), mats)


def _results_ready():
    active = st.session_state.active_section
    return st.session_state.get("calc_sigs", {}).get(active) == _calc_sig()


def _calc_gate():
    """Show the 'press Calculate' prompt and return False when the active
    section's results are stale; return True when they're up to date."""
    if _results_ready():
        return True
    st.info("⚙ Set the load combinations in the **Analysis** controls above, "
            "then click **Calculate results** to compute this tab.")
    return False


# ------------------------------------------------------------ sidebar

AUTOSAVE = Path(os.environ.get("SECTION_AUTOSAVE")
                or (Path(__file__).parent / ".section_autosave.json"))


def _default_project():
    st.session_state.counter = 1
    st.session_state.sections = {
        "Section 1": {"spec": core.Spec(), "code": "AASHTO LRFD 2024"}}
    st.session_state.active_section = "Section 1"


def _init_sections():
    if "sections" not in st.session_state:
        restored = False
        if AUTOSAVE.exists():                # restore last session from disk
            try:
                secs, active, dmnds, mats = core.project_from_json(
                    AUTOSAVE.read_text(encoding="utf-8"))
                st.session_state.sections = secs
                st.session_state.active_section = active
                st.session_state.counter = len(secs)
                st.session_state["demands"] = dmnds
                st.session_state["materials"] = mats
                restored = True
            except Exception:                # corrupt file -> fresh start
                pass
        if not restored:
            _default_project()
    # active_section is a widget key owned by the Sections-page section picker;
    # on the Materials page that widget isn't rendered, so Streamlit drops the
    # key between reruns. Re-anchor it to a valid section on every call so both
    # pages can rely on it (and the Materials early-return never crashes).
    if st.session_state.get("active_section") not in st.session_state.sections:
        st.session_state["active_section"] = next(iter(st.session_state.sections))


def _autosave():
    """Persist the whole library to disk whenever it changes, so sections
    survive a browser reload or a server restart."""
    try:
        text = core.project_to_json(st.session_state.sections,
                                    st.session_state.active_section,
                                    st.session_state.get("demands", {}),
                                    st.session_state.get("materials", {}))
        if st.session_state.get("_autosave_last") != text:
            AUTOSAVE.write_text(text, encoding="utf-8")
            st.session_state["_autosave_last"] = text
    except Exception:
        pass                                 # never let autosave break the app


def _new_project():
    _default_project()
    st.session_state.section_view = "list"


# ---- material spec-field / widget-key tables ----
# Spec fields for a concrete / steel material, and the widget-key suffixes to
# drop so the inputs re-initialise when a code change re-baselines the grade.
_CONC_KEYS = ("fc", "conc_model", "eps_c0", "eps_cu", "fcu_ratio", "fr_model",
              "fr_coeff", "eps_decay", "conc_f1_ratio")
_STEEL_KEYS = ("fy", "Es", "steel_model", "steel_b", "steel_fu_ratio",
               "steel_eps_sh", "steel_eps_su")
# Design-code -> material family (drives the grade-name lookup in the shared
# material library).
_CODE_FAMILY = {"AASHTO LRFD 2024": "ACI", "Eurocode 2": "EC2",
                "IS 456:2000": "IS"}


def _usig():
    return (f"{st.session_state.cfg_force}_{st.session_state.cfg_length}_"
            f"{st.session_state.cfg_stress}")


def _init_config():
    """Project-level Setup config: global units + design code (AdSec-style).
    Held in plain session keys (not widget keys) so the chosen values survive
    on pages where the Setup widgets aren't rendered."""
    ss = st.session_state
    ss.setdefault("cfg_force", "kN")
    ss.setdefault("cfg_length", "m")
    ss.setdefault("cfg_stress", "MPa")
    if ss.get("cfg_code") not in core.CODES:
        rec = ss.sections.get(ss.get("active_section"))
        seed = rec.get("code") if rec else None
        ss["cfg_code"] = seed if seed in core.CODES else core.CODES[0]


def _units():
    ss = st.session_state
    return core.Units(ss.cfg_force, ss.cfg_length, ss.cfg_stress)


# ==================== shared material library (live-link model) =============
# Sections reference named materials from st.session_state["materials"]; those
# properties are resolved into the section spec each run, so editing a material
# updates every section that uses it.
_CONC_DEFAULT = dict(kind="concrete", fc=30e6, conc_model="Kent-Park",
                     eps_c0=0.002, eps_cu=0.0035, fcu_ratio=0.4,
                     fr_model="sqrt", fr_coeff=0.62, eps_decay=1e-3,
                     conc_f1_ratio=0.4)
_STEEL_DEFAULT = dict(kind="steel", fy=500e6, Es=200e9, steel_model="Bilinear",
                      steel_b=0.01, steel_fu_ratio=1.5, steel_eps_sh=0.008,
                      steel_eps_su=0.10)


def _mat_close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= max(1e-9, abs(b) * 1e-6)
    return a == b


def _grade_name(kind, props, code):
    """Friendly library name: match a code grade (C30/37, B500B…) by strength,
    else a strength label."""
    fam = _CODE_FAMILY.get(code, "ACI")
    tbl = core.CODE_GRADES[fam]["concrete" if kind == "concrete" else "steel"]
    val = props["fc"] if kind == "concrete" else props["fy"]
    for gn, gv in tbl.items():
        if abs(gv - val) <= 1e3:
            return gn
    return f"{'Concrete' if kind == 'concrete' else 'Steel'} {val / 1e6:.0f} MPa"


def _find_or_add_material(mats, props, code):
    """Return the name of a library material equal to ``props`` (deduped), else
    create one (uniquely named) and return its name."""
    kind = props["kind"]
    keys = _CONC_KEYS if kind == "concrete" else _STEEL_KEYS
    for n, m in mats.items():
        if m.get("kind") == kind and all(_mat_close(m.get(k), props.get(k))
                                         for k in keys):
            return n
    name = base = _grade_name(kind, props, code)
    i = 2
    while name in mats:
        name = f"{base} ({i})"
        i += 1
    mats[name] = dict(props)
    return name


def _ensure_section_materials(rec):
    """Ensure a section references a valid library concrete + steel material,
    migrating its inline spec material into the library on first sight (no value
    change -- it just becomes a named, shared material)."""
    mats = st.session_state.setdefault("materials", {})
    spec = rec["spec"]
    concs = {n for n, m in mats.items() if m.get("kind") == "concrete"}
    steels = {n for n, m in mats.items() if m.get("kind") == "steel"}
    if rec.get("conc_mat") not in concs:
        props = {"kind": "concrete", **{k: getattr(spec, k) for k in _CONC_KEYS}}
        rec["conc_mat"] = _find_or_add_material(mats, props, rec.get("code", ""))
    if rec.get("steel_mat") not in steels:
        props = {"kind": "steel", **{k: getattr(spec, k) for k in _STEEL_KEYS}}
        rec["steel_mat"] = _find_or_add_material(mats, props, rec.get("code", ""))


def _resolve_materials(rec):
    """Copy referenced library materials' properties into the section spec --
    the live link: edit the material, the section updates."""
    mats = st.session_state.get("materials", {})
    upd = {}
    cm = mats.get(rec.get("conc_mat"))
    if cm and cm.get("kind") == "concrete":
        upd.update({k: cm[k] for k in _CONC_KEYS if k in cm})
    sm = mats.get(rec.get("steel_mat"))
    if sm and sm.get("kind") == "steel":
        upd.update({k: sm[k] for k in _STEEL_KEYS if k in sm})
    if upd:
        rec["spec"] = replace(rec["spec"], **upd)


def _resolve_all_materials():
    """Ensure + resolve materials for every section, once per run."""
    for rec in st.session_state.get("sections", {}).values():
        _ensure_section_materials(rec)
        _resolve_materials(rec)


def _material_in_use(name):
    return any(rec.get("conc_mat") == name or rec.get("steel_mat") == name
               for rec in st.session_state.get("sections", {}).values())


def _rename_material_refs(old, new):
    for rec in st.session_state.get("sections", {}).values():
        if rec.get("conc_mat") == old:
            rec["conc_mat"] = new
        if rec.get("steel_mat") == old:
            rec["steel_mat"] = new


def _concrete_props_editor(mat, u, kp):
    """Edit a library concrete material dict in place."""
    def K(f):
        return f"{kp}_{f}"

    def _cl(v, lo, hi):
        return float(min(hi, max(lo, v)))

    m1 = st.columns(2)
    mat["fc"] = u.to_Pa(m1[0].number_input(
        f"f'c / f_ck [{u.Sl}]", float(u.from_Pa(15e6)), float(u.from_Pa(90e6)),
        float(u.from_Pa(min(90e6, max(15e6, mat.get("fc", 30e6))))),
        key=K("fc")))
    mat["conc_model"] = m1[1].selectbox(
        "Model", core.CONC_MODELS,
        index=(core.CONC_MODELS.index(mat["conc_model"])
               if mat.get("conc_model") in core.CONC_MODELS else 0),
        key=K("cmodel"))
    if mat["conc_model"] == "Trilinear":
        mat["conc_f1_ratio"] = st.number_input(
            "First-knee σ₁/f'c", 0.1, 0.9,
            _cl(mat.get("conc_f1_ratio", 0.4), 0.1, 0.9), 0.05, format="%.2f",
            key=K("f1r"))
    cc = st.columns(2)
    mat["eps_c0"] = cc[0].number_input(
        "Peak strain ε_c0", 0.0010, 0.0050,
        _cl(mat.get("eps_c0", 0.002), 0.001, 0.005), 0.0001, format="%.4f",
        key=K("epsc0"))
    mat["eps_cu"] = cc[1].number_input(
        "Crush strain ε_cu", 0.0025, 0.0200,
        _cl(mat.get("eps_cu", 0.0035), 0.0025, 0.02), 0.0005, format="%.4f",
        key=K("epscu"))
    mat["fcu_ratio"] = cc[0].number_input(
        "Residual f_cu/f'c", 0.0, 1.0, _cl(mat.get("fcu_ratio", 0.4), 0.0, 1.0),
        0.05, format="%.2f", key=K("fcur"))
    _fr = {"k·√f'c": "sqrt", "EC2 f_ctm(f_ck)": "ec2"}
    _lbl = cc[1].selectbox("Rupture", list(_fr),
                           index=1 if mat.get("fr_model") == "ec2" else 0,
                           key=K("frm"))
    mat["fr_model"] = _fr[_lbl]
    if mat["fr_model"] == "sqrt":
        mat["fr_coeff"] = st.number_input(
            "Rupture k  (f_r = k·√f'c)", 0.0, 1.5,
            _cl(mat.get("fr_coeff", 0.62), 0.0, 1.5), 0.02, format="%.2f",
            key=K("frc"))
    mat["eps_decay"] = st.number_input(
        "Tension-stiffening ε_decay", 0.0001, 0.0200,
        _cl(mat.get("eps_decay", 1e-3), 0.0001, 0.02), 0.0001, format="%.4f",
        key=K("epsd"))


def _steel_props_editor(mat, u, kp):
    """Edit a library steel material dict in place."""
    def K(f):
        return f"{kp}_{f}"

    def _cl(v, lo, hi):
        return float(min(hi, max(lo, v)))

    s1 = st.columns(2)
    mat["fy"] = u.to_Pa(s1[0].number_input(
        f"f_y / f_yk [{u.Sl}]", float(u.from_Pa(250e6)),
        float(u.from_Pa(700e6)),
        float(u.from_Pa(min(700e6, max(250e6, mat.get("fy", 500e6))))),
        key=K("fy")))
    mat["steel_model"] = s1[1].selectbox(
        "Model", core.STEEL_MODELS,
        index=(core.STEEL_MODELS.index(mat["steel_model"])
               if mat.get("steel_model") in core.STEEL_MODELS else 0),
        key=K("smodel"))
    sc = st.columns(2)
    mat["Es"] = sc[0].number_input(
        "E_s [GPa]", 150.0, 230.0, _cl(mat.get("Es", 200e9) / 1e9, 150.0, 230.0),
        5.0, format="%.0f", key=K("es")) * 1e9
    if mat["steel_model"] == "Park strain-hardening":
        mat["steel_fu_ratio"] = sc[1].number_input(
            "f_su/f_y", 1.0, 2.0, _cl(mat.get("steel_fu_ratio", 1.5), 1.0, 2.0),
            0.05, format="%.2f", key=K("fu"))
        pc = st.columns(2)
        mat["steel_eps_sh"] = pc[0].number_input(
            "ε_sh", 0.002, 0.05, _cl(mat.get("steel_eps_sh", 0.008), 0.002,
                                     0.05), 0.001, format="%.3f", key=K("esh"))
        mat["steel_eps_su"] = pc[1].number_input(
            "ε_su", 0.02, 0.20, _cl(mat.get("steel_eps_su", 0.10), 0.02, 0.20),
            0.005, format="%.3f", key=K("esu"))
    elif mat["steel_model"] != "Elastic - perfectly plastic":
        mat["steel_b"] = sc[1].number_input(
            "Hardening b", 0.0, 0.1, _cl(mat.get("steel_b", 0.01), 0.0, 0.1),
            0.005, format="%.3f", key=K("sb"))


def _material_assign(rec, u):
    """The section's Materials panel: assign shared library materials (an
    assignment, not a property editor). Properties resolve from the library."""
    active = st.session_state.active_section
    mats = st.session_state.setdefault("materials", {})
    _ensure_section_materials(rec)
    concs = [n for n, m in mats.items() if m.get("kind") == "concrete"]
    steels = [n for n, m in mats.items() if m.get("kind") == "steel"]
    cm, sm = rec.get("conc_mat"), rec.get("steel_mat")
    rec["conc_mat"] = st.selectbox(
        "Concrete material", concs,
        index=concs.index(cm) if cm in concs else 0, key=f"asg_c_{active}")
    rec["steel_mat"] = st.selectbox(
        "Steel material", steels,
        index=steels.index(sm) if sm in steels else 0, key=f"asg_s_{active}")
    _resolve_materials(rec)
    c = mats.get(rec["conc_mat"], {})
    s = mats.get(rec["steel_mat"], {})
    st.caption(
        f"→ concrete **{c.get('fc', 0) / 1e6:.0f} MPa** · "
        f"{c.get('conc_model', '')} · steel **{s.get('fy', 0) / 1e6:.0f} MPa** "
        f"· E_s {s.get('Es', 0) / 1e9:.0f} GPa")
    st.caption("Edit these in the **🧱 Materials** page — changes apply to "
               "every section using them.")


def _add_material(mats, props):
    """Create a new library material (no dedup), returning its unique name."""
    name = base = _grade_name(props["kind"], props, "")
    i = 2
    while name in mats:
        name = f"{base} ({i})"
        i += 1
    mats[name] = dict(props)
    return name


def _material_stress_strain_chart(mat, u):
    """AdSec-style stress-strain diagram for a library material."""
    kind = mat.get("kind", "concrete")
    if kind == "concrete":
        law = core.concrete_uniaxial_from(mat)
        ecu = float(mat.get("eps_cu", 0.0035))
        eps = np.linspace(-ecu * 1.05, 0.0015, 240)
    else:
        law = core.steel_uniaxial_from(mat)
        esu = (float(mat.get("steel_eps_su", 0.05))
               if mat.get("steel_model") == "Park strain-hardening" else 0.02)
        eps = np.linspace(-esu, esu, 240)
    sig = [law.get_response(float(e))[0] / 1e6 for e in eps]
    fig = go.Figure(go.Scatter(
        x=eps * 1e3, y=sig, mode="lines",
        line=dict(color="#2563eb", width=2.5),
        hovertemplate="ε %{x:.2f}‰ · σ %{y:.1f} MPa<extra></extra>"))
    fig.update_xaxes(title_text="Strain [‰]", zeroline=True,
                     zerolinecolor="rgba(120,140,170,.55)")
    fig.update_yaxes(title_text="Stress [MPa]", zeroline=True,
                     zerolinecolor="rgba(120,140,170,.55)")
    fig.update_layout(height=380, margin=dict(l=6, r=6, t=10, b=6),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig


def _apply_project_code():
    """Setup page: apply the chosen design code project-wide (every section)."""
    code = st.session_state.get("w_cfg_code")
    if code not in core.CODES:
        return
    st.session_state["cfg_code"] = code
    for rec in st.session_state.sections.values():
        rec["code"] = code
    _autosave()


def tab_setup():
    """Project Setup (AdSec-style): global display units + design code, shared
    by every section and the Materials library."""
    ss = st.session_state
    st.subheader("Units")
    st.caption("Display units, applied across every section, the Materials "
               "library and the reports.")
    c1, c2, c3 = st.columns(3)
    ss["cfg_force"] = c1.selectbox(
        "Force", list(core.FORCE_N),
        index=list(core.FORCE_N).index(ss["cfg_force"]), key="w_cfg_force")
    ss["cfg_length"] = c2.selectbox(
        "Length", list(core.LENGTH_M),
        index=list(core.LENGTH_M).index(ss["cfg_length"]), key="w_cfg_length")
    ss["cfg_stress"] = c3.selectbox(
        "Stress", list(core.STRESS_PA),
        index=list(core.STRESS_PA).index(ss["cfg_stress"]), key="w_cfg_stress")

    st.divider()
    st.subheader("Design code")
    st.caption("Project-wide — drives the P-M-M stress block, the "
               "load-combination style, the standard material grades on the "
               "🧱 Materials page and the demand / verification checks for "
               "every section.")
    st.selectbox(
        "Design code", core.CODES,
        index=core.CODES.index(ss["cfg_code"]), key="w_cfg_code",
        on_change=_apply_project_code, label_visibility="collapsed")


def tab_materials(u):
    """Project-level Materials library (AdSec-style): grade lists on the left,
    the selected material's properties + stress-strain curve on the right."""
    mats = st.session_state.setdefault("materials", {})
    st.subheader("Materials library")
    st.caption("Shared materials referenced by sections — edit one here and "
               "every section using it updates. Assign a section's materials "
               "in its **Definition** tab (Materials panel).")
    concs = [n for n, m in mats.items() if m.get("kind") == "concrete"]
    steels = [n for n, m in mats.items() if m.get("kind") == "steel"]
    sel = st.session_state.get("mat_edit_sel")
    if sel not in mats:
        sel = (concs + steels)[0] if (concs or steels) else None

    lib, props, chart = st.columns([1, 1.15, 1.4], gap="large")
    with lib:
        st.markdown("**Concrete grades**")
        for n in concs:
            if st.button(f"{n} · {mats[n].get('fc', 0) / 1e6:.0f} MPa",
                         key=f"mtsel_{n}", width="stretch",
                         type="primary" if n == sel else "secondary"):
                st.session_state["mat_edit_sel"] = n
                st.rerun()
        if st.button("＋ New concrete", key="mtnew_c", width="stretch"):
            st.session_state["mat_edit_sel"] = _add_material(
                mats, dict(_CONC_DEFAULT))
            st.rerun()
        st.markdown("**Steel grades**")
        for n in steels:
            if st.button(f"{n} · {mats[n].get('fy', 0) / 1e6:.0f} MPa",
                         key=f"mtsel_{n}", width="stretch",
                         type="primary" if n == sel else "secondary"):
                st.session_state["mat_edit_sel"] = n
                st.rerun()
        if st.button("＋ New steel", key="mtnew_s", width="stretch"):
            st.session_state["mat_edit_sel"] = _add_material(
                mats, dict(_STEEL_DEFAULT))
            st.rerun()

    if not sel:
        props.info("Add a material to begin.")
        return
    mat = mats[sel]
    with props:
        tp = st.columns([3, 1])
        newname = tp[0].text_input("Name", value=sel, key=f"mtname_{sel}")
        used = _material_in_use(sel)
        if tp[1].button("🗑", key=f"mtdel_{sel}", help="Delete material",
                        disabled=used):
            del mats[sel]
            st.session_state.pop("mat_edit_sel", None)
            st.rerun()
        if used:
            st.caption("↪ In use by a section — reassign before deleting.")
        if newname and newname != sel and newname not in mats:
            mats[newname] = mats.pop(sel)
            _rename_material_refs(sel, newname)
            st.session_state["mat_edit_sel"] = newname
            st.rerun()
        st.divider()
        if mat.get("kind") == "concrete":
            _concrete_props_editor(mat, u, f"mt_{sel}")
        else:
            _steel_props_editor(mat, u, f"mt_{sel}")
    with chart:
        st.markdown("**Stress–strain**")
        st.plotly_chart(_material_stress_strain_chart(mat, u),
                        width="stretch", key=f"mtchart_{sel}")


_ARR_CODE = {"Point": "point", "Line": "line", "Arc": "arc",
             "Rectangle": "rect", "Perimeter": "perim"}


def _rebar_arr_label(arr, u):
    typ, dia, p = arr[0], arr[1], arr[3]
    d = f"⌀{dia * 1e3:.0f}"
    if typ == "point":
        return f"Point {d} @ ({u.from_m(p[0]):.3g}, {u.from_m(p[1]):.3g})"
    if typ == "line":
        return f"Line {int(p[0])}·{d}"
    if typ == "arc":
        return f"Arc {int(p[0])}·{d} (r={u.from_m(p[3]):.3g})"
    if typ == "rect":
        return f"Rect {d} ({int(p[0])}×{int(p[1])})"
    return f"Perimeter {int(p[0])}·{d}"


def _rebar_arrangements(rec, u):
    """Add-form + list of rebar arrangements (Phase-2 layout generators)."""
    active = st.session_state.active_section
    spec = rec["spec"]
    st.markdown("**Rebar arrangements**")
    for i, arr in enumerate(list(spec.rebar_arr)):
        c = st.columns([5, 1])
        c[0].caption("• " + _rebar_arr_label(arr, u)
                     + (f" · {arr[2]}" if arr[2] else ""))
        if c[1].button("✕", key=f"delra_{active}_{i}", help="Remove"):
            arrs = list(spec.rebar_arr)
            del arrs[i]
            rec["spec"] = replace(spec, rebar_arr=tuple(arrs))
            st.rerun()

    L = u.Ll

    def k(f):
        return f"ra_{f}_{active}_{u.length}"

    def N(col, lbl, key, dflt_m):
        return u.to_m(col.number_input(f"{lbl} [{L}]", value=float(
            u.from_m(dflt_m)), step=u.len_step(), format="%.4g", key=k(key)))

    typ = st.selectbox("Add", core.REBAR_ARR_TYPES, key=k("type"))
    dia = core.BAR_SIZES[st.selectbox(
        "Bar size", list(core.BAR_SIZES),
        index=list(core.BAR_SIZES).index("25 mm"), key=k("dia"))]
    cc = st.columns(2)
    if typ == "Point":
        params = (N(cc[0], "z", "z", 0.0), N(cc[1], "y", "y", 0.0))
    elif typ == "Line":
        n = int(st.number_input("Bars n", 1, 100, 3, 1, key=k("n")))
        params = (n, N(cc[0], "z1", "z1", -0.15), N(cc[1], "y1", "y1", -0.25),
                  N(cc[0], "z2", "z2", 0.15), N(cc[1], "y2", "y2", -0.25))
    elif typ == "Arc":
        n = int(st.number_input("Bars n", 1, 100, 4, 1, key=k("n")))
        cz, cy = N(cc[0], "cz", "cz", 0.0), N(cc[1], "cy", "cy", 0.0)
        r = N(st, "radius", "r", 0.20)
        a = st.columns(2)
        a1 = a[0].number_input("start °", value=0.0, step=15.0, key=k("a1"))
        a2 = a[1].number_input("end °", value=360.0, step=15.0, key=k("a2"))
        params = (n, cz, cy, r, a1, a2)
    elif typ == "Rectangle":
        nz = int(cc[0].number_input("n / horiz.", 1, 40, 3, 1, key=k("nz")))
        ny = int(cc[1].number_input("n / vert.", 1, 40, 3, 1, key=k("ny")))
        params = (nz, ny, N(cc[0], "cz", "cz", 0.0), N(cc[1], "cy", "cy", 0.0),
                  N(cc[0], "width", "w", 0.30), N(cc[1], "height", "h", 0.50))
    else:                                        # Perimeter
        n = int(cc[0].number_input("Bars n", 3, 100, 8, 1, key=k("n")))
        params = (n, N(cc[1], "cover", "cov", 0.05))
    steels = ["(section steel)"] + [
        n for n, m in st.session_state.get("materials", {}).items()
        if m.get("kind") == "steel"]
    mat = st.selectbox("Material", steels, key=k("mat"),
                       help="Steel material for this arrangement's bars. "
                            "Honoured in the fibre analysis for Composite "
                            "sections; single-material sections use the "
                            "section steel.")
    if st.button("＋ Add arrangement", key=k("add"), width="stretch"):
        matname = "" if mat == "(section steel)" else mat
        new = (_ARR_CODE[typ], dia, matname, tuple(params))
        rec["spec"] = replace(spec, rebar_arr=spec.rebar_arr + (new,))
        st.rerun()


def _tendon_arrangements(rec, u):
    """Add-form + list of tendon arrangements (Point/Line/Arc)."""
    active = st.session_state.active_section
    spec = rec["spec"]
    st.markdown("**Tendon arrangements**")
    for i, arr in enumerate(list(spec.tendon_arr)):
        typ, area, f_pe, _mat, p = arr
        c = st.columns([5, 1])
        c[0].caption(f"• {typ.title()} Aₚ={area * 1e6:.0f}mm² "
                     f"f_pe={u.from_Pa(f_pe):.0f}{u.Sl} "
                     f"({int(p[0]) if typ != 'point' else 1} tendon"
                     f"{'s' if typ != 'point' else ''})")
        if c[1].button("✕", key=f"delta_{active}_{i}", help="Remove"):
            arrs = list(spec.tendon_arr)
            del arrs[i]
            rec["spec"] = replace(spec, tendon_arr=tuple(arrs))
            st.rerun()

    L = u.Ll

    def k(f):
        return f"ta_{f}_{active}_{u.length}"

    def N(col, lbl, key, dflt_m):
        return u.to_m(col.number_input(f"{lbl} [{L}]", value=float(
            u.from_m(dflt_m)), step=u.len_step(), format="%.4g", key=k(key)))

    typ = st.selectbox("Add", core.TENDON_ARR_TYPES, key=k("type"))
    mm = st.columns(2)
    area = mm[0].number_input("Aₚ / tendon [mm²]", 50.0, 5000.0, 140.0, 10.0,
                              key=k("area")) * 1e-6
    f_pe = u.to_Pa(mm[1].number_input(f"f_pe [{u.Sl}]", float(u.from_Pa(500e6)),
                                      float(u.from_Pa(1600e6)),
                                      float(u.from_Pa(1100e6)), key=k("fpe")))
    cc = st.columns(2)
    if typ == "Point":
        params = (N(cc[0], "z", "z", 0.0), N(cc[1], "y", "y", -0.35))
    elif typ == "Line":
        n = int(st.number_input("Tendons n", 1, 60, 4, 1, key=k("n")))
        params = (n, N(cc[0], "z1", "z1", -0.2), N(cc[1], "y1", "y1", -0.35),
                  N(cc[0], "z2", "z2", 0.2), N(cc[1], "y2", "y2", -0.35))
    else:                                        # Arc
        n = int(st.number_input("Tendons n", 1, 60, 4, 1, key=k("n")))
        cz, cy = N(cc[0], "cz", "cz", 0.0), N(cc[1], "cy", "cy", 0.0)
        r = N(st, "radius", "r", 0.30)
        a = st.columns(2)
        a1 = a[0].number_input("start °", value=200.0, step=15.0, key=k("a1"))
        a2 = a[1].number_input("end °", value=340.0, step=15.0, key=k("a2"))
        params = (n, cz, cy, r, a1, a2)
    if st.button("＋ Add tendon group", key=k("add"), width="stretch"):
        new = (_ARR_CODE[typ], area, f_pe, "", tuple(params))
        rec["spec"] = replace(spec, tendon_arr=spec.tendon_arr + (new,))
        st.rerun()


def _new_name():
    st.session_state.counter += 1
    nm = f"Section {st.session_state.counter}"
    while nm in st.session_state.sections:
        st.session_state.counter += 1
        nm = f"Section {st.session_state.counter}"
    return nm


def _load_project():
    f = st.session_state.get("proj_upload")
    if f is None:
        return
    try:
        sections, active, dmnds, mats = core.project_from_json(
            f.getvalue().decode("utf-8"))
        st.session_state.sections = sections
        st.session_state.active_section = active
        st.session_state.counter = len(sections)
        st.session_state["demands"] = dmnds
        st.session_state["materials"] = mats
        st.session_state.section_view = "list"
    except Exception as exc:                       # noqa: BLE001
        st.session_state["proj_error"] = str(exc)


def _unique_name(base):
    if base not in st.session_state.sections:
        return base
    i = 2
    while f"{base} ({i})" in st.session_state.sections:
        i += 1
    return f"{base} ({i})"


def _import_append():
    f = st.session_state.get("proj_import")
    if f is None:
        return
    try:
        imported, _, imp_demands, imp_mats = core.project_from_json(
            f.getvalue().decode("utf-8"))
        st.session_state.setdefault("materials", {}).update(imp_mats)
        dstore = st.session_state.setdefault("demands", {})
        first = None
        for name, rec in imported.items():
            nm = _unique_name(name)          # rename on collision
            st.session_state.sections[nm] = rec
            if name in imp_demands:          # carry combos to the new name
                dstore[nm] = imp_demands[name]
            first = first or nm
        if first:
            st.session_state.active_section = first
        st.session_state.counter = len(st.session_state.sections)
        st.session_state.section_view = "list"
    except Exception as exc:                       # noqa: BLE001
        st.session_state["proj_error"] = str(exc)


# ---- section presets (AdSec-style "Add new…") ----
# Each preset is a ready-to-edit Spec; "Empty section" is a plain concrete
# rectangle (no bars) the user then reinforces.
SECTION_PRESETS = {
    "Empty section": lambda: core.Spec(kind="Rectangular", b=0.40, h=0.60,
                                       n_top=0, n_bot=0, n_side=0),
    "Rectangular RC beam": lambda: core.Spec(
        kind="Rectangular", b=0.30, h=0.60, n_top=2, n_bot=3, n_side=0,
        bar_dia=0.020, cover=0.04),
    "Square RC column": lambda: core.Spec(
        kind="Rectangular", b=0.40, h=0.40, n_top=3, n_bot=3, n_side=1,
        bar_dia=0.025, cover=0.04),
    "Circular RC column": lambda: core.Spec(
        kind="Circular", D=0.50, n_perim=8, bar_dia=0.025, cover=0.04,
        spiral=True),
}


# Button/text callbacks run before the next rerun's widgets; they own
# ``active_section`` (the currently-open section) and ``section_view``
# ("list" gallery vs "detail" definition/analysis).
def _add_preset(preset):
    nm = _new_name()
    st.session_state.sections[nm] = {
        "spec": SECTION_PRESETS[preset](),
        "code": st.session_state.get("cfg_code", "AASHTO LRFD 2024")}
    st.session_state.active_section = nm
    st.session_state.section_view = "detail"


def _open_section(name):
    st.session_state.active_section = name
    st.session_state.section_view = "detail"


def _back_to_list():
    st.session_state.section_view = "list"


def _dup_section(name=None):
    src = name or st.session_state.active_section
    nm = _new_name()
    st.session_state.sections[nm] = copy.deepcopy(
        st.session_state.sections[src])
    st.session_state.active_section = nm


def _del_section(name=None):
    tgt = name or st.session_state.active_section
    if len(st.session_state.sections) > 1 and tgt in st.session_state.sections:
        del st.session_state.sections[tgt]
        st.session_state.active_section = list(st.session_state.sections)[0]


def _rename_section(old):
    new = (st.session_state.get(f"rename_{old}") or "").strip()
    if new and new != old and new not in st.session_state.sections:
        st.session_state.sections = {
            (new if k == old else k): v
            for k, v in st.session_state.sections.items()}
        st.session_state.active_section = new


@st.cache_data(show_spinner=False)
def _thumb_svg(spec_tuple):
    """Small, responsive SVG sketch of a section for the list gallery. Strips
    the fixed width/height so it scales inside its thumbnail box; returns "" if
    the section can't be built yet (e.g. an empty composite/custom)."""
    try:
        svg = core.svg_of(core.build_case(core.Spec(*spec_tuple)))
    except Exception:
        return ""
    # drop the fixed width/height on the opening <svg> tag (keep viewBox) so it
    # scales to fill the thumbnail box.
    svg = re.sub(r"<svg\b[^>]*>",
                 lambda m: re.sub(r'\s(?:width|height)="[^"]*"', "", m.group(0)),
                 svg, count=1)
    return svg


def _prep():
    """Per-run initialisation shared by every page. Returns (units, code)."""
    _init_sections()
    _init_config()
    _resolve_all_materials()      # pull shared-library material props into specs
    st.session_state.setdefault("app_mode", "Sections")
    st.session_state.setdefault("section_view", "list")
    return _units(), st.session_state["cfg_code"]


def _sidebar_nav_and_project(u):
    """The sidebar: primary navigation (Sections | Setup | Materials) and
    project-level I/O only. Per-section inputs live inline on the pages, so a
    user never touches the sidebar to define or analyse a section."""
    st.sidebar.radio(
        "Navigate", ["Setup", "Materials", "Sections"], key="app_mode",
        label_visibility="collapsed",
        format_func=lambda m: {"Setup": "⚙ Setup", "Sections": "📐 Sections",
                               "Materials": "🧱 Materials"}[m])
    st.sidebar.divider()
    st.sidebar.subheader("Project")
    active = st.session_state.active_section
    pcol = st.sidebar.columns(2)
    pcol[0].download_button(
        "Save", core.project_to_json(st.session_state.sections, active,
                                     st.session_state.get("demands", {}),
                                     st.session_state.get("materials", {})),
        "section_project.json", "application/json", width="stretch")
    pcol[1].button("New project", width="stretch", on_click=_new_project)
    st.sidebar.file_uploader("Load project — replaces all (.json)",
                             type=["json"], key="proj_upload",
                             on_change=_load_project)
    st.sidebar.file_uploader("Import & append sections (.json)",
                             type=["json"], key="proj_import",
                             on_change=_import_append)
    err = st.session_state.pop("proj_error", None)
    if err:
        st.sidebar.error(f"Load / import failed: {err}")
    st.sidebar.caption("✓ Autosaved — your sections restore automatically "
                       "on restart.")


def _dimension_inputs(rec, u, code):
    """Shape + parametric dimensions (and PSC strand parameters) for the
    Definition tab's Dimensions panel. Writes the active section's spec.
    Custom / Composite geometry is drawn in their own panels; materials and
    reinforcement have their own panels too."""
    active = st.session_state.active_section
    usig = _usig()
    s0 = rec["spec"]

    # widget keys namespaced by section + unit system, so switching either
    # re-initialises every input from the stored (SI) spec, converted.
    def K(field):
        return f"{field}__{active}__{usig}"

    Ls = u.len_step()

    def LI(col, label, field, default_m, lo, hi):
        dm = min(hi, max(lo, default_m))     # clamp to the widget range
        v = col.number_input(f"{label} [{u.Ll}]", float(u.from_m(lo)),
                             float(u.from_m(hi)), float(u.from_m(dm)), Ls,
                             key=K(field))
        return u.to_m(v)

    d = {}
    _shape_lbl = {k: k for k in core.KINDS}
    _shape_lbl["Custom"] = "General (polygon)"
    kind = st.selectbox(
        "Shape", core.KINDS, index=core.KINDS.index(s0.kind),
        format_func=lambda k: _shape_lbl[k], key=K("kind"),
        help="Basic parametric shapes, or General = an arbitrary polygon "
             "drawn in the Geometry panel.")
    d["kind"] = kind
    c1, c2 = st.columns(2)
    if kind in ("Rectangular", "PSC girder", "Hollow box"):
        d["b"] = LI(c1, "Width b", "b", s0.b, 0.1, 3.0)
        d["h"] = LI(c2, "Depth h", "h", s0.h, 0.1, 4.0)
        if kind == "Hollow box":
            d["wall_t"] = LI(st, "Wall t", "wall_t", s0.wall_t, 0.04, 0.5)
    elif kind == "Circular":
        d["D"] = LI(c1, "Diameter D", "D", s0.D, 0.2, 3.0)
    elif kind == "L-shape":
        d["leg"] = LI(c1, "Leg", "leg", s0.leg, 0.3, 3.0)
        d["thick"] = LI(c2, "Thickness", "thick", s0.thick, 0.1, 1.5)
    elif kind == "T-shape":
        d["h"] = LI(c1, "Depth h", "h", s0.h, 0.2, 3.0)
        d["b"] = LI(c2, "Flange b", "b", s0.b, 0.2, 3.0)
        d["t_f"] = LI(c1, "Flange t_f", "t_f", s0.t_f, 0.08, 0.8)
        d["t_w"] = LI(c2, "Web t_w", "t_w", s0.t_w, 0.08, 0.8)
    elif kind == "Custom":
        st.caption("✏️ Draw the outline in the **Geometry** panel "
                   "(coordinate editor).")
    elif kind == "Composite":
        st.caption("🧩 Add material shapes in the **Shapes** panel. "
                   "Analysis is fibre-based (multi-material).")

    if kind == "PSC girder":
        st.divider()
        st.markdown("**Prestressing strands**")
        p1, p2 = st.columns(2)
        d["n_strand"] = int(p1.number_input("Strands", 1, 40, s0.n_strand, 1,
                                            key=K("nstr")))
        d["strand_area"] = p2.number_input(
            "Aₚ / strand [mm²]", 50.0, 400.0,
            min(400.0, max(50.0, s0.strand_area * 1e6)), 5.0,
            key=K("sa")) * 1e-6
        d["f_pe"] = u.to_Pa(p1.number_input(
            f"f_pe [{u.Sl}]", float(u.from_Pa(500e6)),
            float(u.from_Pa(1600e6)),
            float(u.from_Pa(min(1600e6, max(500e6, s0.f_pe)))), key=K("fpe")))
        half_m = d.get("h", s0.h) / 2
        d["strand_y"] = LI(p2, "Strand y", "sy", s0.strand_y,
                           -half_m + 0.02, half_m - 0.02)

    full = {f: getattr(s0, f) for f in core.Spec.__dataclass_fields__}
    full.update({k: v for k, v in d.items()
                 if k in core.Spec.__dataclass_fields__})
    rec["spec"] = core.Spec(**full)
    _autosave()


def _analysis_inputs(rec, u, code):
    """Loads & analysis controls (mesh density, demand-check method, factored
    load combinations, Calculate) for the Analysis tab — inline, so the
    sidebar stays clear. Writes the section's demands + calc signature."""
    active = st.session_state.active_section
    usig = _usig()
    st.session_state.setdefault("mesh_density", "GSD default")
    c1, c2, c3 = st.columns(3)
    c1.select_slider(
        "Mesh density (P-M-M surface)", list(MESH_GRID), key="mesh_density",
        help="Number of interaction ribs × rings the surface is computed on.")
    c2.radio("Demand check against", ["Design (φ)", "Nominal"], key="dsurf",
             horizontal=True)
    c3.radio(
        "Checking ratio (D/C method)", list(DC_METHODS), key="dc_method_lbl",
        help="How the demand is scaled onto the capacity surface. Keep P "
             "constant (default) = scale moment at fixed axial (flexure). "
             "Keep M constant = scale axial at fixed moment. Keep M/P "
             "constant = radial (scale both).")

    st.markdown("**Load combinations** — factored P, Mz, My")
    pk, mk = u.P_disp(1.0), u.M_disp(1.0)   # display per canonical kN / kN·m
    dstore = st.session_state.setdefault("demands", {})
    dcanon = dstore.get(active, [])
    drows = ([{"Combo": n, f"P [{u.Fl}]": p * pk, f"Mz [{u.Ml}]": mz * mk,
               f"My [{u.Ml}]": my * mk} for (n, p, mz, my) in dcanon]
             or [{"Combo": "", f"P [{u.Fl}]": None, f"Mz [{u.Ml}]": None,
                  f"My [{u.Ml}]": None}])
    ded = st.data_editor(
        pd.DataFrame(drows), num_rows="dynamic", hide_index=True,
        width="stretch", key=f"demand_{active}_{usig}", column_config={
            "Combo": st.column_config.TextColumn("Combo", width="small"),
            f"P [{u.Fl}]": st.column_config.NumberColumn(
                format="%.4g", help="+ compression"),
            f"Mz [{u.Ml}]": st.column_config.NumberColumn(format="%.4g"),
            f"My [{u.Ml}]": st.column_config.NumberColumn(format="%.4g")})

    def _dnum(x):
        return 0.0 if pd.isna(x) else float(x)

    ncanon = []
    for _, r in ded.iterrows():
        P_, Mz_, My_ = r[f"P [{u.Fl}]"], r[f"Mz [{u.Ml}]"], r[f"My [{u.Ml}]"]
        nm = r["Combo"]
        if pd.isna(P_) and pd.isna(Mz_) and pd.isna(My_):
            continue
        ncanon.append(("" if pd.isna(nm) else str(nm),
                       _dnum(P_) / pk, _dnum(Mz_) / mk, _dnum(My_) / mk))
    if ncanon != dcanon:
        dstore[active] = ncanon

    bcol = st.columns([1, 2])
    if bcol[0].button("⚙ Calculate results", type="primary", width="stretch"):
        st.session_state.setdefault("calc_sigs", {})[active] = _calc_sig()
    with bcol[1]:
        if _results_ready():
            st.success("✓ Results up to date")
        else:
            st.warning("Inputs changed — click **Calculate** to update the "
                       "results below.")


# ------------------------------------------------------------ tabs

def _rebar_guard(case) -> bool:
    """True if the section has bars/tendons; else show a prompt and return
    False so the analysis tab can bail out gracefully (custom sections start
    with no reinforcement)."""
    sec = case.section
    has = bool((sec.reinforcement and sec.reinforcement.bars)
               or (getattr(sec, "prestress", None) and sec.prestress.tendons))
    if not has:
        st.info("➕ This section has no reinforcement yet. Add bars in the "
                "**Definition** tab (Reinforcement panel) to compute "
                "interaction, moment-curvature and verification.")
    return has


def _ok_to_analyze(spec) -> bool:
    """Composite sections are valid once they have a shape; others need bars."""
    if spec.kind == "Composite":
        if not spec.shapes:
            st.info("➕ This composite section has no shapes yet — add material "
                    "shapes in the **Definition** tab (Shapes panel).")
            return False
        return True
    return _rebar_guard(core.build_case(spec))


def _custom_editor(u):
    """Coordinate editor for a 'Custom' section: editable vertex + bar
    tables, a fill-perimeter helper, and a live equal-aspect preview.
    Writes back into the active section's Spec and reruns on change."""
    active = st.session_state.active_section
    rec = st.session_state.sections[active]
    spec = rec["spec"]
    st.subheader("Custom section — coordinate editor")
    st.caption(f"Vertices & bar positions in **{u.Ll}**, diameters in mm. "
               "Order vertices around the polygon; the section is auto-"
               "recentred on its centroid. Use ＋ to add rows, select a "
               "row's checkbox then ⌫ to delete.")

    outline = list(spec.custom_outline)
    if len(outline) < 3:                       # seed a starter rectangle
        outline = [(-0.20, -0.30), (0.20, -0.30), (0.20, 0.30), (-0.20, 0.30)]

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Vertices** (ordered)")
        vdf = pd.DataFrame([{"z": u.from_m(z), "y": u.from_m(y)}
                            for (z, y) in outline])
        ved = st.data_editor(
            vdf, num_rows="dynamic", hide_index=True, width="stretch",
            key=f"cv_{active}_{u.length}", column_config={
                "z": st.column_config.NumberColumn(f"z [{u.Ll}]", format="%.4g"),
                "y": st.column_config.NumberColumn(f"y [{u.Ll}]",
                                                   format="%.4g")})
    with e2:
        st.markdown("**Reinforcing bars**")
        bdf = pd.DataFrame([{"z": u.from_m(z), "y": u.from_m(y), "dia": d * 1e3}
                            for (z, y, d) in spec.custom_bars])
        bed = st.data_editor(
            bdf, num_rows="dynamic", hide_index=True, width="stretch",
            key=f"cb_{active}_{u.length}", column_config={
                "z": st.column_config.NumberColumn(f"z [{u.Ll}]", format="%.4g"),
                "y": st.column_config.NumberColumn(f"y [{u.Ll}]", format="%.4g"),
                "dia": st.column_config.NumberColumn(
                    "⌀ [mm]", format="%.0f", min_value=6.0, max_value=60.0)})

    def _rows_xy(ed):
        out = []
        for _, r in ed.iterrows():
            if pd.isna(r["z"]) or pd.isna(r["y"]):
                continue
            out.append((round(u.to_m(float(r["z"])), 6),
                        round(u.to_m(float(r["y"])), 6)))
        return tuple(out)

    new_outline = _rows_xy(ved)
    new_bars = tuple(
        (round(u.to_m(float(r["z"])), 6), round(u.to_m(float(r["y"])), 6),
         round(float(r["dia"]) * 1e-3, 6))
        for _, r in bed.iterrows()
        if not (pd.isna(r["z"]) or pd.isna(r["y"]) or pd.isna(r["dia"])))

    # fill-perimeter helper
    with e2.popover("＋ Fill perimeter with bars", width="stretch"):
        fn = st.number_input("Number of bars", 3, 100, 8, 1, key=f"fpn_{active}")
        fcov = st.number_input(f"Cover [{u.Ll}]", 0.0, 1.0,
                               float(u.from_m(0.05)), key=f"fpc_{active}")
        fdia = st.number_input("⌀ [mm]", 6.0, 60.0, 25.0, 1.0,
                               key=f"fpd_{active}")
        if st.button("Generate", key=f"fpg_{active}", type="primary"):
            gen = core.perimeter_bars(new_outline, int(fn), u.to_m(fcov),
                                      fdia * 1e-3)
            rec["spec"] = replace(spec, custom_outline=new_outline,
                                  custom_bars=gen)
            _autosave()
            st.rerun()

    # live preview (equal aspect)
    fig = go.Figure()
    if len(new_outline) >= 3:
        ox = [u.from_m(z) for z, y in new_outline]
        oy = [u.from_m(y) for z, y in new_outline]
        fig.add_trace(go.Scatter(
            x=ox + [ox[0]], y=oy + [oy[0]], mode="lines+markers",
            fill="toself", line=dict(color="#4472c4"),
            fillcolor="rgba(68,114,196,0.15)", name="outline",
            hovertemplate="z=%{x:.4g}, y=%{y:.4g}<extra></extra>"))
    if new_bars:
        fig.add_trace(go.Scatter(
            x=[u.from_m(z) for z, y, d in new_bars],
            y=[u.from_m(y) for z, y, d in new_bars],
            mode="markers", marker=dict(size=11, color="#222",
                                        line=dict(color="#fff", width=1)),
            name="bars", customdata=[d * 1e3 for z, y, d in new_bars],
            hovertemplate="bar z=%{x:.4g}, y=%{y:.4g}<br>⌀%{customdata:.0f} mm"
                          "<extra></extra>"))
    fig.update_layout(
        height=430, margin=dict(l=0, r=0, t=6, b=0), showlegend=False,
        xaxis_title=f"z [{u.Ll}]", yaxis_title=f"y [{u.Ll}]",
        yaxis=dict(scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig, key=f"cprev_{active}_{u.length}")
    if len(new_outline) < 3:
        st.warning("Add at least 3 vertices to define the polygon.")

    # persist + rerun so sketch/props/analyses pick up the change
    if new_outline != spec.custom_outline or new_bars != spec.custom_bars:
        rec["spec"] = replace(spec, custom_outline=new_outline,
                              custom_bars=new_bars)
        _autosave()
        st.rerun()


def _recover_shape_params(outline):
    """Best-fit ``(styp, cz, cy, w, h, D)`` in metres for a stored composite
    outline, so it can be reloaded into the parametric add/edit form. The
    builder only emits axis-aligned rectangles (4 pts) and circles (48 pts),
    so the bounding box recovers the original parameters exactly."""
    zs = [float(z) for z, _ in outline]
    ys = [float(y) for _, y in outline]
    zmin, zmax, ymin, ymax = min(zs), max(zs), min(ys), max(ys)
    cz, cy = (zmin + zmax) / 2.0, (ymin + ymax) / 2.0
    w, h = zmax - zmin, ymax - ymin
    styp = "Circle" if len(outline) >= 12 else "Rectangle"
    return styp, cz, cy, w, h, max(w, h)


def _match_material_name(mat_kv, mats):
    """Return the library material name whose properties equal ``mat_kv`` (a
    sorted-items tuple), or None when no library material matches -- used to
    pre-select the right material when editing a shape."""
    target = tuple(sorted(dict(mat_kv).items()))
    for name, md in mats.items():
        if tuple(sorted(md.items())) == target:
            return name
    return None


def _composite_editor(u):
    """Composite section shape builder: add / edit / reorder material shapes
    (rectangle/circle), each carrying a material from the library; later shapes
    displace earlier ones on overlap. Analysis is fibre-based (P-M-M & M-φ)."""
    active = st.session_state.active_section
    rec = st.session_state.sections[active]
    spec = rec["spec"]
    mats = st.session_state.get("materials", {})
    L = u.Ll

    def k(f):
        return f"csh_{f}_{active}_{u.length}"

    # A queued reset clears the add/edit form back to defaults. It must run
    # *before* the form widgets below are instantiated: Streamlit forbids
    # mutating a widget's state key after that widget exists in the same run.
    FORM_FIELDS = ("typ", "cz", "cy", "w", "h", "D")
    if st.session_state.pop(k("reset"), False):
        for f in FORM_FIELDS:
            st.session_state.pop(k(f), None)

    editing = st.session_state.get(k("editing"))
    if editing is not None and not (0 <= editing < len(spec.shapes)):
        st.session_state.pop(k("editing"), None)     # stale index -> drop
        editing = None

    def _exit_edit():
        """Leave edit mode and queue a form reset (safe from any handler)."""
        if st.session_state.get(k("editing")) is not None:
            st.session_state.pop(k("editing"), None)
            st.session_state[k("reset")] = True

    st.subheader("Composite section — material shapes")
    st.caption("Each shape carries a material from the library. Later shapes "
               "displace earlier ones on overlap (e.g. a steel core inside "
               "concrete) — use ▲ / ▼ to reorder. Analysis is fibre-based — "
               "see the P-M-M and Moment-curvature tabs. Each rebar "
               "arrangement uses its own steel (set in ▦ Reinforcement); bars "
               "fall back to the section steel when none is chosen.")
    n = len(spec.shapes)
    for i, (outline, mat_kv) in enumerate(list(spec.shapes)):
        md = dict(mat_kv)
        kd = md.get("kind", "concrete")
        strg = (md.get("fc", 0) if kd == "concrete" else md.get("fy", 0)) / 1e6
        c = st.columns([6, 1, 1, 1, 1])
        c[0].caption(f"• Shape {i + 1}: **{kd}** {strg:.0f} MPa · "
                     f"{len(outline)} pts"
                     + ("  ·  ✎ editing" if editing == i else ""))
        if c[1].button("✎", key=f"edsh_{active}_{i}", help="Edit this shape"):
            styp_r, cz_r, cy_r, w_r, h_r, d_r = _recover_shape_params(outline)
            st.session_state[k("editing")] = i
            st.session_state[k("typ")] = styp_r
            st.session_state[k("cz")] = float(u.from_m(cz_r))
            st.session_state[k("cy")] = float(u.from_m(cy_r))
            st.session_state[k("w")] = float(u.from_m(w_r))
            st.session_state[k("h")] = float(u.from_m(h_r))
            st.session_state[k("D")] = float(u.from_m(d_r))
            nm = _match_material_name(mat_kv, mats)
            if nm is not None:
                st.session_state[k("mat")] = nm
            st.rerun()
        if c[2].button("▲", key=f"upsh_{active}_{i}", help="Move up",
                       disabled=(i == 0)):
            sh = list(spec.shapes)
            sh[i - 1], sh[i] = sh[i], sh[i - 1]
            rec["spec"] = replace(spec, shapes=tuple(sh))
            _exit_edit()
            st.rerun()
        if c[3].button("▼", key=f"dnsh_{active}_{i}", help="Move down",
                       disabled=(i == n - 1)):
            sh = list(spec.shapes)
            sh[i + 1], sh[i] = sh[i], sh[i + 1]
            rec["spec"] = replace(spec, shapes=tuple(sh))
            _exit_edit()
            st.rerun()
        if c[4].button("✕", key=f"delsh_{active}_{i}", help="Remove"):
            sh = list(spec.shapes)
            del sh[i]
            rec["spec"] = replace(spec, shapes=tuple(sh))
            _exit_edit()
            st.rerun()

    st.markdown(f"**Edit shape {editing + 1}**" if editing is not None
                else "**Add shape**")

    if mats:
        names = list(mats)
        mk = k("mat")
        if st.session_state.get(mk) not in names:
            st.session_state[mk] = names[0]
        matd = dict(mats[st.selectbox("Material", names, key=mk)])
    else:
        st.caption("Tip: add materials on the **🧱 Materials** page first, "
                   "then assign them here. Using default 30 MPa concrete "
                   "for now.")
        matd = dict(kind="concrete", fc=30e6, conc_model="Kent-Park",
                    eps_c0=0.002, eps_cu=0.0035, fcu_ratio=0.4, fr_model="sqrt",
                    fr_coeff=0.62, eps_decay=1e-3, conc_f1_ratio=0.4)
    st.session_state.setdefault(k("typ"), "Rectangle")
    styp = st.selectbox("Shape", ["Rectangle", "Circle"], key=k("typ"))
    cc = st.columns(2)

    def N(col, lbl, key, dflt):
        kk = k(key)
        st.session_state.setdefault(kk, float(u.from_m(dflt)))
        return u.to_m(col.number_input(f"{lbl} [{L}]", step=u.len_step(),
                                       format="%.4g", key=kk))

    cz, cy = N(cc[0], "center z", "cz", 0.0), N(cc[1], "center y", "cy", 0.0)
    if styp == "Rectangle":
        w, h = N(cc[0], "width", "w", 0.4), N(cc[1], "height", "h", 0.6)
        outline = ((cz - w / 2, cy - h / 2), (cz + w / 2, cy - h / 2),
                   (cz + w / 2, cy + h / 2), (cz - w / 2, cy + h / 2))
    else:
        r = N(st, "diameter", "D", 0.4) / 2
        outline = tuple((float(cz + r * np.cos(a)), float(cy + r * np.sin(a)))
                        for a in np.linspace(0, 2 * np.pi, 48, endpoint=False))

    if editing is None:
        if st.button("＋ Add shape", key=k("add"), width="stretch"):
            rec["spec"] = replace(spec, shapes=spec.shapes
                                  + ((outline, tuple(sorted(matd.items()))),))
            st.rerun()
    else:
        # Keep the shape's original material when there is no library to pick
        # from, so a pure position/size edit never silently swaps its material.
        new_mat_kv = (tuple(sorted(matd.items())) if mats
                      else spec.shapes[editing][1])
        bc = st.columns([3, 1])
        if bc[0].button("✓ Update shape", key=k("upd"), width="stretch",
                        type="primary"):
            sh = list(spec.shapes)
            sh[editing] = (outline, new_mat_kv)
            rec["spec"] = replace(spec, shapes=tuple(sh))
            st.session_state.pop(k("editing"), None)
            st.session_state[k("reset")] = True
            st.rerun()
        if bc[1].button("Cancel", key=k("cancel"), width="stretch"):
            st.session_state.pop(k("editing"), None)
            st.session_state[k("reset")] = True
            st.rerun()

    if spec.shapes:
        st.caption("↔ The **Cross-section** canvas shows these shapes coloured "
                   "by material (grey = concrete, blue = steel).")


def _stress_field(case, spec, u, e_top, e_bot, nz=32, ny=48):
    """Fibre-stress field for a single-material section under a plane-sections
    strain state (strain linear in y, **tension +**). Evaluates the section's
    own concrete / steel constitutive laws (same ``get_response`` the engine
    uses), so the colours are physically consistent. Returns display-unit
    arrays for the canvas."""
    from shapely.geometry import Point
    conc = core.concrete_uniaxial_from(dict(
        fc=spec.fc, conc_model=spec.conc_model, eps_c0=spec.eps_c0,
        eps_cu=spec.eps_cu, fcu_ratio=spec.fcu_ratio, fr_model=spec.fr_model,
        fr_coeff=spec.fr_coeff, eps_decay=spec.eps_decay,
        conc_f1_ratio=spec.conc_f1_ratio))
    steel = core.steel_uniaxial_from(dict(
        fy=spec.fy, Es=spec.Es, steel_model=spec.steel_model,
        steel_b=spec.steel_b, steel_fu_ratio=spec.steel_fu_ratio,
        steel_eps_sh=spec.steel_eps_sh, steel_eps_su=spec.steel_eps_su))
    poly = case.section.geometry.polygon
    minz, miny, maxz, maxy = poly.bounds
    span = (maxy - miny) or 1.0
    cf = u.from_m

    def eps(y):
        return e_bot + (e_top - e_bot) * (y - miny) / span

    Z, Y, S = [], [], []
    for yy in np.linspace(miny, maxy, ny):
        sig = conc.get_response(float(eps(yy)))[0] / 1e6              # MPa
        for zz in np.linspace(minz, maxz, nz):
            if poly.contains(Point(float(zz), float(yy))):
                Z.append(cf(zz)); Y.append(cf(yy)); S.append(sig)
    bars = case.section.reinforcement.bars \
        if case.section.reinforcement else []
    bsig = [steel.get_response(float(eps(b.y)))[0] / 1e6 for b in bars]
    # scale colours to the concrete field so its gradient stays readable; steel
    # (σ is several× higher) clamps to the ends of the same diverging scale.
    clim = max([abs(v) for v in S] + [1.0])
    y_na = None                                       # neutral axis (eps = 0)
    if (e_top > 0) != (e_bot > 0) and e_top != e_bot:
        y0 = miny + (0.0 - e_bot) * span / (e_top - e_bot)
        if miny <= y0 <= maxy:
            y_na = cf(y0)
    return dict(z=Z, y=Y, sig=S, bz=[cf(b.z) for b in bars],
                by=[cf(b.y) for b in bars], bsig=bsig, clim=clim, y_na=y_na)


def _section_canvas(case, u, stress=None, shapes=None):
    """Interactive Plotly cross-section: outline (with holes), rebar, tendons,
    centroid and overall dimensions -- equal aspect, pan/zoom. ``stress`` paints
    the fibre-stress field instead of the plain fill; ``shapes`` (composite
    (outline, material) pairs) draws each material region in its own colour.
    Driven by the engine data model, so it matches svg_of."""
    sec = case.section
    poly = sec.geometry.polygon
    minz, miny, maxz, maxy = poly.bounds
    cf = u.from_m                                   # metres -> display units
    hot = stress is not None

    fig = go.Figure()
    if shapes and not hot:                          # composite: per-material fill
        for (outline, mat_kv) in shapes:
            conc = dict(mat_kv).get("kind", "concrete") == "concrete"
            ox = [cf(z) for z, _y in outline]
            oy = [cf(y) for _z, y in outline]
            fig.add_trace(go.Scatter(
                x=ox + [ox[0]], y=oy + [oy[0]], fill="toself", mode="lines",
                fillcolor=("rgba(150,157,170,0.45)" if conc
                           else "rgba(46,120,210,0.55)"),
                line=dict(color="#5b7aa8", width=1.5), hoverinfo="skip",
                showlegend=False))
    else:
        def _ring(coords):
            pts = list(coords)
            if len(pts) > 1 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if not pts:
                return ""
            d = f"M {cf(pts[0][0])},{cf(pts[0][1])} "
            d += " ".join(f"L {cf(z)},{cf(y)}" for z, y in pts[1:])
            return d + " Z "
        path = _ring(poly.exterior.coords)
        for ring in poly.interiors:
            path += _ring(ring.coords)
        fig.add_shape(
            type="path", path=path, fillrule="evenodd", xref="x", yref="y",
            layer="below", line=dict(color="#5b7aa8", width=2),
            fillcolor=("rgba(0,0,0,0)" if hot else "rgba(37,99,235,0.10)"))

    if hot:                                           # fibre-stress field
        fig.add_trace(go.Scatter(
            x=stress["z"], y=stress["y"], mode="markers", showlegend=False,
            marker=dict(size=7, symbol="square", color=stress["sig"],
                        colorscale="RdBu_r", cmin=-stress["clim"],
                        cmax=stress["clim"], line=dict(width=0),
                        colorbar=dict(title="σ<br>MPa", thickness=12, len=0.85,
                                      outlinewidth=0)),
            hovertemplate="σ %{marker.color:.1f} MPa<extra></extra>"))

    if sec.reinforcement and sec.reinforcement.bars:
        bars = sec.reinforcement.bars
        dia = [2.0 * np.sqrt(b.area / np.pi) * 1e3 for b in bars]   # mm
        bsize = [max(7, min(22, d * 0.34)) for d in dia]
        if hot:
            bmark = dict(size=bsize, color=stress["bsig"], colorscale="RdBu_r",
                         cmin=-stress["clim"], cmax=stress["clim"],
                         line=dict(color="#222", width=1))
            bhov = "σ %{marker.color:.1f} MPa<extra>bar</extra>"
            bx, by = stress["bz"], stress["by"]
        else:
            bmark = dict(size=bsize, color="#e08a00",
                         line=dict(color="#9c6100", width=1))
            bhov = ("z %{x:.4g} · y %{y:.4g}<br>"
                    "⌀%{customdata:.0f} mm<extra>rebar</extra>")
            bx, by = [cf(b.z) for b in bars], [cf(b.y) for b in bars]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="markers", name="rebar", customdata=dia,
            marker=bmark, hovertemplate=bhov, showlegend=False))

    tend = getattr(sec, "prestress", None)
    if tend and tend.tendons and not hot:
        fig.add_trace(go.Scatter(
            x=[cf(t.z) for t in tend.tendons],
            y=[cf(t.y) for t in tend.tendons], mode="markers", name="tendon",
            marker=dict(size=10, color="#0e97ab", symbol="diamond",
                        line=dict(color="#0a6b78", width=1)),
            hovertemplate="tendon z %{x:.4g} · y %{y:.4g}<extra></extra>"))

    cz, cy = sec.geometry.centroid
    fig.add_trace(go.Scatter(
        x=[cf(cz)], y=[cf(cy)], mode="markers", showlegend=False,
        marker=dict(size=10, color="#7a8aa0", symbol="cross"),
        hovertemplate="centroid<extra></extra>"))

    W, H = maxz - minz, maxy - miny
    pad = max(W, H) * 0.10
    # invisible corner points fix the view and keep equal aspect stable
    fig.add_trace(go.Scatter(
        x=[cf(minz - pad), cf(maxz + pad)],
        y=[cf(miny - pad), cf(maxy + pad)], mode="markers",
        marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    _dimfont = dict(size=11, family="IBM Plex Mono, monospace")
    fig.add_annotation(x=cf((minz + maxz) / 2), y=cf(miny), yshift=-18,
                       text=f"{W * 1e3:.0f} mm", showarrow=False, font=_dimfont)
    fig.add_annotation(x=cf(maxz), y=cf((miny + maxy) / 2), xshift=24,
                       textangle=90, text=f"{H * 1e3:.0f} mm", showarrow=False,
                       font=_dimfont)
    if hot and stress.get("y_na") is not None:
        fig.add_hline(y=stress["y_na"], line=dict(dash="dash", width=1.5,
                      color="#e5484d"), annotation_text="N.A.",
                      annotation_position="top right")

    fig.update_xaxes(title_text=f"z [{u.Ll}]", zeroline=True,
                     zerolinecolor="rgba(120,140,170,.45)")
    fig.update_yaxes(title_text=f"y [{u.Ll}]", scaleanchor="x", scaleratio=1,
                     zeroline=True, zerolinecolor="rgba(120,140,170,.45)")
    fig.update_layout(height=470, margin=dict(l=6, r=6, t=6, b=6),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      dragmode="pan")
    return fig


def _reinforcement_editor(rec, u):
    """Reinforcement counts + arrangements for the Section-setup right panel.
    Writes to the spec; the Definition tab rebuilds the section right after, so
    edits show on the canvas immediately."""
    active = st.session_state.active_section
    s0 = rec["spec"]
    kind = s0.kind
    Ls = u.len_step()

    def K(f):
        return f"{f}__{active}__{_usig()}"

    def LI(col, label, field, default_m, lo, hi):
        dm = min(hi, max(lo, default_m))
        v = col.number_input(f"{label} [{u.Ll}]", float(u.from_m(lo)),
                             float(u.from_m(hi)), float(u.from_m(dm)), Ls,
                             key=K(field))
        return u.to_m(v)

    rd = {}
    if kind == "Custom":
        st.caption("Bars are placed individually in the coordinate editor "
                   "below (“Fill perimeter”).")
    else:
        r1, r2 = st.columns(2)
        default_size = f"{int(round(s0.bar_dia * 1e3))} mm"
        if default_size not in core.BAR_SIZES:
            default_size = "25 mm"
        size = r1.selectbox("Bar size", list(core.BAR_SIZES),
                            index=list(core.BAR_SIZES).index(default_size),
                            key=K("bar"))
        rd["bar_dia"] = core.BAR_SIZES[size]
        rd["cover"] = LI(r2, "Cover", "cover", s0.cover, 0.01, 0.3)
        if kind in ("Rectangular", "PSC girder"):
            rd["n_top"] = int(r1.number_input("Top bars", 0, 20, s0.n_top, 1,
                                              key=K("ntop")))
            if kind == "Rectangular":
                rd["n_bot"] = int(r2.number_input(
                    "Bottom bars", 0, 20, s0.n_bot, 1, key=K("nbot")))
                rd["n_side"] = int(r1.number_input(
                    "Side bars / face", 0, 10, s0.n_side, 1, key=K("nside")))
        else:
            rd["n_perim"] = int(st.number_input(
                "Perimeter bars", 4, 60, s0.n_perim, 1, key=K("nperim")))
        if kind == "Circular":
            rd["spiral"] = st.toggle("Spiral (φ cap 0.85)", value=s0.spiral,
                                     key=K("spiral"))
    if rd:
        rec["spec"] = replace(rec["spec"], **rd)
    st.divider()
    _rebar_arrangements(rec, u)


def tab_definition(case, u, code):
    """The section Definition tab: the entire section definition inline —
    shape & dimensions, materials, reinforcement (and tendons), geometry
    editors — beside a live cross-section preview. No sidebar needed."""
    active = st.session_state.active_section
    rec = st.session_state.sections[active]
    rec["code"] = code            # keep the section on the project code
    spec = rec["spec"]
    _kind = spec.kind
    # Panel modes depend on the section kind: Composite/Custom get their
    # geometry editor here as a panel mode, not a full-width block above.
    opts = ["Dimensions"]
    if _kind == "Composite":
        opts.append("Shapes")
    elif _kind == "Custom":
        opts.append("Geometry")
    opts += ["Materials", "Reinforcement", "Properties"]

    pkey = f"panel_{active}"
    cur = st.session_state.get(pkey) or "Dimensions"
    if cur not in opts:
        cur = "Dimensions"
    # geometry editors need room -> a 50/50 split in those modes.
    c1, c2 = st.columns([1, 1] if cur in ("Shapes", "Geometry") else [3, 2],
                        gap="large")
    # ---- right-hand panel: pick what to edit / view (AdSec-style) ----
    with c2:
        panel = st.segmented_control(
            "Panel", opts, default="Dimensions", key=pkey,
            label_visibility="collapsed")
        panel = panel or "Dimensions"
        if panel == "Dimensions":
            _dimension_inputs(rec, u, code)
        elif panel == "Shapes":
            _composite_editor(u)
        elif panel == "Geometry":
            _custom_editor(u)
        elif panel == "Materials":
            _material_assign(rec, u)
        elif panel == "Reinforcement":
            _reinforcement_editor(rec, u)
            st.divider()
            with st.expander("🎗 Tendon arrangements",
                             expanded=(_kind == "PSC girder")):
                _tendon_arrangements(rec, u)
    # the editors above may have mutated the spec -> rebuild so the canvas,
    # properties and reinforcement table all reflect the edit this run.
    spec = rec["spec"]
    case = core.build_case(spec)
    with c1:
        st.subheader("Cross-section")
        stress = None
        if _kind != "Composite":
            mode = st.segmented_control(
                "View", ["Geometry", "Stress"], default="Geometry",
                key=f"geomode_{active}", label_visibility="collapsed")
            if mode == "Stress":
                sc = st.columns([1, 1, 2])
                e_top = sc[0].number_input(
                    "ε top-fibre [‰]", value=-1.50, step=0.10, format="%.2f",
                    key=f"etop_{active}") / 1000.0
                e_bot = sc[1].number_input(
                    "ε bottom-fibre [‰]", value=1.00, step=0.10, format="%.2f",
                    key=f"ebot_{active}") / 1000.0
                sc[2].caption("Plane-sections strain state (tension +). Fibres "
                              "& bars coloured by the material stress law; "
                              "dashed line = neutral axis.")
                stress = _stress_field(case, spec, u, e_top, e_bot)
        st.plotly_chart(
            _section_canvas(case, u, stress,
                            shapes=(spec.shapes if _kind == "Composite"
                                    else None)),
            width="stretch", key=f"seccanvas_{active}_{u.length}")
    with c2:
        if panel == "Properties":
            st.subheader("Section properties")
            g = case.section.geometry
            _cz, cy = g.centroid
            miny = g.polygon.bounds[1]
            m = st.columns(2)
            m[0].metric(f"Gross area A_g [{u.Al}]",
                        f"{u.area_disp(g.area * 1e6):.4g}")
            m[1].metric(f"Centroid ↑ base [{u.Ll}]",
                        f"{u.len_disp((cy - miny) * 1e3):.4g}")
            m[0].metric(f"I_zz [{u.Il}]",
                        f"{u.inertia_disp(g.I_zz * 1e12):.4g}")
            m[1].metric(f"I_yy [{u.Il}]",
                        f"{u.inertia_disp(g.I_yy * 1e12):.4g}")
            n_bar = len(case.section.reinforcement.bars) \
                if case.section.reinforcement else 0
            n_ten = len(case.section.prestress.tendons) \
                if case.section.prestress else 0
            st.caption(f"{n_bar} rebar · {n_ten} tendons. Reference frame at "
                       "the centroid; y vertical (strong axis). Compression +.")

    bars = case.section.reinforcement.bars \
        if case.section.reinforcement else []
    if bars:
        st.markdown("**Reinforcement**")
        dia = [2.0 * np.sqrt(b.area / np.pi) * 1e3 for b in bars]
        df = pd.DataFrame({
            "Bar": list(range(1, len(bars) + 1)),
            f"z [{u.Ll}]": [u.from_m(b.z) for b in bars],
            f"y [{u.Ll}]": [u.from_m(b.y) for b in bars],
            "⌀ [mm]": dia,
            "Area [mm²]": [b.area * 1e6 for b in bars]})
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={
                         f"z [{u.Ll}]":
                             st.column_config.NumberColumn(format="%.4g"),
                         f"y [{u.Ll}]":
                             st.column_config.NumberColumn(format="%.4g"),
                         "⌀ [mm]": st.column_config.NumberColumn(format="%.0f"),
                         "Area [mm²]":
                             st.column_config.NumberColumn(format="%.0f")})


def tab_surface3d(spec, u, code):
    if not _calc_gate():
        return
    if not _ok_to_analyze(spec):
        return
    st.caption(f"Design code: **{code}** · mesh density & load combinations "
               "are set in the Analysis controls above.")
    res = st.session_state.get("mesh_density", "GSD default")
    na, npl = MESH_GRID[res]
    shaded = st.toggle("Shaded surface", value=True, key="s3d_shade")

    mesh = compute_surface3d(astuple(spec), code, na, npl, _mats_frozen())
    ML, FL, LL = u.Ml, u.Fl, u.Ll
    # canonical (kN, kN.m, mm) -> user display units
    Mz = u.M_disp(np.array(mesh["Mz"]))       # (n_plevels, n_angles)
    My = u.M_disp(np.array(mesh["My"]))
    Pl = u.P_disp(np.array(mesh["Plevels"]))  # (n_plevels,)
    thetas = np.array(mesh["thetas"])
    rawP = u.P_disp(np.array(mesh["rawP"]))   # (n_angles, n_depths)
    rawMz = u.M_disp(np.array(mesh["rawMz"]))
    rawMy = u.M_disp(np.array(mesh["rawMy"]))
    rawC = u.len_disp(np.array(mesh["rawC"]))
    rawEps = np.array(mesh["rawEpsT"])
    rawPhi = np.array(mesh["rawPhi"])
    Phi = np.array(mesh["Phi"])
    npl, na = Mz.shape
    nd = rawP.shape[1]

    # The surface is computed on a FIXED grid of neutral-axis angles (ribs)
    # and axial-force levels (rings) whose counts follow the mesh density —
    # so these pick a real computed slice from a dropdown rather than a
    # slider that would snap/interpolate between them.
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**P-M curve** — vertical slice at an angle")
        show_pm = st.checkbox("Show all ribs", value=True, key="s3d_showpm")
        # init in session state (not via index=) so resetting a stale
        # selection after a mesh-density change raises no default-vs-state
        # warning; reset when the option count shrinks past the selection.
        if ("s3d_angi" not in st.session_state
                or st.session_state["s3d_angi"] >= na):
            st.session_state["s3d_angi"] = 0
        i_ang = st.selectbox(
            "Neutral-axis angle [deg]", range(na),
            format_func=lambda i: f"{thetas[i]:.0f}", key="s3d_angi",
            help="Discrete rib angles computed by the mesh (count set by "
                 "Mesh density above).")
        ang = float(thetas[i_ang])
    with c2:
        st.markdown("**M-M curve** — horizontal slice at an axial force")
        show_mm = st.checkbox("Show all rings", value=True, key="s3d_showmm")
        p_def = int(np.argmin(np.abs(Pl)))              # level nearest P=0
        if ("s3d_pi" not in st.session_state
                or st.session_state["s3d_pi"] >= npl):
            st.session_state["s3d_pi"] = p_def
        i_p = st.selectbox(
            f"Axial force P [{FL}]", range(npl),
            format_func=lambda i: f"{Pl[i]:.4g}", key="s3d_pi",
            help="Discrete axial-force levels of the M-M rings (count set by "
                 "Mesh density above).")
        pval = float(Pl[i_p])

    # ---- demand check: read the sidebar's load combinations ----
    design = _dsurf().startswith("Design")
    canon = st.session_state.get("demands", {}).get(
        st.session_state.active_section, [])
    dem_res = (compute_demand(astuple(spec), code, tuple(canon), design, na,
                              npl, _dc_method(), _mats_frozen())
               if canon else [])

    lc = "rgba(70,120,200,0.5)"
    fig = go.Figure()

    # translucent shaded surface (triangulated, wrapping around angles)
    if shaded:
        X = Mz.flatten()
        Y = My.flatten()
        Z = np.repeat(Pl, na)
        Ii, Jj, Kk = [], [], []
        for k in range(npl - 1):
            for i in range(na):
                i2 = (i + 1) % na
                a, b = k * na + i, k * na + i2
                c, d = (k + 1) * na + i, (k + 1) * na + i2
                Ii += [a, b]
                Jj += [b, d]
                Kk += [c, c]
        fig.add_trace(go.Mesh3d(
            x=X, y=Y, z=Z, i=Ii, j=Jj, k=Kk, color="lightskyblue",
            opacity=0.22, hoverinfo="skip", showlegend=False))

    # wireframe ribs (P-M curves) as one trace with None gaps
    if show_pm:
        rx, ry, rz = [], [], []
        for i in range(na):
            rx += [*Mz[:, i], None]
            ry += [*My[:, i], None]
            rz += [*Pl, None]
        fig.add_trace(go.Scatter3d(
            x=rx, y=ry, z=rz, mode="lines", line=dict(color=lc, width=1),
            hoverinfo="skip", showlegend=False))
    # wireframe rings (M-M curves), closed, one trace
    if show_mm:
        hx, hy, hz = [], [], []
        for k in range(npl):
            hx += [*Mz[k], Mz[k][0], None]
            hy += [*My[k], My[k][0], None]
            hz += [*([Pl[k]] * (na + 1)), None]
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz, mode="lines", line=dict(color=lc, width=1),
            hoverinfo="skip", showlegend=False))

    # highlighted P-M curve at the selected angle -- a SINGLE rib from the
    # raw neutral-axis-depth sampling (GSD convention), so it matches the
    # inset/table exactly; one curve per angle, correct for any shape.
    fig.add_trace(go.Scatter3d(
        x=rawMz[i_ang], y=rawMy[i_ang], z=rawP[i_ang], mode="lines",
        line=dict(color="crimson", width=6), name=f"P-M @ {ang:.0f}°"))
    # highlighted M-M curve at the selected axial force (interpolated ring)
    ringz = np.array([float(np.interp(pval, Pl, Mz[:, i])) for i in range(na)])
    ringy = np.array([float(np.interp(pval, Pl, My[:, i])) for i in range(na)])
    fig.add_trace(go.Scatter3d(
        x=list(ringz) + [ringz[0]], y=list(ringy) + [ringy[0]],
        z=[pval] * (na + 1), mode="lines", line=dict(color="orange", width=7),
        name=f"M-M @ P={pval:.4g} {FL}"))

    # per-combination P-M capacity curve, drawn in the vertical plane of that
    # combination's moment vector (β = atan2(My, Mz)) — one P-M interaction
    # curve per load combination, on the checked (design/nominal) surface.
    if dem_res:
        rings_mz = (Mz * Phi) if design else Mz
        rings_my = (My * Phi) if design else My
        for d in dem_res:
            Ps, Ms = core.pm_curve_at_angle(Pl, rings_mz, rings_my,
                                            d["beta_deg"])
            if len(Ps) < 2:
                continue
            br = np.radians(d["beta_deg"])
            lbl = d["name"] or f"{d['beta_deg']:.0f}°"
            fig.add_trace(go.Scatter3d(
                x=[m * float(np.cos(br)) for m in Ms],
                y=[m * float(np.sin(br)) for m in Ms], z=Ps, mode="lines",
                line=dict(color=("#2ca02c" if d["status"] == "OK"
                                 else "#d62728"), width=3, dash="dot"),
                name=f"P-M · {lbl}", hoverinfo="skip"))

    # demand points, coloured by pass/fail
    for grp_name, col in (("OK", "#2ca02c"), ("FAIL", "#d62728")):
        grp = [d for d in dem_res if d["status"] == grp_name]
        if grp:
            fig.add_trace(go.Scatter3d(
                x=[u.M_disp(d["Mz"]) for d in grp],
                y=[u.M_disp(d["My"]) for d in grp],
                z=[u.P_disp(d["P"]) for d in grp],
                mode="markers+text",
                text=[d["name"] for d in grp], textposition="top center",
                marker=dict(size=5, color=col, symbol="diamond",
                            line=dict(color="black", width=1)),
                name=f"Demand {grp_name}",
                hovertext=[f"{d['name']}: util {d['util']*100:.0f}%"
                           for d in grp], hoverinfo="text"))

    fig.update_layout(
        height=650, margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(xaxis_title=f"Mz [{ML}]", yaxis_title=f"My [{ML}]",
                   zaxis_title=f"P [{FL}]", aspectmode="cube"),
        legend=dict(orientation="h", yanchor="bottom", y=0.0))
    st.plotly_chart(fig)
    _demnote = (" Dotted green/red = each load combination's P-M curve at its "
                "moment angle (on the checked surface), with its demand point."
                if dem_res else "")
    st.caption(f"Full P-Mz-My surface — {na}-point M-M rings (constant axial "
               f"force) × {nd}-point P-M ribs (neutral-axis-depth sampled, "
               "GSD convention). Drag to rotate · scroll to zoom. Nominal "
               "surface (φ=1). Red = the P-M curve at the chosen angle; "
               "orange = M-M at the chosen axial force." + _demnote)

    if dem_res:
        gov = max(dem_res, key=lambda d: d["util"])
        gcol = st.columns(3)
        gcol[0].metric("Governing combo", gov["name"] or "—")
        gcol[1].metric("Max utilization", f"{gov['util'] * 100:.1f}%")
        gcol[2].metric("Result", "OK ✅" if gov["util"] <= 1.0 else "FAIL ❌")
        dres = pd.DataFrame([{
            "Combo": d["name"], f"P [{FL}]": u.P_disp(d["P"]),
            f"Mz [{ML}]": u.M_disp(d["Mz"]), f"My [{ML}]": u.M_disp(d["My"]),
            f"M appl [{ML}]": u.M_disp(d["M_res"]), "β [°]": d["beta_deg"],
            f"M cap [{ML}]": u.M_disp(d["M_cap"]),
            "Utilization": min(d["util"], 9.99), "Status": d["status"]}
            for d in dem_res])
        st.dataframe(dres, hide_index=True, width="stretch", column_config={
            f"P [{FL}]": st.column_config.NumberColumn(format="%.4g"),
            f"Mz [{ML}]": st.column_config.NumberColumn(format="%.4g"),
            f"My [{ML}]": st.column_config.NumberColumn(format="%.4g"),
            f"M appl [{ML}]": st.column_config.NumberColumn(format="%.4g"),
            "β [°]": st.column_config.NumberColumn(format="%.0f"),
            f"M cap [{ML}]": st.column_config.NumberColumn(format="%.4g"),
            "Utilization": st.column_config.ProgressColumn(
                "Utilization (D/C)", format="%.2f",
                min_value=0.0, max_value=1.5)})
        _mname = {"P": "keep P constant — scale moment at fixed axial",
                  "M": "keep M constant — scale axial at fixed moment",
                  "MP": "keep M/P constant — radial (scale both)"}[_dc_method()]
        st.caption(f"Utilization (D/C) by the **{_mname}** method, against the "
                   f"**{'φ-reduced design' if design else 'nominal'}** surface. "
                   "Markers: green = OK, red = FAIL. Method & load "
                   "combinations are set in the Analysis controls above.")
    else:
        st.caption("💡 Enter factored load combinations in the Analysis "
                   "controls above to overlay demand points, their P-M "
                   "curves, and utilization ratios.")

    lm = compute_landmarks(astuple(spec), code)
    if lm:
        st.markdown(f"**Interaction landmarks** — strong axis · {code}")
        row = {}
        for name, val, kind in lm:
            if kind == "P":
                row[f"{name} [{FL}]"] = u.P_disp(val)
            else:
                row[f"{name} [{ML}]"] = u.M_disp(val)
        st.dataframe(pd.DataFrame([row]), hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.4g") for c in row})
    else:
        st.caption("Prestressed section — see Verification for the P_o / "
                   "pure-tension axial landmarks.")

    # ---- 2-D slice insets + value tables (nominal + φ-reduced design) ----
    st.divider()

    # P-M curve data (raw, neutral-axis-depth sampled) with design values
    phi_pm = rawPhi[i_ang]
    P_pm, Mz_pm, My_pm = rawP[i_ang], rawMz[i_ang], rawMy[i_ang]
    pm_df = pd.DataFrame({
        "seq": range(nd), "c": rawC[i_ang], "eps_t": rawEps[i_ang],
        "phi": phi_pm, "P": P_pm, "phiP": phi_pm * P_pm,
        "M": np.hypot(Mz_pm, My_pm), "phiM": phi_pm * np.hypot(Mz_pm, My_pm),
        "Mz": Mz_pm, "My": My_pm})

    # M-M curve data (constant axial force) with design values
    zc, yc = np.array(ringz), np.array(ringy)
    phi_mm = np.array([float(np.interp(pval, Pl, Phi[:, i])) for i in range(na)])
    mm_df = pd.DataFrame({
        "angle": np.round(thetas, 1), "phi": phi_mm, "Mz": zc, "My": yc,
        "M": np.hypot(zc, yc), "phiMz": phi_mm * zc, "phiMy": phi_mm * yc})

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"**P-M curve at {int(thetas[i_ang])}°**")
        base = alt.Chart(pm_df)
        nom = base.mark_line(color="crimson").encode(
            x=alt.X("M:Q", title=f"M [{ML}]"),
            y=alt.Y("P:Q", title=f"P [{FL}, + compression]"), order="seq:Q")
        des = base.mark_line(color="crimson", strokeDash=[5, 4],
                             opacity=0.5).encode(
            x="phiM:Q", y="phiP:Q", order="seq:Q")
        pts = base.mark_point(color="crimson", size=20, filled=True).encode(
            x="M:Q", y="P:Q",
            tooltip=[alt.Tooltip("P:Q", format=".4g"),
                     alt.Tooltip("phiP:Q", format=".4g", title="φP"),
                     alt.Tooltip("M:Q", format=".4g"),
                     alt.Tooltip("phiM:Q", format=".4g", title="φM"),
                     alt.Tooltip("c:Q", format=".4g", title=f"c [{LL}]")])
        rule = alt.Chart(pd.DataFrame({"P": [pval]})).mark_rule(
            color="orange", strokeDash=[4, 3]).encode(y="P:Q")
        st.altair_chart((nom + des + pts + rule).properties(height=340))
        st.caption("Solid = nominal · dashed = φ-reduced design")
    with d2:
        st.markdown(f"**M-M curve at P = {pval:.4g} {FL}**")
        mmax = max(1e-9, float(np.max(np.abs(np.concatenate([zc, yc]))))) * 1.15
        nom_df = pd.DataFrame({"Mz": list(zc) + [zc[0]],
                               "My": list(yc) + [yc[0]], "seq": range(na + 1)})
        des_df = pd.DataFrame({"Mz": list(phi_mm * zc) + [float(phi_mm[0] * zc[0])],
                               "My": list(phi_mm * yc) + [float(phi_mm[0] * yc[0])],
                               "seq": range(na + 1)})
        nom_l = alt.Chart(nom_df).mark_line(color="orange").encode(
            x=alt.X("Mz:Q", title=f"Mz [{ML}]",
                    scale=alt.Scale(domain=[-mmax, mmax])),
            y=alt.Y("My:Q", title=f"My [{ML}]",
                    scale=alt.Scale(domain=[-mmax, mmax])), order="seq:Q")
        des_l = alt.Chart(des_df).mark_line(color="orange", strokeDash=[5, 4],
                                            opacity=0.5).encode(
            x="Mz:Q", y="My:Q", order="seq:Q")
        pts_l = alt.Chart(nom_df).mark_point(color="orange", size=20,
                                             filled=True).encode(
            x="Mz:Q", y="My:Q",
            tooltip=[alt.Tooltip("Mz:Q", format=".4g"),
                     alt.Tooltip("My:Q", format=".4g")])
        st.altair_chart((nom_l + des_l + pts_l).properties(height=340, width=340))
        st.caption("Solid = nominal · dashed = φ-reduced design")

    # full-width value tables (all columns; hover top-right to download CSV)
    st.markdown(f"**P-M values at {int(thetas[i_ang])}°** — nominal + φ-design")
    pm_tbl = pm_df[["c", "eps_t", "phi", "P", "phiP",
                    "M", "phiM", "Mz", "My"]].copy()
    pm_tbl.insert(0, "pt", range(1, len(pm_tbl) + 1))
    st.dataframe(pm_tbl, hide_index=True, height=300, column_config={
        "pt": st.column_config.NumberColumn("#", width="small"),
        "c": st.column_config.NumberColumn(f"c [{LL}]", format="%.4g"),
        "eps_t": st.column_config.NumberColumn("εt", format="%.4f"),
        "phi": st.column_config.NumberColumn("φ", format="%.3f"),
        "P": st.column_config.NumberColumn(f"P [{FL}]", format="%.4g"),
        "phiP": st.column_config.NumberColumn(f"φP [{FL}]", format="%.4g"),
        "M": st.column_config.NumberColumn(f"M [{ML}]", format="%.4g"),
        "phiM": st.column_config.NumberColumn(f"φM [{ML}]", format="%.4g"),
        "Mz": st.column_config.NumberColumn(f"Mz [{ML}]", format="%.4g"),
        "My": st.column_config.NumberColumn(f"My [{ML}]", format="%.4g")})
    st.markdown(f"**M-M values at P = {pval:.4g} {FL}** — nominal + φ-design")
    mm_tbl = mm_df[["angle", "phi", "Mz", "My", "M", "phiMz", "phiMy"]].copy()
    mm_tbl.insert(0, "pt", range(1, len(mm_tbl) + 1))
    st.dataframe(mm_tbl, hide_index=True, height=300, column_config={
        "pt": st.column_config.NumberColumn("#", width="small"),
        "angle": st.column_config.NumberColumn("θ [deg]", format="%.1f"),
        "phi": st.column_config.NumberColumn("φ", format="%.3f"),
        "Mz": st.column_config.NumberColumn(f"Mz [{ML}]", format="%.4g"),
        "My": st.column_config.NumberColumn(f"My [{ML}]", format="%.4g"),
        "M": st.column_config.NumberColumn(f"M [{ML}]", format="%.4g"),
        "phiMz": st.column_config.NumberColumn(f"φMz [{ML}]", format="%.4g"),
        "phiMy": st.column_config.NumberColumn(f"φMy [{ML}]", format="%.4g")})


def tab_mphi(spec, u):
    if not _calc_gate():
        return
    if not _ok_to_analyze(spec):
        return
    cP, cA = st.columns([3, 2])
    with cP:
        p_max = float(u.P_disp(8000.0))
        p_user = st.slider(f"Axial load P [{u.Fl}, + compression]",
                           0.0, p_max, 0.0, p_max / 100.0)
    with cA:
        na_angle = st.number_input(
            "Neutral-axis angle θ [deg]", -180.0, 180.0, 0.0, 5.0,
            format="%.1f",
            help="Incline the bending (neutral) axis by θ from the section "
                 "z-axis. 0° = strong-axis, 90° = weak-axis. Rotates the "
                 "whole section; M is reported about the inclined axis.")
    p_kn = p_user * u.fN / 1e3
    m = compute_mphi(astuple(spec), float(p_kn), float(na_angle),
                     _mats_frozen())

    # milestone markers (a=crack, b=first yield, d=ultimate, f=idealized)
    marks = [{"label": ms["label"], "state": ms["state"],
              "kappa": u.curv_disp(ms["kappa"]), "M": u.M_disp(ms["M"])}
             for ms in m["milestones"]]
    if m["ideal"]:
        marks.append({"label": "f", "state": "Idealized yield",
                      "kappa": u.curv_disp(m["ideal"]["kappa"]),
                      "M": u.M_disp(m["ideal"]["M"])})

    df = pd.DataFrame({"kappa": [u.curv_disp(k) for k in m["kappa"]],
                       "M": [u.M_disp(v) for v in m["M"]]})
    line = alt.Chart(df).mark_line(color="#2ca25f").encode(
        x=alt.X("kappa:Q", title=f"curvature κ  [{u.Kl}]"),
        y=alt.Y("M:Q", title=f"moment M  [{u.Ml}]"))
    mdf = pd.DataFrame(marks)
    mpts = alt.Chart(mdf).mark_point(color="crimson", size=90,
                                     filled=True).encode(
        x="kappa:Q", y="M:Q",
        tooltip=[alt.Tooltip("label:N", title="pt"), "state:N",
                 alt.Tooltip("kappa:Q", format=".4g"),
                 alt.Tooltip("M:Q", format=".4g")])
    mtext = alt.Chart(mdf).mark_text(dy=-11, color="crimson",
                                     fontWeight="bold").encode(
        x="kappa:Q", y="M:Q", text="label:N")
    st.altair_chart((line + mpts + mtext).properties(height=420).interactive())

    cols = st.columns(4)
    cols[0].metric("M_cr", f"{u.M_disp(m['M_cr']):.4g} {u.Ml}")
    cols[1].metric("M_y", f"{u.M_disp(m['M_y']):.4g} {u.Ml}"
                   if m["M_y"] else "—")
    cols[2].metric("M_u", f"{u.M_disp(m['M_u']):.4g} {u.Ml}")
    cols[3].metric("μ_φ ductility",
                   f"{m['mu_phi']:.2f}" if m["mu_phi"] else "—")

    st.markdown("**Milestone states**")
    st.dataframe(pd.DataFrame([{
        "Point": x["label"], "State": x["state"],
        "Curvature": x["kappa"], "Moment": x["M"]} for x in marks]),
        hide_index=True, column_config={
            "Curvature": st.column_config.NumberColumn(
                f"Curvature [{u.Kl}]", format="%.4g"),
            "Moment": st.column_config.NumberColumn(
                f"Moment [{u.Ml}]", format="%.4g")})

    # ---- strain diagram at a selected point (strain(y) = eps0 - y·κ) ----
    st.markdown("**Strain diagram**")
    real = m["milestones"]
    if real:
        opts = [f"{ms['label']} · {ms['state']}" for ms in real]
        sel = st.selectbox("At point", opts, index=len(opts) - 1)
        ms = real[opts.index(sel)]
        eps0, kap = ms["eps0"], ms["kappa"]
        y_top, y_bot = m["y_top"], m["y_bot"]
        _ylab = (f"height y from centroid [{u.Ll}]" if abs(na_angle) < 1e-9
                 else f"distance ⊥ neutral axis [{u.Ll}]")
        s1, s2 = st.columns([2, 1])
        with s1:
            prof = pd.DataFrame({
                "strain": [eps0 - y_bot * kap, eps0 - y_top * kap],
                "y": [u.len_disp(y_bot * 1e3), u.len_disp(y_top * 1e3)]})
            zero = alt.Chart(pd.DataFrame({"s": [0.0]})).mark_rule(
                color="gray", strokeDash=[4, 3]).encode(x="s:Q")
            prof_l = alt.Chart(prof).mark_line(
                color="crimson", point=True).encode(
                x=alt.X("strain:Q", title="strain ε  (tension +)"),
                y=alt.Y("y:Q", title=_ylab))
            reb = pd.DataFrame({
                "strain": [eps0 - ry * kap for ry in m["rebar_ys"]],
                "y": [u.len_disp(ry * 1e3) for ry in m["rebar_ys"]]})
            reb_p = alt.Chart(reb).mark_point(
                color="navy", size=45, filled=True).encode(
                x="strain:Q", y="y:Q",
                tooltip=[alt.Tooltip("strain:Q", format="+.5f")])
            st.altair_chart((zero + prof_l + reb_p).properties(height=340))
        with s2:
            na = (y_top - eps0 / kap) if abs(kap) > 1e-9 else None
            st.metric("Curvature", f"{u.curv_disp(kap):.4g} {u.Kl}")
            st.metric("Moment", f"{u.M_disp(ms['M']):.4g} {u.Ml}")
            st.metric("Extreme concrete strain", f"{ms['eps_top']:+.5f}")
            st.metric("Max tension-steel strain", f"{ms['eps_steel']:+.5f}")
            st.metric("Neutral-axis depth",
                      f"{u.len_disp(na * 1e3):.4g} {u.Ll}"
                      if na is not None else "—")

    _ang = (f"Bending about a neutral axis at θ = {na_angle:.1f}° (M and κ "
            "are about that inclined axis). " if abs(na_angle) > 1e-9
            else "")
    if spec.kind == "Composite":
        _laws = ("Material laws come from each shape's library material "
                 "(fibre-based); each rebar arrangement uses its own steel.")
    else:
        _laws = (f"Material laws (**{spec.conc_model}** concrete + tension "
                 f"stiffening, **{spec.steel_model}** steel) are editable on "
                 "the **🧱 Materials** page.")
    st.caption(f"{_ang}Failure mode: {m['failure_mode']}. Milestones: "
               "a=crack · b=first yield · d=ultimate · f=idealized. Strain "
               f"sign: tension +. {_laws}")


def tab_verify(spec, case, u, code):
    if not _calc_gate():
        return
    if not _ok_to_analyze(spec):
        return
    if spec.kind == "Composite":
        st.info("The GSD verification set is single-material. Use the "
                "**P-M-M interaction** and **Moment-curvature** tabs for this "
                "composite section's fibre-based results.")
        return
    st.caption(f"Verification against **{code}** (this section's design "
               "code). Enter each Midas GSD value to get the % difference "
               "and pass/fail against the tolerance.")
    raw = compute_items(astuple(spec), code)

    def _conv(r):
        v, lbl = u.convert_item(r["units"], r["computed"])
        return {"code": r["code"], "quantity": r["quantity"], "units": lbl,
                "femsolver": v, "midas_gsd": float("nan"),
                "tol %": r["tol_pct"]}

    base = pd.DataFrame([_conv(r) for r in raw])
    base["midas_gsd"] = base["midas_gsd"].astype("float64")

    editor_key = (f"gsd_{st.session_state.active_section}"
                  f"_{u.force}{u.length}{u.stress}")
    edited = st.data_editor(
        base, key=editor_key, hide_index=True, width="stretch",
        disabled=["code", "quantity", "units", "femsolver", "tol %"],
        column_config={
            "femsolver": st.column_config.NumberColumn(format="%.4g"),
            "midas_gsd": st.column_config.NumberColumn(
                "Midas GSD", format="%.4g", help="Type the GSD result"),
            "tol %": st.column_config.NumberColumn(format="%.1f"),
        })

    res = edited.copy()

    def _diff(row):
        g = row["midas_gsd"]
        if pd.isna(g):
            return pd.NA
        return (abs(row["femsolver"]) * 100.0 if g == 0
                else (row["femsolver"] - g) / abs(g) * 100.0)

    res["diff %"] = res.apply(_diff, axis=1)
    res["status"] = res.apply(
        lambda r: ("" if pd.isna(r["diff %"])
                   else ("✅ pass" if abs(r["diff %"]) <= r["tol %"]
                         else "❌ FAIL")), axis=1)

    entered = int(res["midas_gsd"].notna().sum())
    if entered:
        npass = int((res["status"] == "✅ pass").sum())
        c = st.columns(3)
        c[0].metric("Entered", entered)
        c[1].metric("Pass", npass)
        c[2].metric("Fail", entered - npass)
        st.dataframe(
            res[res["midas_gsd"].notna()][
                ["code", "quantity", "units", "femsolver", "midas_gsd",
                 "diff %", "tol %", "status"]],
            hide_index=True, column_config={
                "femsolver": st.column_config.NumberColumn(format="%.4g"),
                "midas_gsd": st.column_config.NumberColumn(format="%.4g"),
                "diff %": st.column_config.NumberColumn(format="%+.2f"),
                "tol %": st.column_config.NumberColumn(format="%.1f")})

    st.subheader("Export standard files")
    items = [VerificationItem(
        section_id="LIVE", section_name=spec.kind, code=r["code"],
        quantity=r["quantity"], units=r["units"], computed=r["femsolver"],
        tol_pct=r["tol %"],
        gsd=(None if pd.isna(r["midas_gsd"]) else float(r["midas_gsd"])))
        for _, r in edited.iterrows()]
    compare_items(items)
    sjson = core.section_json_of(case)

    d = st.columns(4)
    d[0].download_button("Verification CSV", core.items_csv(items),
                         "section_verification.csv", "text/csv")
    d[1].download_button("Markdown table", to_markdown(items),
                         "section_verification.md", "text/markdown")
    d[2].download_button("Section JSON", sjson or "{}", "section.json",
                         "application/json", disabled=sjson is None)
    d[3].download_button("Geometry SVG", core.svg_of(case), "section.svg",
                         "image/svg+xml")


# ------------------------------------------------------------ main

def tab_report(spec, case, u, code):
    if not _calc_gate():
        return
    if not _ok_to_analyze(spec):
        return
    if spec.kind == "Composite":
        st.info("The one-click report is for single-material sections. This "
                "composite section's fibre-based results are in the P-M-M and "
                "Moment-curvature tabs.")
        return
    st.caption("One-click calc sheet — section, materials, reinforcement, "
               "P-M interaction, moment-curvature and the demand check. "
               "Download the HTML and **Print → Save as PDF** from your "
               "browser for a submittable report.")
    c = st.columns(4)
    title = c[0].text_input("Report title", "Section Design Report",
                            key="rep_title")
    project = c[1].text_input("Project", "", key="rep_project")
    engineer = c[2].text_input("Engineer", "", key="rep_engineer")
    job = c[3].text_input("Job no.", "", key="rep_job")
    date = st.text_input("Date", _dt.date.today().isoformat(), key="rep_date")

    mphi = compute_mphi(astuple(spec), 0.0)
    design = _dsurf().startswith("Design")
    canon = st.session_state.get("demands", {}).get(
        st.session_state.active_section, [])
    na, npl = MESH_GRID[st.session_state.get("mesh_density", "GSD default")]
    dem_res = (compute_demand(astuple(spec), code, tuple(canon), design,
                              na, npl, _dc_method()) if canon else [])
    meta = {"title": title, "project": project, "engineer": engineer,
            "job": job, "date": date,
            "surface": "φ-reduced design" if design else "nominal",
            "dc_method": {v: k for k, v in DC_METHODS.items()}[_dc_method()]}
    html = core.report_html(case, code, u, mphi=mphi,
                            demand_results=dem_res, meta=meta)
    fname = f"{st.session_state.active_section.replace(' ', '_')}_report.html"
    st.download_button("⬇ Download report (HTML)", html, file_name=fname,
                       mime="text/html", type="primary")
    if not canon:
        st.info("Tip: add load combinations in the **Analysis** controls "
                "above to include a demand/utilization check in the report.")
    b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
    st.iframe(f"data:text/html;base64,{b64}", height=1150)


# ------------------------------------------------------------ pages / router

def _simple_header(sub):
    """Lightweight app header (no active-section chips) for the list / Setup /
    Materials pages."""
    st.markdown(
        f'<div class="app-header"><div class="logo">{_LOGO}</div>'
        f'<div><div class="app-title">Section Designer</div>'
        f'<div class="app-sub">{sub}</div></div></div>',
        unsafe_allow_html=True)


def section_list_page(u, code):
    """AdSec-style Section List: 'Add new…' presets + a gallery of every
    section with a preview thumbnail. Open a section to define / analyse it."""
    _simple_header(f"Section list &middot; {len(st.session_state.sections)} "
                   f"section(s) &middot; {code}")
    st.markdown('<div class="sd-listhdr">Section list</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sd-addlbl">Add new…</div>', unsafe_allow_html=True)
    ac = st.columns(len(SECTION_PRESETS))
    for col, preset in zip(ac, SECTION_PRESETS):
        col.button(preset, width="stretch", key=f"addnew_{preset}",
                   on_click=_add_preset, args=(preset,))
    st.divider()

    widths = [1.4, 3, 2.2, 1.6, 0.7]
    h = st.columns(widths, vertical_alignment="center")
    for col, lbl in zip(h, ["Diagram", "Section", "Components", "Analysis",
                            ""]):
        col.markdown(f'<div class="sd-colhdr">{lbl}</div>',
                     unsafe_allow_html=True)

    names = list(st.session_state.sections)
    demands = st.session_state.get("demands", {})
    for name in names:
        rec = st.session_state.sections[name]
        spec = rec["spec"]
        row = st.columns(widths, vertical_alignment="center")
        with row[0]:
            svg = _thumb_svg(astuple(spec))
            st.markdown(f'<div class="sd-thumb">{svg or "—"}</div>',
                        unsafe_allow_html=True)
        with row[1]:
            st.markdown(
                f'<div class="sd-secname">{name}</div>'
                f'<div class="sd-seckind">{spec.kind}</div>',
                unsafe_allow_html=True)
            st.button("Open ›", key=f"open_{name}", type="primary",
                      on_click=_open_section, args=(name,))
        with row[2]:
            try:
                case = core.build_case(spec)
                n_bar = len(case.section.reinforcement.bars) \
                    if case.section.reinforcement else 0
                n_ten = len(case.section.prestress.tendons) \
                    if getattr(case.section, "prestress", None) else 0
            except Exception:
                n_bar = n_ten = 0
            st.markdown(
                f'<div class="sd-rowmeta">{n_bar} bars &middot; {n_ten} '
                f'tendons<br>{rec.get("conc_mat", "—")}</div>',
                unsafe_allow_html=True)
        with row[3]:
            ncombo = len(demands.get(name, []))
            ready = st.session_state.get("calc_sigs", {}).get(name) is not None
            st.markdown(
                f'<div class="sd-rowmeta">{ncombo} combo(s)<br>'
                f'{"✓ computed" if ready else "not run"}</div>',
                unsafe_allow_html=True)
        with row[4]:
            with st.popover("⋮"):
                st.text_input("Rename", value=name, key=f"rename_{name}",
                              on_change=_rename_section, args=(name,))
                st.button("Duplicate", key=f"dup_{name}", width="stretch",
                          on_click=_dup_section, args=(name,))
                st.button("Delete", key=f"del_{name}", width="stretch",
                          on_click=_del_section, args=(name,),
                          disabled=len(names) <= 1)
    st.caption("Tip: units & design code live on the **⚙ Setup** page; shared "
               "materials on the **🧱 Materials** page.")


def section_detail_page(rec, u, code):
    """One open section: Definition and Analysis tabs (AdSec-style), with a
    back link to the section list. Every input is inline — no sidebar."""
    spec = rec["spec"]
    st.columns([1.4, 6])[0].button(
        "‹ Section list", key="backtolist", width="stretch",
        on_click=_back_to_list)
    _header(spec, code, u)

    def_tab, an_tab = st.tabs([":material/edit_document: Definition",
                               ":material/analytics: Analysis"])
    with def_tab:
        tab_definition(core.build_case(rec["spec"]), u, code)
    # the Definition tab may have mutated the spec -> rebuild so Analysis sees
    # the latest geometry / reinforcement this same run.
    spec = rec["spec"]
    case = core.build_case(spec)
    with an_tab:
        _analysis_inputs(rec, u, code)
        st.divider()
        pmm_tab, mphi_tab, v_tab, r_tab = st.tabs(
            [":material/ssid_chart: P-M-M interaction",
             ":material/show_chart: Moment-curvature",
             ":material/fact_check: Verification",
             ":material/description: Report"])
        with pmm_tab:
            tab_surface3d(spec, u, code)
        with mphi_tab:
            tab_mphi(spec, u)
        with v_tab:
            tab_verify(spec, case, u, code)
        with r_tab:
            tab_report(spec, case, u, code)
    _statusbar(spec, case, code, u)


u, code = _prep()
_sidebar_nav_and_project(u)
mode = st.session_state.app_mode

if mode == "Setup":
    _simple_header("Setup &middot; project units + design code")
    tab_setup()                               # project-level units + code
elif mode == "Materials":
    _simple_header("Materials library")
    tab_materials(u)                          # dedicated Materials page
else:                                          # Sections
    if st.session_state.get("section_view", "list") == "list":
        section_list_page(u, code)
    else:
        active = st.session_state.active_section
        if active not in st.session_state.sections:
            st.session_state.section_view = "list"
            st.rerun()
        section_detail_page(st.session_state.sections[active], u, code)
