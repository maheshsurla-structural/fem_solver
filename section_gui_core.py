"""Pure (no-Streamlit) core for the Section Designer GUI.

Builds sections from a hashable :class:`Spec`, runs the femsolver engine,
and returns plain data (dicts / lists / arrays). Kept free of any
Streamlit import so it can be unit-tested and reused headless; the
Streamlit layer (`streamlit_app.py`) only adds caching + widgets.
"""
from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass, astuple, replace
from pathlib import Path
from typing import Optional

import numpy as np

from femsolver.design.concrete import (
    ConcreteMaterial,
    biaxial_pmm_point_aashto,
    biaxial_pmm_point_ec2,
    biaxial_pmm_point_is456,
    biaxial_pmm_surface_aashto,
    biaxial_pmm_surface_ec2,
    biaxial_pmm_surface_is456,
    moment_curvature,
)
from femsolver.materials.uniaxial import (
    ConcreteKentPark, ConcreteMander, ConcreteParabolaRectangle,
    ConcreteTensionStiffening, ConcreteTrilinear, UniaxialBilinear,
    UniaxialMenegottoPinto, UniaxialReinforcingSteel)
from femsolver.sections import (
    PrestressTendon,
    RebarBar,
    ReinforcementLayout,
    TendonLayout,
    circular_section,
    custom_polygon_section,
    hollow_rect_section,
    rc_rectangular_section,
    section_to_json,
    section_to_svg,
    t_section,
)
from femsolver.sections.section import _discretize_polygon_to_fibers
# U1: section-analysis core lifted to the engine (femsolver.sections.analysis).
# These names are re-exported here so the Streamlit/desktop GUI keep working unchanged.
from femsolver.sections.analysis import (  # noqa: E402,F401
    _GL, _has_confinement_params, _mphi_crossing, _mphi_ctrl,
    _mphi_kappas, _mphi_stop, _rotate_case, _width_bands,
    concrete_uniaxial_from, ec2_fctm, exact_mphi, mander_confinement,
    mphi_data, pmm_slice, section_pm_slice, steel_uniaxial_from,
)
from femsolver.sections.response.fiber import (Fiber, FiberSection2D,
                                               polar_cells, polar_divisions)
from femsolver.materials.uniaxial import PrestressedUniaxial
from femsolver.benchmarks.section_designer import (
    SectionCase,
    VerificationItem,
    _moment_curvature_items,
    _pmm_items_for_code,
    _psc_axial_items,
    _recenter_outline,
    _section_property_items,
    export_csv,
    strong_axis_landmarks,
    E_S,
)

CODES = ["AASHTO LRFD 2024", "Eurocode 2", "IS 456:2000"]

# One-click constitutive presets for the moment-curvature material laws.
# Concrete: eps_c0 (peak strain), eps_cu (crushing strain), fcu_ratio
# (residual/peak stress -> 1.0 = parabola-rectangle plateau, <1 = a
# descending Kent-Park branch), fr_coeff (f_r = k*sqrt(f'c[MPa]) MPa).
# Steel: Es, steel_b (0 = elastic-perfectly-plastic, the design idealization).
MATERIAL_PRESETS = {
    "ACI": dict(eps_c0=0.002, eps_cu=0.003, fcu_ratio=0.2,
                fr_coeff=0.62, fr_model="sqrt", eps_decay=0.001,
                Es=200e9, steel_b=0.0),
    "EC2": dict(eps_c0=0.002, eps_cu=0.0035, fcu_ratio=1.0,
                fr_coeff=0.53, fr_model="ec2", eps_decay=0.001,
                Es=200e9, steel_b=0.0),
    "IS": dict(eps_c0=0.002, eps_cu=0.0035, fcu_ratio=1.0,
               fr_coeff=0.70, fr_model="sqrt", eps_decay=0.001,
               Es=200e9, steel_b=0.0),
}

# Standard material grades offered per design-code family. Picking a grade sets
# the strength; the code-appropriate concrete stress-strain model rides along,
# and the family's constitutive preset (MATERIAL_PRESETS) supplies the curve /
# rupture parameters. Concrete strengths are the cylinder f'c / f_ck (Pa); ACI
# grades are the common psi classes (with the MPa equivalent in the label).
_PSI = 6894.757  # Pa per psi
CODE_GRADES = {
    "ACI": {
        "conc_model": "Mander",
        "concrete": {
            "3000 psi (20.7 MPa)": 3000 * _PSI,
            "4000 psi (27.6 MPa)": 4000 * _PSI,
            "5000 psi (34.5 MPa)": 5000 * _PSI,
            "6000 psi (41.4 MPa)": 6000 * _PSI,
            "8000 psi (55.2 MPa)": 8000 * _PSI,
            "10000 psi (69.0 MPa)": 10000 * _PSI,
        },
        "steel": {
            "Grade 40 (280)": 280e6, "Grade 60 (420)": 420e6,
            "Grade 75 (520)": 520e6, "Grade 80 (550)": 550e6,
        },
    },
    "EC2": {
        "conc_model": "Parabola-rectangle (EC2)",
        "concrete": {
            "C20/25": 20e6, "C25/30": 25e6, "C30/37": 30e6, "C35/45": 35e6,
            "C40/50": 40e6, "C45/55": 45e6, "C50/60": 50e6,
        },
        "steel": {
            "B500A": 500e6, "B500B": 500e6, "B500C": 500e6, "B450C": 450e6,
        },
    },
    "IS": {
        "conc_model": "Kent-Park",
        "concrete": {
            "M20": 20e6, "M25": 25e6, "M30": 30e6, "M35": 35e6,
            "M40": 40e6, "M45": 45e6, "M50": 50e6,
        },
        "steel": {
            "Fe415": 415e6, "Fe500": 500e6, "Fe550": 550e6, "Fe600": 600e6,
        },
    },
}
# The (concrete, steel) grade a section adopts when it first takes on a family.
CODE_DEFAULT_GRADE = {"ACI": ("4000 psi (27.6 MPa)", "Grade 60 (420)"),
                      "EC2": ("C30/37", "B500B"),
                      "IS": ("M30", "Fe500")}


KINDS = ["Rectangular", "Circular", "L-shape", "T-shape", "Hollow box",
         "PSC girder", "Custom", "Composite"]

# Constitutive models for the moment-curvature analysis, each mapped to a real
# femsolver engine class (GSD's other models would need new engine classes).
CONC_MODELS = ["Kent-Park", "Mander", "Parabola-rectangle (EC2)", "Trilinear"]
STEEL_MODELS = ["Bilinear", "Elastic - perfectly plastic", "Menegotto-Pinto",
                "Park strain-hardening"]
BAR_SIZES = {  # label -> diameter (m)
    "12 mm": 0.012, "16 mm": 0.016, "20 mm": 0.020,
    "25 mm": 0.025, "32 mm": 0.032, "40 mm": 0.040,
}

# ---- unit system (Midas-GSD-style force + length, plus a stress unit) ----
FORCE_N = {  # newtons per 1 unit of force
    "kgf": 9.80665, "tonf": 9806.65, "N": 1.0, "kN": 1000.0,
    "lbf": 4.4482216, "kip": 4448.2216,
}
LENGTH_M = {  # metres per 1 unit of length
    "mm": 1e-3, "cm": 1e-2, "m": 1.0, "in": 0.0254, "ft": 0.3048,
}
STRESS_PA = {  # pascals per 1 unit of stress
    "MPa": 1e6, "N/mm²": 1e6, "kPa": 1e3, "Pa": 1.0,
    "ksi": 6.8947573e6, "psi": 6894.7573, "kgf/cm²": 98066.5,
}


class Units:
    """Convert between the engine's SI / canonical values and a user-chosen
    display system, and supply the matching unit labels.

    The core returns values in *canonical* display units (force ``kN``,
    moment ``kN.m``, length ``mm``, area ``mm^2``, inertia ``mm^4``,
    curvature ``1/m``); the ``*_disp`` helpers convert those to the user's
    units. The ``to_m`` / ``to_Pa`` (and ``from_*``) helpers move sidebar
    inputs to/from SI for the :class:`Spec`. Moment, area and inertia are
    derived from length exactly the way Midas GSD does (moment = F.L,
    area = L^2, inertia = L^4).
    """

    def __init__(self, force: str = "kN", length: str = "m",
                 stress: str = "MPa"):
        self.force = force
        self.length = length
        self.stress = stress
        self.fN = FORCE_N[force]
        self.lM = LENGTH_M[length]
        self.sPa = STRESS_PA[stress]

    # ---- canonical core output -> user display ----
    def P_disp(self, kN):
        return kN * 1e3 / self.fN

    def M_disp(self, kNm):
        return kNm * 1e3 / (self.fN * self.lM)

    def len_disp(self, mm):
        return mm * 1e-3 / self.lM

    def area_disp(self, mm2):
        return mm2 * 1e-6 / self.lM ** 2

    def inertia_disp(self, mm4):
        return mm4 * 1e-12 / self.lM ** 4

    def curv_disp(self, per_m):
        return per_m * self.lM

    # ---- sidebar input <-> SI ----
    def to_m(self, v):
        return v * self.lM

    def from_m(self, m):
        return m / self.lM

    def to_Pa(self, v):
        return v * self.sPa

    def from_Pa(self, pa):
        return pa / self.sPa

    def len_step(self):
        """A sensible number-input step (~5 mm) in the current length unit."""
        return round(self.from_m(0.005), 4)

    # ---- labels ----
    @property
    def Fl(self):
        return self.force

    @property
    def Ll(self):
        return self.length

    @property
    def Ml(self):
        return f"{self.force}·{self.length}"

    @property
    def Al(self):
        return f"{self.length}²"

    @property
    def Il(self):
        return f"{self.length}⁴"

    @property
    def Kl(self):
        return f"1/{self.length}"

    @property
    def Sl(self):
        return self.stress

    def convert_item(self, units: str, value: float):
        """Convert a benchmark VerificationItem's (canonical unit, value)
        to (user value, user unit label). Passes through dimensionless."""
        table = {
            "kN": (self.P_disp, self.Fl),
            "kN.m": (self.M_disp, self.Ml),
            "kN·m": (self.M_disp, self.Ml),
            "mm^2": (self.area_disp, self.Al),
            "mm^4": (self.inertia_disp, self.Il),
            "mm": (self.len_disp, self.Ll),
            "MPa": (lambda v: v * 1e6 / self.sPa, self.Sl),
            "1/m": (self.curv_disp, self.Kl),
        }
        if units in table:
            fn, lbl = table[units]
            return fn(value), lbl
        return value, units


@dataclass(frozen=True)
class Spec:
    """Hashable description of a section (drives caching)."""
    kind: str = "Rectangular"
    b: float = 0.40
    h: float = 0.60
    D: float = 0.60
    leg: float = 0.80
    thick: float = 0.40
    t_f: float = 0.15
    t_w: float = 0.15
    wall_t: float = 0.10
    fc: float = 30.0e6
    fy: float = 500.0e6
    bar_dia: float = 0.025
    cover: float = 0.05
    n_top: int = 3
    n_bot: int = 3
    n_side: int = 1
    n_perim: int = 12
    spiral: bool = False
    n_strand: int = 6
    strand_area: float = 140e-6
    f_pe: float = 1100e6
    strand_y: float = -0.38
    # prestressing-steel (strand) material — the tendon base uniaxial law.
    # Defaults reproduce the historical hardcoded strand (E_p 195 GPa, f_py
    # 1675 MPa, b 0.005), so existing specs are unchanged.
    Ep: float = 195.0e9          # prestressing-steel elastic modulus (Pa)
    fpy: float = 1675.0e6        # prestressing-steel yield (Pa)
    ps_b: float = 0.005          # prestressing-steel strain-hardening ratio
    # nonlinear constitutive (moment-curvature / strain analysis)
    eps_c0: float = 0.002        # concrete peak-compression strain
    eps_cu: float = 0.0035       # concrete crushing strain
    fcu_ratio: float = 0.4       # residual crushing stress / f'c
    fr_coeff: float = 0.62       # rupture modulus f_r = k*sqrt(f'c[MPa]) MPa
    fr_model: str = "sqrt"       # "sqrt" (k*sqrt f'c) | "ec2" (0.3 f_ck^2/3)
    eps_decay: float = 1.0e-3    # tension-stiffening decay strain
    Es: float = 200.0e9          # steel elastic modulus (Pa)
    steel_b: float = 0.01        # steel strain-hardening ratio (bilinear/MP)
    steel_fu_ratio: float = 1.5  # f_su/f_y  (Park strain-hardening)
    steel_eps_sh: float = 0.008  # strain-hardening onset strain (Park)
    steel_eps_su: float = 0.10   # strain at ultimate stress (Park)
    kappa_max: float = 0.06      # M-phi curvature sweep limit (1/m)
    conc_model: str = "Kent-Park"   # concrete compression model (CONC_MODELS)
    conc_f1_ratio: float = 0.4      # trilinear first-knee stress / f'c
    steel_model: str = "Bilinear"   # rebar stress-strain model (STEEL_MODELS)
    # custom section (kind == "Custom"): explicit polygon + bars, in metres,
    # stored as nested tuples so the Spec stays hashable (drives caching).
    custom_outline: tuple = ()   # ((z, y), ...) exterior ring
    custom_holes: tuple = ()     # (((z, y), ...), ...) optional holes
    custom_bars: tuple = ()      # ((z, y, dia), ...) individual bars
    # Rebar / tendon ARRANGEMENTS (point/line/arc/rect/perimeter generators),
    # ADDED to the parametric/custom bars. Each is a nested tuple:
    #   rebar:  (type, dia,  mat, (params...))
    #   tendon: (type, area, f_pe, mat, (params...))
    rebar_arr: tuple = ()
    tendon_arr: tuple = ()
    # COMPOSITE section (kind == "Composite"): an ordered list of material
    # shapes ``((outline, mat_kv), ...)`` — outline = ((z,y),...) in metres,
    # mat_kv = tuple(sorted(material_props.items())) (self-contained, hashable;
    # later shapes displace earlier ones on overlap). Rebar comes from the
    # rebar arrangements + the section steel.
    shapes: tuple = ()
    # AdSec-style reinforcement GROUPS — the newer model (each row a group):
    #   (type, pattern, position[, material]) where type is a REBAR_GROUP_TYPE
    #   (Top/Bottom/Sides/Link/Perimeter/Line/Arc/Single), pattern is a bar
    #   description ("4B25", "B16-200", "4#8", "#5-150") and position is a
    #   free-text coord string. When non-empty, these supersede the parametric
    #   counts + rebar_arr for non-Custom/Composite kinds (see build_case).
    rebar_groups: tuple = ()
    # Variable cover (AdSec): when cover_variable, Top/Bottom/Sides use their
    # own covers; else the uniform ``cover`` applies to all faces.
    cover_variable: bool = False
    cover_top: float = 0.05
    cover_bot: float = 0.05
    cover_side: float = 0.05
    # Mander confinement source: by default confinement is auto-read from the
    # section's tie (Link) group + geometry. ``conf_override`` supplies the
    # confinement inputs directly, stored as sorted (key, value) pairs of the
    # ``conf_*`` dict in ``conf_manual``. ``conf_out`` carries Midas-GSD-style
    # per-output overrides (name, value pairs -> ``ov_<name>``); ``conf_ecu_method``
    # is "energy" (area balance) or "experiment" (Mander/Priestley formula).
    conf_override: bool = False
    conf_manual: tuple = ()
    conf_out: tuple = ()
    conf_ecu_method: str = "energy"

    def __post_init__(self):
        # Coerce the variable-length nested-tuple fields even if they arrive as
        # lists (JSON load, widget read-back), so the frozen Spec is always
        # hashable for st.cache_data / astuple round-trips.
        for f in ("custom_outline", "custom_holes", "custom_bars",
                  "rebar_arr", "tendon_arr", "shapes", "rebar_groups",
                  "conf_manual", "conf_out"):
            object.__setattr__(self, f, _deep_tuple(getattr(self, f)))


def _deep_tuple(x):
    """Recursively convert lists/tuples to nested tuples (hashable)."""
    if isinstance(x, (list, tuple)):
        return tuple(_deep_tuple(e) for e in x)
    return x


def bar_area(dia: float) -> float:
    return math.pi / 4.0 * dia ** 2


def perimeter_bars(outline, n: int, cover: float, dia: float) -> tuple:
    """Generate ``n`` bars of diameter ``dia`` evenly around the perimeter of
    the polygon ``outline`` (list of (z, y) in metres), inset by ``cover``.
    Returns ((z, y, dia), ...) in the same (un-recentred) frame as ``outline``
    — a convenience for the custom-section editor's 'fill perimeter' helper."""
    pts = [(float(z), float(y)) for (z, y) in outline]
    if len(pts) < 3 or n < 1:
        return ()
    sec = custom_polygon_section(
        outline=pts, material=ConcreteMaterial(fc_prime=30e6, fy=500e6),
        name="tmp")
    rl = ReinforcementLayout.from_perimeter(
        sec, n_bars=int(n), bar_area=bar_area(dia), cover=float(cover),
        designation=f"{dia*1e3:.0f}mm")
    return tuple((float(b.z), float(b.y), float(dia)) for b in rl.bars)


# ------------------------------------------------- rebar/tendon arrangements

# GSD-style rebar/tendon layout generators. Each arrangement is a nested tuple
# (see Spec.rebar_arr / tendon_arr) that expands to a list of (z, y) points.
REBAR_ARR_TYPES = ["Point", "Line", "Arc", "Rectangle", "Perimeter"]
TENDON_ARR_TYPES = ["Point", "Line", "Arc"]


def _rect_perimeter_points(cz, cy, w, h, nz, ny):
    """Points around a rectangle (centre cz,cy; width w in z, height h in y):
    ``nz`` per horizontal edge, ``ny`` per vertical edge, corners de-duplicated."""
    z0, z1 = cz - w / 2.0, cz + w / 2.0
    y0, y1 = cy - h / 2.0, cy + h / 2.0
    seen = {}
    nz, ny = max(int(nz), 1), max(int(ny), 1)
    for yy in (y0, y1):
        zs = [cz] if nz == 1 else [z0 + (z1 - z0) * i / (nz - 1)
                                   for i in range(nz)]
        for zz in zs:
            seen[(round(zz, 9), round(yy, 9))] = (zz, yy)
    for zz in (z0, z1):
        ys = [cy] if ny == 1 else [y0 + (y1 - y0) * i / (ny - 1)
                                   for i in range(ny)]
        for yy in ys:
            seen[(round(zz, 9), round(yy, 9))] = (zz, yy)
    return list(seen.values())


def _arrangement_points(typ, params, section):
    """Expand one arrangement's parameter tuple to a list of (z, y) points."""
    p = [float(v) for v in params]
    if typ == "point":
        return [(p[0], p[1])]
    if typ == "line":
        n, z1, y1, z2, y2 = int(p[0]), p[1], p[2], p[3], p[4]
        if n <= 1:
            return [((z1 + z2) / 2, (y1 + y2) / 2)]
        return [(z1 + (z2 - z1) * i / (n - 1), y1 + (y2 - y1) * i / (n - 1))
                for i in range(n)]
    if typ == "arc":
        n, cz, cy, r, a1, a2 = int(p[0]), p[1], p[2], p[3], p[4], p[5]
        a1r, a2r = math.radians(a1), math.radians(a2)
        if n <= 1:
            return [(cz + r * math.cos(a1r), cy + r * math.sin(a1r))]
        return [(cz + r * math.cos(a1r + (a2r - a1r) * i / (n - 1)),
                 cy + r * math.sin(a1r + (a2r - a1r) * i / (n - 1)))
                for i in range(n)]
    if typ == "rect":
        nz, ny, cz, cy, w, h = int(p[0]), int(p[1]), p[2], p[3], p[4], p[5]
        return _rect_perimeter_points(cz, cy, w, h, nz, ny)
    if typ == "perim":
        n, cover, dia = int(p[0]), p[1], p[2]
        outline = section.geometry.polygon.exterior.coords[:-1]
        return [(bz, by) for (bz, by, _d) in
                perimeter_bars(outline, n, cover, dia)]
    return []


def bars_from_arrangements(arrangements, section, default_mat=None):
    """Build :class:`RebarBar` objects from the rebar arrangements, in the
    section's (already-centred) frame. ``arrangements`` items are
    ``(type, dia, mat, (params...))``. ``mat`` may be a steel-material
    key-value tuple ``tuple(sorted(props.items()))`` — then that arrangement's
    bars get their own steel law (mixed-material reinforcement); an empty ``""``
    or a bare name falls back to ``default_mat`` (the section steel)."""
    out = []
    for arr in arrangements:
        typ, dia, mat = arr[0], float(arr[1]), arr[2]
        bar_mat = default_mat
        if isinstance(mat, (tuple, list)) and mat:
            bar_mat = steel_uniaxial_from(dict(mat))
        params = list(arr[3])
        if typ == "perim":                       # perimeter also needs dia
            params = list(params) + [dia]
        area = bar_area(dia)
        desig = f"{dia * 1e3:.0f}mm"
        for (z, y) in _arrangement_points(typ, params, section):
            out.append(RebarBar(z=float(z), y=float(y), area=area,
                                material=bar_mat, designation=desig))
    return out


# US (imperial) reinforcing-bar nominal diameters [mm], for #-size notation.
US_BAR_MM = {
    3: 9.525, 4: 12.7, 5: 15.875, 6: 19.05, 7: 22.225, 8: 25.4,
    9: 28.651, 10: 32.258, 11: 35.814, 14: 43.0, 18: 57.33,
}
# AdSec-style reinforcement GROUP types (the newer 'Groups' table model).
REBAR_GROUP_TYPES_RECT = ["Link", "Top", "Bottom", "Sides",
                          "Perimeter", "Line", "Arc", "Single"]
REBAR_GROUP_TYPES_ROUND = ["Perimeter", "Line", "Arc", "Single"]
REBAR_GROUP_TYPES = REBAR_GROUP_TYPES_RECT          # back-compat alias


def parse_bar_desc(desc: str, notation: str = "any"):
    """Parse an AdSec-style bar description into ``(count, dia_m, spacing_m)``:
    ``'4B25'`` -> (4, 0.025, None); ``'B16-200'`` -> (None, 0.016, 0.200);
    ``'B10'`` -> (1, 0.010, None); ``'4#8'`` -> (4, 0.0254, None);
    ``'#5-150'`` -> (None, 0.0159, 0.150). ``notation`` enforces a rebar
    standard: ``"us"`` accepts only US ``#``-sizes, ``"metric"`` only ``B``/⌀mm,
    ``"any"`` both. Raises ValueError on a bad string or notation mismatch."""
    s = str(desc).strip().upper().replace(" ", "")
    if notation == "us" and "B" in s:
        raise ValueError(f"{desc!r} is a metric bar — the US rebar standard "
                         "uses US sizes, e.g. '4#8' or '#5-150'.")
    if notation == "metric" and "#" in s:
        raise ValueError(f"{desc!r} is a US #-size — the metric rebar standard "
                         "uses ⌀mm sizes, e.g. '4B25' or 'B16-200'.")

    def _us(n):
        try:
            return US_BAR_MM[int(n)] / 1e3
        except (KeyError, ValueError):
            raise ValueError(f"unknown US bar size #{n}")

    m = re.fullmatch(r"(\d+)B(\d+(?:\.\d+)?)", s)               # nBd
    if m:
        return int(m.group(1)), float(m.group(2)) / 1e3, None
    m = re.fullmatch(r"(\d+)#(\d+)", s)                         # n#N (US)
    if m:
        return int(m.group(1)), _us(m.group(2)), None
    m = re.fullmatch(r"B(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", s)    # Bd-s
    if m:
        return None, float(m.group(1)) / 1e3, float(m.group(2)) / 1e3
    m = re.fullmatch(r"#(\d+)-(\d+(?:\.\d+)?)", s)              # #N-s (US)
    if m:
        return None, _us(m.group(1)), float(m.group(2)) / 1e3
    m = re.fullmatch(r"B(\d+(?:\.\d+)?)", s)                    # Bd
    if m:
        return 1, float(m.group(1)) / 1e3, None
    m = re.fullmatch(r"#(\d+)", s)                             # #N (US)
    if m:
        return 1, _us(m.group(1)), None
    raise ValueError(f"unrecognised bar description {desc!r} "
                     "(use e.g. '4B25', 'B16-200', '4#8' or '#5-150')")


def _parse_positions(pos):
    """Free-text Position(s) -> flat [floats] (mm lengths / deg angles)."""
    out = []
    for t in re.split(r"[,;\s]+", str(pos or "").strip()):
        if t:
            try:
                out.append(float(t))
            except ValueError:
                pass
    return out


def _line_pts(z1, y1, z2, y2, n):
    if n <= 1:
        return [((z1 + z2) / 2.0, (y1 + y2) / 2.0)]
    return [(z1 + (z2 - z1) * i / (n - 1), y1 + (y2 - y1) * i / (n - 1))
            for i in range(n)]


def _arc_pts(cz, cy, r, a1, a2, n):
    a1r, a2r = math.radians(a1), math.radians(a2)
    if n <= 1:
        return [(cz + r * math.cos(a1r), cy + r * math.sin(a1r))]
    return [(cz + r * math.cos(a1r + (a2r - a1r) * i / (n - 1)),
             cy + r * math.sin(a1r + (a2r - a1r) * i / (n - 1)))
            for i in range(n)]


def bars_from_groups(groups, sec, cover, rect=None, face_covers=None):
    """Build :class:`RebarBar` objects from AdSec-style reinforcement groups —
    each ``(type, description[, position[, material]])``. Position numbers are
    mm / degrees. Face types Top/Bottom/Sides (Link is skipped as a shear tie)
    need ``rect=(b, h)`` (m); Perimeter rings the outline at ``cover``; Line
    needs ``z1,y1;z2,y2`` and Arc ``cz,cy,r,a1,a2``. A steel-material key-value
    tuple in slot 3 gives that group its own law (mixed-material)."""
    bars = []

    def _add(z, y, dia, mat):
        bars.append(RebarBar(z=float(z), y=float(y), area=bar_area(dia),
                             material=mat, designation=f"{dia * 1e3:.0f}mm"))

    for g in groups:
        typ = g[0]
        desc = g[1] if len(g) > 1 else ""
        pos = g[2] if len(g) > 2 else ""
        matslot = g[3] if len(g) > 3 else ""
        mat = (steel_uniaxial_from(dict(matslot))
               if isinstance(matslot, (tuple, list)) and matslot else None)
        try:
            n, dia, sp = parse_bar_desc(desc)
        except ValueError:
            continue
        p = _parse_positions(pos)
        if typ == "Link":
            continue
        if typ in ("Top", "Bottom", "Sides") and rect:
            b, h = rect
            c_top, c_bot, c_side = face_covers or (cover, cover, cover)
            zc = b / 2.0 - c_side
            y_top, y_bot = h / 2.0 - c_top, -h / 2.0 + c_bot
            if typ in ("Top", "Bottom"):
                y = y_top if typ == "Top" else y_bot
                count = max(1, n if n else
                            (int(round((2 * zc) / sp)) + 1 if sp else 1))
                for (z, yy) in _line_pts(-zc, y, zc, y, count):
                    _add(z, yy, dia, mat)
            else:                                   # Sides
                height = y_top - y_bot
                count = (n if n else
                         (max(0, int(round(height / sp)) - 1) if sp else 0))
                for i in range(1, count + 1):
                    yy = y_bot + height * i / (count + 1)
                    _add(-zc, yy, dia, mat)
                    _add(zc, yy, dia, mat)
        elif typ == "Perimeter":
            try:
                rl = ReinforcementLayout.from_perimeter(
                    sec, n_bars=int(n or 8), bar_area=bar_area(dia),
                    cover=float(cover), material=mat,
                    designation=f"{dia * 1e3:.0f}mm")
                bars.extend(rl.bars)
            except Exception:                        # noqa: BLE001
                pass
        elif typ == "Line" and len(p) >= 4:
            for (z, y) in _line_pts(p[0] / 1e3, p[1] / 1e3, p[2] / 1e3,
                                    p[3] / 1e3, n or 2):
                _add(z, y, dia, mat)
        elif typ == "Arc" and len(p) >= 5:
            for (z, y) in _arc_pts(p[0] / 1e3, p[1] / 1e3, p[2] / 1e3,
                                   p[3], p[4], n or 4):
                _add(z, y, dia, mat)
        elif typ == "Single" and len(p) >= 2:
            _add(p[0] / 1e3, p[1] / 1e3, dia, mat)
    return bars


def tendons_from_arrangements(arrangements, section, material):
    """Build :class:`PrestressTendon` objects from the tendon arrangements.
    Items are ``(type, area, f_pe, mat, (params...))``."""
    out = []
    for arr in arrangements:
        typ, area, f_pe = arr[0], float(arr[1]), float(arr[2])
        for (z, y) in _arrangement_points(typ, list(arr[4]), section):
            out.append(PrestressTendon(
                z=float(z), y=float(y), area=area, material=material,
                f_pe=f_pe, designation="tendon"))
    return out


def _composite_fiber_section(spec, na_angle=0.0, n_z=26, n_y=52,
                             materials=None):
    """Build a multi-material :class:`FiberSection2D` for a Composite section.
    Each shape is discretised with its own material; later shapes displace
    earlier ones on overlap (z-order). Everything is recentred on the union
    centroid (and rotated by ``-na_angle``), so ``M`` is about the centroid.
    Each rebar arrangement uses ITS named steel material from ``materials``
    (the library {name: props}); "" or an unknown name falls back to the
    section steel. Returns ``(fiber_section, geom_dict, f_r, E_c, eps_cu)``."""
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import unary_union
    from shapely.affinity import translate, rotate as srotate

    raw = []
    for (outline, mat_kv) in spec.shapes:
        pts = [(float(z), float(y)) for (z, y) in outline]
        if len(pts) >= 3:
            raw.append((SPoly(pts), dict(mat_kv)))
    if not raw:                                          # degenerate default
        raw = [(SPoly([(-0.15, -0.15), (0.15, -0.15),
                       (0.15, 0.15), (-0.15, 0.15)]),
                dict(kind="concrete", fc=spec.fc))]
    union = unary_union([p for p, _ in raw])
    cz, cy = union.centroid.x, union.centroid.y
    th = math.radians(-na_angle)
    cth, sth = math.cos(th), math.sin(th)

    def xf(g):
        g = translate(g, xoff=-cz, yoff=-cy)
        return srotate(g, -na_angle, origin=(0, 0)) if abs(na_angle) > 1e-9 else g

    def xf_pt(z, y):
        z2, y2 = z - cz, y - cy
        return (z2 * cth - y2 * sth, z2 * sth + y2 * cth)

    polys = [(xf(p), m) for p, m in raw]
    union_x = xf(union)
    plist = [p for p, _ in polys]

    fibers = []
    prim_conc = None
    for i, (poly, matd) in enumerate(polys):
        later = unary_union(plist[i + 1:]) if i + 1 < len(plist) else None
        eff = (poly.difference(later)
               if (later is not None and not later.is_empty) else poly)
        if eff.is_empty:
            continue
        if matd.get("kind") == "concrete":
            uni = concrete_uniaxial_from(matd)
            prim_conc = prim_conc or matd
        else:
            uni = steel_uniaxial_from(matd)
        fibers.extend(_discretize_polygon_to_fibers(eff, uni, n_z=n_z, n_y=n_y))

    # rebar from arrangements + any custom bars, transformed. Each arrangement
    # can carry its own named steel material; else the section steel is used.
    sect_steel = dict(
        fy=spec.fy, Es=spec.Es, steel_model=spec.steel_model,
        steel_b=spec.steel_b, steel_fu_ratio=spec.steel_fu_ratio,
        steel_eps_sh=spec.steel_eps_sh, steel_eps_su=spec.steel_eps_su)
    steel = steel_uniaxial_from(sect_steel)
    mats = materials or {}

    def _arr_steel(name):
        m = mats.get(name)
        return steel_uniaxial_from(m) if (m and m.get("kind") == "steel") \
            else steel

    tmp = custom_polygon_section(
        outline=list(union.exterior.coords[:-1]),
        material=ConcreteMaterial(fc_prime=spec.fc, fy=spec.fy), name="tmp")
    rebar_ys = []
    for (z, y, dia) in spec.custom_bars:
        zz, yy = xf_pt(float(z), float(y))
        fibers.append(Fiber(y=yy, z=zz, area=bar_area(dia),
                            material=steel.clone()))
        rebar_ys.append(yy)
    for arr in spec.rebar_arr:
        typ, dia = arr[0], float(arr[1])
        arr_steel = _arr_steel(arr[2])
        params = list(arr[3]) + ([dia] if typ == "perim" else [])
        area = bar_area(dia)
        for (z, y) in _arrangement_points(typ, params, tmp):
            zz, yy = xf_pt(z, y)
            fibers.append(Fiber(y=yy, z=zz, area=area,
                                material=arr_steel.clone()))
            rebar_ys.append(yy)

    fs = FiberSection2D(fibers)
    minz, miny, maxz, maxy = union_x.bounds
    I_zz = sum(f.area * f.y * f.y for f in fibers)
    matd = prim_conc or dict(fc=spec.fc)
    conc = concrete_uniaxial_from(matd)
    geo = dict(A=union.area, I_zz=I_zz, y_top=maxy, y_bot=miny,
               rebar_ys=rebar_ys, n_rebar=len(rebar_ys), n_fiber=len(fibers))
    return fs, geo, conc.f_ct, conc.E_ct, float(matd.get("eps_cu", 0.0035))










def composite_mphi(spec, P_target_kN, *, na_angle=0.0, kappa_max=0.06,
                   materials=None, n_points=65, stop="concrete", **_ignored):
    """Multi-material moment-curvature for a Composite section, returning the
    same dict shape as :func:`mphi_data`. ``n_points`` / ``stop`` are the M4
    analysis controls (curvature resolution / failure criterion)."""
    fs, geo, f_r, E_c, eps_cu = _composite_fiber_section(
        spec, na_angle, materials=materials)
    y_top, y_bot, rebar_ys = geo["y_top"], geo["y_bot"], geo["rebar_ys"]
    N_target = -P_target_kN * 1e3
    eps_y = spec.fy / spec.Es
    kcr = f_r / max(E_c * abs(y_bot), 1e-9)
    kappas = _mphi_kappas(kcr, kappa_max, n_points)
    eps0 = 0.0
    pts = []
    M_y = kappa_y = None
    failure = ""
    for kappa in kappas:
        for _ in range(60):
            s, ks = fs.get_response(np.array([eps0, kappa]))
            resid = s[0] - N_target
            tol = max(1.0, abs(N_target) * 1e-8, 100.0)
            if abs(resid) < tol:
                break
            dN = ks[0, 0]
            eps0 -= resid / (dN if abs(dN) >= 1e3 else 1e6)
        s, _ = fs.get_response(np.array([eps0, kappa]))
        eps_top = eps0 - y_top * kappa
        eps_steel = max((eps0 - ry * kappa for ry in rebar_ys), default=0.0)
        pts.append({"kappa": float(kappa), "M": float(s[1]), "P": float(-s[0]),
                    "axial_strain": float(eps0), "eps_top": float(eps_top),
                    "eps_steel": float(eps_steel)})
        fs.commit_state()
        if M_y is None and eps_steel >= eps_y and kappa > 0:
            M_y, kappa_y = float(s[1]), float(kappa)
        failure = _mphi_stop(stop, crush=(-eps_top >= eps_cu),
                             eps_steel=eps_steel)
        if failure:
            break

    kap = [p["kappa"] for p in pts]
    Ms = [p["M"] for p in pts]
    ipk = int(np.argmax(Ms))
    M_u, kappa_u = Ms[ipk], kap[ipk]
    ctrl = _mphi_ctrl(kap, [m / 1e3 for m in Ms],
                      [-p["eps_top"] for p in pts],
                      [p["eps_steel"] for p in pts], eps_cu)
    if not failure:
        failure = "kappa_max_reached" if ipk == len(pts) - 1 else "M_peak"
    # on-curve cracking point (extreme tension fibre reaches eps_cr)
    eps_cr = f_r / E_c
    kappa_cr = M_cr = None
    prev = None
    for p in pts:
        eb = p["axial_strain"] - y_bot * p["kappa"]
        if p["kappa"] > 0 and eb >= eps_cr:
            if prev is not None:
                eb0 = prev["axial_strain"] - y_bot * prev["kappa"]
                t = min(1.0, max(0.0, (eps_cr - eb0) / (eb - eb0)
                                 if eb > eb0 else 1.0))
                kappa_cr = prev["kappa"] + t * (p["kappa"] - prev["kappa"])
                M_cr = prev["M"] + t * (p["M"] - prev["M"])
            else:
                kappa_cr, M_cr = p["kappa"], p["M"]
            break
        prev = p

    def near(k):
        return min(pts, key=lambda p: abs(p["kappa"] - k)) if k and pts else None

    milestones = []
    for lab, state, k, m in (("a", "Cracking", kappa_cr, M_cr),
                             ("b", "First yield (tension steel)", kappa_y, M_y),
                             ("d", f"Ultimate ({failure})", kappa_u, M_u * 1e3)):
        if k is None or m is None:
            continue
        p = near(k)
        milestones.append({
            "label": lab, "state": state, "kappa": float(k), "M": m / 1e3,
            "eps0": float(p["axial_strain"]) if p else 0.0,
            "eps_top": float(p["eps_top"]) if p else 0.0,
            "eps_steel": float(p["eps_steel"]) if p else 0.0})
    mu = (kappa_u / kappa_y) if kappa_y else None
    return {
        "kappa": kap, "M": [m / 1e3 for m in Ms],
        "M_cr": (M_cr or 0) / 1e3, "kappa_cr": kappa_cr,
        "M_y": (M_y / 1e3) if M_y else None, "kappa_y": kappa_y,
        "M_u": M_u / 1e3, "kappa_u": kappa_u, "mu_phi": mu,
        "failure_mode": failure, "milestones": milestones, "ideal": None,
        "y_top": y_top, "y_bot": y_bot, "rebar_ys": rebar_ys,
        "na_angle": float(na_angle), "ctrl": ctrl,
        "conc_model": spec.conc_model, "steel_model": spec.steel_model}


def _composite_pmm_raw(spec, na, nd, materials=None):
    """Fiber-based ULS interaction for a Composite section: for each neutral-
    axis angle and depth, pin the extreme compression fibre at ε_cu, integrate
    every fibre's stress from ITS material, and return the raw (na × nd) grids
    of (P, Mz, My) in kN / kN·m plus the angles. Nominal strength (fibre
    tension-stiffening included; no code φ)."""
    fs, geo, _f_r, _E_c, eps_cu = _composite_fiber_section(
        spec, 0.0, n_z=16, n_y=32, materials=materials)
    fib = fs.fibers
    zs = np.array([f.z for f in fib])
    ys = np.array([f.y for f in fib])
    As = np.array([f.area for f in fib])
    mats = [f.material for f in fib]
    thetas = np.linspace(0.0, 360.0, na, endpoint=False)
    P = np.zeros((na, nd)); Mz = np.zeros((na, nd)); My = np.zeros((na, nd))
    for a, th in enumerate(thetas):
        thr = math.radians(th)
        w = -zs * math.sin(thr) + ys * math.cos(thr)     # perpendicular to NA
        w_top = float(w.max())
        depth = w_top - float(w.min())
        cs = np.geomspace(0.03 * depth, 12.0 * depth, nd)
        for d, c in enumerate(cs):
            eps = -eps_cu * (w - (w_top - c)) / c        # -ε_cu at extreme comp
            sig = np.fromiter(
                (mats[i].get_response(float(eps[i]))[0] for i in range(len(fib))),
                dtype=float, count=len(fib))
            fA = sig * As
            P[a, d] = -float(np.sum(fA)) / 1e3
            Mz[a, d] = -float(np.sum(ys * fA)) / 1e3
            My[a, d] = float(np.sum(zs * fA)) / 1e3
    return P, Mz, My, thetas


def composite_pmm_mesh(spec, n_angles=24, n_depths=29, materials=None):
    """Composite P-Mz-My surface in the same dict shape as
    :func:`pmm_surface_mesh` (φ=1 nominal), so the P-M-M tab renders it."""
    P, Mz, My, thetas = _composite_pmm_raw(spec, n_angles, n_depths, materials)
    na, nd = P.shape
    Pmin, Pmax = float(P.min()), float(P.max())
    Plevels = np.linspace(Pmin, Pmax, nd)
    Mz2 = np.empty((nd, na)); My2 = np.empty((nd, na))
    for i in range(na):
        o = np.argsort(P[i])
        Mz2[:, i] = np.interp(Plevels, P[i][o], Mz[i][o])
        My2[:, i] = np.interp(Plevels, P[i][o], My[i][o])
    ones_raw = np.ones((na, nd))
    return {
        "rawP": P.tolist(), "rawMz": Mz.tolist(), "rawMy": My.tolist(),
        "rawC": np.zeros((na, nd)).tolist(), "rawEpsT": np.zeros((na, nd)).tolist(),
        "rawPhi": ones_raw.tolist(),
        "Mz": Mz2.tolist(), "My": My2.tolist(),
        "Phi": np.ones((nd, na)).tolist(),
        "Plevels": Plevels.tolist(), "thetas": thetas.tolist(),
        "P_min": Pmin, "P_max": Pmax,
        "n_angles": int(na), "n_plevels": int(nd), "n_depths": int(nd)}


def _rect_bars(b, h, cover, n_top, n_bot, n_side, area, desig, mat=None):
    bars: list[RebarBar] = []

    def _row(n, y):
        if n <= 0:
            return
        if n == 1:
            zs = [0.0]
        else:
            edge = b / 2.0 - cover
            zs = [-edge + 2 * edge * i / (n - 1) for i in range(n)]
        for z in zs:
            bars.append(RebarBar(z=z, y=y, area=area, material=mat,
                                 designation=desig))

    y_bot, y_top = -h / 2.0 + cover, h / 2.0 - cover
    _row(n_bot, y_bot)
    _row(n_top, y_top)
    for i in range(n_side):
        frac = (i + 1) / (n_side + 1)
        y = y_bot + (y_top - y_bot) * frac
        for z in (-(b / 2.0 - cover), +(b / 2.0 - cover)):
            bars.append(RebarBar(z=z, y=y, area=area, material=mat,
                                 designation=desig))
    return bars


def build_case(spec: Spec) -> SectionCase:
    """Turn a :class:`Spec` into a :class:`SectionCase`."""
    cm = ConcreteMaterial(fc_prime=spec.fc, fy=spec.fy)
    area = bar_area(spec.bar_dia)
    desig = f"{spec.bar_dia*1e3:.0f}mm"
    prestressed = spec.kind == "PSC girder"

    if spec.kind in ("Rectangular", "PSC girder"):
        b, h = spec.b, spec.h
        rl = ReinforcementLayout(bars=_rect_bars(
            b, h, spec.cover,
            n_top=spec.n_top,
            n_bot=0 if prestressed else spec.n_bot,
            n_side=0 if prestressed else spec.n_side,
            area=area, desig=desig))
        sec = rc_rectangular_section(b=b, h=h, concrete=cm, reinforcement=rl,
                                     name=f"{spec.kind} {b*1e3:.0f}x{h*1e3:.0f}")
    elif spec.kind == "Circular":
        sec = circular_section(D=spec.D, material=cm, n_sides=120,
                               name=f"Circular D{spec.D*1e3:.0f}")
        sec.reinforcement = ReinforcementLayout.from_perimeter(
            sec, n_bars=spec.n_perim, bar_area=area, cover=spec.cover,
            designation=desig)
    elif spec.kind == "L-shape":
        L, t = spec.leg, spec.thick
        raw = [(0, 0), (L, 0), (L, t), (t, t), (t, L), (0, L)]
        sec = custom_polygon_section(
            outline=_recenter_outline(raw), material=cm,
            name=f"L {L*1e3:.0f}x{L*1e3:.0f} t{t*1e3:.0f}")
        sec.reinforcement = ReinforcementLayout.from_perimeter(
            sec, n_bars=spec.n_perim, bar_area=area, cover=spec.cover,
            designation=desig)
    elif spec.kind == "T-shape":
        sec0 = t_section(h=spec.h, b=spec.b, t_f=spec.t_f, t_w=spec.t_w,
                         material=cm)
        cz, cy = sec0.geometry.centroid
        outline = [(z - cz, y - cy)
                   for (z, y) in sec0.geometry.polygon.exterior.coords[:-1]]
        sec = custom_polygon_section(
            outline=outline, material=cm,
            name=f"T {spec.h*1e3:.0f}x{spec.b*1e3:.0f}")
        sec.reinforcement = ReinforcementLayout.from_perimeter(
            sec, n_bars=spec.n_perim, bar_area=area, cover=spec.cover,
            designation=desig)
    elif spec.kind == "Hollow box":
        sec = hollow_rect_section(b=spec.b, h=spec.h, t=spec.wall_t,
                                  material=cm,
                                  name=f"Box {spec.b*1e3:.0f}x{spec.h*1e3:.0f}")
        sec.reinforcement = ReinforcementLayout.from_perimeter(
            sec, n_bars=spec.n_perim, bar_area=area, cover=spec.cover,
            designation=desig)
    elif spec.kind == "Custom":
        outline = [(float(z), float(y)) for (z, y) in spec.custom_outline]
        holes = [[(float(z), float(y)) for (z, y) in ring]
                 for ring in spec.custom_holes]
        if len(outline) < 3:                    # degenerate -> safe default
            outline = [(-0.15, -0.15), (0.15, -0.15),
                       (0.15, 0.15), (-0.15, 0.15)]
            holes = []
        # recenter on the true centroid (accounts for holes) so the fibre
        # engine's reference axis (y=0) is the centroid, as interaction
        # diagrams and moment-curvature assume.
        tmp = custom_polygon_section(outline=outline, holes=holes or None,
                                     material=cm, name="tmp")
        cz, cy = tmp.geometry.centroid
        outline = [(z - cz, y - cy) for (z, y) in outline]
        holes = [[(z - cz, y - cy) for (z, y) in ring] for ring in holes]
        sec = custom_polygon_section(outline=outline, holes=holes or None,
                                     material=cm, name="Custom section")
        bars = [RebarBar(z=z - cz, y=y - cy, area=bar_area(dia),
                         material=None, designation=f"{dia*1e3:.0f}mm")
                for (z, y, dia) in spec.custom_bars]
        if bars:
            sec.reinforcement = ReinforcementLayout(bars=bars)
    elif spec.kind == "Composite":
        # For DISPLAY / geometry props only — the union outline (recentred) as
        # one polygon plus the rebar. The multi-material ANALYSIS uses the
        # fibre path (composite_mphi / composite_pmm_mesh), not this section.
        from shapely.ops import unary_union
        from shapely.geometry import Polygon as _SP
        polys = [_SP([(float(z), float(y)) for (z, y) in o])
                 for (o, _m) in spec.shapes if len(o) >= 3]
        if polys:
            union = unary_union(polys)
            cz, cy = union.centroid.x, union.centroid.y
            geom = (union.geoms[0] if union.geom_type == "MultiPolygon"
                    else union)
            orig = list(geom.exterior.coords[:-1])
        else:
            cz = cy = 0.0
            orig = [(-0.15, -0.15), (0.15, -0.15), (0.15, 0.15), (-0.15, 0.15)]
        sec = custom_polygon_section(
            outline=[(z - cz, y - cy) for (z, y) in orig], material=cm,
            name="Composite section")
        tmp = custom_polygon_section(
            outline=[(float(z), float(y)) for (z, y) in orig], material=cm,
            name="tmp")                                  # original frame
        rbars = []
        for arr in spec.rebar_arr:
            typ, dia = arr[0], float(arr[1])
            params = list(arr[3]) + ([dia] if typ == "perim" else [])
            for (z, y) in _arrangement_points(typ, params, tmp):
                rbars.append(RebarBar(z=z - cz, y=y - cy, area=bar_area(dia),
                                      material=None,
                                      designation=f"{dia*1e3:.0f}mm"))
        if rbars:
            sec.reinforcement = ReinforcementLayout(bars=rbars)
    else:
        raise ValueError(f"unknown kind {spec.kind!r}")

    case = SectionCase(
        section_id="LIVE", name=sec.name, section=sec,
        f_c_prime=spec.fc, f_y=spec.fy,
        kind="psc" if prestressed else "column",
        spiral=spec.spiral, prestressed=prestressed)

    if prestressed:
        strand_mat = UniaxialBilinear(E=spec.Ep, sigma_y=spec.fpy, b=spec.ps_b)
        edge = spec.b / 2.0 - spec.cover
        n = spec.n_strand
        zs = [0.0] if n == 1 else [
            -edge + 2 * edge * i / (n - 1) for i in range(n)]
        sec.prestress = TendonLayout(tendons=[
            PrestressTendon(z=z, y=spec.strand_y, area=spec.strand_area,
                            material=strand_mat, f_pe=spec.f_pe,
                            designation="0.6in Gr270")
            for z in zs])

    # AdSec-style reinforcement GROUPS (Top/Bottom/Sides/Link/Perimeter/Line/
    # Arc/Single) — the newer model. When present they REPLACE the parametric
    # bars for the section (the desktop zeroes the counts anyway).
    if spec.rebar_groups and spec.kind not in ("Custom", "Composite"):
        rect = ((spec.b, spec.h)
                if spec.kind in ("Rectangular", "Hollow box") else None)
        fc = ((spec.cover_top, spec.cover_bot, spec.cover_side)
              if spec.cover_variable else None)
        gbars = bars_from_groups(spec.rebar_groups, sec, spec.cover, rect, fc)
        sec.reinforcement = ReinforcementLayout(bars=gbars)

    # additive rebar arrangements (point/line/arc/rectangle/perimeter).
    # Composite handles its own (centroid-shifted) rebar in its branch above.
    if spec.rebar_arr and spec.kind != "Composite":
        extra = bars_from_arrangements(spec.rebar_arr, sec)
        base = list(sec.reinforcement.bars) if sec.reinforcement else []
        if base or extra:
            sec.reinforcement = ReinforcementLayout(bars=base + extra)

    # additive tendon arrangements
    if spec.tendon_arr and spec.kind != "Composite":
        strand_mat = UniaxialBilinear(E=spec.Ep, sigma_y=spec.fpy, b=spec.ps_b)
        extra_t = tendons_from_arrangements(spec.tendon_arr, sec, strand_mat)
        base_t = (list(sec.prestress.tendons)
                  if getattr(sec, "prestress", None) else [])
        if base_t or extra_t:
            sec.prestress = TendonLayout(tendons=base_t + extra_t)
            case = replace(case, prestressed=True, kind="psc")
    return case


def pmm_surface_grid(case: SectionCase, code: str,
                     n_angles: int = 28, n_depths: int = 14) -> dict:
    """Full biaxial P-Mz-My surface as an (n_angles x n_depths) grid,
    in display units (kN, kN.m). Rows are neutral-axis angles (0..360),
    columns sweep the NA depth. Feeds the interactive 3-D view."""
    sec = case.section
    fc, fy = case.f_c_prime, case.f_y
    if code == "AASHTO LRFD 2024":
        surf = biaxial_pmm_surface_aashto(
            sec, f_c_prime=fc, f_y=fy, spiral=case.spiral,
            prestressed=case.prestressed, n_angles=n_angles, n_depths=n_depths)
    elif code == "Eurocode 2":
        surf = biaxial_pmm_surface_ec2(
            sec, f_ck=fc, f_yk=fy, n_angles=n_angles, n_depths=n_depths)
    else:
        surf = biaxial_pmm_surface_is456(
            sec, f_ck=fc, f_y=fy, n_angles=n_angles, n_depths=n_depths)

    pts = surf.points          # theta-major, same order as as_arrays()
    na, nd = surf.n_angles, surf.n_depths
    P = (np.array([p.P_n for p in pts]) / 1e3).reshape(na, nd)
    Mz = (np.array([p.M_nz for p in pts]) / 1e3).reshape(na, nd)
    My = (np.array([p.M_ny for p in pts]) / 1e3).reshape(na, nd)
    c_mm = (np.array([p.c for p in pts]) * 1e3).reshape(na, nd)
    eps_t = np.array([p.epsilon_t for p in pts]).reshape(na, nd)
    phi = np.array([p.phi for p in pts]).reshape(na, nd)
    thetas = np.linspace(0.0, 360.0, na, endpoint=False)
    return {
        "P": P.tolist(), "Mz": Mz.tolist(), "My": My.tolist(),
        "c": c_mm.tolist(), "eps_t": eps_t.tolist(), "phi": phi.tolist(),
        "thetas": thetas.tolist(),
        "P_min": float(P.min()), "P_max": float(P.max()),
        "n_angles": int(na), "n_depths": int(nd),
    }


def mm_contour(grid: dict, P_target_kN: float):
    """Biaxial moment (Mz, My) interaction contour at a constant axial
    force, interpolated across the surface's angle ribs. Returns
    (Mz_list, My_list), closed into a ring."""
    P = np.array(grid["P"])
    Mz = np.array(grid["Mz"])
    My = np.array(grid["My"])
    mz, my = [], []
    for i in range(P.shape[0]):
        order = np.argsort(P[i])
        ps = P[i][order]
        if P_target_kN < ps[0] or P_target_kN > ps[-1]:
            continue
        mz.append(float(np.interp(P_target_kN, ps, Mz[i][order])))
        my.append(float(np.interp(P_target_kN, ps, My[i][order])))
    if mz:
        mz.append(mz[0])
        my.append(my[0])
    return mz, my


def pmm_surface_mesh(case: SectionCase, code: str,
                     n_angles: int = 24, n_depths: int = 29) -> dict:
    """Full P-Mz-My surface in the two forms Midas GSD uses.

    * **P-M curves** (``raw*``): each constant-angle rib is the raw engine
      output, sampled by **neutral-axis depth** -- the GSD convention, so
      the ``n_depths`` points are NOT uniform in axial force. Carries the
      neutral-axis depth ``c`` (mm) and extreme-tension strain ``eps_t``
      per point for row-by-row comparison.
    * **M-M curves / 3-D surface** (``Mz`` / ``My`` / ``Plevels``): the same
      surface resampled onto a regular grid of ``n_depths`` **constant
      axial-force** levels, so each ring is a true M-M contour of
      ``n_angles`` points.

    Display units (kN, kN.m, mm).
    """
    raw = pmm_surface_grid(case, code, n_angles=n_angles, n_depths=n_depths)
    P = np.array(raw["P"])
    Mz = np.array(raw["Mz"])
    My = np.array(raw["My"])
    Pmin, Pmax = raw["P_min"], raw["P_max"]
    Phi = np.array(raw["phi"])
    n_plevels = n_depths
    Plevels = np.linspace(Pmin, Pmax, n_plevels)
    na = P.shape[0]
    Mz2 = np.empty((n_plevels, na))
    My2 = np.empty((n_plevels, na))
    Phi2 = np.empty((n_plevels, na))
    for i in range(na):
        order = np.argsort(P[i])
        ps = P[i][order]
        Mz2[:, i] = np.interp(Plevels, ps, Mz[i][order])
        My2[:, i] = np.interp(Plevels, ps, My[i][order])
        Phi2[:, i] = np.interp(Plevels, ps, Phi[i][order])
    return {
        # raw ribs, sampled by neutral-axis depth -> P-M curves (GSD)
        "rawP": raw["P"], "rawMz": raw["Mz"], "rawMy": raw["My"],
        "rawC": raw["c"], "rawEpsT": raw["eps_t"], "rawPhi": raw["phi"],
        # resampled to constant axial-force levels -> M-M rings + surface
        "Mz": Mz2.tolist(), "My": My2.tolist(), "Phi": Phi2.tolist(),
        "Plevels": Plevels.tolist(), "thetas": raw["thetas"],
        "P_min": float(Pmin), "P_max": float(Pmax),
        "n_angles": int(na), "n_plevels": int(n_plevels),
        "n_depths": int(n_depths),
    }


def mphi_props(spec: "Spec") -> dict:
    """Nonlinear constitutive parameters (for moment-curvature) pulled
    off a :class:`Spec`, as keyword args for :func:`mphi_data`."""
    return {
        "eps_c0": spec.eps_c0, "eps_cu": spec.eps_cu,
        "fcu_ratio": spec.fcu_ratio, "fr_coeff": spec.fr_coeff,
        "fr_model": spec.fr_model, "eps_decay": spec.eps_decay,
        "E_s": spec.Es, "steel_b": spec.steel_b, "kappa_max": spec.kappa_max,
        "conc_model": spec.conc_model, "conc_f1_ratio": spec.conc_f1_ratio,
        "steel_model": spec.steel_model,
        "steel_fu_ratio": spec.steel_fu_ratio,
        "steel_eps_sh": spec.steel_eps_sh, "steel_eps_su": spec.steel_eps_su,
    }














# Gauss-Legendre nodes/weights for the exact concrete integration (M3).








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
        _n, dia_h, s = parse_bar_desc(link[1] if len(link) > 1 else "")
    except Exception:                              # noqa: BLE001
        return None, "bad-pattern"
    if not s or s <= 0:
        return None, "no-spacing"           # a tie needs a spacing (e.g. B10-150)
    asp = bar_area(dia_h)
    sp = max(s - dia_h, 1e-3)               # clear spacing s' = s − hoop dia
    hoopmat = materials.get(link[3]) if len(link) > 3 else None
    fyh = float((hoopmat or {}).get("fy", 400e6))
    esu = float((hoopmat or {}).get("steel_eps_su", 0.10))
    es_h = float((hoopmat or {}).get("Es", 200e9))
    ny, nz = _parse_legs(link[2] if len(link) > 2 else "")
    cover = float(spec.cover)
    try:
        sec = build_case(spec).section
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


def _confined_fiber_section(spec, conf, na_angle=0.0, n_z=28, n_y=56):
    """Two-zone fibre section for a confined Circular / Rectangular column: an
    unconfined COVER ring plus a confined Mander CORE (the outline inset by the
    cover) plus the longitudinal rebar. ``conf`` is the section-derived
    confinement dict (``conf_*`` keys, e.g. from the tie Link group). Returns
    ``(fiber_section, geo, f_r, E_c, eps_cu_core)``; ``geo`` carries the core
    edges so crushing can be judged there, not at the (spalling) cover edge."""
    from shapely.affinity import rotate as srotate
    case = build_case(spec)
    outline = case.section.geometry.polygon
    cover = float(spec.cover)
    base = dict(fc=spec.fc, eps_c0=spec.eps_c0, eps_cu=spec.eps_cu,
                fcu_ratio=spec.fcu_ratio, fr_model=spec.fr_model,
                fr_coeff=spec.fr_coeff, eps_decay=spec.eps_decay,
                conc_f1_ratio=spec.conc_f1_ratio)
    cover_law = concrete_uniaxial_from(dict(base, conc_model="Mander"))
    core_props = dict(base, conc_model="Mander", **(conf or {}))
    core_law = concrete_uniaxial_from(core_props)
    eps_cu_core = (mander_confinement(core_props)["eps_cu"]
                   if _has_confinement_params(core_props)
                   else float(spec.eps_cu))

    def xf(g):
        return srotate(g, -na_angle, origin=(0, 0)) if abs(na_angle) > 1e-9 else g

    core = outline.buffer(-cover, join_style=2)
    fibers = []
    if core.is_empty or core.area <= 1e-9:
        core = None
        fibers += _discretize_polygon_to_fibers(xf(outline), cover_law,
                                                n_z=n_z, n_y=n_y)
    else:
        ring = outline.difference(core)
        if not ring.is_empty:
            fibers += _discretize_polygon_to_fibers(xf(ring), cover_law,
                                                    n_z=n_z, n_y=n_y)
        fibers += _discretize_polygon_to_fibers(xf(core), core_law,
                                                n_z=n_z, n_y=n_y)

    steel = steel_uniaxial_from(dict(
        fy=spec.fy, Es=spec.Es, steel_model=spec.steel_model,
        steel_b=spec.steel_b, steel_fu_ratio=spec.steel_fu_ratio,
        steel_eps_sh=spec.steel_eps_sh, steel_eps_su=spec.steel_eps_su))
    th = math.radians(-na_angle)
    cth, sth = math.cos(th), math.sin(th)
    rebar_ys = []
    bars = case.section.reinforcement.bars if case.section.reinforcement else []
    for b in bars:
        z, y = float(b.z), float(b.y)
        if abs(na_angle) > 1e-9:
            z, y = z * cth - y * sth, z * sth + y * cth
        fibers.append(Fiber(y=y, z=z, area=float(b.area),
                            material=steel.clone()))
        rebar_ys.append(y)

    fs = FiberSection2D(fibers)
    _minz, miny, _maxz, maxy = xf(outline).bounds
    geo = dict(y_top=maxy, y_bot=miny, A=float(outline.area),
               y_core_top=(maxy - cover) if core is not None else maxy,
               y_core_bot=(miny + cover) if core is not None else miny,
               rebar_ys=rebar_ys, n_fiber=len(fibers))
    return fs, geo, core_law.f_ct, core_law.E_ct, eps_cu_core


def confined_mphi(spec, conf, P_target_kN, *, na_angle=0.0, kappa_max=0.06,
                  n_points=65, stop="concrete", **_ignored):
    """Moment-curvature for a confined Circular / Rectangular column on a
    two-zone (confined core + unconfined cover) fibre section — the Midas-GSD
    model. Same return shape as :func:`mphi_data`. Crushing is judged at the
    CORE edge reaching the confined ε_cu, so cover spalling (handled by the
    cover fibres' own descending branch) doesn't prematurely end the curve."""
    fs, geo, f_r, E_c, eps_cu = _confined_fiber_section(spec, conf, na_angle)
    y_top, y_bot = geo["y_top"], geo["y_bot"]
    y_core_top, rebar_ys = geo["y_core_top"], geo["rebar_ys"]
    N_target = -P_target_kN * 1e3
    eps_y = spec.fy / spec.Es
    A = max(geo.get("A", 0.0), 1e-6)
    kcr = f_r / max(E_c * abs(y_bot), 1e-9)
    kappas = _mphi_kappas(kcr, kappa_max, n_points)
    eps0 = N_target / max(E_c * A, 1e6)          # elastic axial warm start
    pts = []
    M_y = kappa_y = None
    failure = ""
    for kappa in kappas:
        # damped, clamped Newton on the reference strain to hold P = N_target;
        # the warm start + step cap avoid the spurious fully-crushed root.
        for _ in range(80):
            s, ks = fs.get_response(np.array([eps0, kappa]))
            resid = s[0] - N_target
            tol = max(1.0, abs(N_target) * 1e-8, 100.0)
            if abs(resid) < tol:
                break
            dN = ks[0, 0]
            step = resid / (dN if abs(dN) >= 1e6 else 1e8)
            eps0 = min(0.05, max(-0.05, eps0 - max(-2e-3, min(2e-3, step))))
        s, _ = fs.get_response(np.array([eps0, kappa]))
        eps_top = eps0 - y_top * kappa            # extreme (cover) fibre
        eps_core = eps0 - y_core_top * kappa      # core edge — governs crushing
        eps_steel = max((eps0 - ry * kappa for ry in rebar_ys), default=0.0)
        pts.append({"kappa": float(kappa), "M": float(s[1]), "P": float(-s[0]),
                    "axial_strain": float(eps0), "eps_top": float(eps_top),
                    "eps_core": float(eps_core), "eps_steel": float(eps_steel)})
        fs.commit_state()
        if M_y is None and eps_steel >= eps_y and kappa > 0:
            M_y, kappa_y = float(s[1]), float(kappa)
        if stop != "peak":
            if stop != "steel" and -eps_core >= eps_cu:
                failure = "core_crushing"
                break
            if eps_steel >= 0.05:
                failure = "steel_rupture"
                break

    kap = [p["kappa"] for p in pts]
    Ms = [p["M"] for p in pts]
    ipk = int(np.argmax(Ms)) if Ms else 0
    M_u, kappa_u = (Ms[ipk], kap[ipk]) if pts else (0.0, 0.0)
    ctrl = _mphi_ctrl(kap, [m / 1e3 for m in Ms],
                      [-p["eps_core"] for p in pts],
                      [p["eps_steel"] for p in pts], eps_cu)
    if not failure:
        failure = "kappa_max_reached" if ipk == len(pts) - 1 else "M_peak"
    eps_cr = f_r / E_c
    kappa_cr = M_cr = None
    prev = None
    for p in pts:
        eb = p["axial_strain"] - y_bot * p["kappa"]
        if p["kappa"] > 0 and eb >= eps_cr:
            if prev is not None:
                eb0 = prev["axial_strain"] - y_bot * prev["kappa"]
                t = min(1.0, max(0.0, (eps_cr - eb0) / (eb - eb0)
                                 if eb > eb0 else 1.0))
                kappa_cr = prev["kappa"] + t * (p["kappa"] - prev["kappa"])
                M_cr = prev["M"] + t * (p["M"] - prev["M"])
            else:
                kappa_cr, M_cr = p["kappa"], p["M"]
            break
        prev = p

    def near(k):
        return min(pts, key=lambda p: abs(p["kappa"] - k)) if k and pts else None

    milestones = []
    for lab, state, k, m in (("a", "Cracking", kappa_cr, M_cr),
                             ("b", "First yield (tension steel)", kappa_y, M_y),
                             ("d", f"Ultimate ({failure})", kappa_u, M_u * 1e3)):
        if k is None or m is None:
            continue
        p = near(k)
        milestones.append({
            "label": lab, "state": state, "kappa": float(k), "M": m / 1e3,
            "eps0": float(p["axial_strain"]) if p else 0.0,
            "eps_top": float(p["eps_top"]) if p else 0.0,
            "eps_steel": float(p["eps_steel"]) if p else 0.0})
    mu = (kappa_u / kappa_y) if kappa_y else None
    return {
        "kappa": kap, "M": [m / 1e3 for m in Ms],
        "M_cr": (M_cr or 0) / 1e3, "kappa_cr": kappa_cr,
        "M_y": (M_y / 1e3) if M_y else None, "kappa_y": kappa_y,
        "M_u": M_u / 1e3, "kappa_u": kappa_u, "mu_phi": mu,
        "failure_mode": failure, "milestones": milestones, "ideal": None,
        "y_top": y_top, "y_bot": y_bot, "rebar_ys": rebar_ys,
        "na_angle": float(na_angle), "ctrl": ctrl,
        "conc_model": spec.conc_model, "steel_model": spec.steel_model}


def _surface_for_code(case: SectionCase, code: str, na: int, nd: int):
    """Build the BiaxialPMMSurface for one code (dispatch helper)."""
    sec, fc, fy = case.section, case.f_c_prime, case.f_y
    if code == "AASHTO LRFD 2024":
        return biaxial_pmm_surface_aashto(
            sec, f_c_prime=fc, f_y=fy, spiral=case.spiral,
            prestressed=case.prestressed, n_angles=na, n_depths=nd)
    if code == "Eurocode 2":
        return biaxial_pmm_surface_ec2(
            sec, f_ck=fc, f_yk=fy, n_angles=na, n_depths=nd)
    return biaxial_pmm_surface_is456(
        sec, f_ck=fc, f_y=fy, n_angles=na, n_depths=nd)


def _ray_radius(mz, my, dz, dy):
    """Distance from the origin to the closed polygon (mz, my) along the
    ray in direction (dz, dy). Used to size the biaxial moment capacity in
    the direction of an applied moment vector. ``None`` if no crossing."""
    d = math.hypot(dz, dy)
    if d < 1e-12 or len(mz) < 3:
        return None
    ux, uy = dz / d, dy / d
    n = len(mz)
    best = None
    for i in range(n):
        x1, y1 = mz[i], my[i]
        x2, y2 = mz[(i + 1) % n], my[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        det = ex * uy - ey * ux
        if abs(det) < 1e-12:
            continue
        t = (ex * y1 - ey * x1) / det     # ray parameter (distance)
        s = (ux * y1 - uy * x1) / det     # segment parameter in [0, 1]
        if t >= -1e-9 and -1e-9 <= s <= 1 + 1e-9:
            if best is None or t < best:
                best = t
    return best


def pm_curve_at_angle(Plevels, rings_mz, rings_my, beta_deg):
    """P-M capacity curve (|M| vs P) in the direction ``beta_deg`` — the
    vertical slice of the interaction surface in the plane of an applied
    moment vector. For each axial-force ring level, ``_ray_radius`` gives the
    capacity moment magnitude in that direction. ``rings_mz`` / ``rings_my``
    are (n_levels, n_angles) arrays. Returns ``(P_list, M_cap_list)`` (levels
    with no crossing are dropped). Used to draw one P-M curve per load
    combination at that combination's moment angle."""
    b = math.radians(beta_deg)
    dz, dy = math.cos(b), math.sin(b)
    Ps, Ms = [], []
    for k in range(len(Plevels)):
        r = _ray_radius(list(rings_mz[k]), list(rings_my[k]), dz, dy)
        if r is not None and r > 0:
            Ps.append(float(Plevels[k]))
            Ms.append(float(r))
    return Ps, Ms


def _ray_polyline(xs, ys, dx, dy):
    """Distance from the origin to the OPEN polyline (xs, ys) along the ray
    in direction (dx, dy). Used for the radial (keep-M/P-constant) D/C in the
    (M, P) plane. ``None`` if no forward crossing."""
    d = math.hypot(dx, dy)
    if d < 1e-12 or len(xs) < 2:
        return None
    ux, uy = dx / d, dy / d
    best = None
    for i in range(len(xs) - 1):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i + 1], ys[i + 1]
        ex, ey = x2 - x1, y2 - y1
        det = ex * uy - ey * ux
        if abs(det) < 1e-12:
            continue
        t = (ex * y1 - ey * x1) / det
        s = (ux * y1 - uy * x1) / det
        if t >= -1e-9 and -1e-9 <= s <= 1 + 1e-9:
            if best is None or t < best:
                best = t
    return best


def _dc_utilization(Pc, Mc, Pd, M_res, method):
    """Demand/capacity ratio from the P-M boundary curve ``(Pc, Mc)`` (Pc
    ascending, Mc the capacity moment in the demand's direction at each P).
    ``method`` selects how the demand is scaled onto the surface:

    * ``"P"``  keep P constant, scale the moment  -> util = M_res / M_cap(Pd)
    * ``"M"``  keep M constant, scale the axial   -> util = Pd / P_cap(M_res)
    * ``"MP"`` keep M/P constant (radial)         -> util = |D| / |C|

    Returns ``(util, M_cap, P_cap, govern)`` with ``M_cap`` = moment capacity
    at ``Pd`` (a common reference) and ``P_cap`` the method's axial capacity.
    """
    Pc = np.asarray(Pc, float)
    Mc = np.asarray(Mc, float)
    if len(Pc) < 3:
        return math.inf, 0.0, 0.0, "biaxial"
    P_lo, P_hi = float(Pc[0]), float(Pc[-1])
    M_cap = float(np.interp(Pd, Pc, Mc)) if P_lo <= Pd <= P_hi else 0.0

    # Pure axial demand, or axial outside the section's axial range -> axial
    # governs regardless of the chosen method.
    if M_res <= 1e-9 or not (P_lo <= Pd <= P_hi):
        denom = P_hi if Pd >= 0 else P_lo
        util = abs(Pd / denom) if abs(denom) > 1e-9 else (
            0.0 if M_res <= 1e-9 else math.inf)
        return util, M_cap, denom, "axial"

    if method == "M":                       # keep M constant, scale P
        # A demand whose moment already exceeds the capacity at its own axial
        # load is flexure-governed and fails -- keep-M cannot scale it to
        # safety, so report the flexural overload directly (> 1).
        if M_cap <= 1e-9 or M_res > M_cap + 1e-6:
            return (math.inf if M_cap <= 1e-9 else M_res / M_cap), M_cap, Pd, \
                "moment"
        # Feasible: the capacity is the axial crossing at M_res in the load's
        # direction (upper/compression for Pd>=0, lower/tension otherwise);
        # util is then the axial capacity ratio.
        ipk = int(np.argmax(Mc))
        if Pd >= 0:
            P_cap = float(np.interp(M_res, Mc[ipk:][::-1], Pc[ipk:][::-1]))
        else:
            P_cap = float(np.interp(M_res, Mc[:ipk + 1], Pc[:ipk + 1]))
        util = abs(Pd / P_cap) if abs(P_cap) > 1e-9 else 0.0
        return util, M_cap, P_cap, "axial"

    if method == "MP":                      # radial in (M, P): |D| / |C|
        t = _ray_polyline(list(Mc), list(Pc), M_res, Pd)
        Dd = math.hypot(M_res, Pd)
        if t is None or t <= 0 or Dd <= 0:
            return math.inf, M_cap, 0.0, "radial"
        return Dd / t, (M_res / Dd) * t, (Pd / Dd) * t, "radial"

    # method == "P" (default): keep P constant, scale the moment
    if M_cap <= 0:
        return math.inf, M_cap, Pd, "biaxial"
    return M_res / M_cap, M_cap, Pd, "moment"


def demand_check(case: SectionCase, code: str, demands, *,
                 design: bool = True, na: int = 24, nd: int = 29,
                 method: str = "P", spec=None, materials=None):
    """Check a list of applied (P, Mz, My) load combinations against the
    section's interaction surface (spColumn / Midas GSD).

    ``demands`` is a list of dicts ``{"name", "P", "Mz", "My"}`` in canonical
    units (kN, kN.m; P compression-positive). With ``design=True`` the check
    is against the phi-reduced design surface (EC2/IS phi=1). ``method`` picks
    the checking-ratio convention: ``"P"`` (keep P constant, default), ``"M"``
    (keep M constant) or ``"MP"`` (keep M/P constant, radial). For a Composite
    ``spec`` the fibre-based nominal surface is used (φ=1). Returns one result
    dict per demand: capacity moment at Pd, the utilization ratio, and status.
    """
    if spec is not None and spec.kind == "Composite":
        P, Mz, My, _th = _composite_pmm_raw(spec, na, nd, materials)
        D, A = nd, na
    else:
        surf = _surface_for_code(case, code, na, nd)
        pts = surf.points
        A, D = surf.n_angles, surf.n_depths
        P = np.array([p.P_n for p in pts]).reshape(A, D) / 1e3      # kN
        Mz = np.array([p.M_nz for p in pts]).reshape(A, D) / 1e3    # kN.m
        My = np.array([p.M_ny for p in pts]).reshape(A, D) / 1e3
        phi = np.array([p.phi for p in pts]).reshape(A, D)
        if design:
            P, Mz, My = P * phi, Mz * phi, My * phi
    # Resample to constant-axial-force rings so a P-M boundary curve can be
    # built in any moment-vector direction (needed by all three methods).
    Plevels = np.linspace(float(P.min()), float(P.max()), D)
    Mz2 = np.empty((D, A))
    My2 = np.empty((D, A))
    for i in range(A):
        order = np.argsort(P[i])
        ps = P[i][order]
        Mz2[:, i] = np.interp(Plevels, ps, Mz[i][order])
        My2[:, i] = np.interp(Plevels, ps, My[i][order])

    out = []
    for dem in demands:
        Pd = float(dem["P"]); Mzd = float(dem["Mz"]); Myd = float(dem["My"])
        M_res = math.hypot(Mzd, Myd)
        beta = math.degrees(math.atan2(Myd, Mzd))
        Pc, Mc = pm_curve_at_angle(Plevels, Mz2, My2, beta)
        util, M_cap, P_cap, govern = _dc_utilization(Pc, Mc, Pd, M_res, method)
        out.append({
            "name": dem.get("name", ""), "P": Pd, "Mz": Mzd, "My": Myd,
            "M_res": M_res, "beta_deg": beta, "M_cap": float(M_cap),
            "P_cap": float(P_cap), "util": util, "govern": govern,
            "status": "OK" if util <= 1.0 + 1e-9 else "FAIL"})
    return out


def items_data(case: SectionCase, code: str,
               mphi_props: Optional[dict] = None,
               axis_labels: tuple = ("z", "y")) -> list[dict]:
    """Verification quantities for one section evaluated in ONE design
    code (Midas-GSD style: each section carries its own code). Values in
    canonical display units (kN, kN.m, mm, mm^2, mm^4, 1/m). ``axis_labels`` is
    ``(horizontal, vertical)`` axis letters for the inertia subscripts (the
    horizontal-axis inertia is engine ``I_zz``, the vertical-axis one ``I_yy``);
    default ``("z","y")`` keeps the engine convention."""
    g = case.section.geometry
    _cz, cy = g.centroid
    miny = g.polygon.bounds[1]
    h, v = axis_labels
    rows = [
        ("-", "Gross area A_g", "mm^2", g.area * 1e6, 1.0),
        ("-", f"I_{h}{h} (about centroidal {h})", "mm^4", g.I_zz * 1e12, 1.0),
        ("-", f"I_{v}{v} (about centroidal {v})", "mm^4", g.I_yy * 1e12, 1.0),
        ("-", "Centroid above bottom fibre", "mm", (cy - miny) * 1e3, 1.0),
    ]
    if case.prestressed:
        surf = _surface_for_code(case, code, 4, 6)
        rows.append((code, "P_o squash (incl. tendon f_pu)", "kN",
                     surf.P_o / 1e3, 1.0))
        rows.append((code, "P pure tension", "kN",
                     surf.P_pure_tension / 1e3, 2.0))
    else:
        _curve, lm = pmm_slice(case, code)
        for name, val, kind in (lm or []):
            units = "kN" if kind == "P" else "kN.m"
            rows.append((code, name, units, val, 1.0 if kind == "P" else 2.0))

    mp = mphi_data(case, 0.0, **(mphi_props or {}))
    rows.append(("M-φ", "Cracking moment M_cr", "kN.m", mp["M_cr"], 3.0))
    if mp["M_y"] is not None:
        rows.append(("M-φ", "First-yield moment M_y", "kN.m",
                     mp["M_y"], 3.0))
    rows.append(("M-φ", "Ultimate moment M_u", "kN.m", mp["M_u"], 3.0))
    if mp["mu_phi"] is not None:
        rows.append(("M-φ", "Curvature ductility mu_phi", "-",
                     mp["mu_phi"], 3.0))
    return [{
        "section_id": "LIVE", "section": case.name, "code": c,
        "quantity": q, "units": un, "computed": v, "tol_pct": t, "note": "",
    } for (c, q, un, v, t) in rows]


def _mat_label(matd: dict) -> str:
    """Short material label for the fibre view, e.g. 'Concrete 30' / 'Steel 355'."""
    kind = str(matd.get("kind", "concrete")).title()
    strength = matd.get("fc") if kind == "Concrete" else matd.get("fy")
    try:
        return f"{kind} {float(strength) / 1e6:.0f}" if strength else kind
    except (TypeError, ValueError):
        return kind


def _composite_cell_fibers(spec, csz_z, csz_y, eps0=None, kappa=None) -> list:
    """Composite section cells, discretised per shape with each shape's own
    material (z-order: later shapes displace earlier), recentred on the union
    centroid — matching :func:`_composite_fiber_section`. Cell size ≈
    ``csz_z × csz_y`` so all shapes share one grid. Returns fibre dicts labelled
    by material; with ``eps0``/``kappa`` each also carries strain + stress from
    its shape's constitutive law."""
    from shapely.geometry import Polygon as SPoly
    from shapely.ops import unary_union
    from shapely.affinity import translate

    raw = [(SPoly([(float(z), float(y)) for (z, y) in o]), dict(m))
           for (o, m) in spec.shapes if len(o) >= 3]
    if not raw:
        return []
    union = unary_union([p for p, _ in raw])
    cz, cy = union.centroid.x, union.centroid.y
    polys = [(translate(p, xoff=-cz, yoff=-cy), m) for p, m in raw]
    plist = [p for p, _ in polys]
    dummy = concrete_uniaxial_from(dict(fc=spec.fc, conc_model="Kent-Park",
                                        eps_c0=spec.eps_c0, eps_cu=spec.eps_cu,
                                        fcu_ratio=spec.fcu_ratio))
    do_state = eps0 is not None and kappa is not None
    rows = []
    for i, (poly, matd) in enumerate(polys):
        later = unary_union(plist[i + 1:]) if i + 1 < len(plist) else None
        eff = (poly.difference(later)
               if (later is not None and not later.is_empty) else poly)
        if eff.is_empty:
            continue
        b = eff.bounds
        nz_s = max(2, int(round((b[2] - b[0]) / max(csz_z, 1e-9))))
        ny_s = max(2, int(round((b[3] - b[1]) / max(csz_y, 1e-9))))
        label = _mat_label(matd)
        law = (concrete_uniaxial_from(matd) if matd.get("kind") == "concrete"
               else steel_uniaxial_from(matd)) if do_state else dummy
        for f in _discretize_polygon_to_fibers(eff, dummy, n_z=nz_s, n_y=ny_s):
            row = {"area": float(f.area), "z": float(f.z),
                   "y": float(f.y), "mat": label, "cell": True}
            if do_state:
                eps = eps0 - float(f.y) * kappa
                row["strain"] = eps
                row["stress"] = float(law.get_response(eps)[0])
            rows.append(row)
    return rows


# Polar mesh divisions + annular-sector cells now live in the engine
# (femsolver.sections.response.fiber) so the analysis fibers and this fibre view
# share one implementation of the maths (plan §15 U2). Kept aliased under the
# old private name for the call sites below.
_polar_divisions = polar_divisions


def section_fiber_mesh(spec: "Spec", target: int = 1400) -> dict:
    """The fibre discretisation grid clipped to the section — the cell
    boundaries, for drawing the actual mesh (CSiBridge-style). Returns
    ``{"segments": [((z0,y0),(z1,y1)), ...], "n_z", "n_y"}`` in SI metres.
    Circular sections use a polar (rings + spokes) mesh; everything else uses the
    Cartesian n_z × n_y grid intersected with the (holed) polygon so only the
    parts inside the section are returned."""
    from shapely.geometry import LineString

    if spec.kind == "Circular":
        R = float(spec.D) / 2.0
        n_r, n_theta = _polar_divisions(target)
        dr, dth = R / n_r, 2.0 * math.pi / n_theta
        segments = []
        for i in range(1, n_r + 1):                    # rings (arcs)
            r = i * dr
            pts = [(r * math.cos(k * dth), r * math.sin(k * dth))
                   for k in range(n_theta + 1)]
            segments.extend(zip(pts[:-1], pts[1:]))
        for j in range(n_theta):                       # spokes
            th = j * dth
            segments.append(((0.0, 0.0),
                             (R * math.cos(th), R * math.sin(th))))
        return {"segments": segments, "n_z": n_r, "n_y": n_theta}

    case = build_case(spec)
    poly = case.section.geometry.polygon
    minz, miny, maxz, maxy = poly.bounds
    w = max(maxz - minz, 1e-9)
    h = max(maxy - miny, 1e-9)
    n_z = max(6, int(round((target * w / h) ** 0.5)))
    n_y = max(6, int(round((target * h / w) ** 0.5)))
    dz, dy = w / n_z, h / n_y
    segments = []

    def _collect(inter):
        geoms = getattr(inter, "geoms", None) or [inter]
        for g in geoms:
            cs = list(getattr(g, "coords", []))
            for a, b in zip(cs[:-1], cs[1:]):
                segments.append(((float(a[0]), float(a[1])),
                                 (float(b[0]), float(b[1]))))

    for i in range(n_z + 1):
        zi = minz + i * dz
        _collect(poly.intersection(LineString([(zi, miny), (zi, maxy)])))
    for j in range(n_y + 1):
        yj = miny + j * dy
        _collect(poly.intersection(LineString([(minz, yj), (maxz, yj)])))
    return {"segments": segments, "n_z": n_z, "n_y": n_y}


def section_fibers(spec: "Spec", target: int = 1400,
                   eps0=None, kappa=None) -> dict:
    """Discretise the section the way the fibre analysis does and return the
    fibre list + fibre-derived properties, for a CSiBridge-style fibre view.

    Returns ``{"fibers": [...], "n_z", "n_y", "fiber_props", "solid_props",
    "has_state"}``. Each fibre is ``{"area", "z", "y", "mat", "cell"}`` in SI
    (m², m); with a strain plane ``eps0`` (axial) + ``kappa`` (curvature, about
    the horizontal axis) each fibre also gets ``"strain"`` = eps0 − y·κ and
    ``"stress"`` = its material's response (Pa). Cells come from grid-sampling
    the (holed) polygon — one material for a single-material section, per-shape
    for Composite; each rebar / tendon is one point fibre. Properties integrate
    over the CELL fibres. ``target`` is the approximate cell count."""
    case = build_case(spec)
    sec = case.section
    poly = sec.geometry.polygon
    minz, miny, maxz, maxy = poly.bounds
    w = max(maxz - minz, 1e-9)
    h = max(maxy - miny, 1e-9)
    n_z = max(6, int(round((target * w / h) ** 0.5)))
    n_y = max(6, int(round((target * h / w) ** 0.5)))
    do_state = eps0 is not None and kappa is not None

    conc_law = concrete_uniaxial_from(dict(
        fc=spec.fc, conc_model=spec.conc_model, eps_c0=spec.eps_c0,
        eps_cu=spec.eps_cu, fcu_ratio=spec.fcu_ratio,
        fr_model=spec.fr_model, fr_coeff=spec.fr_coeff,
        eps_decay=spec.eps_decay, conc_f1_ratio=spec.conc_f1_ratio))
    steel_law = steel_uniaxial_from(dict(
        fy=spec.fy, Es=spec.Es, steel_model=spec.steel_model,
        steel_b=spec.steel_b, steel_fu_ratio=spec.steel_fu_ratio,
        steel_eps_sh=spec.steel_eps_sh, steel_eps_su=spec.steel_eps_su))

    def _state(row, law):
        if do_state:
            eps = eps0 - row["y"] * kappa
            row["strain"] = eps
            row["stress"] = float(law.get_response(eps)[0])
        return row

    if spec.kind == "Circular":
        # polar (rings × sectors) discretisation via the shared engine mesher
        R = float(spec.D) / 2.0
        n_z, n_y = polar_divisions(target)             # rings, sectors
        cells = [_state({"area": area, "z": z, "y": y,
                         "mat": "Concrete", "cell": True}, conc_law)
                 for (y, z, area) in polar_cells(0.0, R, n_z, n_y)]
    elif spec.kind == "Composite":
        cells = _composite_cell_fibers(spec, w / n_z, h / n_y, eps0, kappa)
    else:
        cells = [_state({"area": float(f.area), "z": float(f.z),
                         "y": float(f.y), "mat": "Concrete", "cell": True},
                        conc_law)
                 for f in _discretize_polygon_to_fibers(poly, conc_law,
                                                        n_z=n_z, n_y=n_y)]
    fibers = list(cells)
    bars = sec.reinforcement.bars if sec.reinforcement else []
    for b in bars:
        fibers.append(_state(
            {"area": float(b.area), "z": float(b.z), "y": float(b.y),
             "mat": "Steel", "cell": False},
            getattr(b, "material", None) or steel_law))
    tendons = (sec.prestress.tendons
               if getattr(sec, "prestress", None) else [])
    for t in tendons:
        law = getattr(t, "material", None) or steel_law
        fibers.append(_state(
            {"area": float(t.area), "z": float(t.z), "y": float(t.y),
             "mat": "Tendon", "cell": False}, law))

    A = sum(f["area"] for f in cells)
    if A > 0:
        cz = sum(f["area"] * f["z"] for f in cells) / A
        cy = sum(f["area"] * f["y"] for f in cells) / A
        I_zz = sum(f["area"] * (f["y"] - cy) ** 2 for f in cells)
        I_yy = sum(f["area"] * (f["z"] - cz) ** 2 for f in cells)
    else:
        cz = cy = I_zz = I_yy = 0.0
    g = sec.geometry
    gcz, gcy = g.centroid
    return {
        "fibers": fibers, "n_z": n_z, "n_y": n_y, "has_state": do_state,
        "fiber_props": {"A": A, "cz": cz, "cy": cy, "I_zz": I_zz, "I_yy": I_yy},
        "solid_props": {"A": g.area, "cz": gcz, "cy": gcy,
                        "I_zz": g.I_zz, "I_yy": g.I_yy},
    }


def props_of(case: SectionCase) -> dict:
    g = case.section.geometry
    cz, cy = g.centroid
    return {
        "Gross area A_g [mm²]": g.area * 1e6,
        "I_zz [mm⁴]": g.I_zz * 1e12,
        "I_yy [mm⁴]": g.I_yy * 1e12,
        "Centroid above bottom [mm]": (cy - g.polygon.bounds[1]) * 1e3,
    }


def svg_of(case: SectionCase, *, show_axes: bool = False,
           axis_labels: tuple = ("z", "y")) -> str:
    return section_to_svg(case.section, width_px=380, show_rebar=True,
                          show_dimensions=True, show_axes=show_axes,
                          axis_labels=axis_labels)


def _pm_diagram_svg(curve: dict, u: "Units", W: int = 470, H: int = 330) -> str:
    """Inline-SVG P-M interaction diagram (nominal solid + φ-design dashed)
    for the calc report. Self-contained vector graphic, no matplotlib."""
    Mn = [u.M_disp(m) for m in curve["M_nom"]]
    Pn = [u.P_disp(p) for p in curve["P_nom"]]
    has_d = bool(curve.get("has_design"))
    Md = [u.M_disp(m) for m in curve["M_des"]] if has_d else []
    Pd = [u.P_disp(p) for p in curve["P_des"]] if has_d else []
    Mmax = (max(Mn + Md + [0.0]) or 1.0) * 1.08
    allP = Pn + Pd or [0.0, 1.0]
    Pmin, Pmax = min(allP), max(allP)
    if Pmax == Pmin:
        Pmax = Pmin + 1.0
    pad = (Pmax - Pmin) * 0.06
    Pmin -= pad; Pmax += pad
    ml, mr, mt, mb = 64, 14, 14, 40
    pw, ph = W - ml - mr, H - mt - mb

    def sx(M):
        return ml + (M / Mmax) * pw

    def sy(P):
        return mt + (Pmax - P) / (Pmax - Pmin) * ph

    def poly(Ms, Ps, color, dash=""):
        pts = " ".join(f"{sx(m):.1f},{sy(p):.1f}" for m, p in zip(Ms, Ps))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<polyline points="{pts}" fill="none" stroke="{color}" '
                f'stroke-width="2"{d}/>')

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             f'font-family="sans-serif" font-size="11">']
    parts.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" '
                 'fill="#fbfbfd" stroke="#ccc"/>')
    # zero-P line
    if Pmin < 0 < Pmax:
        y0 = sy(0.0)
        parts.append(f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml + pw}" '
                     f'y2="{y0:.1f}" stroke="#bbb" stroke-dasharray="3 3"/>')
    # ticks: P at min/0/max, M at 0/max
    for P in sorted({Pmin + pad, 0.0 if Pmin < 0 < Pmax else Pmin + pad,
                     Pmax - pad}):
        y = sy(P)
        parts.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end" '
                     f'fill="#555">{P:.4g}</text>')
    for M in (0.0, Mmax / (1.08)):
        x = sx(M)
        parts.append(f'<text x="{x:.1f}" y="{mt + ph + 15}" '
                     f'text-anchor="middle" fill="#555">{M:.4g}</text>')
    parts.append(poly(Mn, Pn, "#c0392b"))
    if has_d:
        parts.append(poly(Md, Pd, "#c0392b", dash="5 4"))
    # axis titles
    parts.append(f'<text x="{ml + pw / 2:.0f}" y="{H - 4}" '
                 f'text-anchor="middle" fill="#333">M [{u.Ml}]</text>')
    parts.append(f'<text x="14" y="{mt + ph / 2:.0f}" text-anchor="middle" '
                 f'fill="#333" transform="rotate(-90 14 {mt + ph / 2:.0f})">'
                 f'P [{u.Fl}, + comp.]</text>')
    parts.append('</svg>')
    return "".join(parts)


def fiber_mesh_svg(spec: "Spec", target: int = 1000,
                   W: int = 300, H: int = 300) -> str:
    """Inline-SVG of the fibre discretisation (mesh cell boundaries + the
    section outline) for the calc report. Self-contained vector graphic."""
    try:
        mesh = section_fiber_mesh(spec, target)
        case = build_case(spec)
        poly = case.section.geometry.polygon
        ext = list(poly.exterior.coords)
        interiors = [list(r.coords) for r in poly.interiors]
    except Exception:                                  # noqa: BLE001
        return ""
    segs = mesh.get("segments") or []
    xs = [p[0] for p in ext] or [-0.2, 0.2]
    ys = [p[1] for p in ext] or [-0.3, 0.3]
    minz, maxz, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w = max(maxz - minz, 1e-9)
    h = max(maxy - miny, 1e-9)
    pad = 10
    sc = min((W - 2 * pad) / w, (H - 2 * pad) / h)
    ox = (W - w * sc) / 2.0
    oy = (H - h * sc) / 2.0

    def X(z):
        return ox + (z - minz) * sc

    def Y(y):
        return H - oy - (y - miny) * sc            # y up

    def ring(pts):
        d = "M" + " L".join(f"{X(z):.1f},{Y(y):.1f}" for z, y in pts) + " Z"
        return d

    mesh_d = "".join(
        f"M{X(a[0]):.1f},{Y(a[1]):.1f} L{X(b[0]):.1f},{Y(b[1]):.1f}"
        for a, b in segs)
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(f'<path d="{ring(ext)}" fill="#eef4fb" stroke="none"/>')
    if mesh_d:
        parts.append(f'<path d="{mesh_d}" fill="none" stroke="#9fb0c3" '
                     'stroke-width="0.5"/>')
    parts.append(f'<path d="{ring(ext)}" fill="none" stroke="#1f4f73" '
                 'stroke-width="1.4"/>')
    for r in interiors:
        parts.append(f'<path d="{ring(r)}" fill="#ffffff" stroke="#1f4f73" '
                     'stroke-width="1.2"/>')
    parts.append('</svg>')
    return "".join(parts)


_REPORT_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif;
       color: #1a1a1a; margin: 0; font-size: 13px; line-height: 1.45; }
.sheet { max-width: 900px; margin: 0 auto; padding: 26px 30px; }
.rhead { display: flex; justify-content: space-between; align-items: flex-start;
         border-bottom: 3px solid #c0392b; padding-bottom: 10px; }
.rhead h1 { font-size: 20px; margin: 0 0 3px; }
.rhead .sub { color: #666; font-size: 12px; }
.brandmark { display: flex; align-items: center; gap: 7px; margin-bottom: 6px; }
.brandmark svg { width: 22px; height: 22px; }
.brandmark span { font-size: 15px; font-weight: 700; letter-spacing: -0.01em;
                  color: #c0392b; }
.meta { text-align: right; font-size: 11px; color: #444; }
.meta b { color: #111; }
h2 { font-size: 14px; margin: 22px 0 8px; padding-bottom: 4px;
     border-bottom: 1px solid #ddd; color: #c0392b; }
.cols { display: flex; gap: 22px; align-items: flex-start; }
.col { flex: 1; }
table { border-collapse: collapse; width: 100%; margin: 4px 0; }
th, td { border: 1px solid #dcdcdc; padding: 4px 8px; text-align: left; }
th { background: #f4f4f6; font-weight: 600; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
.kv th { width: 55%; background: #fafafa; font-weight: 500; }
.ok { color: #157a3f; font-weight: 700; }
.fail { color: #c0392b; font-weight: 700; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 4px;
         font-weight: 700; font-size: 12px; }
.badge.ok { background: #e5f5eb; }
.badge.fail { background: #fbe6e4; }
.foot { margin-top: 26px; padding-top: 10px; border-top: 1px solid #ddd;
        color: #888; font-size: 10.5px; }
.sketch svg, .diagram svg { max-width: 100%; height: auto; }
@media print { .sheet { max-width: none; } h2 { break-after: avoid; }
               table, .cols { break-inside: avoid; } }
"""


def report_html(case: SectionCase, code: str, u: "Units", *,
                mphi: Optional[dict] = None,
                mphi_exact: Optional[dict] = None,
                demand_results: Optional[list] = None,
                meta: Optional[dict] = None,
                axis_labels: tuple = ("z", "y"),
                brand: str = "",
                logo_svg: str = "",
                fiber_svg: str = "") -> str:
    """A self-contained HTML calc sheet for one section: sketch, materials,
    reinforcement schedule, P-M interaction diagram + capacity landmarks,
    moment-curvature summary, and (if provided) the demand/utilization check.
    Prints cleanly to PDF from a browser. ``axis_labels`` is ``(horizontal,
    vertical)`` axis letters for the inertia subscripts (the strong/horizontal-
    axis inertia is engine ``I_zz``, the weak/vertical one ``I_yy``); default
    ``("z","y")`` keeps the engine convention."""
    meta = meta or {}
    g = case.section.geometry
    esc = _html_escape
    h, v = axis_labels

    # ---- properties / materials ----
    A_g = u.area_disp(g.area * 1e6)
    props = [
        (f"Gross area A_g [{u.Al}]", A_g),
        (f"Depth × width [{u.Ll}]",
         f"{u.len_disp(g.depth * 1e3):.4g} × {u.len_disp(g.width * 1e3):.4g}"),
        (f"I_{h}{h} (strong) [{u.Il}]", f"{u.inertia_disp(g.I_zz * 1e12):.4g}"),
        (f"I_{v}{v} (weak) [{u.Il}]", f"{u.inertia_disp(g.I_yy * 1e12):.4g}"),
    ]
    mats = [
        (f"Concrete f'c / f_ck [{u.Sl}]", f"{u.from_Pa(case.f_c_prime):.4g}"),
        (f"Steel f_y / f_yk [{u.Sl}]", f"{u.from_Pa(case.f_y):.4g}"),
        ("Section type", esc(case.kind)),
        ("Prestressed", "yes" if case.prestressed else "no"),
    ]

    # ---- reinforcement schedule ----
    bars = (case.section.reinforcement.bars
            if case.section.reinforcement else [])
    As = sum(b.area for b in bars)
    rho = (As / g.area * 100.0) if g.area > 0 else 0.0
    reb_rows = "".join(
        f'<tr><td>{esc(b.designation)}</td>'
        f'<td class="n">{u.len_disp(b.z * 1e3):.4g}</td>'
        f'<td class="n">{u.len_disp(b.y * 1e3):.4g}</td>'
        f'<td class="n">{u.area_disp(b.area * 1e6):.4g}</td></tr>'
        for b in bars)
    tendons = (case.section.prestress.tendons
               if getattr(case.section, "prestress", None) else [])
    reb_html = ""
    if bars:
        reb_html = (
            f'<table><thead><tr><th>Bar</th><th class="n">z [{u.Ll}]</th>'
            f'<th class="n">y [{u.Ll}]</th><th class="n">A [{u.Al}]</th>'
            f'</tr></thead><tbody>{reb_rows}</tbody></table>'
            f'<p>{len(bars)} bars · total A_s = {u.area_disp(As * 1e6):.4g} '
            f'{u.Al} · ρ = {rho:.2f}%'
            + (f' · {len(tendons)} tendons' if tendons else '') + '</p>')

    # ---- interaction diagram + landmarks ----
    curve, lm = pmm_slice(case, code)
    diagram = _pm_diagram_svg(curve, u)
    lm_html = ""
    if lm:
        lm_rows = "".join(
            f'<tr><th>{esc(name)}</th><td class="n">'
            f'{(u.P_disp(val) if kind == "P" else u.M_disp(val)):.4g} '
            f'{u.Fl if kind == "P" else u.Ml}</td></tr>'
            for name, val, kind in lm)
        lm_html = f'<table class="kv">{lm_rows}</table>'

    # ---- moment-curvature summary (fibre; + exact-integration column when
    # provided — the two agree as the fibre grid refines, so it doubles as a
    # convergence cross-check on the calc sheet) ----
    if mphi is None:
        mphi = mphi_data(case, 0.0)

    def _mphi_vals(m):
        return {
            "M_cr": f"{u.M_disp(m.get('M_cr') or 0):.4g}",
            "M_y": f"{u.M_disp(m['M_y']):.4g}" if m.get("M_y") else "—",
            "M_u": f"{u.M_disp(m.get('M_u') or 0):.4g}",
            "mu": f"{m['mu_phi']:.2f}" if m.get("mu_phi") else "—",
            "fail": esc(m.get("failure_mode", "")),
        }

    _mphi_labels = (
        (f"Cracking M_cr [{u.Ml}]", "M_cr"),
        (f"First-yield M_y [{u.Ml}]", "M_y"),
        (f"Ultimate M_u [{u.Ml}]", "M_u"),
        ("Curvature ductility μ_φ", "mu"),
        ("Failure mode", "fail"),
    )
    fv = _mphi_vals(mphi)
    if mphi_exact:
        ev = _mphi_vals(mphi_exact)
        cmp_rows = "".join(
            f'<tr><td>{lab}</td><td class="n">{fv[k]}</td>'
            f'<td class="n">{ev[k]}</td></tr>' for lab, k in _mphi_labels)
        mphi_html = (
            f'<table><thead><tr><th>Quantity</th>'
            f'<th class="n">Fibre model</th><th class="n">Exact integ.</th>'
            f'</tr></thead><tbody>{cmp_rows}</tbody></table>'
            '<p style="color:#888;font-size:10.5px">Fibre = discretised '
            'fibre grid; Exact = Gauss-quadrature integration over the true '
            'section (no discretisation error). Agreement confirms the fibre '
            'mesh has converged.</p>')
    else:
        mphi_rows = "".join(
            f'<tr><th>{lab}</th><td class="n">{fv[k]}</td></tr>'
            for lab, k in _mphi_labels)
        mphi_html = f'<table class="kv">{mphi_rows}</table>'

    # ---- demand check ----
    dem_html = ""
    if demand_results:
        gov = max(demand_results, key=lambda d: d["util"])
        verdict = "OK" if gov["util"] <= 1.0 else "FAIL"
        cls = "ok" if verdict == "OK" else "fail"
        drows = "".join(
            f'<tr><td>{esc(d["name"])}</td>'
            f'<td class="n">{u.P_disp(d["P"]):.4g}</td>'
            f'<td class="n">{u.M_disp(d["Mz"]):.4g}</td>'
            f'<td class="n">{u.M_disp(d["My"]):.4g}</td>'
            f'<td class="n">{u.M_disp(d["M_res"]):.4g}</td>'
            f'<td class="n">{u.M_disp(d["M_cap"]):.4g}</td>'
            f'<td class="n">{d["util"] * 100:.1f}%</td>'
            f'<td class="{"ok" if d["status"] == "OK" else "fail"}">'
            f'{d["status"]}</td></tr>'
            for d in demand_results)
        dem_html = (
            f'<h2>Demand check — {esc(meta.get("surface", "design"))} '
            f'surface, {esc(meta.get("dc_method", "keep P constant"))}</h2>'
            f'<p>Governing: <b>{esc(gov["name"] or "—")}</b> at '
            f'<span class="badge {cls}">{gov["util"] * 100:.1f}% · '
            f'{verdict}</span></p>'
            f'<table><thead><tr><th>Combo</th><th class="n">P [{u.Fl}]</th>'
            f'<th class="n">Mz [{u.Ml}]</th><th class="n">My [{u.Ml}]</th>'
            f'<th class="n">M appl [{u.Ml}]</th>'
            f'<th class="n">M cap [{u.Ml}]</th><th class="n">D/C</th>'
            f'<th>Status</th></tr></thead><tbody>{drows}</tbody></table>'
            '<p style="color:#888;font-size:10.5px">Utilization = |applied '
            'moment| ÷ capacity at constant axial load (load-contour method).'
            '</p>')

    kv = lambda pairs: "".join(  # noqa: E731
        f'<tr><th>{k}</th><td class="n">{v}</td></tr>' for k, v in pairs)
    title = esc(meta.get("title") or "Section Design Report")
    brand_html = ""
    if brand or logo_svg:
        brand_html = (f'<div class="brandmark">{logo_svg}'
                      f'<span>{esc(brand)}</span></div>')
    meta_rows = [("Company", meta.get("company")),
                 ("Project", meta.get("project")),
                 ("Engineer", meta.get("engineer")),
                 ("Job no.", meta.get("job")),
                 ("Date", meta.get("date"))]
    meta_html = "".join(f"<b>{k}:</b> {esc(str(v))}<br>"
                        for k, v in meta_rows if v)
    fiber_html = ""
    if fiber_svg:
        fiber_html = (
            f'<div class="col diagram"><h2>Fibre discretisation</h2>{fiber_svg}'
            '<p style="color:#888;font-size:10.5px">Section fibres used for the '
            'moment-curvature and biaxial capacity (cell mesh).</p></div>')
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{esc(case.name)} — report</title><style>{_REPORT_CSS}</style></head>
<body><div class="sheet">
<div class="rhead"><div>{brand_html}<h1>{title}</h1>
<div class="sub">{esc(case.name)} · design code <b>{esc(code)}</b></div></div>
<div class="meta">{meta_html or '—'}</div></div>

<div class="cols"><div class="col sketch"><h2>Cross-section</h2>{svg_of(case)}</div>
<div class="col"><h2>Geometry</h2><table class="kv">{kv(props)}</table>
<h2>Materials</h2><table class="kv">{kv(mats)}</table></div></div>

<h2>Reinforcement</h2>{reb_html or '<p>No bars.</p>'}

<div class="cols"><div class="col diagram"><h2>P-M interaction ({esc(code)})</h2>
{diagram}<p style="color:#888;font-size:10.5px">Solid = nominal · dashed =
φ-reduced design (strong axis, θ=0).</p></div>
<div class="col"><h2>Capacity landmarks</h2>{lm_html or '<p>—</p>'}
<h2>Moment-curvature</h2>{mphi_html}</div></div>

{f'<div class="cols">{fiber_html}<div class="col"></div></div>' if fiber_html else ''}

{dem_html}

<div class="foot">Generated by {esc(brand) + ' · ' if brand else ''}the femsolver
desktop · fiber-section
biaxial P-M-M (Whitney/parabolic stress block per code) and moment-curvature
({esc(mphi.get("conc_model", "Kent-Park"))} concrete + tension stiffening,
{esc(mphi.get("steel_model", "Bilinear"))} steel). Engineering check to be
reviewed by a qualified engineer.</div>
</div></body></html>"""


def _html_escape(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def section_json_of(case: SectionCase) -> str | None:
    try:
        return section_to_json(case.section)
    except Exception:
        return None


def items_csv(items: list[VerificationItem]) -> str:
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        export_csv(items, path)
        return Path(path).read_text(encoding="utf-8")
    finally:
        os.remove(path)


# ---- whole-project save / load (the section library) -------------------

def project_to_json(sections: dict, active: str, demands: dict = None,
                    materials: dict = None) -> str:
    """Serialise the whole section library ({name: {"spec", "code"}}), each
    section's factored load combinations (``demands`` =
    ``{section: [(combo, P, Mz, My), ...]}``, canonical kN/kN·m), and the
    project-level named materials library (``materials`` = {name: {kind,
    ...props}}) to a JSON string."""
    import dataclasses
    import json
    demands = demands or {}
    data = {
        "app": "femsolver-section-designer", "version": 3, "active": active,
        "materials": materials or {},
        "sections": {
            name: {"code": rec["code"],
                   "conc_mat": rec.get("conc_mat"),
                   "steel_mat": rec.get("steel_mat"),
                   "spec": dataclasses.asdict(rec["spec"]),
                   "demands": [list(d) for d in demands.get(name, [])]}
            for name, rec in sections.items()},
    }
    return json.dumps(data, indent=2)


def project_from_json(text: str):
    """Parse a saved project back to ``(sections, active, demands, materials)``.
    Unknown spec fields are dropped and missing ones fall back to :class:`Spec`
    defaults, so older/newer files still load; ``demands`` come back as tuples
    (hashable for caching); ``materials`` is the named library ({} if absent)."""
    import json
    data = json.loads(text)
    fields = set(Spec.__dataclass_fields__)
    sections: dict = {}
    demands: dict = {}
    for name, rec in (data.get("sections") or {}).items():
        nm = str(name)
        sd = {k: v for k, v in (rec.get("spec") or {}).items() if k in fields}
        sections[nm] = {
            "code": rec.get("code", "AASHTO LRFD 2024"),
            "conc_mat": rec.get("conc_mat"),
            "steel_mat": rec.get("steel_mat"),
            "spec": Spec(**sd)}
        dlist = rec.get("demands") or []
        combos = [(str(c), float(p), float(mz), float(my))
                  for (c, p, mz, my) in dlist]
        if combos:
            demands[nm] = combos
    if not sections:
        raise ValueError("no sections found in file")
    active = data.get("active")
    if active not in sections:
        active = next(iter(sections))
    materials = dict(data.get("materials") or {})
    return sections, active, demands, materials
