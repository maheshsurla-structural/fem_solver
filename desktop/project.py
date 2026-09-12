"""The project — the serializable information model, the app's single source of truth.

A ``Project`` is a *declarative* description of the structure (nodes, members
and their sections + materials, supports, loads) plus metadata (name, units,
design code). It is what the app opens and saves. ``build_model()`` compiles
it into a transient ``femsolver.Model`` for analysis; results are never stored
in the project — they are recomputed from it.

This is the spine every future view (analysis, design, drawings, and the
desktop Section Designer) reads and writes. Today it compiles 2-D frames
(``BeamColumn2D``); element/section breadth grows from here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

SCHEMA = "femsolver-project/1"


@dataclass
class Material:
    id: int
    name: str
    E: float
    nu: float = 0.3
    kind: str = "elastic_isotropic"
    fy: float = 345.0e6           # yield (A992 = 345 MPa) — for design checks
    fu: float = 448.0e6           # ultimate (A992 = 448 MPa)
    # Inelastic (fiber) constitutive parameters, per ``kind`` — e.g. concrete
    # {fc, eps_c0, eps_cu, Ec, fpcu_ratio}, steel {E, fy, fu, eps_sh, eps_su}.
    # Elastic materials leave this empty. Consumed by desktop.materials
    # (`uniaxial_law` / `stress_strain_curve`) to build the engine law.
    params: dict = field(default_factory=dict)


@dataclass
class Section:
    id: int
    name: str
    A: float
    Iz: float
    shape: str = ""               # AISC W-shape (e.g. "W12x65"); drives design
    Iy: float = 0.0               # 3-D weak-axis inertia (from shape if named)
    J: float = 0.0                # 3-D torsion constant (from shape if named)
    # A concrete/PSC/composite section authored in the General Section Designer.
    # When present, ``gsd_spec`` (a serialized ``section_gui_core.Spec``) is the
    # source of truth for A / Iz / Iy / J (see ``_resolve_section``) and carries
    # the reinforcement for future P-M-M member design. ``gsd_code`` is its
    # Section-Designer design code.
    gsd_spec: dict | None = None
    gsd_code: str = ""


@dataclass
class Node:
    id: int
    x: float
    y: float
    z: float = 0.0
    supports: tuple = ()          # fixity mask (len ndf); empty = free


@dataclass
class Member:
    id: int
    n1: int
    n2: int
    section: int
    material: int
    kind: str = "beamcolumn2d"


# Load "nature" -> ASCE 7 pattern key used by the code combinations. ``None``
# means the case carries no code factor of its own (user combos only).
NATURE_ASCE = {"dead": "D", "live": "L", "roof_live": "Lr", "snow": "S",
               "rain": "R", "wind": "W", "seismic": "E", "other": None}
NATURE_LABELS = {"dead": "Dead", "live": "Live", "roof_live": "Roof live",
                 "snow": "Snow", "rain": "Rain", "wind": "Wind",
                 "seismic": "Seismic", "other": "Other"}


@dataclass
class LoadCase:
    """A named physical load case (Dead, Live, Wind, …). Loads belong to a
    case; combinations factor cases together. ``nature`` maps the case to an
    ASCE 7 pattern key so code combinations can be generated automatically."""
    id: int
    name: str
    nature: str = "dead"          # key of NATURE_ASCE


@dataclass
class Load:
    node: int
    values: tuple                 # nodal load vector (len ndf)
    case: int = 1                 # owning LoadCase id


@dataclass
class MemberLoad:
    """A uniform transverse line load on a member, in the member's local axes
    (N/m). ``wz`` is used in 3-D only."""
    member: int
    wy: float = 0.0               # local-y UDL (N/m)
    wz: float = 0.0               # local-z UDL (N/m), 3-D only
    case: int = 1                 # owning LoadCase id


@dataclass
class LoadCombination:
    """A weighted sum of load cases. ``factors`` maps case id -> factor."""
    id: int
    name: str
    factors: dict = field(default_factory=dict)   # {case_id: factor}


@dataclass
class Project:
    name: str = "Untitled"
    ndm: int = 2
    ndf: int = 3
    force_unit: str = "N"          # project stores SI base units (N, m, Pa)
    length_unit: str = "m"
    design_code: str = "AISC 360"
    materials: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    members: list = field(default_factory=list)
    load_cases: list = field(default_factory=list)    # LoadCase
    loads: list = field(default_factory=list)          # nodal Load
    member_loads: list = field(default_factory=list)   # MemberLoad (line loads)
    combinations: list = field(default_factory=list)   # LoadCombination

    def __post_init__(self):
        if not self.load_cases:
            self.load_cases.append(LoadCase(id=1, name="Dead", nature="dead"))

    # ------------------------------------------------------------- load cases
    def case(self, case_id):
        return next((c for c in self.load_cases if c.id == case_id), None)

    def combination(self, combo_id):
        return next((c for c in self.combinations if c.id == combo_id), None)

    def default_case_id(self) -> int:
        return self.load_cases[0].id if self.load_cases else 1

    def generate_asce7_combinations(self) -> list:
        """ASCE 7-22 LRFD strength combinations as project ``LoadCombination``s,
        mapping each case to its pattern key (D, L, W, …) via its nature. Combos
        that reduce to the same factor set — e.g. a project with no Lr/S/R cases
        — are de-duplicated. Returns the new combinations (caller adds them)."""
        from femsolver.analysis.load_combinations import asce7_lrfd_combinations
        key_to_cases: dict = {}
        for c in self.load_cases:
            key = NATURE_ASCE.get(c.nature)
            if key:
                key_to_cases.setdefault(key, []).append(c.id)
        out, seen = [], set()
        next_id = max((c.id for c in self.combinations), default=0) + 1
        for ec in asce7_lrfd_combinations():
            factors: dict = {}
            for key, f in ec.factors.items():
                for cid in key_to_cases.get(key, []):
                    factors[cid] = factors.get(cid, 0.0) + f
            if not factors:
                continue
            sig = frozenset(factors.items())
            if sig in seen:
                continue
            seen.add(sig)
            out.append(LoadCombination(id=next_id, name=ec.name,
                                       factors=dict(factors)))
            next_id += 1
        return out

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {"schema": SCHEMA, **asdict(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        cases = [LoadCase(**c) for c in d.get("load_cases", [])]
        if not cases:                       # migrate pre-load-case projects
            cases = [LoadCase(id=1, name="Dead", nature="dead")]
        default_id = cases[0].id
        return cls(
            name=d.get("name", "Untitled"),
            ndm=int(d.get("ndm", 2)),
            ndf=int(d.get("ndf", 3)),
            force_unit=d.get("force_unit", "kN"),
            length_unit=d.get("length_unit", "m"),
            design_code=d.get("design_code", "AISC 360"),
            materials=[Material(**m) for m in d.get("materials", [])],
            sections=[Section(**s) for s in d.get("sections", [])],
            nodes=[Node(**_coerce_node(n)) for n in d.get("nodes", [])],
            members=[Member(**m) for m in d.get("members", [])],
            load_cases=cases,
            loads=[Load(node=x["node"], values=tuple(x["values"]),
                        case=x.get("case", default_id))
                   for x in d.get("loads", [])],
            member_loads=[MemberLoad(member=x["member"], wy=x.get("wy", 0.0),
                                     wz=x.get("wz", 0.0),
                                     case=x.get("case", default_id))
                          for x in d.get("member_loads", [])],
            combinations=[LoadCombination(
                id=c["id"], name=c["name"],
                factors={int(k): v for k, v in c.get("factors", {}).items()})
                for c in d.get("combinations", [])],
        )

    @classmethod
    def from_json(cls, text: str) -> "Project":
        return cls.from_dict(json.loads(text))

    def save(self, path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Project":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # -------------------------------------------------------- compile to solver
    def build_model(self, with_loads: bool = True):
        """Compile this declarative project into a transient femsolver.Model.

        With ``with_loads`` the unfactored sum of every load case is applied
        (ready to solve as-is). Pass ``with_loads=False`` to build geometry +
        supports only, then apply a specific case/combination via
        :meth:`apply_loads` — what the combination/envelope design does."""
        from femsolver import (BeamColumn2D, BeamColumn3D, ElasticIsotropic,
                               Model)

        m = Model(ndm=self.ndm, ndf=self.ndf)
        mats = {}
        for mat in self.materials:
            obj = ElasticIsotropic(mat.id, E=mat.E, nu=mat.nu)
            m.add_material(obj)
            mats[mat.id] = obj
        secs = {s.id: s for s in self.sections}
        for nd in self.nodes:
            coords = (nd.x, nd.y) if self.ndm == 2 else (nd.x, nd.y, nd.z)
            m.add_node(nd.id, *coords)
        gsd_mats: dict = {}          # section id -> concrete ElasticIsotropic
        for mb in self.members:
            sec = secs[mb.section]
            A, Iz, Iy, J = _resolve_section(sec)
            material = mats[mb.material]
            # a Section-Designer (concrete/PSC) section drives its own modulus,
            # so the member's stiffness reflects concrete E_c, not whatever
            # material was assigned in the model.
            if getattr(sec, "gsd_spec", None):
                cm = gsd_mats.get(sec.id)
                if cm is None:
                    try:
                        Ec = _gsd_modulus(sec.gsd_spec)
                    except Exception:
                        Ec = None
                    if Ec:
                        cm = ElasticIsotropic(10_000 + sec.id, E=Ec, nu=0.2)
                        m.add_material(cm)
                        gsd_mats[sec.id] = cm
                if cm is not None:
                    material = cm
            if self.ndm == 3:
                m.add_element(BeamColumn3D(mb.id, (mb.n1, mb.n2),
                                           material, A, Iy, Iz, J))
            else:
                m.add_element(BeamColumn2D(mb.id, (mb.n1, mb.n2),
                                           material, A, Iz))
        for nd in self.nodes:
            if nd.supports and any(nd.supports):
                m.fix(nd.id, list(nd.supports))
        if with_loads:
            self.apply_loads(m, ("all", None))
        return m

    # ------------------------------------------------- load application
    def _selection_factors(self, selection) -> dict:
        """{case_id: factor} for a load selection (solving or display)."""
        kind = selection[0] if selection else "all"
        if kind == "case":
            return {selection[1]: 1.0}
        if kind == "combination":
            combo = self.combination(selection[1])
            return dict(combo.factors) if combo else {}
        return {c.id: 1.0 for c in self.load_cases}

    def resolved_loads(self, selection=("all", None)):
        """(nodal, member) factored applied-load maps for a selection —
        nodal = {node_id: [components]}, member = {member_id: (wy, wz)}."""
        factors = self._selection_factors(selection)
        nodal: dict = {}
        for ld in self.loads:
            f = factors.get(ld.case, 0.0)
            if not f:
                continue
            acc = nodal.setdefault(ld.node, [0.0] * self.ndf)
            for k, v in enumerate(ld.values):
                if k < len(acc):
                    acc[k] += v * f
        member: dict = {}
        for ml in self.member_loads:
            f = factors.get(ml.case, 0.0)
            if not f:
                continue
            wv = member.setdefault(ml.member, [0.0, 0.0])
            wv[0] += ml.wy * f
            wv[1] += ml.wz * f
        return nodal, member

    def apply_loads(self, model, selection=("all", None)) -> None:
        """Clear the model's loads and apply one selection: ``("all", None)``
        (every case ×1), ``("case", id)``, or ``("combination", id)``."""
        model.clear_loads()
        kind = selection[0] if selection else "all"
        if kind == "case":
            self.apply_case(model, selection[1], 1.0)
        elif kind == "combination":
            combo = self.combination(selection[1])
            if combo:
                for cid, f in combo.factors.items():
                    if f:
                        self.apply_case(model, cid, f)
        else:                                    # "all" (or unknown → all)
            for c in self.load_cases:
                self.apply_case(model, c.id, 1.0)

    def _apply_member_load(self, model, ml, factor: float) -> None:
        try:
            el = model.element(ml.member)
        except KeyError:
            return
        if not hasattr(el, "add_uniform_load"):
            return
        if self.ndm == 3:
            el.add_uniform_load(ml.wy * factor, ml.wz * factor)
        else:
            el.add_uniform_load(ml.wy * factor)

    def apply_case(self, model, case_id: int, factor: float = 1.0) -> None:
        """Add one case's nodal + line loads to a built model, scaled by
        ``factor`` (additive — pair with ``model.clear_loads()`` between
        combos)."""
        for ld in self.loads:
            if ld.case == case_id:
                model.add_nodal_load(ld.node, [v * factor for v in ld.values])
        for ml in self.member_loads:
            if ml.case == case_id:
                self._apply_member_load(model, ml, factor)

    def load_patterns(self) -> dict:
        """{str(case_id): LoadPattern} for the engine's combination/envelope
        drivers — one named, scalable pattern per load case."""
        from femsolver.analysis.load_combinations import LoadPattern
        return {str(c.id): LoadPattern(
            str(c.id),
            lambda model, factor=1.0, cid=c.id: self.apply_case(
                model, cid, factor))
            for c in self.load_cases}


def _coerce_node(n: dict) -> dict:
    n = dict(n)
    if n.get("supports") is not None:
        n["supports"] = tuple(n["supports"])
    return n


def _resolve_section(section):
    """(A, Iz, Iy, J) for analysis — from a General-Section-Designer section
    when the section carries a ``gsd_spec`` (concrete/PSC/composite: gross A and
    inertias from the built section), else from the AISC catalog when it names a
    W-shape (Iz = strong-axis Ix), else the section's own values with positive
    fallbacks for the 3-D-only Iy / J."""
    if getattr(section, "gsd_spec", None):
        try:
            return _gsd_section_props(section.gsd_spec)
        except Exception:
            pass
    if section.shape:
        try:
            from femsolver.design.steel.sections import get_section
            ss = get_section(section.shape.replace("X", "x"))
            return ss.A, ss.Ix, ss.Iy, ss.J
        except Exception:
            pass
    Iy = section.Iy or section.Iz
    J = section.J or (0.1 * section.Iz)
    return section.A, section.Iz, Iy, J


def _spec_from_gsd(gsd_spec: dict):
    """Deserialize a stored ``gsd_spec`` dict into a ``section_gui_core.Spec``.
    Imported lazily so the core FEM model never hard-depends on the
    Section-Designer engine; repo root is put on sys.path since the SD engine
    lives one level above ``desktop/``."""
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import section_gui_core as core
    fields = set(core.Spec.__dataclass_fields__)
    return core.Spec(**{k: v for k, v in gsd_spec.items() if k in fields})


@lru_cache(maxsize=128)
def _gsd_built_section(spec):
    """The built ``section_gui_core`` Section for a Spec (cached; Spec is a
    frozen/hashable dataclass, so props + modulus share one build)."""
    import section_gui_core as core
    return core.build_case(spec).section


def _gsd_section_props(gsd_spec: dict):
    """(A, Iz, Iy, J) in SI for a General-Section-Designer section, built from
    its serialized ``section_gui_core.Spec``."""
    sec = _gsd_built_section(_spec_from_gsd(gsd_spec))
    Iz, Iy = sec.I_zz, sec.I_yy
    J = sec.J or (Iz + Iy)          # St-Venant fallback ~ polar for solid shapes
    return sec.area, Iz, Iy, J


def _gsd_modulus(gsd_spec: dict):
    """Concrete elastic modulus E_c [Pa] of a General-Section-Designer section
    (its section's ``primary_material.Ec`` — e.g. ACI 4700·√f'c), so a concrete
    frame member's stiffness reflects concrete, not a steel material assigned in
    the model. Returns None when no concrete modulus is available."""
    sec = _gsd_built_section(_spec_from_gsd(gsd_spec))
    Ec = getattr(getattr(sec, "primary_material", None), "Ec", None)
    return float(Ec) if Ec else None
