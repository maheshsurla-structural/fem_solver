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
    rho: float = 0.0              # mass density (kg/m³) — self-mass for modal/RS
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
class Hinge:
    """A finite-length fiber plastic-hinge *property* (plan §14 GUI-3).

    Assigned to a member (``Member.hinge``) it turns that member's fiber
    (Section-Designer) section into a lumped ``FiberHingeBeamColumn2D`` with a
    plastic hinge of length ``lp`` at each end (the CSI "Fiber P-M2-M3" / Midas
    lumped-hinge idiom). ``relative`` makes ``lp`` / ``lp_j`` fractions of the
    member length (CSI "relative hinge length"); otherwise they are absolute
    lengths in project units. ``lp_j`` ``None`` means end J equals end I
    (a symmetric hinge)."""
    id: int
    name: str
    lp: float = 0.1               # end-I hinge length (ratio if ``relative``)
    lp_j: float | None = None     # end-J hinge length; None = same as end I
    relative: bool = True


@dataclass
class Member:
    id: int
    n1: int
    n2: int
    section: int
    material: int
    kind: str = "beamcolumn2d"
    hinge: int | None = None      # Hinge property id (None = distributed/elastic)


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
class NonlinearCase:
    """A saved nonlinear (fiber) analysis case (plan §14 GUI-4) — the desktop
    counterpart of a Midas/CSI nonlinear static load case.

    Displacement-controlled: ``control_node`` DOF ``control_dof`` (0=Ux, 1=Uy,
    2=Rz) is driven to ``target`` following ``protocol``:

    * ``"monotonic"`` — a ramp 0→target over ``n_steps``.
    * ``"cyclic"`` — reversed cycles of growing amplitude; the peaks are
      ``amplitudes`` (fractions) × ``target``, ``cycles`` full cycles each,
      ``pts_per_cycle`` points per cycle.

    ``axial`` (compression magnitude at ``axial_node``/``axial_dof``) is applied
    first and held constant (staged). ``continue_from`` (another case's id)
    continues this case from that case's committed state (staged construction).
    ``tol`` / ``max_iter`` are the Newton controls."""
    id: int
    name: str
    control_node: int = 0
    control_dof: int = 1
    target: float = 0.05
    n_steps: int = 40
    protocol: str = "monotonic"           # "monotonic" | "cyclic"
    amplitudes: list = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    cycles: int = 1
    pts_per_cycle: int = 40
    axial: float = 0.0
    axial_node: int | None = None
    axial_dof: int = 0
    continue_from: int | None = None      # id of a prior NonlinearCase
    tol: float = 1.0e-6
    max_iter: int = 60
    notes: str = ""                        # free-text case notes (GUI only)


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
class Stage:
    """One construction stage: the members that become active ("born") in it,
    in construction order. Members in no stage are treated as built before the
    sequence (initially active). Drives the incremental staged analysis + camber
    (bridge GUI plan G3)."""
    id: int
    name: str
    add_members: list = field(default_factory=list)   # member ids born this stage


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
    hinges: list = field(default_factory=list)         # Hinge properties (GUI-3)
    nodes: list = field(default_factory=list)
    members: list = field(default_factory=list)
    load_cases: list = field(default_factory=list)    # LoadCase
    loads: list = field(default_factory=list)          # nodal Load
    member_loads: list = field(default_factory=list)   # MemberLoad (line loads)
    combinations: list = field(default_factory=list)   # LoadCombination
    stages: list = field(default_factory=list)          # Stage (construction seq)
    nonlinear_cases: list = field(default_factory=list)  # NonlinearCase (GUI-4)
    # Saved nonlinear-run results (plan §16 G-S2) — the expensive, run-specific
    # exception to "results are recomputed": each is a lean ``nl_runs.RunRecord``
    # (curve + summary + ASCE 41 milestones). Not consumed by ``build_model``.
    runs: list = field(default_factory=list)

    def __post_init__(self):
        if not self.load_cases:
            self.load_cases.append(LoadCase(id=1, name="Dead", nature="dead"))

    # ------------------------------------------------------------- load cases
    def case(self, case_id):
        return next((c for c in self.load_cases if c.id == case_id), None)

    def combination(self, combo_id):
        return next((c for c in self.combinations if c.id == combo_id), None)

    def nonlinear_case(self, case_id):
        return next((c for c in self.nonlinear_cases if c.id == case_id), None)

    def default_case_id(self) -> int:
        return self.load_cases[0].id if self.load_cases else 1

    # ------------------------------------------------------------- hinges (GUI-3)
    def hinge(self, hinge_id):
        return next((h for h in self.hinges if h.id == hinge_id), None)

    def member_length(self, member) -> float:
        """Geometric length of ``member`` from its end-node coordinates."""
        nodes = {n.id: n for n in self.nodes}
        a, b = nodes[member.n1], nodes[member.n2]
        d2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2
        if self.ndm == 3:
            d2 += (a.z - b.z) ** 2
        return d2 ** 0.5

    def resolve_hinge_lengths(self, member, hinge) -> tuple:
        """Absolute (lp_i, lp_j) plastic-hinge lengths for ``member`` under
        ``hinge`` — ratios × member length when ``hinge.relative``, else the
        values as given — clamped positive and so lp_i + lp_j < the member
        length (the element requires lp_i + lp_j ≤ L)."""
        L = self.member_length(member) or 1.0
        raw_i = float(hinge.lp)
        raw_j = raw_i if hinge.lp_j is None else float(hinge.lp_j)
        lp_i = raw_i * L if hinge.relative else raw_i
        lp_j = raw_j * L if hinge.relative else raw_j
        lp_i = max(lp_i, 1.0e-9)
        lp_j = max(lp_j, 1.0e-9)
        if lp_i + lp_j >= L:                       # scale down to fit the member
            s = 0.98 * L / (lp_i + lp_j)
            lp_i *= s
            lp_j *= s
        return lp_i, lp_j

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
            force_unit=d.get("force_unit", "N"),   # SI base default (matches the
            length_unit=d.get("length_unit", "m"),  # dataclass; was "kN" — a bug)
            design_code=d.get("design_code", "AISC 360"),
            materials=[Material(**m) for m in d.get("materials", [])],
            sections=[Section(**s) for s in d.get("sections", [])],
            hinges=[Hinge(**h) for h in d.get("hinges", [])],
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
            stages=[Stage(id=s["id"], name=s.get("name", ""),
                          add_members=list(s.get("add_members", [])))
                    for s in d.get("stages", [])],
            nonlinear_cases=[NonlinearCase(**c)
                             for c in d.get("nonlinear_cases", [])],
            runs=_load_runs(d.get("runs", [])),
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
            obj = ElasticIsotropic(mat.id, E=mat.E, nu=mat.nu,
                                   rho=getattr(mat, "rho", 0.0))
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
            # a pin-ended cable / truss carries axial force only
            if getattr(mb, "kind", "") == "cable":
                from femsolver import Truss2D, Truss3D
                if self.ndm == 3:
                    m.add_element(Truss3D(mb.id, (mb.n1, mb.n2), material, A))
                else:
                    m.add_element(Truss2D(mb.id, (mb.n1, mb.n2), material, A))
                continue
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
                        cm = ElasticIsotropic(10_000 + sec.id, E=Ec, nu=0.2,
                                              rho=getattr(material, "rho", 0.0))
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

    def build_buckling_model(self, selection=("all", None),
                             subdivisions: int = 6):
        """Compile a model for linear (eigenvalue) buckling: each member is
        meshed into ``subdivisions`` sub-elements so member (Euler) buckling
        between joints is captured, not just global sway. The reference load
        ``selection`` (``("all", None)`` / ``("case", id)`` /
        ``("combination", id)``) is applied — the buckling factor λ multiplies
        it. 2-D only for now (``BeamColumn2D`` carries the geometric stiffness).

        Returns ``(model, member_subelems)`` where ``member_subelems`` maps
        each original member id to its list of sub-element ids. Works for 2-D
        (``BeamColumn2D``) and 3-D (``BeamColumn3D``) — both carry the
        geometric stiffness eigenvalue buckling needs."""
        from femsolver import (BeamColumn2D, BeamColumn3D, ElasticIsotropic,
                               Model, Truss2D, Truss3D)
        import numpy as np

        d3 = self.ndm == 3
        k = max(1, int(subdivisions))

        m = Model(ndm=self.ndm, ndf=self.ndf)
        mats = {}
        for mat in self.materials:
            obj = ElasticIsotropic(mat.id, E=mat.E, nu=mat.nu,
                                   rho=getattr(mat, "rho", 0.0))
            m.add_material(obj)
            mats[mat.id] = obj
        secs = {s.id: s for s in self.sections}
        gsd_mats: dict = {}

        coord = {}
        for nd in self.nodes:
            c = (nd.x, nd.y, nd.z) if d3 else (nd.x, nd.y)
            m.add_node(nd.id, *c)
            coord[nd.id] = np.array(c, dtype=float)
        nid = max((nd.id for nd in self.nodes), default=0) + 1
        eid = max((mb.id for mb in self.members), default=0) + 1

        member_subelems: dict = {}
        for mb in self.members:
            sec = secs[mb.section]
            A, Iz, Iy, J = _resolve_section(sec)
            material = mats[mb.material]
            # a cable / truss stays a single pin-ended element (no sub-division,
            # no bending — carries axial only)
            if getattr(mb, "kind", "") == "cable":
                cls = Truss3D if d3 else Truss2D
                m.add_element(cls(eid, (mb.n1, mb.n2), material, A))
                member_subelems[mb.id] = [eid]
                eid += 1
                continue
            if getattr(sec, "gsd_spec", None):
                cm = gsd_mats.get(sec.id)
                if cm is None:
                    try:
                        Ec = _gsd_modulus(sec.gsd_spec)
                    except Exception:
                        Ec = None
                    if Ec:
                        cm = ElasticIsotropic(10_000 + sec.id, E=Ec, nu=0.2,
                                              rho=getattr(material, "rho", 0.0))
                        m.add_material(cm)
                        gsd_mats[sec.id] = cm
                if cm is not None:
                    material = cm
            p1, p2 = coord[mb.n1], coord[mb.n2]
            chain = [mb.n1]
            for j in range(1, k):
                t = j / k
                m.add_node(nid, *(p1 + t * (p2 - p1)))
                chain.append(nid)
                nid += 1
            chain.append(mb.n2)
            subs = []
            for a, b in zip(chain[:-1], chain[1:]):
                if d3:
                    m.add_element(BeamColumn3D(eid, (a, b), material,
                                               A, Iy, Iz, J))
                else:
                    m.add_element(BeamColumn2D(eid, (a, b), material, A, Iz))
                subs.append(eid)
                eid += 1
            member_subelems[mb.id] = subs

        for nd in self.nodes:
            if nd.supports and any(nd.supports):
                m.fix(nd.id, list(nd.supports))

        nodal, member = self.resolved_loads(selection)
        for node_id, comps in nodal.items():
            m.add_nodal_load(node_id, list(comps))
        for mid, (wy, wz) in member.items():
            for sid in member_subelems.get(mid, []):
                el = m.element(sid)
                if not hasattr(el, "add_uniform_load"):
                    continue
                if d3:
                    el.add_uniform_load(wy, wz)
                else:
                    el.add_uniform_load(wy)
        return m, member_subelems

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


def _load_runs(raw):
    """Deserialize stored run records (plan §16 G-S2) into ``RunRecord``s,
    tolerant of unknown keys. Imported lazily to keep the project model free of
    a hard dependency on the (desktop-only) results layer."""
    if not raw:
        return []
    from nl_runs import RunRecord
    return [RunRecord.from_dict(r) for r in raw]


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
