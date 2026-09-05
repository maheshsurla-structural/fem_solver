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


@dataclass
class Load:
    node: int
    values: tuple                 # nodal load vector (len ndf)


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
    loads: list = field(default_factory=list)

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {"schema": SCHEMA, **asdict(self)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
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
            loads=[Load(node=x["node"], values=tuple(x["values"]))
                   for x in d.get("loads", [])],
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
    def build_model(self):
        """Compile this declarative project into a transient femsolver.Model."""
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
        for ld in self.loads:
            m.add_nodal_load(ld.node, list(ld.values))
        return m


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
