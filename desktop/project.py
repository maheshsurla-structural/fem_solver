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
        from femsolver import BeamColumn2D, ElasticIsotropic, Model

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
        for mb in self.members:
            A, Iz = _section_props(secs[mb.section])
            m.add_element(BeamColumn2D(mb.id, (mb.n1, mb.n2),
                                       mats[mb.material], A, Iz))
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


def _section_props(section):
    """(A, Iz) for analysis — from the AISC catalog when the section names a
    W-shape (so the shape drives analysis too), else the section's own A / Iz."""
    if section.shape:
        try:
            from femsolver.design.steel.sections import get_section
            ss = get_section(section.shape.replace("X", "x"))
            return ss.A, ss.Ix
        except Exception:
            pass
    return section.A, section.Iz
