"""JSON serialization of Model. Phase-1 supported elements only."""
from __future__ import annotations

import json
from pathlib import Path

from femsolver.constraints import (
    EqualDOF,
    MPConstraint,
    RigidDiaphragm,
    RigidLink,
)
from femsolver.core.model import Model
from femsolver.elements.beam import BeamColumn2D, BeamColumn3D
from femsolver.elements.plane import Quad4
from femsolver.elements.truss import Truss2D, Truss3D
from femsolver.materials.elastic import ElasticIsotropic


_ELEMENT_REGISTRY = {
    "Truss2D": Truss2D,
    "Truss3D": Truss3D,
    "BeamColumn2D": BeamColumn2D,
    "BeamColumn3D": BeamColumn3D,
    "Quad4": Quad4,
}

_MATERIAL_REGISTRY = {
    "ElasticIsotropic": ElasticIsotropic,
}


# bumped to 0.2 with the addition of MP constraints — old files (no
# "constraints" key) still load cleanly under the same parser.
_FILE_VERSION = "0.2"
_SUPPORTED_VERSIONS = ("0.1", "0.2")


def save_model_json(model: Model, path: str | Path) -> None:
    data = {
        "version": _FILE_VERSION,
        "ndm": model.ndm,
        "ndf": model.ndf,
        "nodes": [
            {
                "tag": n.tag,
                "coords": n.coords.tolist(),
                "fixity": n.fixity.astype(int).tolist(),
                "load": n._load.tolist(),
            }
            for n in model.nodes.values()
        ],
        "materials": [_serialize_material(m) for m in model._materials.values()],
        "elements": [_serialize_element(e) for e in model.elements.values()],
        "constraints": [_serialize_constraint(c) for c in model.mp_constraints],
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_model_json(path: str | Path) -> Model:
    data = json.loads(Path(path).read_text())
    version = data.get("version")
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported model file version {version!r}")
    m = Model(ndm=data["ndm"], ndf=data["ndf"])
    for nd in data["nodes"]:
        node = m.add_node(nd["tag"], *nd["coords"])
        node.fix(nd.get("fixity", [0] * len(nd["coords"])))
        load = nd.get("load")
        if load and any(load):
            node.add_load(load)
    mat_lookup: dict[int, object] = {}
    for md in data["materials"]:
        cls = _MATERIAL_REGISTRY[md["type"]]
        mat = cls(**md["params"])
        m.add_material(mat)
        mat_lookup[mat.tag] = mat
    for ed in data["elements"]:
        cls = _ELEMENT_REGISTRY[ed["type"]]
        params = dict(ed["params"])
        params["material"] = mat_lookup[params.pop("material_tag")]
        m.add_element(cls(**params))
    for cd in data.get("constraints", []):
        m.add_mp_constraint(_deserialize_constraint(cd))
    return m


def _serialize_material(mat) -> dict:
    if isinstance(mat, ElasticIsotropic):
        return {
            "type": "ElasticIsotropic",
            "params": {
                "tag": mat.tag,
                "E": mat.E,
                "nu": mat.nu,
                "rho": mat.rho,
            },
        }
    raise NotImplementedError(f"cannot serialize material type {type(mat).__name__}")


def _serialize_element(e) -> dict:
    common = {"tag": e.tag, "nodes": list(e.node_tags), "material_tag": e.material.tag}
    if isinstance(e, (Truss2D, Truss3D)):
        common["area"] = e.area
        return {"type": type(e).__name__, "params": common}
    if isinstance(e, BeamColumn2D):
        common["area"] = e.area
        common["Iz"] = e.Iz
        return {"type": "BeamColumn2D", "params": common}
    if isinstance(e, BeamColumn3D):
        common["area"] = e.area
        common["Iy"] = e.Iy
        common["Iz"] = e.Iz
        common["J"] = e.J
        if e._vecxz_user is not None:
            common["vecxz"] = e._vecxz_user.tolist()
        return {"type": "BeamColumn3D", "params": common}
    if isinstance(e, Quad4):
        common["thickness"] = e.thickness
        common["state"] = e.state
        common["quadrature"] = e.quadrature
        return {"type": "Quad4", "params": common}
    raise NotImplementedError(f"cannot serialize element type {type(e).__name__}")


def _serialize_constraint(c) -> dict:
    if isinstance(c, EqualDOF):
        return {
            "type": "EqualDOF",
            "params": {
                "retained": c.retained,
                "constrained": c.constrained,
                "dofs": list(c.dofs),
            },
        }
    if isinstance(c, RigidLink):
        return {
            "type": "RigidLink",
            "params": {
                "retained": c.retained,
                "constrained": c.constrained,
                "kind": c.kind,
            },
        }
    if isinstance(c, RigidDiaphragm):
        return {
            "type": "RigidDiaphragm",
            "params": {
                "master": c.master,
                "slaves": list(c.slaves),
                "perp_dir": c.perp_dir,
            },
        }
    if isinstance(c, MPConstraint):
        return {
            "type": "MPConstraint",
            "params": {
                "constrained": [c.c_node, c.c_dof],
                "retained": [list(t) for t in c.r_terms],
                "g": c.g,
            },
        }
    raise NotImplementedError(f"cannot serialize constraint type {type(c).__name__}")


def _deserialize_constraint(cd: dict):
    kind = cd["type"]
    p = cd["params"]
    if kind == "EqualDOF":
        return EqualDOF(retained=p["retained"], constrained=p["constrained"], dofs=p["dofs"])
    if kind == "RigidLink":
        return RigidLink(
            retained=p["retained"], constrained=p["constrained"], kind=p.get("kind", "beam")
        )
    if kind == "RigidDiaphragm":
        return RigidDiaphragm(
            master=p["master"], slaves=p["slaves"], perp_dir=p.get("perp_dir", 2)
        )
    if kind == "MPConstraint":
        return MPConstraint(
            constrained=tuple(p["constrained"]),
            retained=[tuple(t) for t in p["retained"]],
            g=p.get("g", 0.0),
        )
    raise NotImplementedError(f"unknown constraint type {kind!r}")
