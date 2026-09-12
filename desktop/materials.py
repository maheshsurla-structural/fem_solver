"""Bridge desktop ``Material`` records to engine uniaxial laws + σ-ε sampling.

One engine-backed source for both the fiber sections / hinges (GUI-2/3, later)
and the material editor's stress-strain preview (GUI-1). Maps a declarative
:class:`project.Material` to a :class:`femsolver.materials.uniaxial` law via
``uniaxial_law``, and samples its monotonic σ-ε curve via
``stress_strain_curve`` — so the desktop never re-implements constitutive
maths (plan §15 unification).

Units are the project's own consistent unit system (the desktop stores SI).
"""
from __future__ import annotations

import numpy as np

from femsolver.materials.uniaxial.concrete import (ConcreteKentPark,
                                                   ConcreteMander)
from femsolver.materials.uniaxial.elastic import UniaxialElastic
from femsolver.materials.uniaxial.reinforcing import (ReinforcingSteelKinematic,
                                                      UniaxialReinforcingSteel)

# kind -> human label (drives the editor's combo + is the canonical registry)
MATERIAL_KINDS: dict[str, str] = {
    "elastic_isotropic": "Elastic isotropic",
    "concrete_kentpark": "Concrete — Kent-Park",
    "concrete_mander": "Concrete — Mander",
    "reinforcing_steel": "Reinforcing steel — Park (monotonic)",
    "cyclic_steel": "Reinforcing steel — kinematic (cyclic)",
}

# per-kind parameter defaults (SI) — the editor seeds new materials from these
KIND_DEFAULTS: dict[str, dict] = {
    "elastic_isotropic": {},
    "concrete_kentpark": {"fc": 30.0e6, "eps_c0": 0.002, "fpcu_ratio": 0.2,
                          "eps_cu": 0.0035},
    "concrete_mander": {"fc": 34.5e6, "eps_c0": 0.002219, "Ec": 24.9e9,
                        "eps_cu": 0.02},
    "reinforcing_steel": {"E": 200.0e9, "fy": 469.0e6, "fu": 655.0e6,
                          "eps_sh": 0.0075, "eps_su": 0.09},
    "cyclic_steel": {"E": 200.0e9, "fy": 469.0e6, "fu": 655.0e6,
                     "eps_sh": 0.0075, "eps_su": 0.09},
}


def is_inelastic(kind: str) -> bool:
    return kind != "elastic_isotropic"


def uniaxial_law(material):
    """Build the engine :class:`UniaxialMaterial` for a desktop ``Material``.

    Inelastic-kind parameters are read from ``material.params`` (falling back
    to :data:`KIND_DEFAULTS`); elastic uses ``material.E``.
    """
    kind = material.kind
    p = {**KIND_DEFAULTS.get(kind, {}), **(material.params or {})}

    if kind == "elastic_isotropic":
        return UniaxialElastic(float(material.E))

    if kind == "concrete_kentpark":
        fc = float(p["fc"])
        return ConcreteKentPark(fpc=fc, eps_c0=float(p["eps_c0"]),
                                fpcu=float(p.get("fpcu_ratio", 0.2)) * fc,
                                eps_cu=float(p["eps_cu"]))

    if kind == "concrete_mander":
        Ec = float(p.get("Ec") or material.E)
        return ConcreteMander(fpc=float(p["fc"]), eps_c0=float(p["eps_c0"]),
                              Ec=Ec)

    if kind in ("reinforcing_steel", "cyclic_steel"):
        cls = (ReinforcingSteelKinematic if kind == "cyclic_steel"
               else UniaxialReinforcingSteel)
        return cls(E=float(p.get("E", material.E)), f_y=float(p["fy"]),
                   f_su=float(p["fu"]), eps_sh=float(p["eps_sh"]),
                   eps_su=float(p["eps_su"]))

    raise ValueError(f"unknown material kind {kind!r}")


def stress_strain_curve(material, *, n: int = 240):
    """Sample the monotonic σ-ε response, returning ``(eps, sigma)`` arrays.

    Range is chosen per kind: concrete swept in compression to ~1.15·ε_cu
    (compression-negative), steel in tension to ~1.1·ε_su, elastic over a
    small symmetric band. The law is committed after each step so path-
    dependent models trace their monotonic backbone.
    """
    law = uniaxial_law(material)
    kind = material.kind
    p = {**KIND_DEFAULTS.get(kind, {}), **(material.params or {})}

    if kind.startswith("concrete"):
        eps_cu = float(p.get("eps_cu", 0.0035))
        eps = np.linspace(0.0, -1.15 * max(eps_cu, 0.003), n)
    elif kind in ("reinforcing_steel", "cyclic_steel"):
        eps_su = float(p.get("eps_su", 0.10))
        eps = np.linspace(0.0, 1.1 * eps_su, n)
    else:  # elastic
        eps = np.linspace(-0.002, 0.002, n)

    sig = np.empty(n)
    for i, e in enumerate(eps):
        s, _ = law.get_response(float(e))
        sig[i] = s
        commit = getattr(law, "commit_state", None)
        if commit is not None:
            commit()
    return eps, sig
