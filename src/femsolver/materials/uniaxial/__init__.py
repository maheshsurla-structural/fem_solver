"""Uniaxial constitutive models — the inner constitutive layer for fiber
sections.

Each model maps a scalar strain ``eps`` to a scalar stress ``sigma`` and
returns its tangent modulus ``Et = d sigma / d eps``. State-bearing
models (e.g. plasticity) carry committed/trial pairs and respond to the
``commit_state`` / ``revert_state`` lifecycle calls forwarded by the
fiber section, just like :class:`BilinearMomentRotationSpring` does at
the element-end-spring level.
"""
from femsolver.materials.uniaxial.base import UniaxialMaterial
from femsolver.materials.uniaxial.elastic import UniaxialElastic
from femsolver.materials.uniaxial.bilinear import UniaxialBilinear

__all__ = [
    "UniaxialMaterial",
    "UniaxialElastic",
    "UniaxialBilinear",
]
