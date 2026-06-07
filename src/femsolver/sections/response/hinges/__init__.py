"""Concentrated-plasticity components: zero-length M-theta springs.

These are the building blocks for FEMA-356 / ASCE-41 style "skeleton"
hinges. A hinge is a rotational spring with a 1-D backbone curve
(currently bilinear with kinematic hardening, which subsumes the
elastic-perfectly-plastic case at ``b = 0``). The
:class:`HingedBeamColumn2D` element combines an elastic Euler-Bernoulli
core with one or two such springs at the ends.
"""
from femsolver.sections.response.hinges.spring import (
    BilinearMomentRotationSpring,
)

__all__ = [
    "BilinearMomentRotationSpring",
]
