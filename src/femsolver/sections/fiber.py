"""2-D fiber cross-section: distributed plasticity from individual fibers.

A fiber section discretizes the cross-section into a list of small
strips of area, each holding a uniaxial constitutive law. The section
response is the area-weighted aggregate of fiber responses — axial
force from sigma * area, bending moment from -y * sigma * area, plus
their tangent counterparts.

Strain / force ordering (matches :class:`ElasticSection2D`):

    e = [eps_axial, kappa_z]^T
    s = [N, Mz]^T

The tangent stiffness picks up an off-diagonal term once different
fibers carry different tangent moduli (e.g., when the compression side
yields before the tension side):

    ks = [[ EA,  -ES ],
          [-ES,  EI  ]]

with ``EA = sum(Et_f A_f)``, ``ES = sum(Et_f A_f y_f)``,
``EI = sum(Et_f A_f y_f^2)``. This off-diagonal coupling is what gives
fiber sections their P-M interaction post-yield — pure bending develops
an axial-force component and vice versa.

State
-----
Each fiber holds an independent :class:`UniaxialMaterial`. The section
forwards lifecycle calls (:meth:`commit_state` / :meth:`revert_state`)
to every fiber's material. :meth:`clone` deep-copies the fibers so each
integration point of a beam element gets its own independent state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femsolver.materials.uniaxial.base import UniaxialMaterial
from femsolver.sections.base import SectionBase


@dataclass
class Fiber:
    """A single fiber: position, area, and uniaxial material.

    ``y`` is measured from the section centroid, perpendicular to the
    bending axis (positive on the +y side of the local frame).
    ``z`` is the out-of-plane coordinate; it is unused by
    :class:`FiberSection2D` but kept on the dataclass so the same
    Fiber type can later feed a 3-D fiber section.
    """
    y: float
    z: float
    area: float
    material: UniaxialMaterial


class FiberSection2D(SectionBase):
    """Fiber-discretized 2-D cross-section."""

    n_resultants = 2
    is_stateful = True   # per-fiber materials carry history

    def __init__(self, fibers: list[Fiber]):
        if not fibers:
            raise ValueError("FiberSection2D needs at least one fiber")
        for f in fibers:
            if f.area <= 0.0:
                raise ValueError(f"fiber at y={f.y} has non-positive area")
        self.fibers = list(fibers)

    # --------------------------------------------------------- gross props
    @property
    def gross_area(self) -> float:
        return float(sum(f.area for f in self.fibers))

    @property
    def centroid_y(self) -> float:
        A = self.gross_area
        if A == 0.0:
            return 0.0
        return float(sum(f.area * f.y for f in self.fibers) / A)

    @property
    def gross_Iz(self) -> float:
        """Second moment of area about the centroidal z-axis (parallel-
        axis-theorem-corrected)."""
        yc = self.centroid_y
        return float(sum(f.area * (f.y - yc) ** 2 for f in self.fibers))

    # -------------------------------------------------------- response
    def get_response(self, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate fiber responses into section forces and tangent.

        Fiber strain at offset ``y`` from the centroid:

            eps_f(y) = eps_axial - y * kappa_z

        (the minus sign is the standard plane-sections beam kinematics:
        positive curvature compresses the +y fibers).
        """
        eps_a = float(e[0])
        kappa = float(e[1])
        N = 0.0
        M = 0.0
        EA = 0.0
        ES = 0.0
        EI = 0.0
        for f in self.fibers:
            eps_f = eps_a - f.y * kappa
            sigma, Et = f.material.get_response(eps_f)
            dF = sigma * f.area
            dEt_A = Et * f.area
            N += dF
            M -= f.y * dF
            EA += dEt_A
            ES += dEt_A * f.y
            EI += dEt_A * f.y * f.y
        s = np.array([N, M])
        ks = np.array([[EA, -ES], [-ES, EI]])
        return s, ks

    # -------------------------------------------------------- lifecycle
    def commit_state(self) -> None:
        for f in self.fibers:
            f.material.commit_state()

    def revert_state(self) -> None:
        for f in self.fibers:
            f.material.revert_state()

    # ------------------------------------------------------------- clone
    def clone(self) -> "FiberSection2D":
        """Deep copy of fibers with independent material state."""
        return FiberSection2D([
            Fiber(y=f.y, z=f.z, area=f.area, material=f.material.clone())
            for f in self.fibers
        ])

    # --------------------------------------------------------- factories
    @classmethod
    def rectangular(
        cls,
        width: float,
        height: float,
        n_fibers: int,
        material: UniaxialMaterial,
        *,
        centroid_y: float = 0.0,
    ) -> "FiberSection2D":
        """Build a rectangular section discretized into ``n_fibers``
        equal-area strips along the height direction (``y``).

        Parameters
        ----------
        width : float
            Cross-section width (in z-direction).
        height : float
            Cross-section height (in y-direction). Strips are stacked
            along this direction.
        n_fibers : int
            Number of strips, each of area ``width * height / n_fibers``.
        material : UniaxialMaterial
            Material assigned to every fiber. Each fiber gets its own
            cloned copy so state is independent.
        centroid_y : float, default 0.0
            Y-coordinate of the section centroid; useful when the
            section is offset from the beam neutral axis.
        """
        if width <= 0.0 or height <= 0.0:
            raise ValueError("width and height must be positive")
        if n_fibers < 2:
            raise ValueError("need at least 2 fibers")
        strip_h = height / n_fibers
        fiber_area = width * strip_h
        # strips centred at y_i, evenly spaced from -h/2 to +h/2 about centroid
        y_top = centroid_y + 0.5 * height - 0.5 * strip_h
        fibers = [
            Fiber(
                y=y_top - i * strip_h,
                z=0.0,
                area=fiber_area,
                material=material.clone(),
            )
            for i in range(n_fibers)
        ]
        return cls(fibers)

    def __repr__(self) -> str:
        return (
            f"FiberSection2D(n_fibers={len(self.fibers)}, "
            f"A={self.gross_area:g}, Iz={self.gross_Iz:g})"
        )
