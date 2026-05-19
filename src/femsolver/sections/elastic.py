"""Elastic beam-column sections.

These are the simplest possible sections: linear, uncoupled, stateless.
Their tangent stiffness is constant, so :meth:`get_response` is just a
matrix-vector product. They exist primarily so that downstream elements
talk to a uniform :class:`SectionBase` interface, regardless of whether
the section is elastic, hinged, or made of fibers.

For the linear-static path the element can read ``EA``, ``EIy``, ``EIz``,
``GJ`` directly off the section to keep the closed-form stiffness exact.
For numerically-integrated formulations (Phase 2 onward) the element
calls :meth:`get_response` at each Gauss point.
"""
from __future__ import annotations

import numpy as np

from femsolver.sections.base import SectionBase


class ElasticSection2D(SectionBase):
    """Linear elastic 2-D beam-column section.

    Strain ordering: ``[eps_axial, kappa_z]``.
    Force ordering:  ``[N, Mz]``.
    """

    n_resultants = 2
    is_stateful = False  # constant ks, no history — safe to share across IPs

    def clone(self) -> "ElasticSection2D":
        # No state to make independent — sharing is safe and zero-cost.
        return self

    def __init__(self, E: float, A: float, Iz: float):
        if E <= 0 or A <= 0 or Iz <= 0:
            raise ValueError("E, A, Iz must be positive")
        self.E = float(E)
        self.A = float(A)
        self.Iz = float(Iz)
        self.EA = self.E * self.A
        self.EIz = self.E * self.Iz
        self._ks = np.diag([self.EA, self.EIz])

    def get_response(self, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = np.asarray(e, dtype=float)
        s = self._ks @ e
        return s, self._ks.copy()

    def __repr__(self) -> str:
        return f"ElasticSection2D(E={self.E:g}, A={self.A:g}, Iz={self.Iz:g})"


class ElasticSection3D(SectionBase):
    """Linear elastic 3-D beam-column section.

    Strain ordering: ``[eps_axial, kappa_z, kappa_y, gamma_torsion]``.
    Force ordering:  ``[N, Mz, My, T]``.

    The shear modulus :math:`G` is taken directly so the section is
    independent of any particular material container.
    """

    n_resultants = 4
    is_stateful = False

    def clone(self) -> "ElasticSection3D":
        return self

    def __init__(self, E: float, G: float, A: float, Iy: float, Iz: float, J: float):
        if min(E, G, A, Iy, Iz, J) <= 0:
            raise ValueError("E, G, A, Iy, Iz, J must all be positive")
        self.E = float(E)
        self.G = float(G)
        self.A = float(A)
        self.Iy = float(Iy)
        self.Iz = float(Iz)
        self.J = float(J)
        self.EA = self.E * self.A
        self.EIz = self.E * self.Iz
        self.EIy = self.E * self.Iy
        self.GJ = self.G * self.J
        self._ks = np.diag([self.EA, self.EIz, self.EIy, self.GJ])

    def get_response(self, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = np.asarray(e, dtype=float)
        s = self._ks @ e
        return s, self._ks.copy()

    def __repr__(self) -> str:
        return (
            f"ElasticSection3D(E={self.E:g}, G={self.G:g}, A={self.A:g}, "
            f"Iy={self.Iy:g}, Iz={self.Iz:g}, J={self.J:g})"
        )
