"""Truss elements (2D and 3D), pin-jointed, axial only."""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element


class _TrussBase(Element):
    n_nodes = 2

    def __init__(self, tag: int, nodes, material, area: float):
        super().__init__(tag, nodes, material)
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}")
        self.area = float(area)
        self.axial_force = 0.0
        self.axial_stress = 0.0

    def length_and_dirs(self) -> tuple[float, np.ndarray]:
        coords = self.node_coords()
        d = coords[1] - coords[0]
        L = float(np.linalg.norm(d))
        if L == 0.0:
            raise ValueError(f"truss element {self.tag} has zero length")
        return L, d / L


class Truss2D(_TrussBase):
    """2-node planar truss, 2 DOF/node (ux, uy)."""

    dofs_per_node = 2

    def K_global(self) -> np.ndarray:
        L, n = self.length_and_dirs()
        c, s = float(n[0]), float(n[1])
        EAoL = self.material.E * self.area / L
        block = np.array([[c * c, c * s], [c * s, s * s]])
        K = np.zeros((4, 4))
        K[0:2, 0:2] = block
        K[2:4, 2:4] = block
        K[0:2, 2:4] = -block
        K[2:4, 0:2] = -block
        return EAoL * K

    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = self.material.rho
        if rho == 0.0:
            return np.zeros((4, 4))
        L, _ = self.length_and_dirs()
        m_total = rho * self.area * L
        if lumped:
            return 0.5 * m_total * np.eye(4)
        # consistent mass — translation in any direction is decoupled across
        # x and y because the truss has no rotational coupling, so the same
        # 2-node line-element form applies independently to each component.
        I2 = np.eye(2)
        return (m_total / 6.0) * np.block([[2.0 * I2, 1.0 * I2], [1.0 * I2, 2.0 * I2]])

    def recover(self) -> None:
        L, n = self.length_and_dirs()
        u = self.gather_u()
        # axial elongation = (u_node2 - u_node1) . n_hat
        du = u[2:4] - u[0:2]
        elong = float(du @ n)
        self.axial_force = self.material.E * self.area * elong / L
        self.axial_stress = self.material.E * elong / L


class Truss3D(_TrussBase):
    """2-node spatial truss, 3 DOF/node (ux, uy, uz)."""

    dofs_per_node = 3

    def K_global(self) -> np.ndarray:
        L, n = self.length_and_dirs()
        EAoL = self.material.E * self.area / L
        block = np.outer(n, n)  # 3x3
        K = np.zeros((6, 6))
        K[0:3, 0:3] = block
        K[3:6, 3:6] = block
        K[0:3, 3:6] = -block
        K[3:6, 0:3] = -block
        return EAoL * K

    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        rho = self.material.rho
        if rho == 0.0:
            return np.zeros((6, 6))
        L, _ = self.length_and_dirs()
        m_total = rho * self.area * L
        if lumped:
            return 0.5 * m_total * np.eye(6)
        I3 = np.eye(3)
        return (m_total / 6.0) * np.block([[2.0 * I3, 1.0 * I3], [1.0 * I3, 2.0 * I3]])

    def recover(self) -> None:
        L, n = self.length_and_dirs()
        u = self.gather_u()
        du = u[3:6] - u[0:3]
        elong = float(du @ n)
        self.axial_force = self.material.E * self.area * elong / L
        self.axial_stress = self.material.E * elong / L
