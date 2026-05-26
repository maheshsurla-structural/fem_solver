"""Zero-length connector element with per-DOF uniaxial materials.

The ``ZeroLengthElement`` connects two (typically coincident) nodes
with independent springs in any subset of DOFs. Each spring's
force-displacement behaviour is supplied as a
:class:`~femsolver.materials.uniaxial.UniaxialMaterial`, so the
existing library of uniaxials (Elastic, Bilinear, Hysteretic,
Menegotto-Pinto, Concrete, Gap, ...) all become connector building
blocks.

This is the workhorse element for:

* Base isolators (lead-rubber, friction-pendulum)
* Foundation uplift / pounding gaps
* Discrete plastic hinges
* Generic spring-dashpot connectors
* Bonded / glued joints (with high-stiffness springs)

Sign convention
---------------
Per DOF ``d``: the **strain** ``epsilon = u_j[d] - u_i[d]`` is the
"j minus i" displacement difference, so positive ``epsilon`` is
elongation. The material returns the stress (= force) ``sigma`` in
that DOF:

* ``f_int_i[d] = +sigma`` (force on i pulls *toward* j when sigma > 0)
* ``f_int_j[d] = -sigma`` (reaction on j)

For a uniaxial elastic spring with stiffness ``K``: ``sigma = K * epsilon``
gives the standard ``[[K, -K], [-K, K]]`` 2-DOF block.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from femsolver.elements.base import Element
from femsolver.materials.uniaxial.base import UniaxialMaterial


class ZeroLengthElement(Element):
    """2-node connector with per-DOF uniaxial springs.

    Parameters
    ----------
    tag : int
    nodes : sequence of 2 node tags (i, j). Coincident coordinates are
        typical but not required; the element does not rotate to any
        local frame -- all directions are *global*.
    material : Material, optional
        Used only for the standard ``Element.material`` slot (e.g. mass
        density if you want a non-zero element mass via the lumped-mass
        path). May be ``None``. The connector's stiffness comes from
        the ``materials`` argument.
    materials : dict[int, UniaxialMaterial]
        Map from global DOF index (0=ux, 1=uy, 2=uz, 3=rx, 4=ry, 5=rz)
        to the uniaxial spring used in that DOF. DOFs not in the map
        have zero stiffness (free).
    dofs_per_node : int, default 6
        Number of DOFs per node in the connector. Must be >= max(DOF
        index in ``materials``) + 1. The element will pull
        ``dofs_per_node`` DOFs from each node when assembled, so the
        value should match the model's ``ndf`` (2, 3, or 6).
    """

    n_nodes = 2

    def __init__(
        self,
        tag: int,
        nodes,
        material=None,
        *,
        materials: Mapping[int, UniaxialMaterial] | None = None,
        dofs_per_node: int = 6,
    ):
        # Allow material=None (no global material). Use a stub if needed.
        # We set dofs_per_node as an instance attribute before super().__init__
        # so the base class can validate n_nodes correctly.
        if materials is None or len(materials) == 0:
            raise ValueError(
                "ZeroLengthElement needs at least one (DOF -> material) entry "
                "in 'materials'"
            )
        max_dof = max(int(d) for d in materials.keys())
        if max_dof >= dofs_per_node:
            raise ValueError(
                f"materials reference DOF {max_dof} but dofs_per_node="
                f"{dofs_per_node}; pass a larger dofs_per_node"
            )
        for d in materials.keys():
            if d < 0:
                raise ValueError(f"DOF index must be >= 0, got {d}")
        # Use a None material slot is fine; base only stores it.
        # However, Element's __init__ expects 'material' positionally.
        # We pass the user-supplied material (may be None).
        self.dofs_per_node = int(dofs_per_node)
        super().__init__(tag, nodes, material)
        # Sorted DOF directions for deterministic iteration order.
        self.materials: dict[int, UniaxialMaterial] = dict(
            sorted((int(k), v) for k, v in materials.items())
        )

    # ----------------------------------------------------- helpers
    def _delta(self) -> np.ndarray:
        """Per-DOF strain epsilon[d] = u_j[d] - u_i[d]."""
        u = self.gather_u()
        n = self.dofs_per_node
        return u[n:2 * n] - u[0:n]

    # ----------------------------------------------------- K_global
    def K_global(self) -> np.ndarray:
        """Initial (elastic) stiffness -- each material evaluated at
        zero strain to extract its initial tangent."""
        n = self.dofs_per_node
        K = np.zeros((2 * n, 2 * n))
        for d, mat in self.materials.items():
            # Probe at eps = 0 then revert so initial state is unchanged.
            _, Et = mat.get_response(0.0)
            mat.revert_state()
            K[d, d] += Et
            K[n + d, n + d] += Et
            K[d, n + d] -= Et
            K[n + d, d] -= Et
        return K

    def K_tangent_global(self) -> np.ndarray:
        """Tangent stiffness at the current state."""
        n = self.dofs_per_node
        eps = self._delta()
        K = np.zeros((2 * n, 2 * n))
        for d, mat in self.materials.items():
            _, Et = mat.get_response(float(eps[d]))
            K[d, d] += Et
            K[n + d, n + d] += Et
            K[d, n + d] -= Et
            K[n + d, d] -= Et
        return K

    def f_int_global(self) -> np.ndarray:
        """Internal resisting force vector (size 2*ndf).

        Sign convention matches ``K @ u`` of an equivalent
        ``[[K, -K], [-K, K]]`` 1-D spring: for ``epsilon = u_j - u_i > 0``,
        ``f_int[i] = -sigma`` (force pulls i toward j) and
        ``f_int[j] = +sigma`` (force pulls j toward i in the
        resisting-force / external-load sense).
        """
        n = self.dofs_per_node
        eps = self._delta()
        f = np.zeros(2 * n)
        for d, mat in self.materials.items():
            sigma, _ = mat.get_response(float(eps[d]))
            f[d] -= sigma           # on node i
            f[n + d] += sigma       # on node j
        return f

    # ----------------------------------------------------- mass
    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        """Zero-length elements carry no mass by default."""
        n = self.dofs_per_node
        return np.zeros((2 * n, 2 * n))

    # ----------------------------------------------------- state
    def commit_state(self) -> None:
        # Forward to materials. Materials' trial states were updated by
        # the most recent get_response call during residual / tangent
        # evaluation, so committing now persists them.
        for mat in self.materials.values():
            mat.commit_state()

    def revert_state(self) -> None:
        for mat in self.materials.values():
            mat.revert_state()

    def recover(self) -> None:
        """Re-evaluate each material at the committed displacement so
        the materials' internal stress / strain attributes reflect the
        current state."""
        eps = self._delta()
        for d, mat in self.materials.items():
            mat.get_response(float(eps[d]))
