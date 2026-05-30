"""Node-to-rigid-surface contact with Coulomb friction (penalty method).

The simplest workable contact model: each "slave" node ``s`` is
checked against a rigid target plane defined by a point ``p_0`` and
a normal ``n``. The signed gap is

    g_N = (p_s - p_0) · n,

with positive ``g_N`` meaning the node is OFF the surface (no contact)
and negative meaning penetration. Contact is enforced by an
**outward normal penalty** force

    F_N = K_N · max(-g_N, 0),

returning the node toward the surface. Tangential friction is
enforced by a **stick-slip** penalty:

    F_T_trial = K_T · u_T_total
    if ‖F_T_trial‖ ≤ μ · |F_N|:    F_T = F_T_trial      (stick)
    else:                            F_T = μ · |F_N| · t̂  (slip)

Choice of penalty stiffnesses ``K_N`` and ``K_T`` is a trade-off:
too low and the bodies interpenetrate; too high and the global
system becomes ill-conditioned. Use ``K_N ~ 100-1000 × E_local``.

This module ships :class:`ContactNodeToPlane3D` — one element per
slave node. The element provides:

* ``f_int_global`` -- the contact reaction at the slave node;
* ``K_tangent_global`` -- the penalty stiffness at the slave node;
* ``commit_state`` / ``revert_state`` -- friction-history bookkeeping.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element


class ContactNodeToPlane3D(Element):
    """Node-to-rigid-plane contact with optional Coulomb friction.

    Parameters
    ----------
    tag : int
    node : int
        Slave-node tag.
    plane_point : (3,) array-like
        A point on the rigid target plane (m).
    plane_normal : (3,) array-like
        Outward-pointing unit normal (will be normalised).
    K_N : float
        Normal penalty stiffness (N/m). Rule of thumb:
        ``K_N ~ 1e3 · E_local · L_char``.
    mu : float, default 0.0
        Coulomb friction coefficient (>= 0, typical 0.1-0.5).
    K_T : float, optional
        Tangential penalty stiffness. Defaults to ``K_N``.

    Notes
    -----
    The element operates in 3D (``dofs_per_node = 3``); use it on a
    single slave node by passing ``nodes=(node, node)`` or wrap an
    auxiliary fixed dummy. To keep the API uniform with other 2-node
    elements we accept a (1,) node tuple and internally double it.
    """

    n_nodes = 1
    dofs_per_node = 3

    def __init__(
        self,
        tag: int,
        nodes,
        material=None,
        *,
        plane_point,
        plane_normal,
        K_N: float,
        mu: float = 0.0,
        K_T: float | None = None,
    ):
        super().__init__(tag, nodes, material)
        if K_N <= 0.0:
            raise ValueError("K_N must be > 0")
        if mu < 0.0:
            raise ValueError("mu must be >= 0")
        self.p_0 = np.asarray(plane_point, dtype=float).ravel()
        n = np.asarray(plane_normal, dtype=float).ravel()
        norm = np.linalg.norm(n)
        if norm == 0.0:
            raise ValueError("plane_normal must be non-zero")
        self.n_hat = n / norm
        self.K_N = float(K_N)
        self.K_T = float(K_T if K_T is not None else K_N)
        self.mu = float(mu)
        # Tangent-frame basis (two perpendicular tangents)
        self._build_tangent_basis()
        # Friction history
        self.u_T_committed = np.zeros(2)        # tangent-frame slip
        self.u_T_trial = np.zeros(2)

    def _build_tangent_basis(self) -> None:
        """Construct two unit vectors ``t1, t2`` spanning the plane
        orthogonal to ``n_hat``."""
        # Pick the global axis least aligned with n_hat
        ref = np.array([1.0, 0.0, 0.0])
        if abs(self.n_hat @ ref) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        t1 = np.cross(self.n_hat, ref)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(self.n_hat, t1)
        self.t1, self.t2 = t1, t2

    def _slave_position_and_disp(self) -> tuple[np.ndarray, np.ndarray]:
        node = self.model.node(self.node_tags[0])
        return node.coords[:3], node.disp[:3]

    def _gap_and_slip(self) -> tuple[float, np.ndarray]:
        """Return ``(g_N, u_T)`` -- normal gap and 2-component
        tangential displacement.

        ``g_N < 0`` means penetration; ``g_N >= 0`` means no contact.
        """
        X, u = self._slave_position_and_disp()
        p = X + u
        d = p - self.p_0                # vector from plane point
        g_N = float(d @ self.n_hat)
        d_T_global = u - (u @ self.n_hat) * self.n_hat
        u_T = np.array([d_T_global @ self.t1, d_T_global @ self.t2])
        return g_N, u_T

    # -------------------------------------------------- internal force

    def f_int_global(self) -> np.ndarray:
        """Return the contact-reaction vector at the slave node (3,)."""
        g_N, u_T = self._gap_and_slip()
        if g_N >= 0.0:
            # No contact
            self.u_T_trial = self.u_T_committed.copy()
            return np.zeros(3)
        # Normal force (penalty)
        F_N = self.K_N * (-g_N)
        # Tangential trial force (penalty on TOTAL slip since
        # contact start, with stick-slip)
        # Use the tangential displacement from the committed
        # configuration as the elastic slip; if the trial exceeds
        # the Coulomb cap, slip and update committed.
        F_T_trial = self.K_T * (u_T - self.u_T_committed)
        F_T_mag_trial = np.linalg.norm(F_T_trial)
        cap = self.mu * F_N
        if self.mu == 0.0 or F_T_mag_trial <= cap:
            # Stick
            F_T = F_T_trial
            self.u_T_trial = u_T
        else:
            # Slip
            t_dir = F_T_trial / F_T_mag_trial
            F_T = cap * t_dir
            # Update the stored slip-from-stuck reference (committed
            # frame moves so future trials measure incremental stick)
            self.u_T_trial = u_T - (cap / self.K_T) * t_dir
        # Assemble global force (push slave node away from surface
        # along +n, and tangentially)
        f_global = F_N * self.n_hat
        f_global += F_T[0] * self.t1 + F_T[1] * self.t2
        return f_global

    def K_global(self) -> np.ndarray:
        """Initial (no-contact) stiffness -- zero matrix.

        The actual penalty stiffness is engaged only when contact is
        active and is exposed via :meth:`K_tangent_global`.
        """
        return np.zeros((3, 3))

    def K_tangent_global(self) -> np.ndarray:
        """Current-state tangent: K_N n n^T + K_T (t1 t1^T + t2 t2^T)
        when in contact, zeros otherwise.
        """
        g_N, _ = self._gap_and_slip()
        if g_N >= 0.0:
            return np.zeros((3, 3))
        K = self.K_N * np.outer(self.n_hat, self.n_hat)
        if self.mu == 0.0:
            # Frictionless: still add full tangential penalty to
            # avoid rigid-body motion along the surface in Newton.
            K += self.K_T * (np.outer(self.t1, self.t1)
                              + np.outer(self.t2, self.t2))
        else:
            # Conservative: include the full tangential penalty
            # (sufficient for slip-detected returns, not the
            # rigorous "slip-only" tangent which drops the t parts).
            K += self.K_T * (np.outer(self.t1, self.t1)
                              + np.outer(self.t2, self.t2))
        return K

    # -------------------------------------------------- state lifecycle

    def commit_state(self) -> None:
        self.u_T_committed = self.u_T_trial.copy()

    def revert_state(self) -> None:
        self.u_T_trial = self.u_T_committed.copy()
