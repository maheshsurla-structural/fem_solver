"""Force-based 3-D beam-column (small-displacement) — genuine P-M2-M3.

The 3-D counterpart of :class:`ForceBeamColumn2DCorotational`, and the element
that turns a :class:`~femsolver.sections.response.fiber.FiberSection3D` into a
biaxial (P-M2-M3) fiber beam-column with one element per member (plan G5 / P7).

Like the 2-D force-based element it assumes the *force* distribution along the
member and integrates section *flexibility* -- so the moment diagram is exact
under equilibrium even when the section response is highly nonlinear. Geometry
is linear (small displacement), matching the benchmark's ``GeoNonLin=None``
(plan §2.6); P-Delta can be layered on later.

Basic (natural) system -- 6 deformations / forces after removing the 6
rigid-body modes of the 12-DOF local element::

    v = [u_axial, theta_z1, theta_z2, theta_y1, theta_y2, phi_torsion]
    q = [N,       Mz1,      Mz2,      My1,      My2,      T]

Force interpolation ``b(x)`` distributes ``q`` to the section forces
``s(x) = [N, Mz(x), My(x), T]`` (bending linear between ends, N and T
constant), with the same ``-(1-x/L), x/L`` sign convention the 2-D element
uses so the elastic flexibility inverts to the standard prismatic stiffness.
The constant basic-to-local matrix ``a`` maps ``v = a u_local`` and
``f_local = a^T q``; state determination is the Neuenhofer-Filippou (1997)
algorithm, identical in structure to the 2-D case but with 6 basic DOFs and 4
section resultants.

For an elastic prismatic section the force-based and displacement-based
stiffnesses are identical (both exact), and the result is invariant to the
number of integration points -- both are checked in the tests. Under uniaxial
bending it reduces to the 2-D force-based response.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.beam import BeamColumn3D
from femsolver.numerics.quadrature import gauss_lobatto_1d
from femsolver.sections.response.base import SectionBase


class ForceBeamColumn3D(BeamColumn3D):
    """Force-based (flexibility-based) small-displacement 3-D beam-column.

    Drop-in replacement for :class:`BeamColumn3D`; same constructor. Pass a
    :class:`FiberSection3D` (e.g. ``FiberSection3D.circular``) for P-M2-M3.
    """

    state_det_tol: float = 1.0e-12
    state_det_max_iter: int = 30

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        area: float | None = None,
        Iy: float | None = None,
        Iz: float | None = None,
        J: float | None = None,
        vecxz=None,
        *,
        section: SectionBase | None = None,
    ):
        super().__init__(tag, nodes, material, area, Iy, Iz, J, vecxz,
                         section=section)
        self._q_committed: np.ndarray = np.zeros(6)
        self._e_committed: np.ndarray | None = None
        self._K_b_last: np.ndarray | None = None
        self._v_committed: np.ndarray = np.zeros(6)
        # per-(gather_u) cache of the state-determination result
        self._cache_key = None
        self._cache_val = None

    # ------------------------------------------ basic-to-local matrix a
    @staticmethod
    def _a_matrix(L: float) -> np.ndarray:
        """Constant ``(6, 12)`` matrix ``a`` with ``v = a @ u_local``.

        Local DOF order: ``[u,v,w,theta_x,theta_y,theta_z]`` per node.
        Basic order: ``[u_axial, theta_z1, theta_z2, theta_y1, theta_y2,
        phi_torsion]``. Bending rotations are taken relative to the chord
        (``rho_z = (v2-v1)/L``; for y, ``dw/dx = -theta_y`` so the chord
        contributes ``+(w2-w1)/L``)."""
        a = np.zeros((6, 12))
        a[0, 0], a[0, 6] = -1.0, 1.0        # u_axial = u2 - u1
        a[1, 1], a[1, 5], a[1, 7] = 1.0 / L, 1.0, -1.0 / L   # theta_z1 - rho_z
        a[2, 1], a[2, 7], a[2, 11] = 1.0 / L, -1.0 / L, 1.0  # theta_z2 - rho_z
        a[3, 2], a[3, 4], a[3, 8] = -1.0 / L, 1.0, 1.0 / L   # theta_y1 + rho_y
        a[4, 2], a[4, 8], a[4, 10] = -1.0 / L, 1.0 / L, 1.0  # theta_y2 + rho_y
        a[5, 3], a[5, 9] = -1.0, 1.0        # phi = theta_x2 - theta_x1
        return a

    # ------------------------------------------ force interpolation b(x)
    @staticmethod
    def _b_matrix(xi: float) -> np.ndarray:
        """``(4, 6)`` force interpolation at natural coordinate ``xi`` in
        ``[-1, 1]``: ``s(x) = [N, Mz(x), My(x), T] = b @ q``."""
        s = 0.5 * (1.0 + xi)                 # x / L in [0, 1]
        b = np.zeros((4, 6))
        b[0, 0] = 1.0                        # N
        b[1, 1], b[1, 2] = -(1.0 - s), s     # Mz(x)
        b[2, 3], b[2, 4] = -(1.0 - s), s     # My(x)
        b[3, 5] = 1.0                        # T
        return b

    # ------------------------------------------ section-level inversion
    @staticmethod
    def _section_strain_for_force(section, s_target, e_start, *,
                                  max_iter: int = 20, tol: float = 1.0e-12):
        e = np.asarray(e_start, dtype=float).copy()
        s_curr, k_s = section.get_response(e)
        for _ in range(max_iter):
            r = s_target - s_curr
            if float(np.max(np.abs(r))) < tol:
                return e, k_s
            try:
                de = np.linalg.solve(k_s, r)
            except np.linalg.LinAlgError:
                de, *_ = np.linalg.lstsq(k_s, r, rcond=None)
            e = e + de
            s_curr, k_s = section.get_response(e)
        return e, k_s

    # ------------------------------------------ Neuenhofer-Filippou loop
    def _basic_response(self):
        """Return ``(q, K_basic, e_trial, a, L)`` from force-based state
        determination at the current displacements. Cached per ``gather_u``
        so ``f_int_global`` and ``K_tangent_global`` share one solve."""
        u_g = self.gather_u()
        key = u_g.tobytes()
        if self._cache_key == key and self._cache_val is not None:
            return self._cache_val

        L, _, _, _ = self.length_and_axes()
        T = self.transform_matrix()
        u_l = T @ u_g
        a = self._a_matrix(L)
        v_target = a @ u_l

        self._ensure_sections_length()
        xi_pts, w_pts = gauss_lobatto_1d(self.n_int)
        n_r = self.sections[0].n_resultants
        jac = 0.5 * L
        if self._e_committed is None:
            self._e_committed = np.zeros((self.n_int, n_r))

        if self._K_b_last is not None:
            q = self._q_committed + self._K_b_last @ (v_target - self._v_committed)
        else:
            q = self._q_committed.copy()

        e_trial = self._e_committed.copy()
        K_b = None
        for _it in range(self.state_det_max_iter):
            F_b = np.zeros((6, 6))
            v_computed = np.zeros(6)
            for i, (xi, w) in enumerate(zip(xi_pts, w_pts)):
                b = self._b_matrix(xi)
                s_target = b @ q
                e_i, k_s_i = self._section_strain_for_force(
                    self.sections[i], s_target, self._e_committed[i])
                e_trial[i] = e_i
                try:
                    f_s_b = np.linalg.solve(k_s_i, b)
                except np.linalg.LinAlgError:
                    f_s_b, *_ = np.linalg.lstsq(k_s_i, b, rcond=None)
                F_b += (w * jac) * (b.T @ f_s_b)
                v_computed += (w * jac) * (b.T @ e_i)
            dv = v_target - v_computed
            try:
                K_b = np.linalg.inv(F_b)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    f"force-based beam {self.tag}: element flexibility became "
                    f"singular during state determination ({exc}). Likely "
                    "cause: a section lost stiffness in a needed direction "
                    "(fully plastic). Use a hardening section or smaller steps."
                ) from exc
            if float(np.max(np.abs(dv))) < self.state_det_tol:
                break
            q = q + K_b @ dv
        else:
            raise RuntimeError(
                f"force-based beam {self.tag}: state determination did not "
                f"converge in {self.state_det_max_iter} iterations "
                f"(||dv|| = {float(np.max(np.abs(dv))):.3e})")

        self._q_trial = q
        self._e_trial = e_trial
        self._K_b_last = K_b
        self._cache_key = key
        self._cache_val = (q, K_b, e_trial, a, L)
        return self._cache_val

    # ---------------------------------------------------- force / tangent
    def f_int_global(self) -> np.ndarray:
        q, _K_b, _e, a, _L = self._basic_response()
        f_local = a.T @ q
        return self.transform_matrix().T @ f_local

    def K_tangent_global(self) -> np.ndarray:
        _q, K_b, _e, a, _L = self._basic_response()
        T = self.transform_matrix()
        K_local = a.T @ K_b @ a
        return T.T @ K_local @ T

    def K_global(self) -> np.ndarray:
        # Force-based stiffness at the current (committed) state; at zero
        # state with an elastic section this equals the displacement-based
        # closed-form K.
        return self.K_tangent_global()

    # ---------------------------------------------------------- lifecycle
    def commit_state(self) -> None:
        super().commit_state()
        if hasattr(self, "_q_trial"):
            self._q_committed = self._q_trial.copy()
        if hasattr(self, "_e_trial"):
            self._e_committed = self._e_trial.copy()
        L, _, _, _ = self.length_and_axes()
        u_l = self.transform_matrix() @ self.gather_u()
        self._v_committed = self._a_matrix(L) @ u_l
        self._cache_key = None

    def revert_state(self) -> None:
        super().revert_state()
        if hasattr(self, "_q_trial"):
            del self._q_trial
        if hasattr(self, "_e_trial"):
            del self._e_trial
        self._cache_key = None

    # ------------------------------------------------------------- recover
    def recover(self) -> None:
        """Populate element end forces and per-IP section response from the
        converged force-based state."""
        q, _K_b, e_trial, a, L = self._basic_response()
        # end forces in the local frame: f_local = a^T q
        self.end_forces_local = a.T @ q - self.f_eq_local()
        xi_pts, _ = gauss_lobatto_1d(self.n_int)
        n_r = self.sections[0].n_resultants
        self.section_locations = np.empty(self.n_int)
        self.section_strains = np.empty((self.n_int, n_r))
        self.section_forces = np.empty((self.n_int, n_r))
        for i, xi in enumerate(xi_pts):
            self.section_locations[i] = 0.5 * L * (1.0 + xi)
            self.section_strains[i] = e_trial[i]
            self.section_forces[i] = self._b_matrix(xi) @ q
