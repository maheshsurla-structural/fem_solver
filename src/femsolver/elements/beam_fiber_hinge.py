"""Finite-length fiber-hinge beam-column (plan G4 / P9).

A beam-with-hinges element: an elastic member with a **finite-length fiber
plastic hinge** of length ``lp`` at each end, mirroring the CSI "Fiber
P-M2-M3" hinge (a fiber section assigned over a relative hinge length) and the
Midas lumped inelastic hinge. Inside the hinge regions the response comes from
the actual fiber section (confined-core concrete, Park/kinematic steel, ...);
the interior is elastic.

Formulation (force-based, exact-elastic + localized plastic hinge)
------------------------------------------------------------------
Work in the basic (natural) 2-D system ``v = [u, theta_1, theta_2]`` /
``q = [N, M_1, M_2]`` (same as :class:`ForceBeamColumn2DCorotational`). The
element flexibility is the exact elastic flexibility of the full member plus a
plastic correction localized at each end hinge::

    F = F_el + sum_h  lp_h * b(x_h)^T (f_fiber,h - f_el) b(x_h)

and the basic deformation is::

    v = F_el q + sum_h  lp_h * b(x_h)^T (e_fiber,h - f_el s_h),   s_h = b(x_h) q

where ``f_el`` is the section's elastic flexibility (from its initial tangent),
``b(x)`` the force-interpolation matrix, and ``e_fiber,h``/``f_fiber,h`` the
fiber section's deformation/flexibility driven to the hinge force ``s_h``. The
hinge points sit at the member ends (``x = 0`` and ``x = L``). Because the
elastic part is the exact member flexibility, an elastic fiber section
reproduces the analytic prismatic stiffness for any ``lp``; plastic
deformation is spread over the length ``lp`` you assign.

This keeps the external 6-DOF interface identical to
:class:`BeamColumn2DCorotational` and reuses its corotational wrapping and the
force-based section inversion. ``lp -> L/2`` per end approaches the
distributed :class:`ForceBeamColumn2DCorotational`; a short ``lp`` concentrates
the plasticity (Caltrans-style plastic-hinge length).

Coupling note: ``f_el`` is taken as the full 2x2 inverse of the section's
initial tangent, so axial-moment coupling in the elastic range is retained.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.beam_force import ForceBeamColumn2DCorotational
from femsolver.sections.response.base import SectionBase


class FiberHingeBeamColumn2D(ForceBeamColumn2DCorotational):
    """Elastic member with finite-length fiber hinges (length ``lp``) at the
    ends.

    Parameters
    ----------
    tag, nodes, material :
        As :class:`BeamColumn2DCorotational`. ``material`` supplies the
        corotational/geometry frame (and mass); the constitutive response
        comes from ``section`` in the hinges and its elastic tangent elsewhere.
    section :
        The fiber section (e.g. from ``rc_circular_column_section``) placed in
        both hinge regions.
    lp : float
        Plastic-hinge length at end I (and end J if ``lp_j`` is not given), in
        model length units. Must satisfy ``lp_i + lp_j <= L``.
    lp_j : float, optional
        Hinge length at end J; defaults to ``lp``.
    """

    def __init__(self, tag, nodes, material, *, section: SectionBase,
                 lp: float, lp_j: float | None = None):
        if section is None:
            raise ValueError("FiberHingeBeamColumn2D requires a fiber section")
        super().__init__(tag, nodes, material, section=section)
        self.lp_i = float(lp)
        self.lp_j = float(lp) if lp_j is None else float(lp_j)
        if self.lp_i <= 0.0 or self.lp_j <= 0.0:
            raise ValueError("hinge lengths lp must be positive")
        # Two hinge integration points -> two fiber sections (ends I, J).
        self.n_int = 2
        clone = section.clone
        self.sections = [clone(), clone()]
        self._stateful_sections = True
        # Elastic section flexibility (retains any axial-moment coupling),
        # constant. Probe at a small *compressive* axial strain, not zero:
        # compression-only concrete laws (ConcreteMander) report a zero tangent
        # at exactly zero strain, which would drop the concrete's contribution
        # to EA/EI. A tiny strain (well below eps_c0) engages the initial
        # modulus; revert so the committed section state stays pristine.
        n_r = section.n_resultants
        e_probe = np.zeros(n_r)
        e_probe[0] = -1.0e-6
        _s0, ks0 = section.get_response(e_probe)
        section.revert_state()
        self._f_el = np.linalg.inv(np.asarray(ks0, dtype=float))
        self._e_committed = np.zeros((2, section.n_resultants))
        self._F_el_basic = None            # built lazily (needs L0)

    # --------------------------------------------- elastic basic flexibility
    def _elastic_basic_flexibility(self, L0: float) -> np.ndarray:
        """F_el = integral b^T f_el b dx over the member (constant; 3x3).
        Integrated with Gauss-Legendre (exact for the quadratic integrand)."""
        xg, wg = np.polynomial.legendre.leggauss(4)
        F = np.zeros((3, 3))
        jac = 0.5 * L0
        for xi, w in zip(xg, wg):
            b = self._b_matrix(xi, L0)                # 2x3
            F += (w * jac) * (b.T @ self._f_el @ b)
        return F

    # ------------------------------------------------- natural response
    def _natural_response(self):
        L0, L, c, s, _, _, alpha = self._current_geometry()
        u_g = self.gather_u()
        v_target = np.array([L - L0,
                             float(u_g[2]) - alpha,
                             float(u_g[5]) - alpha])

        if self._F_el_basic is None:
            self._F_el_basic = self._elastic_basic_flexibility(L0)
        F_el = self._F_el_basic
        f_el = self._f_el

        # hinge points at the member ends (xi = -1 -> x=0, xi = +1 -> x=L)
        hinges = ((self._b_matrix(-1.0, L0), self.lp_i, 0),
                  (self._b_matrix(+1.0, L0), self.lp_j, 1))

        # predictor from committed state
        if self._K_b_last is not None:
            q = self._q_committed + self._K_b_last @ (
                v_target - getattr(self, "_v_committed", np.zeros(3)))
        else:
            q = self._q_committed.copy()

        e_trial = self._e_committed.copy()
        K_b = None
        for _it in range(self.state_det_max_iter):
            v = F_el @ q
            F_tan = F_el.copy()
            for b_h, lp_h, idx in hinges:
                s_h = b_h @ q
                e_h, ks_h = self._section_strain_for_force(
                    self.sections[idx], s_h, self._e_committed[idx])
                e_trial[idx] = e_h
                try:
                    f_fiber = np.linalg.inv(ks_h)
                except np.linalg.LinAlgError:
                    f_fiber = np.linalg.pinv(ks_h)
                v = v + lp_h * (b_h.T @ (e_h - f_el @ s_h))
                F_tan = F_tan + lp_h * (b_h.T @ (f_fiber - f_el) @ b_h)
            dv = v_target - v
            try:
                K_b = np.linalg.inv(F_tan)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    f"fiber-hinge beam {self.tag}: element flexibility became "
                    f"singular during state determination ({exc})."
                ) from exc
            if self._state_det_converged(dv, v_target):
                break
            q = q + K_b @ dv
        else:
            raise RuntimeError(
                f"fiber-hinge beam {self.tag}: state determination did not "
                f"converge in {self.state_det_max_iter} iterations "
                f"(||dv|| = {float(np.max(np.abs(dv))):.3e})")

        self._q_trial = q
        self._e_trial = e_trial
        self._K_b_last = K_b
        return q, K_b, c, s, L, L0

    def K_global(self) -> np.ndarray:
        # Initial/elastic stiffness via the hinge state determination (the
        # inherited closed-form path does not know the hinge formulation).
        return self.K_tangent_global()

    # ------------------------------------------------------------- recover
    def recover(self) -> None:
        """Populate per-hinge section response and element end forces."""
        q, _K_b, c, s, L, L0 = self._natural_response()
        self.section_locations = np.array([0.0, L0])
        n_r = self.sections[0].n_resultants
        self.section_strains = np.array(self._e_trial)
        self.section_forces = np.empty((2, n_r))
        for i, xi in enumerate((-1.0, 1.0)):
            self.section_forces[i] = self._b_matrix(xi, L0) @ q
