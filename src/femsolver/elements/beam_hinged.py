"""2-D Euler-Bernoulli beam-column with optional concentrated plastic
hinges at one or both ends.

Architecture
------------
The element wraps an elastic prismatic beam (described by ``EA`` and
``EIz``, exactly like :class:`BeamColumn2D`) and one or two zero-length
rotational springs at the ends. From the outside it advertises the same
6 node DOFs as the elastic beam, so it composes cleanly with the rest of
the model. Internally it carries up to two extra rotational degrees of
freedom — the *interior* beam-end rotations on the beam side of each
spring. These interior DOFs are eliminated by static (Guyan) condensation
before the element returns its tangent stiffness to the assembler.

Sign convention for a hinge at end I::

      [node I]---||spring i||---[beam end I]------ ...
        theta_n_i               theta_b_i

    theta_h_i = theta_n_i - theta_b_i        (hinge rotation)
    M_h_i = M(theta_h_i)                     (spring response)

Equilibrium at the interior DOF ``theta_b_i`` requires the beam-side
moment to equal the spring moment:

    K_beam_local[2, :] @ u_local  ==  M_h_i

When the spring is in its elastic regime this is a single linear
equation per hinge. Once the spring yields the equation is nonlinear,
and the element runs an internal Newton iteration on
``(theta_b_i, theta_b_j)`` to drive the residual to zero before
reporting its tangent and internal force to the global solver.

Limitations (current phase)
---------------------------
* No distributed loads. Add nodal loads to the model.
* 2-D only. The 3-D variant follows the same pattern but is not yet
  implemented.
* Mass is taken from the underlying elastic beam. The hinges contribute
  zero mass.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.elements.beam import BeamColumn2D
from femsolver.sections.elastic import ElasticSection2D
from femsolver.sections.hinges.spring import BilinearMomentRotationSpring


class HingedBeamColumn2D(Element):
    """Elastic beam with concentrated plastic hinges at the ends.

    Parameters
    ----------
    tag : int
    nodes : (int, int)
    material : Material
    area, Iz : float, optional
        Cross-section properties. Either both or ``section=`` must be given.
    section : ElasticSection2D, optional
        Pre-built elastic section, alternative to ``(area, Iz)``.
    hinge_i, hinge_j : BilinearMomentRotationSpring, optional
        Springs at end I (node 1 side) and end J (node 2 side). At least
        one must be supplied — otherwise use :class:`BeamColumn2D`.
    """

    n_nodes = 2
    dofs_per_node = 3

    # Internal-Newton settings. The tolerance is on the absolute residual
    # (units: moment), so it must be lenient enough to absorb the
    # round-off of beam-stiffness terms (order ``EI / L``) yet strict
    # enough to make the global Newton residual converge to its own tol.
    # 1e-9 is a reasonable default for typical structural-engineering
    # scales (kN-m, MPa, mm).
    _internal_max_iter: int = 50
    _internal_tol: float = 1.0e-9

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        area: float | None = None,
        Iz: float | None = None,
        *,
        section: ElasticSection2D | None = None,
        hinge_i: BilinearMomentRotationSpring | None = None,
        hinge_j: BilinearMomentRotationSpring | None = None,
    ):
        super().__init__(tag, nodes, material)
        if hinge_i is None and hinge_j is None:
            raise ValueError(
                "HingedBeamColumn2D needs at least one hinge — for a fully "
                "elastic beam, use BeamColumn2D instead"
            )
        if section is not None:
            if area is not None or Iz is not None:
                raise ValueError(
                    "HingedBeamColumn2D: pass either (area, Iz) or section=, not both"
                )
            self.section = section
            self.area = float(section.A)
            self.Iz = float(section.Iz)
        else:
            if area is None or Iz is None:
                raise ValueError(
                    "HingedBeamColumn2D: provide (area, Iz) or section="
                )
            self.area = float(area)
            self.Iz = float(Iz)
            self.section = ElasticSection2D(material.E, self.area, self.Iz)
        self.hinge_i = hinge_i
        self.hinge_j = hinge_j
        self.end_forces_local = np.zeros(6)
        # Trial / committed values of the *interior* beam-end rotations.
        # These are what the internal Newton iteration solves for.
        self._theta_bi_committed: float = 0.0
        self._theta_bi_trial: float = 0.0
        self._theta_bj_committed: float = 0.0
        self._theta_bj_trial: float = 0.0

    # ------------------------------------------------------------- geometry
    def length_and_angle(self) -> tuple[float, float, float]:
        c = self.node_coords()
        d = c[1] - c[0]
        L = float(np.linalg.norm(d))
        if L == 0.0:
            raise ValueError(f"hinged beam {self.tag} has zero length")
        return L, d[0] / L, d[1] / L

    def transform_matrix(self) -> np.ndarray:
        L, c, s = self.length_and_angle()
        R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        T = np.zeros((6, 6))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        return T

    # --------------------------------------------------- elastic beam K_local
    def _beam_K_local(self) -> np.ndarray:
        """6 x 6 stiffness of the elastic prismatic beam in local frame.

        DOF order is interpreted as
        ``[u1, v1, theta_b_i, u2, v2, theta_b_j]`` — the rotational DOFs
        are the *beam-side* rotations, not the node rotations. They become
        node rotations whenever the corresponding hinge is absent.
        """
        L, _, _ = self.length_and_angle()
        E = self.material.E
        A = self.area
        Iz = self.Iz
        a = E * A / L
        b3 = 12.0 * E * Iz / (L ** 3)
        b2 = 6.0 * E * Iz / (L ** 2)
        b1 = 4.0 * E * Iz / L
        b1h = 2.0 * E * Iz / L
        K = np.zeros((6, 6))
        K[0, 0] = a; K[0, 3] = -a
        K[3, 0] = -a; K[3, 3] = a
        K[1, 1] = b3; K[1, 2] = b2; K[1, 4] = -b3; K[1, 5] = b2
        K[2, 1] = b2; K[2, 2] = b1; K[2, 4] = -b2; K[2, 5] = b1h
        K[4, 1] = -b3; K[4, 2] = -b2; K[4, 4] = b3; K[4, 5] = -b2
        K[5, 1] = b2; K[5, 2] = b1h; K[5, 4] = -b2; K[5, 5] = b1
        return K

    # ------------------------------------------------ state determination
    def _u_local_full(self, theta_bi: float, theta_bj: float) -> np.ndarray:
        """Build the 6-vector that the elastic beam sees:
        ``[u1, v1, theta_b_i, u2, v2, theta_b_j]``.

        For DOFs without a hinge the beam rotation equals the node
        rotation, so we read it straight off the node displacements.
        """
        u_l_node = self.transform_matrix() @ self.gather_u()
        # u_l_node has DOF order [u1, v1, theta_n_i, u2, v2, theta_n_j]
        u_b = u_l_node.copy()
        if self.hinge_i is not None:
            u_b[2] = theta_bi
        if self.hinge_j is not None:
            u_b[5] = theta_bj
        return u_b, u_l_node

    def _state_determination(self) -> dict:
        """Run internal Newton on the beam-end rotations and return a
        dict with all the quantities the global solver needs.

        Returned keys
        -------------
        ``u_l_node`` (size 6)            local displacements at node DOFs
        ``u_b``      (size 6)            beam-side u_local with theta_b
                                         at the end positions
        ``K_b``      (6x6)               elastic beam stiffness
        ``M_h_i``, ``K_h_i``             converged spring response at I
                                         (``None`` if no hinge)
        ``M_h_j``, ``K_h_j``             converged spring response at J
        ``theta_b_i``, ``theta_b_j``     converged interior rotations

        Algorithm
        ---------
        Newton on the equilibrium residual at the interior DOFs:

            R_i = K_b[2, :] @ u_b - M_h_i(theta_n_i - theta_b_i) = 0
            R_j = K_b[5, :] @ u_b - M_h_j(theta_n_j - theta_b_j) = 0

        For an elastic-perfectly-plastic spring the residual is
        piecewise-linear in ``theta_b`` with a discontinuity in slope at
        the elastic-plastic boundary. Pure Newton can overshoot the
        regime each step and oscillate, so we wrap the update in a
        backtracking line search: try the full Newton step; if the
        residual norm does not decrease, halve and retry.
        """
        K_b = self._beam_K_local()
        theta_bi = self._theta_bi_trial if self.hinge_i is not None else 0.0
        theta_bj = self._theta_bj_trial if self.hinge_j is not None else 0.0

        def _evaluate(tbi: float, tbj: float):
            """Compute (residual, K_h_i, K_h_j, M_h_i, M_h_j, u_l_node, u_b)
            at the trial rotations. Each call mutates the spring's *trial*
            state — that is fine because the committed state is unchanged
            and the spring is deterministic in the input."""
            u_b, u_l_node = self._u_local_full(tbi, tbj)
            theta_n_i = u_l_node[2]
            theta_n_j = u_l_node[5]
            M_h_i, K_h_i = (
                self.hinge_i.get_response(theta_n_i - tbi)
                if self.hinge_i is not None else (None, None)
            )
            M_h_j, K_h_j = (
                self.hinge_j.get_response(theta_n_j - tbj)
                if self.hinge_j is not None else (None, None)
            )
            r = []
            if self.hinge_i is not None:
                r.append(K_b[2, :] @ u_b - M_h_i)
            if self.hinge_j is not None:
                r.append(K_b[5, :] @ u_b - M_h_j)
            return np.array(r), K_h_i, K_h_j, M_h_i, M_h_j, u_l_node, u_b

        r, K_h_i, K_h_j, M_h_i, M_h_j, u_l_node, u_b = _evaluate(theta_bi, theta_bj)

        for _ in range(self._internal_max_iter):
            norm_r = float(np.max(np.abs(r)))
            if norm_r < self._internal_tol:
                break
            # Newton tangent at the current state
            if self.hinge_i is not None and self.hinge_j is not None:
                Jm = np.array([
                    [K_b[2, 2] + K_h_i, K_b[2, 5]],
                    [K_b[5, 2],         K_b[5, 5] + K_h_j],
                ])
                d = np.linalg.solve(Jm, -r)
                d_bi, d_bj = float(d[0]), float(d[1])
            elif self.hinge_i is not None:
                d_bi = float(-r[0] / (K_b[2, 2] + K_h_i))
                d_bj = 0.0
            else:
                d_bi = 0.0
                d_bj = float(-r[0] / (K_b[5, 5] + K_h_j))
            # Backtracking line search — halve the step until the residual
            # norm goes down. Twenty backtracks let alpha shrink to ~1e-6,
            # well below any physically reasonable step size.
            alpha = 1.0
            for _bt in range(20):
                tbi_try = theta_bi + alpha * d_bi
                tbj_try = theta_bj + alpha * d_bj
                r_try, K_h_i_try, K_h_j_try, M_h_i_try, M_h_j_try, u_l_node_try, u_b_try = (
                    _evaluate(tbi_try, tbj_try)
                )
                norm_r_try = float(np.max(np.abs(r_try)))
                if norm_r_try < (1.0 - 1.0e-4 * alpha) * norm_r:
                    # accept
                    theta_bi, theta_bj = tbi_try, tbj_try
                    r = r_try
                    K_h_i, K_h_j = K_h_i_try, K_h_j_try
                    M_h_i, M_h_j = M_h_i_try, M_h_j_try
                    u_l_node, u_b = u_l_node_try, u_b_try
                    break
                alpha *= 0.5
            else:
                # Line search exhausted without finding a descent direction.
                # Re-evaluate at the current iterate so the springs leave
                # consistent trial state, then bail out.
                _evaluate(theta_bi, theta_bj)
                raise RuntimeError(
                    f"hinged beam {self.tag}: internal line search failed "
                    f"(|R| = {norm_r})"
                )
        else:
            raise RuntimeError(
                f"hinged beam {self.tag}: internal Newton did not converge "
                f"in {self._internal_max_iter} iterations "
                f"(|R| = {float(np.max(np.abs(r)))})"
            )
        self._theta_bi_trial = theta_bi
        self._theta_bj_trial = theta_bj
        return dict(
            u_l_node=u_l_node, u_b=u_b, K_b=K_b,
            M_h_i=M_h_i, K_h_i=K_h_i, M_h_j=M_h_j, K_h_j=K_h_j,
            theta_b_i=theta_bi, theta_b_j=theta_bj,
        )

    # ------------------------------------------------- stiffness / forces
    def _condensed_K_local(self, K_b: np.ndarray, K_h_i, K_h_j) -> np.ndarray:
        """Build the augmented K, partition, and condense to 6x6 in node
        DOFs. The hinge tangent stiffnesses ``K_h_i`` / ``K_h_j`` are
        evaluated at the converged state and may be zero (perfectly
        plastic) — both cases are handled by ``np.linalg.solve``.
        """
        # Augmented DOF layout: [u1, v1, theta_n_i, u2, v2, theta_n_j |
        #                        theta_b_i?, theta_b_j?]
        n_int = (1 if self.hinge_i is not None else 0) + (1 if self.hinge_j is not None else 0)
        n = 6 + n_int
        K_aug = np.zeros((n, n))
        # beam dof -> aug dof mapping
        next_int = 6
        if self.hinge_i is not None:
            beam_idx_i = next_int
            next_int += 1
        else:
            beam_idx_i = 2
        if self.hinge_j is not None:
            beam_idx_j = next_int
        else:
            beam_idx_j = 5
        beam_dofs = (0, 1, beam_idx_i, 3, 4, beam_idx_j)
        for ii in range(6):
            for jj in range(6):
                K_aug[beam_dofs[ii], beam_dofs[jj]] += K_b[ii, jj]
        # springs: K_h is the *current tangent*, supplied by caller
        if self.hinge_i is not None:
            n_idx, b_idx = 2, beam_idx_i
            K_aug[n_idx, n_idx] += K_h_i
            K_aug[n_idx, b_idx] -= K_h_i
            K_aug[b_idx, n_idx] -= K_h_i
            K_aug[b_idx, b_idx] += K_h_i
        if self.hinge_j is not None:
            n_idx, b_idx = 5, beam_idx_j
            K_aug[n_idx, n_idx] += K_h_j
            K_aug[n_idx, b_idx] -= K_h_j
            K_aug[b_idx, n_idx] -= K_h_j
            K_aug[b_idx, b_idx] += K_h_j
        # Condense out the internal DOFs
        if n_int == 0:
            return K_aug
        ext = list(range(6))
        intern = list(range(6, n))
        K_ee = K_aug[np.ix_(ext, ext)]
        K_ei = K_aug[np.ix_(ext, intern)]
        K_ie = K_aug[np.ix_(intern, ext)]
        K_ii = K_aug[np.ix_(intern, intern)]
        # K_ii can be near-singular when the spring is perfectly plastic
        # *and* the beam contribution at that DOF is the only term left.
        # In practice the beam contributes 4EI/L on the diagonal, which
        # keeps K_ii well-conditioned even for K_h = 0.
        K_cond = K_ee - K_ei @ np.linalg.solve(K_ii, K_ie)
        return K_cond

    def K_local(self) -> np.ndarray:
        """Initial (elastic) stiffness — used for linear analysis and the
        first iteration of modified Newton."""
        K_b = self._beam_K_local()
        K_h_i = self.hinge_i.K0 if self.hinge_i is not None else None
        K_h_j = self.hinge_j.K0 if self.hinge_j is not None else None
        return self._condensed_K_local(K_b, K_h_i, K_h_j)

    def K_global(self) -> np.ndarray:
        T = self.transform_matrix()
        return T.T @ self.K_local() @ T

    def K_tangent_global(self) -> np.ndarray:
        """Tangent stiffness at the *current* state.

        Runs internal state determination first so the spring tangents
        reflect the converged hinge rotations under the current node
        displacements.
        """
        sd = self._state_determination()
        K_cond = self._condensed_K_local(sd["K_b"], sd["K_h_i"], sd["K_h_j"])
        T = self.transform_matrix()
        return T.T @ K_cond @ T

    def f_int_global(self) -> np.ndarray:
        """Internal nodal force in global coords at the current state."""
        sd = self._state_determination()
        K_b = sd["K_b"]
        u_b = sd["u_b"]
        # internal force at the 6 node DOFs in local coords
        f_l = np.zeros(6)
        # translational DOFs come straight from the beam
        f_l[0] = K_b[0, :] @ u_b
        f_l[1] = K_b[1, :] @ u_b
        f_l[3] = K_b[3, :] @ u_b
        f_l[4] = K_b[4, :] @ u_b
        # rotational DOFs at the node:
        #   if there's a hinge -> moment exerted on the node by the spring
        #   if there's no hinge -> beam moment at that end (rotational dof
        #                          is the *node* rotation, identical to beam-end)
        if self.hinge_i is not None:
            f_l[2] = sd["M_h_i"]
        else:
            f_l[2] = K_b[2, :] @ u_b
        if self.hinge_j is not None:
            f_l[5] = sd["M_h_j"]
        else:
            f_l[5] = K_b[5, :] @ u_b
        T = self.transform_matrix()
        return T.T @ f_l

    # ----------------------------------------------------------------- mass
    def M_local(self, *, lumped: bool = False) -> np.ndarray:
        """Mass of the underlying elastic beam. Hinges contribute zero
        mass (they are zero-length, mass-less rotational springs).
        """
        # Re-use BeamColumn2D's mass formulation by delegating to a
        # temporary elastic beam — keeps the consistent-mass formula in
        # one place.
        rho = self.material.rho
        if rho == 0.0:
            return np.zeros((6, 6))
        # Ad hoc proxy: build a BeamColumn2D and call M_local. We can't
        # share state because the proxy is unbound, so just inline the
        # formulae using the same pattern as BeamColumn2D.
        L, _, _ = self.length_and_angle()
        m_total = rho * self.area * L
        if lumped:
            return np.diag([0.5, 0.5, 0.0, 0.5, 0.5, 0.0]) * m_total
        M = np.zeros((6, 6))
        M[0, 0] = M[3, 3] = m_total / 3.0
        M[0, 3] = M[3, 0] = m_total / 6.0
        f = m_total / 420.0
        Mt = f * np.array([
            [156.0,    22.0 * L,    54.0,   -13.0 * L],
            [ 22.0 * L,  4.0 * L * L, 13.0 * L, -3.0 * L * L],
            [ 54.0,    13.0 * L,   156.0,   -22.0 * L],
            [-13.0 * L, -3.0 * L * L, -22.0 * L,  4.0 * L * L],
        ])
        idx = [1, 2, 4, 5]
        for i, ig in enumerate(idx):
            for j, jg in enumerate(idx):
                M[ig, jg] = Mt[i, j]
        return M

    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        T = self.transform_matrix()
        return T.T @ self.M_local(lumped=lumped) @ T

    # ---------------------------------------------------------------- state
    def commit_state(self) -> None:
        if self.hinge_i is not None:
            self.hinge_i.commit_state()
        if self.hinge_j is not None:
            self.hinge_j.commit_state()
        self._theta_bi_committed = self._theta_bi_trial
        self._theta_bj_committed = self._theta_bj_trial

    def revert_state(self) -> None:
        if self.hinge_i is not None:
            self.hinge_i.revert_state()
        if self.hinge_j is not None:
            self.hinge_j.revert_state()
        self._theta_bi_trial = self._theta_bi_committed
        self._theta_bj_trial = self._theta_bj_committed

    # -------------------------------------------------------------- recovery
    def recover(self) -> None:
        sd = self._state_determination()
        K_b = sd["K_b"]
        u_b = sd["u_b"]
        # element-end forces in local coords, in the same convention as
        # BeamColumn2D: [F_x1, F_y1, M_z1, F_x2, F_y2, M_z2].
        f_l = np.zeros(6)
        f_l[0] = K_b[0, :] @ u_b
        f_l[1] = K_b[1, :] @ u_b
        f_l[2] = sd["M_h_i"] if self.hinge_i is not None else K_b[2, :] @ u_b
        f_l[3] = K_b[3, :] @ u_b
        f_l[4] = K_b[4, :] @ u_b
        f_l[5] = sd["M_h_j"] if self.hinge_j is not None else K_b[5, :] @ u_b
        self.end_forces_local = f_l
