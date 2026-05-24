"""Euler-Bernoulli beam-column elements (2D and 3D).

Local axis convention (3D):
    x : along the element axis (node1 -> node2)
    y : local transverse, in the plane defined by local-x and local-z, by y = z x x
    z : determined from the user-supplied orientation vector `vecxz` whose
        projection orthogonal to local-x lies along +z

`vecxz` defaults to a heuristic: (0, 0, 1) unless the element is vertical, in
which case (1, 0, 0). For a customized orientation pass the `vecxz` argument.
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.base import Element
from femsolver.numerics.quadrature import gauss_lobatto_1d
from femsolver.sections.base import SectionBase
from femsolver.sections.elastic import ElasticSection2D, ElasticSection3D


class BeamColumn2D(Element):
    """2-node 2D Euler-Bernoulli beam-column. 3 DOF/node (ux, uy, rz).

    Construction
    ------------
    Three constructors are supported, matched to three different
    use-cases:

    * ``BeamColumn2D(tag, nodes, mat, area, Iz)`` — legacy API: an
      :class:`ElasticSection2D` is built internally.
    * ``BeamColumn2D(tag, nodes, mat, section=ElasticSection2D(...))`` —
      explicit elastic section, identical behaviour to the legacy path.
    * ``BeamColumn2D(tag, nodes, mat, section=FiberSection2D(...))`` —
      fiber section. The section is detected as stateful via
      :attr:`SectionBase.is_stateful` and is *cloned per integration
      point*, so each Gauss-Lobatto point along the element gets its
      own independent state. ``use_numerical_integration`` is forced to
      ``True`` because a fiber response cannot be reduced to a
      closed-form ``EI``.

    Two stiffness paths are available:

    * ``use_numerical_integration = False`` (default for elastic) —
      closed-form 6 x 6 ``K_local``. Fastest; exact for an elastic
      prismatic Euler-Bernoulli beam.
    * ``use_numerical_integration = True`` — Gauss-Lobatto integration
      of :math:`\\int_0^L B^T k_s B \\,dx`. For elastic sections this
      reproduces the closed form to machine precision once
      ``n_int >= 3``; for fiber sections it carries the per-integration-
      point constitutive state needed for distributed plasticity.
    """

    n_nodes = 2
    dofs_per_node = 3

    # --- numerical-integration controls (class-level defaults; settable on
    # instances). Default closed-form so all existing analyses are unaffected.
    use_numerical_integration: bool = False
    n_int: int = 5    # Gauss-Lobatto points; n >= 3 needed for exact bending

    def __init__(
        self,
        tag: int,
        nodes,
        material,
        area: float | None = None,
        Iz: float | None = None,
        *,
        section: SectionBase | None = None,
    ):
        super().__init__(tag, nodes, material)

        # Section input handling:
        #
        # Did the user pass a section object?
        # │
        # ├── Yes
        # │   │
        # │   ├── Stateful/nonlinear section
        # │   │   ├── Use gross_area and gross_Iz
        # │   │   ├── Enable numerical integration
        # │   │   └── Clone section for each integration point
        # │   │
        # │   └── Stateless/elastic section
        # │       ├── Use A and Iz
        # │       └── Share same section across integration points
        # │
        # └── No
        #     ├── Require area and Iz
        #     ├── Create ElasticSection2D automatically
        #     └── Share same elastic section across integration points

        if section is not None:
            if area is not None or Iz is not None:
                raise ValueError(
                    "BeamColumn2D: pass either (area, Iz) or section=, not both"
                )

            self.section = section

            # Duck-typed sections (not subclassing SectionBase) may not
            # declare ``is_stateful``. Default to True — that is the safe
            # choice; a stateless section that wants the elastic fast path
            # has to advertise itself explicitly.
            is_stateful = getattr(section, "is_stateful", True)

            if is_stateful and hasattr(section, "gross_area"):
                # Real stateful section (e.g. FiberSection2D): needs
                # per-IP state and numerical integration.
                self.area = float(section.gross_area)
                self.Iz = float(section.gross_Iz)
                self.use_numerical_integration = True
                self._stateful_sections = True
                clone = getattr(section, "clone", None)
                self.sections: list[SectionBase] = [
                    (clone() if clone is not None else section)
                    for _ in range(self.n_int)
                ]

            else:
                # Elastic/stateless sections can be shared across IPs.
                if not hasattr(section, "A") or not hasattr(section, "Iz"):
                    raise ValueError(
                        "BeamColumn2D: stateless section must expose "
                        "'A' and 'Iz' attributes"
                    )

                self.area = float(section.A)
                self.Iz = float(section.Iz)
                self._stateful_sections = False
                self.sections = [section] * self.n_int

        else:
            # No section object was given, so create a simple elastic section.
            if area is None or Iz is None:
                raise ValueError(
                    "BeamColumn2D: provide (area, Iz) or a section= argument"
                )

            if area <= 0 or Iz <= 0:
                raise ValueError("area and Iz must be positive")

            self.area = float(area)
            self.Iz = float(Iz)
            self.section = ElasticSection2D(material.E, self.area, self.Iz)
            self._stateful_sections = False
            self.sections = [self.section] * self.n_int

        self._wy_local: float = 0.0  # transverse uniform distributed load, local y
        self.end_forces_local = np.zeros(6)

    # ------------------------------------------------------------------- geom
    def length_and_angle(self) -> tuple[float, float, float]:
        c = self.node_coords()
        d = c[1] - c[0]
        L = float(np.linalg.norm(d))
        if L == 0.0:
            raise ValueError(f"beam {self.tag} has zero length")
        return L, d[0] / L, d[1] / L  # L, cos(theta), sin(theta)

    def transform_matrix(self) -> np.ndarray:
        L, c, s = self.length_and_angle()
        R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        T = np.zeros((6, 6))
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        return T

    # -------------------------------------------------------------- stiffness
    def K_local(self) -> np.ndarray:
        """Local stiffness — dispatches to closed-form or numerical path."""
        if self.use_numerical_integration:
            return self._K_local_numerical()
        return self._K_local_closed_form()

    def _K_local_closed_form(self) -> np.ndarray:
        """Analytical 6 x 6 stiffness for an elastic prismatic beam.

        DOF order (local): ``[u1, v1, theta1, u2, v2, theta2]``.
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

    def _strain_disp_matrix(self, xi: float, L: float) -> np.ndarray:
        """Section strain-displacement matrix at natural coordinate xi.

        Returns the (2, 6) matrix ``B`` such that the section generalized
        strain ``e = [eps_axial, kappa_z] = B @ u_local``, with DOF order
        ``[u1, v1, theta1, u2, v2, theta2]`` and natural coordinate
        ``xi in [-1, 1]`` mapped to ``x in [0, L]`` via ``x = L (1 + xi) / 2``.

        Derivation
        ----------
        Axial strain uses linear shape functions on ``u``:

            u(x) = (1 - x/L) u1 + (x/L) u2     =>   du/dx = (u2 - u1) / L

        Bending uses the Hermite cubic basis on ``v``:

            v(x) = H1(x) v1 + H2(x) theta1 + H3(x) v2 + H4(x) theta2

        with ``H1, ..., H4`` the standard cubics on ``[0, L]``. Curvature
        is ``kappa_z = d^2 v / dx^2``. Substituting ``x = L(1+xi)/2``
        collapses the resulting expressions to the linear-in-xi forms used
        below.
        """
        B = np.zeros((2, 6))
        # axial: constant in xi
        B[0, 0] = -1.0 / L
        B[0, 3] = 1.0 / L
        # bending (Hermite cubic second derivative), DOFs (1, 2, 4, 5)
        B[1, 1] = 6.0 * xi / (L * L)
        B[1, 2] = (3.0 * xi - 1.0) / L
        B[1, 4] = -6.0 * xi / (L * L)
        B[1, 5] = (3.0 * xi + 1.0) / L
        return B

    def _ensure_sections_length(self) -> None:
        """Make sure ``self.sections`` has ``n_int`` entries.

        Users may flip ``self.n_int`` after construction (the test suite
        does, parametrizing over quadrature orders). For *shared*
        sections that is harmless — we just resize the list. For
        per-IP stateful sections it is not safe: each entry carries
        independent history and growing the list would silently start
        empty fibers mid-analysis.
        """
        if len(self.sections) == self.n_int:
            return
        if self._stateful_sections:
            raise RuntimeError(
                f"BeamColumn2D {self.tag}: cannot change n_int after "
                f"construction for stateful sections "
                f"(have {len(self.sections)} per-IP states, asked for {self.n_int})"
            )
        # Stateless / shared section: rebuild the list, all entries point
        # at the same object (cost is one Python list, not n_int copies).
        self.sections = [self.section] * self.n_int

    def _K_local_numerical(self, *, use_current_state: bool = False) -> np.ndarray:
        """Numerically-integrated local stiffness via Gauss-Lobatto.

            K = integral_{0}^{L} B^T(x) k_s(x) B(x) dx
              ~ sum_i w_i (L/2) B^T(xi_i) k_s(xi_i) B(xi_i)

        Parameters
        ----------
        use_current_state : bool, default False
            When ``False`` (the legacy / "initial elastic" path) the
            section is queried at zero strain — for an elastic section
            this reproduces the closed-form K to machine precision once
            ``n_int >= 3``.  When ``True`` the section is queried at the
            actual strain ``B(xi) @ u_local``, giving the *consistent
            tangent* needed by Newton-Raphson when the section carries
            state (fiber plasticity, etc.).
        """
        self._ensure_sections_length()
        L, _, _ = self.length_and_angle()
        xi_pts, w_pts = gauss_lobatto_1d(self.n_int)
        K = np.zeros((6, 6))
        jac = 0.5 * L  # dx/dxi
        if use_current_state:
            T = self.transform_matrix()
            u_l = T @ self.gather_u()
            for i, (xi, w) in enumerate(zip(xi_pts, w_pts)):
                B = self._strain_disp_matrix(xi, L)
                e_i = B @ u_l
                _, ks = self.sections[i].get_response(e_i)
                K += (w * jac) * (B.T @ ks @ B)
        else:
            zero_strain = np.zeros(self.section.n_resultants)
            for i, (xi, w) in enumerate(zip(xi_pts, w_pts)):
                B = self._strain_disp_matrix(xi, L)
                _, ks = self.sections[i].get_response(zero_strain)
                K += (w * jac) * (B.T @ ks @ B)
        return K

    def K_global(self) -> np.ndarray:
        T = self.transform_matrix()
        return T.T @ self.K_local() @ T

    # ------------------------------------------------------ tangent / f_int
    def K_tangent_global(self) -> np.ndarray:
        """Tangent stiffness at the *current* deformation state.

        For elastic (stateless) sections this is identical to
        :meth:`K_global` and we delegate to it. For stateful (e.g.
        fiber) sections we re-evaluate the section response at every
        Gauss-Lobatto point using the actual strain ``B(xi) @ u_local``
        so the returned matrix is the consistent linearisation of
        :meth:`f_int_global`.
        """
        if not self._stateful_sections:
            return self.K_global()
        T = self.transform_matrix()
        K_loc = self._K_local_numerical(use_current_state=True)
        return T.T @ K_loc @ T

    def f_int_global(self) -> np.ndarray:
        """Internal nodal force at the current state.

        For elastic sections this falls back to the default
        ``K_global() @ u`` (correct for linear materials). For stateful
        sections we integrate the section forces along the element via
        Gauss-Lobatto:

            f_int_local = integral B^T(x) s(x) dx

        where ``s(x)`` is the section force vector returned by the
        section's ``get_response`` at the current local strain.
        """
        if not self._stateful_sections:
            return super().f_int_global()
        self._ensure_sections_length()
        L, _, _ = self.length_and_angle()
        T = self.transform_matrix()
        u_l = T @ self.gather_u()
        xi_pts, w_pts = gauss_lobatto_1d(self.n_int)
        f_l = np.zeros(6)
        jac = 0.5 * L
        for i, (xi, w) in enumerate(zip(xi_pts, w_pts)):
            B = self._strain_disp_matrix(xi, L)
            e_i = B @ u_l
            s_i, _ = self.sections[i].get_response(e_i)
            f_l += (w * jac) * (B.T @ s_i)
        return T.T @ f_l

    # ----------------------------------------------------------------- mass
    def M_local(self, *, lumped: bool = False) -> np.ndarray:
        """Local consistent / lumped mass matrix for a 2D Euler-Bernoulli
        beam. DOF order: u1, v1, theta1, u2, v2, theta2.
        """
        rho = self.material.rho
        if rho == 0.0:
            return np.zeros((6, 6))
        L, _, _ = self.length_and_angle()
        m_total = rho * self.area * L
        if lumped:
            # translation only — rotational lumped mass is zero (a common
            # choice; the HRZ alternative is left for future extension)
            return np.diag([0.5, 0.5, 0.0, 0.5, 0.5, 0.0]) * m_total
        # consistent — axial uses linear shape, transverse + rotation uses
        # the cubic Hermite mass:
        #     M_trans = rho A L / 420 *
        #         [[156,  22 L,   54, -13 L],
        #          [ 22 L,  4 L^2, 13 L, -3 L^2],
        #          [ 54,  13 L,  156, -22 L],
        #          [-13 L, -3 L^2, -22 L,  4 L^2]]
        M = np.zeros((6, 6))
        # axial (rows/cols 0 and 3)
        M[0, 0] = M[3, 3] = m_total / 3.0
        M[0, 3] = M[3, 0] = m_total / 6.0
        # transverse + rotational (rows/cols 1, 2, 4, 5)
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

    # ----------------------------------------------------------------- loads
    def add_uniform_load(self, wy_local: float) -> None:
        """Uniform transverse load (force per unit length) along local +y."""
        self._wy_local += float(wy_local)

    def clear_distributed_loads(self) -> None:
        self._wy_local = 0.0

    def f_eq_local(self) -> np.ndarray:
        L, _, _ = self.length_and_angle()
        wy = self._wy_local
        if wy == 0.0:
            return np.zeros(6)
        return np.array([
            0.0,
            wy * L / 2.0,
            wy * L * L / 12.0,
            0.0,
            wy * L / 2.0,
            -wy * L * L / 12.0,
        ])

    def f_eq_global(self) -> np.ndarray:
        return self.transform_matrix().T @ self.f_eq_local()

    # -------------------------------------------------------------- recovery
    def recover(self) -> None:
        """Compute element-end forces and per-section response.

        Two outputs are produced:

        * ``self.end_forces_local`` (length 6) — internal nodal forces in
          the local frame, computed as ``K_local @ u_local - f_eq_local``.
          This is the long-standing element-level result.
        * ``self.section_locations`` (length ``n_int``),
          ``self.section_strains`` (shape ``(n_int, n_resultants)``),
          ``self.section_forces`` (same shape) — section response at each
          Gauss-Lobatto point along the element. Useful for moment
          diagrams in linear analysis and the foundation for state
          determination once non-elastic sections (hinges, fibers) are
          introduced.
        """
        T = self.transform_matrix()
        u_g = self.gather_u()
        u_l = T @ u_g
        # element-end forces in local coordinates: K_local @ u_local - f_eq_local
        # (Internal nodal force = K u - f_eq, sign convention follows F = K u for "what the structure does to the element")
        self.end_forces_local = self.K_local() @ u_l - self.f_eq_local()
        # Per-section response at Gauss-Lobatto integration points.
        L, _, _ = self.length_and_angle()
        self._evaluate_sections_along_length(u_l, L)

    def _evaluate_sections_along_length(self, u_l: np.ndarray, L: float) -> None:
        """Populate ``section_locations``, ``section_strains``,
        ``section_forces`` by evaluating the section at each integration
        point. Uses the per-IP section so fiber state is correctly
        reflected; for elastic (shared section) all entries reduce to
        the same call.
        """
        self._ensure_sections_length()
        xi_pts, _ = gauss_lobatto_1d(self.n_int)
        n_r = self.section.n_resultants
        self.section_locations = np.empty(self.n_int)
        self.section_strains = np.empty((self.n_int, n_r))
        self.section_forces = np.empty((self.n_int, n_r))
        for i, xi in enumerate(xi_pts):
            B = self._strain_disp_matrix(xi, L)
            e_i = B @ u_l
            s_i, _ = self.sections[i].get_response(e_i)
            self.section_locations[i] = 0.5 * L * (1.0 + xi)
            self.section_strains[i] = e_i
            self.section_forces[i] = s_i

    # ---------------------------------------------------------------- state
    def commit_state(self) -> None:
        """Forward converged state to the section(s).

        For stateful sections each integration point holds its own
        independent state, so we iterate over ``self.sections``. For
        stateless sections all entries point to the same shared object
        and the loop is a no-op (the elastic section's
        ``commit_state`` is itself a no-op).
        """
        if self._stateful_sections:
            for sec in self.sections:
                sec.commit_state()
        else:
            self.section.commit_state()

    def revert_state(self) -> None:
        if self._stateful_sections:
            for sec in self.sections:
                sec.revert_state()
        else:
            self.section.revert_state()


class BeamColumn3D(Element):
    """2-node 3D Euler-Bernoulli beam-column. 6 DOF/node (ux, uy, uz, rx, ry, rz).

    Three construction paths, mirroring :class:`BeamColumn2D`:

    * ``BeamColumn3D(tag, nodes, mat, area, Iy, Iz, J)`` — legacy
      elastic; an internal :class:`ElasticSection3D` is built.
    * ``BeamColumn3D(tag, nodes, mat, section=ElasticSection3D(...))``
      — explicit elastic section, identical to the legacy path.
    * ``BeamColumn3D(tag, nodes, mat, section=FiberSection3D(...))``
      — stateful fiber section, cloned per integration point.
      ``use_numerical_integration`` is forced ``True``; section
      response (and tangent) is queried at every Gauss-Lobatto point.

    ``use_numerical_integration`` switches the local stiffness from
    the closed-form 12 x 12 expression to a Gauss-Lobatto integration
    of :math:`\\int_0^L B^T k_s B \\,dx`. For elastic sections the
    two paths agree to machine precision once ``n_int >= 3``; for
    fiber sections only the numerical path is meaningful.
    """

    n_nodes = 2
    dofs_per_node = 6

    use_numerical_integration: bool = False
    n_int: int = 5

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
        super().__init__(tag, nodes, material)
        if section is not None:
            if any(v is not None for v in (area, Iy, Iz, J)):
                raise ValueError(
                    "BeamColumn3D: pass either (area, Iy, Iz, J) or section=, not both"
                )
            self.section = section
            # Duck-typed sections (not subclassing SectionBase) may not
            # declare is_stateful — default to True (the safe choice).
            is_stateful = getattr(section, "is_stateful", True)
            if is_stateful and hasattr(section, "gross_area"):
                # Stateful (fiber) section: per-IP clones, numerical
                # integration, gross properties for the closed-form
                # fallback / mass matrix.
                self.area = float(section.gross_area)
                self.Iz = float(section.gross_Iz)
                self.Iy = float(section.gross_Iy)
                # GJ is the meaningful torsional input; if the section
                # exposes a usable ``J``, use it; otherwise back out
                # GJ / material.G.
                if hasattr(section, "GJ"):
                    self.J = float(section.GJ) / float(material.G)
                else:
                    self.J = float(getattr(section, "J", 0.0)) or 1.0
                self.use_numerical_integration = True
                self._stateful_sections = True
                clone = getattr(section, "clone", None)
                self.sections: list[SectionBase] = [
                    (clone() if clone is not None else section)
                    for _ in range(self.n_int)
                ]
            else:
                # Stateless elastic section — share across IPs.
                for attr in ("A", "Iy", "Iz", "J"):
                    if not hasattr(section, attr):
                        raise ValueError(
                            f"BeamColumn3D: section must expose '{attr}' attribute"
                        )
                self.area = float(section.A)
                self.Iy = float(section.Iy)
                self.Iz = float(section.Iz)
                self.J = float(section.J)
                self._stateful_sections = False
                self.sections = [section] * self.n_int
        else:
            if area is None or Iy is None or Iz is None or J is None:
                raise ValueError(
                    "BeamColumn3D: provide (area, Iy, Iz, J) or a section= argument"
                )
            if area <= 0 or Iy <= 0 or Iz <= 0 or J <= 0:
                raise ValueError("area, Iy, Iz, J must be positive")
            self.area = float(area)
            self.Iy = float(Iy)
            self.Iz = float(Iz)
            self.J = float(J)
            self.section = ElasticSection3D(
                material.E, material.G, self.area, self.Iy, self.Iz, self.J
            )
            self._stateful_sections = False
            self.sections = [self.section] * self.n_int
        self._vecxz_user = np.asarray(vecxz, dtype=float) if vecxz is not None else None
        self._wy_local: float = 0.0
        self._wz_local: float = 0.0
        self.end_forces_local = np.zeros(12)

    def length_and_axes(self) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        coords = self.node_coords()
        d = coords[1] - coords[0]
        if d.size == 2:
            d = np.array([d[0], d[1], 0.0])
        L = float(np.linalg.norm(d))
        if L == 0.0:
            raise ValueError(f"beam {self.tag} has zero length")
        ex = d / L
        if self._vecxz_user is not None:
            v = self._vecxz_user
        else:
            # pick a default not parallel to ex
            if abs(ex[2]) < 0.999:
                v = np.array([0.0, 0.0, 1.0])
            else:
                v = np.array([1.0, 0.0, 0.0])
        # Make ez perpendicular to ex by removing projection
        ez = v - (v @ ex) * ex
        nz = np.linalg.norm(ez)
        if nz < 1e-12:
            raise ValueError(
                f"beam {self.tag}: vecxz is parallel to element axis"
            )
        ez = ez / nz
        ey = np.cross(ez, ex)
        return L, ex, ey, ez

    def rotation_matrix(self) -> np.ndarray:
        _, ex, ey, ez = self.length_and_axes()
        R = np.vstack([ex, ey, ez])  # rows are local axes in global coords
        return R

    def transform_matrix(self) -> np.ndarray:
        R = self.rotation_matrix()
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = R
        return T

    def K_local(self) -> np.ndarray:
        """Local stiffness — dispatches to closed-form or numerical path."""
        if self.use_numerical_integration:
            return self._K_local_numerical()
        return self._K_local_closed_form()

    def _K_local_closed_form(self) -> np.ndarray:
        """Analytical 12 x 12 stiffness for an elastic prismatic beam.

        DOF order (local): ``[u, v, w, theta_x, theta_y, theta_z]`` per
        node, nodes 1 then 2.
        """
        L, _, _, _ = self.length_and_axes()
        E = self.material.E
        G = self.material.G
        A = self.area
        Iy, Iz, J = self.Iy, self.Iz, self.J
        a = E * A / L
        t = G * J / L
        bz3 = 12.0 * E * Iz / (L ** 3)
        bz2 = 6.0 * E * Iz / (L ** 2)
        bz1 = 4.0 * E * Iz / L
        bz1h = 2.0 * E * Iz / L
        by3 = 12.0 * E * Iy / (L ** 3)
        by2 = 6.0 * E * Iy / (L ** 2)
        by1 = 4.0 * E * Iy / L
        by1h = 2.0 * E * Iy / L

        K = np.zeros((12, 12))
        # axial (u1-u7)
        K[0, 0] = a; K[0, 6] = -a
        K[6, 0] = -a; K[6, 6] = a
        # torsion (rx1-rx2 = dofs 3,9)
        K[3, 3] = t; K[3, 9] = -t
        K[9, 3] = -t; K[9, 9] = t
        # bending about z (uy, rz: dofs 1,5,7,11)
        idx_z = [1, 5, 7, 11]
        Kz = np.array([
            [bz3,  bz2, -bz3,  bz2],
            [bz2,  bz1, -bz2,  bz1h],
            [-bz3, -bz2, bz3, -bz2],
            [bz2,  bz1h, -bz2, bz1],
        ])
        for i, ii in enumerate(idx_z):
            for j, jj in enumerate(idx_z):
                K[ii, jj] = Kz[i, j]
        # bending about y (uz, ry: dofs 2,4,8,10) — sign convention flipped on coupling
        idx_y = [2, 4, 8, 10]
        Ky = np.array([
            [by3,  -by2, -by3, -by2],
            [-by2, by1,  by2,  by1h],
            [-by3, by2,  by3,  by2],
            [-by2, by1h, by2,  by1],
        ])
        for i, ii in enumerate(idx_y):
            for j, jj in enumerate(idx_y):
                K[ii, jj] = Ky[i, j]
        return K

    def _strain_disp_matrix(self, xi: float, L: float) -> np.ndarray:
        """Section strain-displacement matrix at natural coordinate xi.

        Returns the (4, 12) matrix ``B`` such that ``e = B @ u_local``.
        Strain ordering: ``[eps_axial, kappa_z, kappa_y, gamma_torsion]``.
        DOF ordering: ``[u, v, w, theta_x, theta_y, theta_z]`` per node.

        The bending-about-z block uses the same Hermite cubic 2nd
        derivatives as the 2D element (DOFs ``v``, ``theta_z``). The
        bending-about-y block uses the same shape-function magnitudes but
        with **flipped signs on the rotational columns** because, with
        the local axis convention ``y = z x x``, ``dw/dx = -theta_y``
        (whereas ``dv/dx = +theta_z``). This single sign convention is
        what produces the off-diagonal sign pattern observed in the
        closed-form ``Ky`` block.

        Torsion is linear in ``theta_x``: ``gamma = d theta_x / dx``.
        """
        B = np.zeros((4, 12))
        # axial (DOFs 0, 6)
        B[0, 0] = -1.0 / L
        B[0, 6] = 1.0 / L
        # bending about z (DOFs 1, 5, 7, 11): v at 1,7 and theta_z at 5,11
        B[1, 1] = 6.0 * xi / (L * L)
        B[1, 5] = (3.0 * xi - 1.0) / L
        B[1, 7] = -6.0 * xi / (L * L)
        B[1, 11] = (3.0 * xi + 1.0) / L
        # bending about y (DOFs 2, 4, 8, 10): w at 2,8 and theta_y at 4,10
        # rotational columns flip sign because dw/dx = -theta_y
        B[2, 2] = 6.0 * xi / (L * L)
        B[2, 4] = -(3.0 * xi - 1.0) / L
        B[2, 8] = -6.0 * xi / (L * L)
        B[2, 10] = -(3.0 * xi + 1.0) / L
        # torsion (DOFs 3, 9)
        B[3, 3] = -1.0 / L
        B[3, 9] = 1.0 / L
        return B

    def _ensure_sections_length(self) -> None:
        """Make sure ``self.sections`` has ``n_int`` entries (mirrors
        BeamColumn2D)."""
        if len(self.sections) == self.n_int:
            return
        if self._stateful_sections:
            raise RuntimeError(
                f"BeamColumn3D {self.tag}: cannot change n_int after "
                f"construction for stateful sections "
                f"(have {len(self.sections)} per-IP states, asked for {self.n_int})"
            )
        self.sections = [self.section] * self.n_int

    def _K_local_numerical(self, *, use_current_state: bool = False) -> np.ndarray:
        """Numerically-integrated local stiffness via Gauss-Lobatto.

        With ``use_current_state=False`` (default), the section is
        queried at zero strain — for an elastic section this matches
        the closed-form K to machine precision. With
        ``use_current_state=True`` the section is queried at the
        actual strain ``B(xi) @ u_local``, giving the consistent
        tangent for the current state (needed for fiber-section
        plasticity).
        """
        self._ensure_sections_length()
        L, _, _, _ = self.length_and_axes()
        xi_pts, w_pts = gauss_lobatto_1d(self.n_int)
        K = np.zeros((12, 12))
        jac = 0.5 * L
        if use_current_state:
            T = self.transform_matrix()
            u_l = T @ self.gather_u()
            for i, (xi, w) in enumerate(zip(xi_pts, w_pts)):
                B = self._strain_disp_matrix(xi, L)
                e_i = B @ u_l
                _, ks = self.sections[i].get_response(e_i)
                K += (w * jac) * (B.T @ ks @ B)
        else:
            zero_strain = np.zeros(self.section.n_resultants)
            for i, (xi, w) in enumerate(zip(xi_pts, w_pts)):
                B = self._strain_disp_matrix(xi, L)
                _, ks = self.sections[i].get_response(zero_strain)
                K += (w * jac) * (B.T @ ks @ B)
        return K

    def K_global(self) -> np.ndarray:
        T = self.transform_matrix()
        return T.T @ self.K_local() @ T

    # ------------------------------------------------------ tangent / f_int
    def K_tangent_global(self) -> np.ndarray:
        """Tangent stiffness at the current state. For elastic
        sections this is identical to :meth:`K_global`; for stateful
        sections it re-evaluates the section response at every IP
        with the actual current strain."""
        if not self._stateful_sections:
            return self.K_global()
        T = self.transform_matrix()
        K_loc = self._K_local_numerical(use_current_state=True)
        return T.T @ K_loc @ T

    def f_int_global(self) -> np.ndarray:
        """Internal nodal force at the current state. For stateful
        sections, integrates ``B^T s(x)`` along the element using
        Gauss-Lobatto. For elastic / linear elements, falls back to
        ``K_global() @ u``."""
        if not self._stateful_sections:
            return super().f_int_global()
        self._ensure_sections_length()
        L, _, _, _ = self.length_and_axes()
        T = self.transform_matrix()
        u_l = T @ self.gather_u()
        xi_pts, w_pts = gauss_lobatto_1d(self.n_int)
        f_l = np.zeros(12)
        jac = 0.5 * L
        for i, (xi, w) in enumerate(zip(xi_pts, w_pts)):
            B = self._strain_disp_matrix(xi, L)
            e_i = B @ u_l
            s_i, _ = self.sections[i].get_response(e_i)
            f_l += (w * jac) * (B.T @ s_i)
        return T.T @ f_l

    # ----------------------------------------------------------------- state
    def commit_state(self) -> None:
        """Forward converged state to the section(s). For stateful
        sections each IP holds its own state; for stateless sections
        all entries point to the same shared object."""
        if self._stateful_sections:
            for sec in self.sections:
                sec.commit_state()
        else:
            self.section.commit_state()

    def revert_state(self) -> None:
        if self._stateful_sections:
            for sec in self.sections:
                sec.revert_state()
        else:
            self.section.revert_state()

    # ----------------------------------------------------------------- mass
    def M_local(self, *, lumped: bool = False) -> np.ndarray:
        """Local consistent / lumped mass matrix (12x12).

        DOF order per node: u, v, w, theta_x, theta_y, theta_z.
        """
        rho = self.material.rho
        if rho == 0.0:
            return np.zeros((12, 12))
        L, _, _, _ = self.length_and_axes()
        m_total = rho * self.area * L
        if lumped:
            # translation only — rotational DOFs get zero (HRZ deferred)
            d = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0,
                          0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
            return np.diag(d) * m_total
        # consistent — combine axial (linear shape), torsion (linear),
        # bending about z (cubic Hermite, transverse v + theta_z) and
        # bending about y (cubic Hermite, transverse w + theta_y).
        M = np.zeros((12, 12))
        # axial: dofs 0, 6
        M[0, 0] = M[6, 6] = m_total / 3.0
        M[0, 6] = M[6, 0] = m_total / 6.0
        # torsion (rotational inertia about local x): use polar moment ratio
        #   m_t_total = rho * J * L (rotational mass per unit angular vel.)
        # consistent torsion mass = (m_t_total / 6) * [[2, 1], [1, 2]]
        m_torsion = rho * self.J * L
        M[3, 3] = M[9, 9] = m_torsion / 3.0
        M[3, 9] = M[9, 3] = m_torsion / 6.0
        # bending about z (transverse v at dofs 1, 7 + theta_z at dofs 5, 11)
        f = m_total / 420.0
        Mz = f * np.array([
            [156.0,    22.0 * L,    54.0,   -13.0 * L],
            [ 22.0 * L,  4.0 * L * L, 13.0 * L, -3.0 * L * L],
            [ 54.0,    13.0 * L,   156.0,   -22.0 * L],
            [-13.0 * L, -3.0 * L * L, -22.0 * L,  4.0 * L * L],
        ])
        idx_z = [1, 5, 7, 11]
        for i, ii in enumerate(idx_z):
            for j, jj in enumerate(idx_z):
                M[ii, jj] = Mz[i, j]
        # bending about y (transverse w at dofs 2, 8 + theta_y at dofs 4, 10)
        # — sign convention flips the cross-coupling between w and theta_y
        My = f * np.array([
            [156.0,    -22.0 * L,    54.0,    13.0 * L],
            [-22.0 * L,   4.0 * L * L, -13.0 * L, -3.0 * L * L],
            [ 54.0,    -13.0 * L,   156.0,    22.0 * L],
            [ 13.0 * L,  -3.0 * L * L,  22.0 * L,  4.0 * L * L],
        ])
        idx_y = [2, 4, 8, 10]
        for i, ii in enumerate(idx_y):
            for j, jj in enumerate(idx_y):
                M[ii, jj] = My[i, j]
        return M

    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        T = self.transform_matrix()
        return T.T @ self.M_local(lumped=lumped) @ T

    # ----------------------------------------------------------------- loads
    def add_uniform_load(self, wy_local: float = 0.0, wz_local: float = 0.0) -> None:
        self._wy_local += float(wy_local)
        self._wz_local += float(wz_local)

    def clear_distributed_loads(self) -> None:
        self._wy_local = 0.0
        self._wz_local = 0.0

    def f_eq_local(self) -> np.ndarray:
        L, _, _, _ = self.length_and_axes()
        f = np.zeros(12)
        wy, wz = self._wy_local, self._wz_local
        if wy != 0.0:
            f[1] += wy * L / 2.0
            f[5] += wy * L * L / 12.0
            f[7] += wy * L / 2.0
            f[11] += -wy * L * L / 12.0
        if wz != 0.0:
            f[2] += wz * L / 2.0
            f[4] += -wz * L * L / 12.0
            f[8] += wz * L / 2.0
            f[10] += wz * L * L / 12.0
        return f

    def f_eq_global(self) -> np.ndarray:
        return self.transform_matrix().T @ self.f_eq_local()

    def recover(self) -> None:
        """Compute element-end forces and per-section response.

        Mirrors the 2D variant. ``self.section_strains[i]`` has length 4
        with ordering ``[eps_axial, kappa_z, kappa_y, gamma_torsion]`` and
        ``self.section_forces[i]`` carries ``[N, Mz, My, T]``.
        """
        T = self.transform_matrix()
        u_g = self.gather_u()
        u_l = T @ u_g
        self.end_forces_local = self.K_local() @ u_l - self.f_eq_local()
        L, _, _, _ = self.length_and_axes()
        self._evaluate_sections_along_length(u_l, L)

    def _evaluate_sections_along_length(self, u_l: np.ndarray, L: float) -> None:
        """Per-IP section response, used by ``recover()``. Uses
        ``self.sections[i]`` so that for stateful (fiber) sections each
        IP's independent state is correctly read out."""
        self._ensure_sections_length()
        xi_pts, _ = gauss_lobatto_1d(self.n_int)
        n_r = self.section.n_resultants
        self.section_locations = np.empty(self.n_int)
        self.section_strains = np.empty((self.n_int, n_r))
        self.section_forces = np.empty((self.n_int, n_r))
        for i, xi in enumerate(xi_pts):
            B = self._strain_disp_matrix(xi, L)
            e_i = B @ u_l
            s_i, _ = self.sections[i].get_response(e_i)
            self.section_locations[i] = 0.5 * L * (1.0 + xi)
            self.section_strains[i] = e_i
            self.section_forces[i] = s_i

    # NOTE: ``commit_state`` and ``revert_state`` are defined earlier
    # in the class (the Phase 5.5 stateful-section-aware versions).
    # The original Phase 3 versions that only forwarded to
    # ``self.section`` would have been overwritten by the later
    # definitions and led to silent loss of per-IP fiber-state commit.
