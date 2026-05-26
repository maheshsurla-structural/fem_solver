"""Response-spectrum analysis — modal superposition under a
user-supplied pseudo-acceleration spectrum.

A *response spectrum* gives the peak response of an SDOF oscillator
(of given period and damping) under a particular ground-motion record.
For a multi-DOF structure, the peak response in each mode is
estimated as

    u_i_peak = (phi_i^T M iota) / (phi_i^T M phi_i) * phi_i * Sa(T_i) / omega_i^2

where ``iota`` is the influence vector for the ground-motion direction.
The peak nodal responses are then combined across modes using either:

* **SRSS** (Square-Root-of-Sum-of-Squares) — appropriate when modal
  natural periods are well separated. ``u_max = sqrt(sum u_i^2)``.
* **CQC** (Complete Quadratic Combination, Der Kiureghian 1981) —
  the standard for closely-spaced modes. Uses a cross-correlation
  coefficient ``rho_ik`` that depends on the frequency ratio and
  damping ratio.

This is the workhorse design-level seismic analysis in every commercial
structural-FE code (SAP2000, ETABS, MIDAS, STAAD). The output is a
single peak-response state, not a time history — much cheaper than
direct integration and adequate for code-prescribed design checks
where time-history is not required.
"""
from __future__ import annotations

import math

import numpy as np

from femsolver.analysis.assembler import assemble_mass
from femsolver.analysis.eigen import EigenAnalysis


class ResponseSpectrum:
    """A pseudo-acceleration response spectrum.

    Parameters
    ----------
    periods : array-like
        Period values (s), must be monotonically increasing.
    accelerations : array-like
        Pseudo-spectral accelerations ``Sa`` at each period (units of
        m/s^2 typically, or g · acceleration_of_gravity).
    damping_ratio : float, default 0.05
        Damping ratio for which the spectrum is defined. The same
        damping is used in any subsequent modal combination.

    Notes
    -----
    The lookup uses linear interpolation between the supplied table
    points and clamps to the endpoint values outside the tabulated
    range. For typical design spectra (ASCE 7, EC 8, IS 1893) the
    spectrum can be discretised on a logarithmic period axis.
    """

    def __init__(self, periods, accelerations, *,
                 damping_ratio: float = 0.05):
        periods = np.asarray(periods, dtype=float).ravel()
        accel = np.asarray(accelerations, dtype=float).ravel()
        if periods.size != accel.size:
            raise ValueError(
                f"periods (size {periods.size}) and accelerations "
                f"(size {accel.size}) must have the same length"
            )
        if periods.size < 2:
            raise ValueError("need at least 2 points to define a spectrum")
        if np.any(np.diff(periods) <= 0.0):
            raise ValueError("periods must be strictly increasing")
        if damping_ratio < 0.0 or damping_ratio >= 1.0:
            raise ValueError("damping_ratio must be in [0, 1)")
        self.periods = periods
        self.Sa_values = accel
        self.damping_ratio = float(damping_ratio)

    def Sa(self, T: float) -> float:
        """Spectral acceleration at period ``T``.

        Beyond the tabulated range we clamp to the nearest endpoint —
        a deliberate choice for design analyses (the spectrum is
        assumed defined over the periods of interest; modes with
        periods well outside the range contribute negligibly).
        """
        return float(np.interp(T, self.periods, self.Sa_values))

    @classmethod
    def from_function(
        cls,
        Sa_func,
        *,
        T_min: float = 0.01,
        T_max: float = 10.0,
        n_points: int = 200,
        damping_ratio: float = 0.05,
    ) -> "ResponseSpectrum":
        """Build a tabulated spectrum from a callable ``Sa_func(T)``."""
        periods = np.logspace(math.log10(T_min), math.log10(T_max), n_points)
        accel = np.array([Sa_func(T) for T in periods])
        return cls(periods, accel, damping_ratio=damping_ratio)


def cqc_correlation_coefficient(
    omega_i: float, omega_j: float,
    zeta_i: float, zeta_j: float,
) -> float:
    """Cross-correlation coefficient for CQC modal combination
    (Der Kiureghian 1981).

    For ``omega_i == omega_j`` returns 1; for well-separated
    frequencies returns ~0. Symmetric in ``(i, j)`` swap.
    """
    if omega_i <= 0.0 or omega_j <= 0.0:
        raise ValueError("frequencies must be positive")
    r = omega_j / omega_i
    num = 8.0 * math.sqrt(zeta_i * zeta_j) * (zeta_i + r * zeta_j) * r ** 1.5
    den = (
        (1.0 - r ** 2) ** 2
        + 4.0 * zeta_i * zeta_j * r * (1.0 + r ** 2)
        + 4.0 * (zeta_i ** 2 + zeta_j ** 2) * r ** 2
    )
    return num / den


def _influence_vector(model, direction: str) -> np.ndarray:
    """Build the influence vector ``iota`` for the given ground-motion
    direction. ``iota[eq] = 1`` if the equation corresponds to a free
    DOF in the chosen direction at any node, else ``0``.
    """
    direction_to_index = {"x": 0, "y": 1, "z": 2}
    if direction not in direction_to_index:
        raise ValueError(f"unknown direction {direction!r} — use 'x', 'y', or 'z'")
    idx = direction_to_index[direction]
    iota = np.zeros(model.neq)
    for node in model.nodes.values():
        if idx < node.ndf:
            eq = int(node.eqn[idx])
            if eq >= 0:
                iota[eq] = 1.0
    return iota


class ResponseSpectrumAnalysis:
    """Modal-superposition seismic analysis.

    Parameters
    ----------
    model : Model
    spectrum : ResponseSpectrum
    num_modes : int, default 10
        Number of modes to extract and combine. The first ``num_modes``
        modes typically capture >90% of the modal mass for a typical
        building, which is what design codes require.
    direction : ``'x'``, ``'y'`` or ``'z'``, default ``'x'``
        Direction of the ground motion.
    combination : ``'srss'`` or ``'cqc'``, default ``'cqc'``
        Modal combination rule. CQC is the default — it reduces to
        SRSS for well-separated modes and is more accurate when modes
        are closely spaced.
    damping_ratio : float, optional
        Damping ratio to use for each mode. If ``None`` (default),
        uses the spectrum's damping ratio (recommended).
    """

    def __init__(
        self,
        model,
        spectrum: ResponseSpectrum,
        *,
        num_modes: int = 10,
        direction: str = "x",
        combination: str = "cqc",
        damping_ratio: float | None = None,
    ):
        if num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        if combination not in ("srss", "cqc"):
            raise ValueError(f"unknown combination {combination!r}")
        self.model = model
        self.spectrum = spectrum
        self.num_modes = int(num_modes)
        self.direction = direction
        self.combination = combination
        self.damping_ratio = (
            float(damping_ratio) if damping_ratio is not None
            else spectrum.damping_ratio
        )
        # Results, populated by run()
        self.modal_results: list[dict] = []
        self.peak_disp: np.ndarray | None = None

    # ------------------------------------------------------------ run
    def run(self) -> dict:
        m = self.model
        m.reset_results()
        m.number_dofs()
        if m.neq == 0:
            raise RuntimeError("no free DOFs — fully-constrained model")

        # --- Eigen analysis: periods, frequencies, mode shapes ---
        eig = EigenAnalysis(m, num_modes=self.num_modes).run()
        T_modes = np.array(eig["periods_s"])
        omega_modes = 2.0 * math.pi / T_modes

        # Build free-DOF mode shape matrix from Node.mode_disp
        Phi = np.zeros((m.neq, self.num_modes))
        for node in m.nodes.values():
            for i in range(node.ndf):
                eq = int(node.eqn[i])
                if eq >= 0:
                    Phi[eq, :] = node.mode_disp[i, :self.num_modes]

        # --- mass-normalize the eigenvectors and form Γ_i ---
        M = assemble_mass(m)
        M_dense = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
        iota = _influence_vector(m, self.direction)

        # phi_i^T M phi_i (modal mass)
        modal_mass = np.array([
            float(Phi[:, i] @ (M_dense @ Phi[:, i]))
            for i in range(self.num_modes)
        ])
        # Mass-normalize so Phi_norm^T M Phi_norm = I
        Phi_norm = Phi / np.sqrt(modal_mass)[None, :]
        # Participation factor: Γ_i = phi_norm_i^T M iota
        Gamma = np.array([
            float(Phi_norm[:, i] @ (M_dense @ iota))
            for i in range(self.num_modes)
        ])

        # --- per-mode peak modal response ---
        self.modal_results = []
        u_modes = np.zeros_like(Phi_norm)
        for i in range(self.num_modes):
            T_i = T_modes[i]
            omega_i = omega_modes[i]
            Sa_i = self.spectrum.Sa(T_i)
            # Peak displacement amplitude in mode i:
            # u_i = Γ_i · phi_norm_i · Sa_i / omega_i^2
            u_i = Gamma[i] * Phi_norm[:, i] * Sa_i / (omega_i ** 2)
            u_modes[:, i] = u_i
            # Effective modal mass (fraction of total participating mass)
            m_eff = Gamma[i] ** 2
            self.modal_results.append(dict(
                mode=i + 1,
                period=float(T_i),
                omega=float(omega_i),
                Sa=float(Sa_i),
                Gamma=float(Gamma[i]),
                modal_mass_eff=float(m_eff),
                u_max_dof=float(np.max(np.abs(u_i))),
            ))

        # --- modal combination ---
        if self.combination == "srss":
            # u_max[j] = sqrt(sum_i u_modes[j, i]^2)
            self.peak_disp = np.sqrt(np.sum(u_modes ** 2, axis=1))
        else:  # cqc
            rho = np.empty((self.num_modes, self.num_modes))
            zeta = self.damping_ratio
            for i in range(self.num_modes):
                for j in range(self.num_modes):
                    rho[i, j] = cqc_correlation_coefficient(
                        omega_modes[i], omega_modes[j], zeta, zeta,
                    )
            # u_max[j]^2 = sum_{i,k} rho[i,k] * u[j,i] * u[j,k]
            # This is a sum over a 3-D tensor product; we use the
            # quadratic form per DOF.
            self.peak_disp = np.sqrt(np.einsum(
                "ji,ik,jk->j", u_modes, rho, u_modes,
            ))

        # --- scatter peak displacements to Node.disp for inspection ---
        for node in m.nodes.values():
            for i in range(node.ndf):
                eq = int(node.eqn[i])
                if eq >= 0:
                    node.disp[i] = self.peak_disp[eq]

        # --- total participating modal mass ---
        total_part = float(np.sum([
            r["modal_mass_eff"] for r in self.modal_results
        ]))
        return {
            "neq": int(m.neq),
            "num_modes": int(self.num_modes),
            "combination": self.combination,
            "direction": self.direction,
            "damping_ratio": self.damping_ratio,
            "modal_results": list(self.modal_results),
            "total_participating_mass": total_part,
        }


# ============================================================ ground motion

def ground_motion_force(
    model,
    direction: str = "x",
    accel_function=None,
):
    """Construct a ``load_function`` suitable for ``TransientAnalysis``
    representing base excitation in the given direction.

    Returns a callable ``F(t)`` of size ``model.neq`` such that

        F(t) = -M iota * accel_function(t)

    where ``iota`` is the influence vector for the chosen direction.
    This is the standard formulation for direct integration under
    rigid-base ground acceleration: the inertia force ``-m_i ü_g(t)``
    is applied at every mass DOF along the ground-motion direction.

    Parameters
    ----------
    model : Model
        Model with mass-bearing elements. ``model.number_dofs()`` will
        be called if it hasn't been already.
    direction : {'x', 'y', 'z'}, default 'x'
    accel_function : callable
        ``accel_function(t) -> float`` returning the ground
        acceleration (m/s^2) at time ``t``.

    Examples
    --------
    >>> g_load = ground_motion_force(model, 'x', accel_function=ricker_pulse)
    >>> TransientAnalysis(model, num_steps=500, dt=0.01,
    ...                    load_function=g_load).run()
    """
    if accel_function is None:
        raise ValueError("accel_function must be supplied")
    # We defer M-iota assembly to the first call so that the model has
    # been DOF-numbered by the analysis driver.
    cache: dict = {}

    def _force(t: float) -> np.ndarray:
        if "Mr" not in cache:
            if model.neq == 0:
                model.number_dofs()
            M = assemble_mass(model)
            iota = _influence_vector(model, direction)
            cache["Mr"] = np.asarray(M @ iota).ravel()
        return -cache["Mr"] * float(accel_function(t))

    return _force


# ============================================================ multi-support

def _influence_vector_for_nodes(model, direction: str,
                                 node_tags) -> np.ndarray:
    """Influence vector restricted to a given subset of nodes (instead
    of the entire model). Free DOFs at the listed nodes in the chosen
    direction get a 1; everything else gets 0. Use this to express
    per-support motion for multi-support excitation."""
    direction_to_index = {"x": 0, "y": 1, "z": 2}
    if direction not in direction_to_index:
        raise ValueError(
            f"unknown direction {direction!r} -- use 'x', 'y', or 'z'"
        )
    idx = direction_to_index[direction]
    iota = np.zeros(model.neq)
    node_set = set(int(t) for t in node_tags)
    for node in model.nodes.values():
        if node.tag not in node_set:
            continue
        if idx < node.ndf:
            eq = int(node.eqn[idx])
            if eq >= 0:
                iota[eq] = 1.0
    return iota


def multi_support_ground_motion_force(model, supports):
    """Build a ``load_function`` for transient analysis where each
    support (or group of nodes) follows a *different* ground-motion
    history.

    Parameters
    ----------
    model : Model
    supports : sequence of dicts
        Each item describes one support group::

            {
                "direction": "x",       # 'x' | 'y' | 'z'
                "accel_function": fn,   # callable t -> u_ddot_g(t)
                "nodes": [tag1, tag2, ...] | None
            }

        If ``"nodes"`` is ``None`` or absent, the entire model is
        considered to move with this support (equivalent to the
        single-support :func:`ground_motion_force`). Otherwise only
        the listed free DOFs in the chosen direction follow this
        support's motion.

    Returns
    -------
    callable
        ``F(t) -> ndarray`` of length ``model.neq`` giving
        ``-sum_j M iota_j * a_g_j(t)``.

    Examples
    --------
    Two bridge piers, each with its own time-history file::

        supports = [
            {"direction": "x", "accel_function": pier_west_ag,
             "nodes": [1, 2, 3]},
            {"direction": "x", "accel_function": pier_east_ag,
             "nodes": [7, 8, 9]},
        ]
        load = multi_support_ground_motion_force(model, supports)
        TransientAnalysis(model, num_steps=N, dt=dt,
                            load_function=load).run()

    Notes
    -----
    This implementation uses the simplest "kinematic influence vector"
    approach: each support's influence vector marks the free DOFs that
    are taken to *follow rigidly* in the relevant direction. For
    spatially-varying inputs without rigid-body kinematics (long
    bridges with travelling waves) a more elaborate static-condensation
    construction of the per-support influence vectors is needed -- a
    future refinement.
    """
    if not supports:
        raise ValueError("multi_support_ground_motion_force needs >= 1 support")
    specs = []
    for spec in supports:
        if "accel_function" not in spec or spec["accel_function"] is None:
            raise ValueError("each support spec must provide 'accel_function'")
        specs.append({
            "direction": spec.get("direction", "x"),
            "accel_function": spec["accel_function"],
            "nodes": spec.get("nodes"),
        })
    cache: dict = {}

    def _force(t: float) -> np.ndarray:
        if "Mr_list" not in cache:
            if model.neq == 0:
                model.number_dofs()
            M = assemble_mass(model)
            mr_list = []
            for s in specs:
                if s["nodes"] is None:
                    iota = _influence_vector(model, s["direction"])
                else:
                    iota = _influence_vector_for_nodes(
                        model, s["direction"], s["nodes"]
                    )
                mr_list.append(np.asarray(M @ iota).ravel())
            cache["Mr_list"] = mr_list
        F = np.zeros(model.neq)
        for mr, s in zip(cache["Mr_list"], specs):
            F -= mr * float(s["accel_function"](t))
        return F

    return _force
