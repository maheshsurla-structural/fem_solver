"""Modal Pushover Analysis (Chopra & Goel, 2002).

Standard single-mode nonlinear-static pushover assumes the structure
responds in one (usually fundamental) mode shape. For tall or
plan-irregular buildings this misses higher-mode contributions that
dominate upper-story EDPs (e.g. interstory drift in stories above the
half-height, beam moments in the upper third). Modal Pushover Analysis
(MPA) addresses the gap by performing *N* independent pushovers --
one per significant mode -- each with an invariant force pattern

    s_n = M phi_n

derived from the mass matrix and the n-th mode shape. The MDOF
capacity curve from each modal pushover is converted to an equivalent
SDOF via the modal participation factor Gamma_n, bilinearized, and
its target displacement determined against the design spectrum
(either by the N2 method or the ASCE 41 coefficient method). The
modal nodal responses at each target are then combined by SRSS or
CQC.

Theoretical foundation
----------------------
For an undamped MDOF system the inertial force in mode n under
ground excitation u_g_ddot is

    f_n(t) = - s_n * Gamma_n * A_n(t)

where ``A_n(t)`` is the n-th-mode pseudo-acceleration. The
peak response in mode n is therefore obtained by applying the
invariant pattern ``s_n = M phi_n`` and pushing to the peak modal
displacement, which corresponds to the equivalent SDOF target
displacement scaled by Gamma_n.

Reference: Chopra & Goel, "A modal pushover analysis procedure for
estimating seismic demands for buildings," Earthquake Engineering &
Structural Dynamics, 31(3), 2002, pp. 561-582.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from femsolver.analysis.assembler import assemble_mass
from femsolver.analysis.capacity_design import (
    BilinearCurve,
    EquivalentSDOF,
    bilinearize_capacity_curve,
    coefficient_method_target,
    equivalent_sdof,
    n2_target_displacement,
)
from femsolver.analysis.eigen import EigenAnalysis
from femsolver.analysis.integrator import DisplacementControl
from femsolver.analysis.nonlinear_static import NonlinearStaticAnalysis
from femsolver.analysis.response_spectrum import cqc_correlation_coefficient


# ============================================================ helpers

_DIRECTION_INDEX = {"x": 0, "y": 1, "z": 2}


def _influence_vector(model, direction_index: int) -> np.ndarray:
    """Build the influence vector ``iota`` (size ``neq``): a 1 at each
    free DOF whose nodal DOF index equals ``direction_index``, else 0.
    Matches the convention used by ResponseSpectrumAnalysis.
    """
    iota = np.zeros(model.neq)
    for node in model.nodes.values():
        if direction_index < node.ndf:
            eq = int(node.eqn[direction_index])
            if eq >= 0:
                iota[eq] = 1.0
    return iota


def _scatter_mode_shape(model, phi_free: np.ndarray) -> dict[int, np.ndarray]:
    """Map a free-DOF mode-shape column onto per-node displacement
    arrays."""
    nodal = {}
    for node in model.nodes.values():
        v = np.zeros(node.ndf, dtype=float)
        for i in range(node.ndf):
            eq = int(node.eqn[i])
            if eq >= 0:
                v[i] = phi_free[eq]
        nodal[node.tag] = v
    return nodal


def _interp_disp_snapshot(
    drifts: np.ndarray,
    snapshots: list[dict[int, np.ndarray]],
    d_target: float,
) -> dict[int, np.ndarray]:
    """Linearly interpolate nodal displacement snapshots at the target
    drift. Snapshots are committed-state nodal disps captured after
    each pushover step; drifts is the control-DOF reading at each step.
    Clamps to the last snapshot if d_target exceeds the pushover range.
    """
    abs_drifts = np.abs(drifts)
    abs_target = abs(d_target)
    if abs_target <= abs_drifts[0]:
        return {tag: arr.copy() for tag, arr in snapshots[0].items()}
    if abs_target >= abs_drifts[-1]:
        return {tag: arr.copy() for tag, arr in snapshots[-1].items()}
    idx = int(np.searchsorted(abs_drifts, abs_target))
    # idx is the first step with abs_drift >= abs_target
    if idx <= 0:
        return {tag: arr.copy() for tag, arr in snapshots[0].items()}
    d1 = abs_drifts[idx - 1]
    d2 = abs_drifts[idx]
    if d2 == d1:
        return {tag: arr.copy() for tag, arr in snapshots[idx].items()}
    t = (abs_target - d1) / (d2 - d1)
    interp = {}
    for tag, arr2 in snapshots[idx].items():
        arr1 = snapshots[idx - 1][tag]
        interp[tag] = arr1 + t * (arr2 - arr1)
    return interp


# ============================================================ driver

class ModalPushoverAnalysis:
    """Chopra-Goel Modal Pushover Analysis.

    Parameters
    ----------
    model_factory : Callable[[], Model]
        Zero-argument callable that returns a *fresh* Model with mesh,
        materials, sections, supports, and mass distribution already
        defined but **no** nodal loads. The factory is called once per
        modal pushover so each mode's analysis starts from undeformed
        state -- this is the core MPA assumption (modal responses are
        independent and superposed only at the post-processing stage).
    spectrum : ResponseSpectrum
        Elastic design spectrum (pseudo-acceleration Sa(T)).
    control_node : int
        Node tag at which to track drift (typically the roof).
    control_dof : int
        DOF index at the control node (0 = x, 1 = y, ...).
    direction : ``"x"`` | ``"y"`` | ``"z"`` (default ``"x"``)
        Ground motion direction. Used to (a) build the influence vector
        iota for modal participation factors, and (b) restrict the
        modal force pattern to the lateral direction.
    max_drift : float
        The pushover sweeps the control DOF from 0 to this displacement
        magnitude in each mode. Should be set comfortably past the
        expected target (e.g. 2-3x the estimated target).
    num_modes : int, default 3
        Number of modes to retain. 3 is typical for buildings up to
        ~15 stories; taller / irregular structures may need 4-6.
    num_steps : int, default 100
        Pushover steps from 0 to max_drift, per mode.
    combination : ``"srss"`` | ``"cqc"`` (default ``"srss"``)
        Modal-response combination rule. SRSS is the Chopra-Goel
        recommendation; CQC accounts for closely-spaced modes using
        damping-based correlation coefficients.
    target_method : ``"n2"`` | ``"coefficient"`` (default ``"n2"``)
        How to obtain each mode's target displacement from the SDOF
        capacity curve.
    Tc : float, default 0.5
        N2-method corner period (only used if ``target_method="n2"``).
    damping_ratio : float, default 0.05
        Modal damping ratio (used for CQC and for spectrum lookup if
        the spectrum has a different damping).
    lateral_only_pattern : bool, default True
        If True the modal force pattern is restricted to nodal DOFs
        in ``direction`` (the standard MPA pattern: only lateral
        inertia forces are applied). If False, the full ``M phi_n``
        vector is applied at every DOF (rotational + vertical
        components included).
    algorithm : str, default ``"newton"``
    convergence : str, default ``"unbalance"``
    tol : float, default 1e-6
    max_iter : int, default 25
    secant_ratio : float, default 0.6
        Secant-stiffness fraction for bilinearization.

    Notes
    -----
    The implementation uses mass-normalized mode shapes
    (``phi_n^T M phi_n = 1``) throughout, so ``Gamma_n = phi_n^T M iota``
    and ``m_eff_n = Gamma_n^2`` directly.
    """

    def __init__(
        self,
        model_factory: Callable,
        spectrum,
        *,
        control_node: int,
        control_dof: int,
        max_drift: float,
        direction: str = "x",
        num_modes: int = 3,
        num_steps: int = 100,
        combination: str = "srss",
        target_method: str = "n2",
        Tc: float = 0.5,
        damping_ratio: float = 0.05,
        lateral_only_pattern: bool = True,
        algorithm: str = "newton",
        convergence: str = "unbalance",
        tol: float = 1.0e-6,
        max_iter: int = 25,
        secant_ratio: float = 0.6,
    ):
        if not callable(model_factory):
            raise TypeError("model_factory must be a zero-argument callable "
                              "returning a fresh Model")
        if num_modes < 1:
            raise ValueError(f"num_modes must be >= 1, got {num_modes}")
        if num_steps < 2:
            raise ValueError(f"num_steps must be >= 2, got {num_steps}")
        if combination not in ("srss", "cqc"):
            raise ValueError(
                f"unknown combination {combination!r}; expected 'srss' or 'cqc'"
            )
        if target_method not in ("n2", "coefficient"):
            raise ValueError(
                f"unknown target_method {target_method!r}; expected 'n2' or "
                "'coefficient'"
            )
        if direction not in _DIRECTION_INDEX:
            raise ValueError(
                f"unknown direction {direction!r}; expected 'x', 'y', or 'z'"
            )
        if max_drift == 0.0:
            raise ValueError("max_drift must be nonzero")
        if not (0.0 <= damping_ratio < 1.0):
            raise ValueError(
                f"damping_ratio must be in [0, 1), got {damping_ratio}"
            )

        self.model_factory = model_factory
        self.spectrum = spectrum
        self.control_node = int(control_node)
        self.control_dof = int(control_dof)
        self.max_drift = float(max_drift)
        self.direction = direction
        self.num_modes = int(num_modes)
        self.num_steps = int(num_steps)
        self.combination = combination
        self.target_method = target_method
        self.Tc = float(Tc)
        self.damping_ratio = float(damping_ratio)
        self.lateral_only_pattern = bool(lateral_only_pattern)
        self.algorithm = algorithm
        self.convergence = convergence
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.secant_ratio = float(secant_ratio)

        # populated after run()
        self.modal_results: list[dict] = []
        self.combined_nodal_disps: dict[int, np.ndarray] = {}

    # =============================================== run
    def run(self) -> dict:
        # --- 1. Eigen analysis on a fresh model ---
        eig_model = self.model_factory()
        eig = EigenAnalysis(eig_model, num_modes=self.num_modes)
        eig.run()

        # Mass matrix in the eigen model's DOF numbering
        M_eig = assemble_mass(eig_model)
        M_dense = M_eig.toarray() if hasattr(M_eig, "toarray") else np.asarray(M_eig)
        iota_eig = _influence_vector(eig_model, _DIRECTION_INDEX[self.direction])

        # Mass-normalize the eigenvectors so phi_n^T M phi_n = 1
        Phi = np.asarray(eig.mode_shapes)         # (neq, num_modes)
        modal_mass = np.array([
            float(Phi[:, i] @ (M_dense @ Phi[:, i]))
            for i in range(self.num_modes)
        ])
        if np.any(modal_mass <= 0.0):
            raise RuntimeError(
                "non-positive modal mass detected -- check that elements "
                "have positive density"
            )
        Phi_n = Phi / np.sqrt(modal_mass)[None, :]  # mass-normalized

        # Participation factors and effective modal masses
        Gamma = np.array([
            float(Phi_n[:, i] @ (M_dense @ iota_eig))
            for i in range(self.num_modes)
        ])
        m_eff = Gamma * Gamma     # since phi_n^T M phi_n = 1
        periods = np.asarray(eig.periods)
        omegas = np.where(periods > 0, 2.0 * math.pi / periods, np.inf)

        # Map the mass-normalized mode shapes (in the eigen model's DOF
        # numbering) onto per-node arrays, keyed by node tag. This works
        # *if* the model_factory produces models with the same node tags
        # and DOF ordering each time (which is the standard pattern).
        modal_shapes_nodal = [
            _scatter_mode_shape(eig_model, Phi_n[:, i])
            for i in range(self.num_modes)
        ]
        # If the mode shape has negative control-DOF entry, flip it so
        # the pushover proceeds in the +control_dof direction. We also
        # flip Gamma for consistency.
        for i in range(self.num_modes):
            ctrl_val = float(modal_shapes_nodal[i][self.control_node][self.control_dof])
            if ctrl_val < 0.0:
                for tag in modal_shapes_nodal[i]:
                    modal_shapes_nodal[i][tag] *= -1.0
                Gamma[i] = -Gamma[i]
                Phi_n[:, i] = -Phi_n[:, i]

        # ``Gamma_conv[n]`` is the dimensionless participation factor used
        # by the existing N2-method machinery -- it equals the mass-norm
        # Gamma multiplied by the mode-shape entry at the control DOF (a
        # rescaling equivalent to choosing the conventional "phi[top] = 1"
        # normalization). This makes ``d_star = u_top / Gamma_conv`` the
        # actual SDOF displacement, in length units. ``m_eff`` is
        # scale-invariant.
        phi_rn = np.array([
            float(modal_shapes_nodal[i][self.control_node][self.control_dof])
            for i in range(self.num_modes)
        ])
        Gamma_conv = Gamma * phi_rn

        # --- 2. Per-mode pushover ---
        self.modal_results = []
        for n in range(self.num_modes):
            res = self._run_one_mode(
                mode_index=n,
                phi_n_nodal=modal_shapes_nodal[n],
                Gamma_n=float(Gamma[n]),
                Gamma_conv_n=float(Gamma_conv[n]),
                m_eff_n=float(m_eff[n]),
                T_n=float(periods[n]),
                omega_n=float(omegas[n]),
            )
            self.modal_results.append(res)

        # --- 3. Modal combination ---
        self.combined_nodal_disps = self._combine_modal_responses(omegas)

        # Scatter combined displacements onto the eigen model's nodes
        # for inspection convenience.
        for tag, vec in self.combined_nodal_disps.items():
            node = eig_model.node(tag)
            node.disp[:vec.size] = vec

        # --- 4. Summary ---
        return {
            "num_modes": self.num_modes,
            "direction": self.direction,
            "combination": self.combination,
            "target_method": self.target_method,
            "periods_s": [float(p) for p in periods],
            "Gamma": Gamma.tolist(),
            "m_eff": m_eff.tolist(),
            "total_participating_mass": float(np.sum(m_eff)),
            "modal_results": self.modal_results,
        }

    # =============================================== single mode
    def _run_one_mode(
        self,
        *,
        mode_index: int,
        phi_n_nodal: dict[int, np.ndarray],
        Gamma_n: float,
        Gamma_conv_n: float,
        m_eff_n: float,
        T_n: float,
        omega_n: float,
    ) -> dict:
        """Build a fresh model, apply the modal force pattern, sweep
        the control DOF, then post-process to a modal target displacement
        and capture nodal disps at that target."""
        m = self.model_factory()

        # Build s_n = M @ phi_n in the fresh model's DOF numbering.
        # We re-number DOFs and build the influence + M @ phi vector.
        m.number_dofs()
        if m.neq == 0:
            raise RuntimeError("model has no free DOFs -- check supports")

        # Reconstruct the free-DOF phi vector for this model from the
        # per-node mapping.
        phi_free = np.zeros(m.neq)
        for node in m.nodes.values():
            v = phi_n_nodal.get(node.tag)
            if v is None:
                continue
            for i in range(node.ndf):
                eq = int(node.eqn[i])
                if eq >= 0 and i < v.size:
                    phi_free[eq] = float(v[i])

        M_sp = assemble_mass(m)
        M_dense = M_sp.toarray() if hasattr(M_sp, "toarray") else np.asarray(M_sp)
        s_free = M_dense @ phi_free     # size (neq,)

        # Scatter s_free back onto nodes as nodal loads. If
        # lateral_only_pattern is True, zero out everything except the
        # direction-of-interest DOF -- this is the standard MPA pattern.
        direction_index = _DIRECTION_INDEX[self.direction]
        for node in m.nodes.values():
            for i in range(node.ndf):
                if self.lateral_only_pattern and i != direction_index:
                    continue
                eq = int(node.eqn[i])
                if eq >= 0:
                    node._load[i] = float(s_free[eq])

        # Verify the control DOF will move in the +direction. The applied
        # force at the control DOF should be positive (we already flipped
        # phi_n's sign to make this so, but verify -- a zero control-DOF
        # mass would make this break).
        ctrl_load = float(m.node(self.control_node)._load[self.control_dof])
        if ctrl_load == 0.0:
            raise RuntimeError(
                f"mode {mode_index + 1}: applied load at control DOF "
                f"({self.control_node}, {self.control_dof}) is zero -- this "
                "mode has no inertial force at the control DOF (check that "
                "the control node has mass in the chosen direction)"
            )

        # --- displacement-controlled pushover ---
        du_step = self.max_drift / self.num_steps
        integrator = DisplacementControl(
            node_tag=self.control_node,
            dof_index=self.control_dof,
            du_step=du_step,
        )
        analysis = NonlinearStaticAnalysis(
            m, num_steps=self.num_steps,
            integrator=integrator,
            algorithm=self.algorithm,
            convergence=self.convergence,
            tol=self.tol, max_iter=self.max_iter,
            track=(self.control_node, self.control_dof),
        )

        # Capture nodal-displacement snapshots after each step by
        # monkey-patching the analysis loop. The cleanest implementation
        # would have NonlinearStaticAnalysis take a per-step callback;
        # for now we re-implement the step loop here in place.
        snapshots = self._run_with_snapshots(analysis, m)

        drifts = np.asarray(snapshots["drifts"], dtype=float)
        applied_forces = np.asarray(snapshots["lambdas"], dtype=float) * ctrl_load
        # Base shear in MPA is the total applied force in `direction`:
        #     V_b = sum over nodes of f_n[direction] * lambda
        # That sum equals s_free's `direction` components dot 1, scaled
        # by lambda. Equivalent to lambda * (sum of all applied loads in
        # `direction`).
        F_ref_sum = 0.0
        for node in m.nodes.values():
            F_ref_sum += float(node._load[direction_index])
        base_shear = np.asarray(snapshots["lambdas"], dtype=float) * F_ref_sum

        # --- equivalent SDOF, bilinearization, target ---
        # Use absolute values since pushover may go in negative direction.
        d_top = drifts
        V_base = base_shear
        # Sort by drift sign so bilinearize sees monotonic ascending
        sign = 1.0 if d_top[-1] >= 0.0 else -1.0
        d_top_abs = np.abs(d_top)
        V_base_abs = np.abs(V_base)

        sdof = equivalent_sdof(
            d_top_abs, V_base_abs,
            Gamma=abs(Gamma_conv_n), m_eff=m_eff_n,
        )
        bilinear = bilinearize_capacity_curve(
            sdof.d_star, sdof.F_star, secant_ratio=self.secant_ratio,
        )
        if self.target_method == "n2":
            target_info = n2_target_displacement(
                self.spectrum, sdof, bilinear, Tc=self.Tc,
            )
            d_t_top = target_info["d_t_top"]
        else:
            target_info = coefficient_method_target(
                self.spectrum, T_eff=sdof.T_eff,
            )
            d_t_top = target_info["d_t_top"]

        # Interpolate nodal-disp snapshot at the target drift (in the
        # +direction)
        d_t_top_signed = sign * d_t_top
        nodal_at_target = _interp_disp_snapshot(
            drifts, snapshots["nodal"], d_t_top_signed,
        )

        return {
            "mode": mode_index + 1,
            "period": T_n,
            "omega": omega_n,
            "Gamma": Gamma_n,           # mass-normalized convention
            "Gamma_conv": Gamma_conv_n, # phi[control]=1 convention (N2)
            "m_eff": m_eff_n,
            "drift_curve": d_top,
            "force_curve": V_base,
            "sdof_d_star": sdof.d_star,
            "sdof_F_star": sdof.F_star,
            "T_eff": sdof.T_eff,
            "bilinear": bilinear,
            "target_info": target_info,
            "d_t_top": float(d_t_top_signed),
            "nodal_disps_at_target": nodal_at_target,
        }

    def _run_with_snapshots(self, analysis: NonlinearStaticAnalysis,
                              m) -> dict:
        """Run the pushover step-by-step, capturing nodal disp
        snapshots after each committed step. Replicates the body of
        NonlinearStaticAnalysis.run() to inject snapshot capture.
        """
        from femsolver.analysis.algorithm import NotConvergedError

        m.reset_results()
        m.number_dofs()
        if m.neq == 0:
            raise RuntimeError(
                "no free DOFs -- model is fully constrained or empty"
            )
        analysis.integrator.bind(m)

        def scatter_du(du: np.ndarray) -> None:
            for node in m.nodes.values():
                for i in range(node.ndf):
                    eq = int(node.eqn[i])
                    if eq >= 0:
                        node.disp[i] += du[eq]

        drifts: list[float] = []
        lambdas: list[float] = []
        nodal_snapshots: list[dict[int, np.ndarray]] = []

        for step in range(1, analysis.num_steps + 1):
            analysis.integrator.new_step()
            try:
                analysis.algorithm.solve_step(
                    analysis.integrator, analysis.convergence,
                    scatter_du=scatter_du,
                )
            except NotConvergedError:
                analysis.integrator.revert_step()
                for e in m.elements.values():
                    e.revert_state()
                # Truncate: keep what converged
                break

            for e in m.elements.values():
                e.commit_state()
            analysis.integrator.commit_step()

            drifts.append(float(m.node(self.control_node).disp[self.control_dof]))
            lambdas.append(float(analysis.integrator.lambd))
            # snapshot every node's disp
            snap = {}
            for node in m.nodes.values():
                snap[node.tag] = node.disp.copy()
            nodal_snapshots.append(snap)

        if not drifts:
            raise RuntimeError(
                "modal pushover failed at the first step -- check tolerances "
                "or supports"
            )
        return {
            "drifts": drifts,
            "lambdas": lambdas,
            "nodal": nodal_snapshots,
        }

    # =============================================== modal combination
    def _combine_modal_responses(
        self,
        omegas: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Combine per-mode nodal displacements at each mode's target
        into a single peak-response field per node.

        SRSS: u_peak[node, dof] = sqrt(sum_n u_n[node, dof]^2)
        CQC:  u_peak[node, dof]^2 = sum_{n, k} rho_nk * u_n * u_k
        """
        # Gather all node tags
        node_tags = set()
        for r in self.modal_results:
            node_tags.update(r["nodal_disps_at_target"].keys())
        # Determine ndf per node (use the eigen model)
        # Each result's nodal_disps_at_target maps tag -> ndarray(ndf,)
        ndf_by_tag: dict[int, int] = {}
        for r in self.modal_results:
            for tag, arr in r["nodal_disps_at_target"].items():
                if tag not in ndf_by_tag:
                    ndf_by_tag[tag] = int(arr.size)

        # Build per-mode arrays of shape (sum_ndf,) across all (node, dof) pairs
        # We process node by node to keep it readable.
        combined: dict[int, np.ndarray] = {}
        if self.combination == "srss":
            for tag in node_tags:
                ndf = ndf_by_tag[tag]
                modal_vecs = np.zeros((ndf, len(self.modal_results)))
                for i, r in enumerate(self.modal_results):
                    v = r["nodal_disps_at_target"].get(tag)
                    if v is None:
                        continue
                    modal_vecs[:v.size, i] = v
                combined[tag] = np.sqrt(np.sum(modal_vecs ** 2, axis=1))
        else:    # cqc
            N = len(self.modal_results)
            rho = np.empty((N, N))
            zeta = self.damping_ratio
            for i in range(N):
                for j in range(N):
                    if omegas[i] > 0 and omegas[j] > 0:
                        rho[i, j] = cqc_correlation_coefficient(
                            float(omegas[i]), float(omegas[j]), zeta, zeta,
                        )
                    else:
                        rho[i, j] = 1.0 if i == j else 0.0
            for tag in node_tags:
                ndf = ndf_by_tag[tag]
                modal_vecs = np.zeros((ndf, N))
                for i, r in enumerate(self.modal_results):
                    v = r["nodal_disps_at_target"].get(tag)
                    if v is None:
                        continue
                    modal_vecs[:v.size, i] = v
                # u_peak[j]^2 = sum_{i,k} rho[i,k] * modal_vecs[j, i] * modal_vecs[j, k]
                # Use signed values inside CQC; only the SQRT at the end is
                # absolute. (Standard CQC for RSA.)
                quad = np.einsum(
                    "ji,ik,jk->j", modal_vecs, rho, modal_vecs,
                )
                # quad should be non-negative; clamp small negatives
                quad = np.where(quad < 0.0, 0.0, quad)
                combined[tag] = np.sqrt(quad)
        return combined
