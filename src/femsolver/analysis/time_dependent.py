"""Structural effects of concrete creep and shrinkage.

Creep and shrinkage are not just material properties -- in a real
structure they produce **forces** (where deformation is restrained)
and **deformations** (where it is free):

* **Creep** multiplies the elastic deflection under sustained load and,
  in indeterminate / composite structures, redistributes internal
  forces. The Age-Adjusted Effective Modulus (AAEM, Trost-Bazant) is
  the standard one-step tool::

      E_eff(t, t0) = E_c / (1 + chi · phi(t, t0))     (chi ~ 0.8)

* **Shrinkage** ``ε_cs`` is a stress-free imposed strain. In an
  unrestrained member it just shortens it; in a restrained or
  statically-indeterminate member it induces a tensile restraint force
  (and, when it differs through the depth, curvature and self-
  equilibrated stresses).

This module provides the member-level closed forms plus a turnkey
``apply_shrinkage_load`` that imposes a uniform shrinkage strain on a
finite-element model as an eigenstrain (reusing the thermal-strain /
initial-stress machinery), giving the restraint forces and
deformations directly.

The creep coefficient ``phi`` and shrinkage strain ``ε_cs`` come from
:mod:`femsolver.bridges.creep_shrinkage`; the age-dependent moduli from
:mod:`femsolver.materials.concrete.time_dependent`. Staged-construction creep
redistribution is handled by
:class:`femsolver.bridges.IncrementalStagedAnalysis` (per-stage EMM
stiffness factor).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


# ============================================================ AAEM / creep

def age_adjusted_modulus(*, E_c: float, phi: float, chi: float = 0.8) -> float:
    """Age-Adjusted Effective Modulus ``E_eff = E_c / (1 + chi·phi)``
    (Trost 1967 / Bazant 1972). ``chi`` is the ageing coefficient
    (~0.8 for long-term sustained load; use 1.0 to recover the simple
    Effective Modulus Method)."""
    if E_c <= 0:
        raise ValueError("E_c must be > 0")
    if phi < 0:
        raise ValueError("phi must be >= 0")
    if not (0.0 < chi <= 1.0):
        raise ValueError("chi must be in (0, 1]")
    return float(E_c / (1.0 + chi * phi))


def creep_deflection(*, instantaneous: float, phi: float) -> float:
    """Long-term deflection under sustained load including creep:
    ``δ_∞ = δ_inst · (1 + phi)``.

    (Equivalent to scaling the elastic stiffness by the effective
    modulus ``E_c/(1+phi)`` for a load sustained from ``t0``.)
    """
    if phi < 0:
        raise ValueError("phi must be >= 0")
    return float(instantaneous * (1.0 + phi))


# ============================================================ shrinkage (member)

def shrinkage_axial_force(*, E_eff: float, A: float, eps_sh: float) -> float:
    """Axial restraint force in a **fully restrained** member due to a
    free shrinkage strain ``ε_cs`` (negative).

        N = -E_eff · A · ε_cs        (tension positive)

    A shortening shrinkage (``ε_cs < 0``) held by the supports develops
    a *tensile* restraint force. ``E_eff`` should be the age-adjusted
    effective modulus (creep relaxes the restraint force over time).
    """
    if E_eff <= 0 or A <= 0:
        raise ValueError("E_eff and A must be > 0")
    return float(-E_eff * A * eps_sh)


def differential_shrinkage_curvature(
    *, eps_top: float, eps_bot: float, h: float,
) -> float:
    """Curvature induced by a linear shrinkage gradient across depth
    ``h``: ``κ = (ε_bot - ε_top) / h``.

    For a composite deck-on-girder where the deck shrinks more than the
    girder, the differential drives a hogging/sagging curvature (and,
    if restrained, self-equilibrated stresses + secondary moments).
    """
    if h <= 0:
        raise ValueError("h must be > 0")
    return float((eps_bot - eps_top) / h)


# ============================================================ structural (FE) shrinkage

def apply_shrinkage_load(model, *, eps_sh: float, alpha: float = 1.0) -> int:
    """Impose a uniform shrinkage strain on a finite-element model as an
    eigenstrain, adding the equivalent nodal loads.

    Shrinkage is mathematically a thermal-type eigenstrain
    ``ε_0 = ε_sh`` (isotropic, stress-free), so this delegates to the
    thermal-strain equivalent-load machinery with ``α·ΔT = ε_sh``. On a
    restrained / indeterminate model the resulting solve gives the
    shrinkage **restraint forces and reactions**; on a free model it
    gives the **shortening deformation**.

    Parameters
    ----------
    model : Model
        A 2-D plane (``Quad4``) or 3-D solid (``Hex8``) model.
    eps_sh : float
        Free shrinkage strain (negative for shrinkage).
    alpha : float, default 1.0
        Scale (use 1.0 to pass ``eps_sh`` directly as the eigenstrain).

    Returns
    -------
    int : number of elements processed.
    """
    from femsolver.analysis.thermal_strain import apply_thermal_load
    temps = {tag: eps_sh for tag in model.nodes}
    return apply_thermal_load(model, temperatures=temps, T_ref=0.0,
                              alpha=alpha)


def beam_shrinkage_axial_force(*, E_eff: float, A: float, eps_sh: float) -> float:
    """Alias of :func:`shrinkage_axial_force` for a fully axially
    restrained beam segment (``N = -E_eff·A·ε_cs``)."""
    return shrinkage_axial_force(E_eff=E_eff, A=A, eps_sh=eps_sh)


# ============================================================ step-by-step creep

@dataclass
class CreepHistory:
    """Time history from a step-by-step creep / relaxation solve.

    Attributes
    ----------
    times : np.ndarray
        Ages (days) -- the time grid (``times[0]`` is the loading age).
    stress : np.ndarray
        Stress at each time (Pa).
    strain : np.ndarray
        Total (mechanical + shrinkage) strain at each time.
    eps_shrinkage : np.ndarray
        Free shrinkage strain at each time.
    mode : str
        ``"stress-controlled"`` or ``"relaxation"``.
    """

    times: np.ndarray
    stress: np.ndarray
    strain: np.ndarray
    eps_shrinkage: np.ndarray
    mode: str = ""


class StepByStepCreep:
    """Rigorous step-by-step integration of the uniaxial concrete creep
    superposition (Volterra) integral.

    The constitutive law of ageing linear viscoelasticity is

        ε(t) = ∫ J(t, t') dσ(t')  +  ε_sh(t)

    with the creep compliance ``J(t, t') = [1 + φ(t, t')] / E_c``. The
    integral is discretised on a time grid into stress increments
    ``Δσ_j`` applied at the mid-step ages ``τ_j``:

        ε(t_n) = Σ_{j≤n} Δσ_j · J(t_n, τ_j) + ε_sh(t_n)

    From this single relation:

    * **stress-controlled** -- given σ(t), the strain follows by direct
      summation (creep under a varying stress history);
    * **relaxation** -- given ε(t), the unknown stress increment each
      step is

        Δσ_n = [ε(t_n) - ε_sh(t_n) - Σ_{j<n} Δσ_j J(t_n,τ_j)] / J(t_n,τ_n)

      i.e. true stress relaxation (no EMM/AAEM ageing-coefficient
      approximation).

    This is the engine behind real step-by-step time-dependent analysis
    (creep deflection growth, relaxation of restraint forces, creep
    redistribution in indeterminate / composite / staged structures).

    Parameters
    ----------
    E_c : float
        Elastic modulus at the reference (28-day) age (Pa). The
        compliance uses ``J = (1+φ)/E_c``; supply a ``phi`` consistent
        with this convention.
    phi : Callable[[float, float], float]
        Creep coefficient ``φ(t, t0)`` (e.g. a wrapper around
        :func:`femsolver.bridges.cebfip_creep_coefficient`). Returns 0
        for ``t ≤ t0``.
    shrinkage : Callable[[float], float], optional
        Free shrinkage strain ``ε_sh(t)`` (negative). Defaults to none.
    """

    def __init__(
        self,
        *,
        E_c: float,
        phi: Callable[[float, float], float],
        shrinkage: Optional[Callable[[float], float]] = None,
    ):
        if E_c <= 0:
            raise ValueError("E_c must be > 0")
        self.E_c = float(E_c)
        self._phi = phi
        self._sh = shrinkage

    def phi(self, t: float, t0: float) -> float:
        if t <= t0:
            return 0.0
        return float(self._phi(t, t0))

    def compliance(self, t: float, t0: float) -> float:
        """Creep compliance ``J(t, t0) = (1 + φ(t, t0)) / E_c``."""
        return (1.0 + self.phi(t, t0)) / self.E_c

    def eps_sh(self, t: float) -> float:
        return float(self._sh(t)) if self._sh is not None else 0.0

    @staticmethod
    def _tau(times: np.ndarray) -> np.ndarray:
        """Effective age of each stress increment: the loading age for
        the first, mid-step ages thereafter."""
        tau = np.empty_like(times)
        tau[0] = times[0]
        tau[1:] = 0.5 * (times[:-1] + times[1:])
        return tau

    def strain_history(self, times, stress) -> CreepHistory:
        """Strain history under a prescribed **stress** history."""
        t = np.asarray(times, dtype=float).ravel()
        sig = np.asarray(stress, dtype=float).ravel()
        if t.size != sig.size or t.size < 1:
            raise ValueError("times and stress must be equal-length (>=1)")
        tau = self._tau(t)
        dsig = np.empty_like(sig)
        dsig[0] = sig[0]
        dsig[1:] = np.diff(sig)
        eps = np.zeros_like(t)
        sh = np.array([self.eps_sh(ti) for ti in t])
        for n in range(t.size):
            s = 0.0
            for j in range(n + 1):
                s += dsig[j] * self.compliance(t[n], tau[j])
            eps[n] = s + sh[n]
        return CreepHistory(times=t, stress=sig, strain=eps,
                            eps_shrinkage=sh, mode="stress-controlled")

    def relaxation_history(self, times, strain) -> CreepHistory:
        """Stress history under a prescribed **total strain** history
        (relaxation). ``strain`` is the imposed mechanical+shrinkage
        strain; the free shrinkage part is removed internally so the
        creep relaxation acts on the stress-producing strain."""
        t = np.asarray(times, dtype=float).ravel()
        eps = np.asarray(strain, dtype=float).ravel()
        if t.size != eps.size or t.size < 1:
            raise ValueError("times and strain must be equal-length (>=1)")
        tau = self._tau(t)
        sh = np.array([self.eps_sh(ti) for ti in t])
        dsig = np.zeros_like(t)
        for n in range(t.size):
            past = 0.0
            for j in range(n):
                past += dsig[j] * self.compliance(t[n], tau[j])
            J_nn = self.compliance(t[n], tau[n])
            dsig[n] = (eps[n] - sh[n] - past) / J_nn
        sigma = np.cumsum(dsig)
        return CreepHistory(times=t, stress=sigma, strain=eps,
                            eps_shrinkage=sh, mode="relaxation")


def _continuum_creep_kernel(model, elem):
    """Return (D, B_centroid, unit_voigt, dofs_per_node) for a continuum
    element, used to recover a representative (centroid) stress/strain."""
    from femsolver.elements.plane import Quad4
    from femsolver.elements.solid import Hex8, _hex8_dN_dxi

    if isinstance(elem, Quad4):
        X = elem.node_coords()
        _, _, dN_dx = elem.jacobian(0.0, 0.0, X)
        B = elem.B_matrix(dN_dx)
        D = elem.D()
        unit = np.array([1.0, 1.0, 0.0])
        return D, B, unit, 2
    if isinstance(elem, Hex8):
        X = elem.node_coords()
        dN = _hex8_dN_dxi(0.0, 0.0, 0.0)
        J = dN @ X
        dN_dx = np.linalg.solve(J, dN)
        B = np.zeros((6, 24))
        for i in range(8):
            dNx, dNy, dNz = dN_dx[0, i], dN_dx[1, i], dN_dx[2, i]
            B[0, 3 * i + 0] = dNx
            B[1, 3 * i + 1] = dNy
            B[2, 3 * i + 2] = dNz
            B[3, 3 * i + 0] = dNy
            B[3, 3 * i + 1] = dNx
            B[4, 3 * i + 1] = dNz
            B[4, 3 * i + 2] = dNy
            B[5, 3 * i + 0] = dNz
            B[5, 3 * i + 2] = dNx
        D = elem.D() if hasattr(elem, "D") else elem.material.D_3d()
        unit = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return D, B, unit, 3
    raise TypeError(
        f"StepByStepCreepFE supports Quad4 / Hex8 continuum elements; "
        f"got {type(elem).__name__}"
    )


@dataclass
class CreepFEResult:
    """Time history from :meth:`StepByStepCreepFE.run`.

    Attributes
    ----------
    times : np.ndarray
    disp : dict[(int, int), np.ndarray]
        Tracked nodal-displacement histories, keyed by ``(node_tag, dof)``.
    reactions : dict[(int, int), np.ndarray]
        Tracked support-reaction histories, keyed by ``(node_tag, dof)``.
    element_stress : dict[int, np.ndarray]
        Per-element representative (centroid) stress at each time
        (Voigt). Shows creep redistribution.
    """

    times: np.ndarray
    disp: dict
    reactions: dict
    element_stress: dict


class StepByStepCreepFE:
    """Structural step-by-step creep / shrinkage time-march on a
    continuum finite-element model (``Quad4`` / ``Hex8``).

    At each time step the **creep strain increment** of every concrete
    element (from its stored stress-increment history through the
    superposition integral) and the **shrinkage strain increment** are
    imposed as an eigenstrain. The resulting equivalent-load solve:

    * in a statically **determinate** structure, is fully accommodated
      -- the element stress is unchanged and the deflection simply grows
      (``δ → δ(1+φ)``);
    * in an **indeterminate** structure, is restrained -- the redundant
      reactions and internal forces **redistribute** with time.

    The elastic stiffness uses the instantaneous modulus ``E_c`` (the
    creep compliance increment is carried entirely by the eigenstrain --
    the standard general step-by-step / initial-strain method). Stress
    is tracked at the element centroid (exact for constant-strain
    elements / uniform-stress fields).

    Parameters
    ----------
    model : Model
        2-D plane or 3-D solid model of continuum (Quad4/Hex8) elements.
    E_c : float
        Instantaneous concrete modulus (Pa).
    phi : Callable[[float, float], float]
        Creep coefficient ``φ(t, t0)``.
    shrinkage : Callable[[float], float], optional
        Free shrinkage strain ``ε_sh(t)``.
    """

    def __init__(self, model, *, E_c, phi, shrinkage=None):
        self.model = model
        self.E_c = float(E_c)
        self._phi = phi
        self._sh = shrinkage

    def phi(self, t, t0):
        return 0.0 if t <= t0 else float(self._phi(t, t0))

    def eps_sh(self, t):
        return float(self._sh(t)) if self._sh is not None else 0.0

    @staticmethod
    def _tau(times):
        tau = np.empty_like(times)
        tau[0] = times[0]
        tau[1:] = 0.5 * (times[:-1] + times[1:])
        return tau

    def _gather_u(self, elem, dpn):
        out = []
        for nt in elem.node_tags:
            out.extend(self.model.node(nt).disp[:dpn].tolist())
        return np.asarray(out, dtype=float)

    def run(self, times, *, sustained_loads, track=(), track_reactions=()):
        from femsolver.analysis.linear_static import LinearStaticAnalysis
        from femsolver.analysis.initial_stress import apply_initial_stress
        from femsolver.elements.plane import Quad4
        from femsolver.elements.solid import Hex8

        m = self.model
        t = np.asarray(times, dtype=float).ravel()
        tau = self._tau(t)
        elems = [e for e in m.elements.values()
                 if isinstance(e, (Quad4, Hex8))]
        kern = {e.tag: _continuum_creep_kernel(m, e) for e in elems}
        Dinv = {e.tag: np.linalg.inv(kern[e.tag][0]) for e in elems}
        hist = {e.tag: [] for e in elems}        # [(dsig_voigt, tau_j), ...]
        sigma = {e.tag: np.zeros(kern[e.tag][0].shape[0]) for e in elems}

        disp_out = {k: [] for k in track}
        reac_out = {k: [] for k in track_reactions}
        es_out = {e.tag: [] for e in elems}
        u_tot = {n.tag: np.zeros(n.ndf) for n in m.nodes.values()}
        r_tot = {n.tag: np.zeros(n.ndf) for n in m.nodes.values()}

        for n_idx, t_n in enumerate(t):
            if n_idx == 0:
                m.clear_loads()
                sustained_loads(m)
                LinearStaticAnalysis(m).run()
                for e in elems:
                    D, B, _, dpn = kern[e.tag]
                    sig = D @ (B @ self._gather_u(e, dpn))
                    hist[e.tag].append((sig.copy(), tau[0]))
                    sigma[e.tag] = sig.copy()
            else:
                t_prev = t[n_idx - 1]
                deimp = {}
                sig0 = {}
                for e in elems:
                    D, _, unit, _ = kern[e.tag]
                    de_cr = np.zeros(D.shape[0])
                    for (dsig_j, tau_j) in hist[e.tag]:
                        dphi = self.phi(t_n, tau_j) - self.phi(t_prev, tau_j)
                        if dphi != 0.0:
                            de_cr += (Dinv[e.tag] @ dsig_j) * dphi
                    de_sh = (self.eps_sh(t_n) - self.eps_sh(t_prev)) * unit
                    de = de_cr + de_sh
                    deimp[e.tag] = de
                    sig0[e.tag] = -(D @ de)      # apply_initial_stress sign
                m.clear_loads()
                apply_initial_stress(m, sig0)
                LinearStaticAnalysis(m).run()     # node.disp = Δu_n
                for e in elems:
                    D, B, _, dpn = kern[e.tag]
                    de_tot = B @ self._gather_u(e, dpn)
                    dsig = D @ (de_tot - deimp[e.tag])
                    hist[e.tag].append((dsig.copy(), tau[n_idx]))
                    sigma[e.tag] = sigma[e.tag] + dsig
            # accumulate displacement + reaction increments from this solve
            for node in m.nodes.values():
                u_tot[node.tag] = u_tot[node.tag] + node.disp
                r_tot[node.tag] = r_tot[node.tag] + node.reaction
            for (nd, dof) in track:
                disp_out[(nd, dof)].append(float(u_tot[nd][dof]))
            for (nd, dof) in track_reactions:
                reac_out[(nd, dof)].append(float(r_tot[nd][dof]))
            for e in elems:
                es_out[e.tag].append(sigma[e.tag].copy())

        return CreepFEResult(
            times=t,
            disp={k: np.asarray(v) for k, v in disp_out.items()},
            reactions={k: np.asarray(v) for k, v in reac_out.items()},
            element_stress={k: np.asarray(v) for k, v in es_out.items()},
        )


# ============================================================ step-by-step creep on FRAMES

@dataclass
class CreepFrameResult:
    """Time history from :meth:`StepByStepCreepFrame.run`.

    Attributes
    ----------
    times : np.ndarray
    disp : dict[(int, int), np.ndarray]
        Tracked nodal-displacement histories, keyed by ``(node_tag, dof)``.
    reactions : dict[(int, int), np.ndarray]
        Tracked support-reaction histories, keyed by ``(node_tag, dof)``.
    axial : dict[int, np.ndarray]
        Per-element axial-force ``N(t)`` history (Pa·m² = N).
    moment : dict[int, np.ndarray]
        Per-element (average) bending-moment ``M(t)`` history (N·m).
    """

    times: np.ndarray
    disp: dict
    reactions: dict
    axial: dict
    moment: dict


class StepByStepCreepFrame:
    """Step-by-step creep / shrinkage time-march on a 2-D **frame**
    (``BeamColumn2D``) model — the frame counterpart of
    :class:`StepByStepCreepFE` (plan C1b).

    Each concrete beam element carries a generalised section state — axial
    force ``N`` (from axial strain ``ε``) and average bending moment ``M``
    (from average curvature ``κ = (θ₂−θ₁)/L``). At each time step the creep
    generalised-strain increment (from the element's own stress-increment
    history through the ageing-viscoelastic superposition integral) and the
    axial shrinkage increment are imposed as an **eigenstrain** — a per-element
    ``(ε₀, κ₀)`` applied as work-equivalent nodal loads exactly as
    :func:`femsolver.bridges.thermal_gradient.apply_beam_thermal_actions` does
    for temperature. The resulting solve:

    * in a statically **determinate** frame, is fully accommodated — the
      section forces are unchanged and the deflection simply grows
      (``δ → δ(1+φ)``);
    * in an **indeterminate** frame, is restrained wherever the free creep
      deformation is incompatible, so internal forces **redistribute** — the
      classic effects: differential-creep force transfer between members of
      different age / material, and relaxation of a restraint force.

    The elastic stiffness uses the instantaneous modulus (the element's own
    ``E``); the creep compliance increment is carried entirely by the
    eigenstrain (the general step-by-step / initial-strain method). Section
    state is tracked with a constant axial strain (exact) and an average
    curvature (exact for constant-moment elements; refine the mesh where the
    moment varies along a member).

    Parameters
    ----------
    model : Model
        A 2-D frame (``ndm=2, ndf=3``) of ``BeamColumn2D`` elements.
    phi : Callable[[float, float], float]
        Creep coefficient ``φ(t, t0)`` in analysis-time days (returns 0 for
        ``t ≤ t0``).
    shrinkage : Callable[[float], float], optional
        Free axial shrinkage strain ``ε_sh(t)`` (negative).
    element_phi : dict[int, Callable], optional
        Per-element creep-coefficient override (for **differential creep** —
        members cast at different ages / of different concrete). Elements not
        listed use ``phi``.

    Notes
    -----
    Small-displacement, linear-elastic instantaneous stiffness. MP constraints
    are handled by the underlying :class:`LinearStaticAnalysis`. Non-beam
    elements are ignored by the creep march (they still contribute stiffness).
    """

    def __init__(self, model, *, phi, shrinkage=None, element_phi=None):
        self.model = model
        self._phi = phi
        self._sh = shrinkage
        self._element_phi = dict(element_phi or {})

    def _phi_of(self, tag):
        return self._element_phi.get(tag, self._phi)

    def phi(self, tag, t, t0):
        f = self._phi_of(tag)
        return 0.0 if t <= t0 else float(f(t, t0))

    def eps_sh(self, t):
        return float(self._sh(t)) if self._sh is not None else 0.0

    @staticmethod
    def _tau(times):
        tau = np.empty_like(times)
        tau[0] = times[0]
        tau[1:] = 0.5 * (times[:-1] + times[1:])
        return tau

    def _beam_elems(self):
        from femsolver.elements.beam import BeamColumn2D
        return [e for e in self.model.elements.values()
                if isinstance(e, BeamColumn2D)]

    def _gen_strain(self, el):
        """Current-increment generalised strain ``(ε_axial, κ_avg)`` of a beam
        element, from its nodes' displacement increment (this step's solve).

        The curvature sign is taken as ``(θ₁−θ₂)/L`` so it matches the
        curvature convention of the imposed eigenstrain load in
        :meth:`_apply_eigenstrain` (verified: a free element under an imposed
        ``κ₀`` recovers exactly ``κ₀``, so a determinate member develops no
        stress under creep)."""
        T = el.transform_matrix()
        u_g = np.concatenate([self.model.node(nt).disp[:3]
                              for nt in el.node_tags])
        u_l = T @ u_g
        L = el.length_and_angle()[0]
        return (u_l[3] - u_l[0]) / L, (u_l[2] - u_l[5]) / L

    def _apply_eigenstrain(self, el, eps0, kappa):
        """Add the work-equivalent nodal loads for an imposed ``(ε₀, κ₀)`` on
        one beam element (the temperature-action form)."""
        E = el.material.E
        EA, EI = E * el.area, E * el.Iz
        f_local = np.array([-EA * eps0, 0.0, +EI * kappa,
                            +EA * eps0, 0.0, -EI * kappa])
        f_global = el.transform_matrix().T @ f_local
        n1, n2 = el.node_tags
        self.model.add_nodal_load(n1, list(f_global[0:3]))
        self.model.add_nodal_load(n2, list(f_global[3:6]))

    def run(self, times, *, sustained_loads, track=(), track_reactions=()):
        """March the creep/shrinkage response over ``times`` (days).

        Parameters
        ----------
        times : array-like
            Ages (days); ``times[0]`` is the loading age.
        sustained_loads : callable
            ``sustained_loads(model)`` sets the constant nodal/member loads
            applied at ``times[0]`` and held (via creep) thereafter.
        track, track_reactions : iterable[(node_tag, dof)]
            Nodal-displacement / support-reaction histories to record.
        """
        from femsolver.analysis.linear_static import LinearStaticAnalysis

        m = self.model
        t = np.asarray(times, dtype=float).ravel()
        if t.size < 1:
            raise ValueError("need at least one time")
        tau = self._tau(t)
        elems = self._beam_elems()
        if not elems:
            raise TypeError("StepByStepCreepFrame needs BeamColumn2D elements")

        # per-element elastic generalised-strain increment history + state
        hist = {e.tag: [] for e in elems}        # [(deps, dkappa, tau_j), ...]
        N = {e.tag: 0.0 for e in elems}
        Mn = {e.tag: 0.0 for e in elems}

        disp_out = {k: [] for k in track}
        reac_out = {k: [] for k in track_reactions}
        ax_out = {e.tag: [] for e in elems}
        mo_out = {e.tag: [] for e in elems}
        u_tot = {n.tag: np.zeros(n.ndf) for n in m.nodes.values()}
        r_tot = {n.tag: np.zeros(n.ndf) for n in m.nodes.values()}

        for n_idx, t_n in enumerate(t):
            if n_idx == 0:
                m.clear_loads()
                sustained_loads(m)
                LinearStaticAnalysis(m).run()
                for e in elems:
                    de, dk = self._gen_strain(e)
                    hist[e.tag].append((de, dk, tau[0]))
                    N[e.tag] = e.material.E * e.area * de
                    Mn[e.tag] = e.material.E * e.Iz * dk
            else:
                t_prev = t[n_idx - 1]
                imposed = {}
                for e in elems:
                    de_cr = dk_cr = 0.0
                    for (de_j, dk_j, tau_j) in hist[e.tag]:
                        dphi = (self.phi(e.tag, t_n, tau_j)
                                - self.phi(e.tag, t_prev, tau_j))
                        if dphi != 0.0:
                            de_cr += de_j * dphi
                            dk_cr += dk_j * dphi
                    de_sh = self.eps_sh(t_n) - self.eps_sh(t_prev)
                    imposed[e.tag] = (de_cr + de_sh, dk_cr)
                m.clear_loads()
                for e in elems:
                    self._apply_eigenstrain(e, *imposed[e.tag])
                LinearStaticAnalysis(m).run()         # node.disp = Δu_n
                for e in elems:
                    de_tot, dk_tot = self._gen_strain(e)
                    eps0, kap0 = imposed[e.tag]
                    de = de_tot - eps0                # stress-producing part
                    dk = dk_tot - kap0
                    hist[e.tag].append((de, dk, tau[n_idx]))
                    N[e.tag] += e.material.E * e.area * de
                    Mn[e.tag] += e.material.E * e.Iz * dk
            # accumulate tracked histories from this solve's increment
            for node in m.nodes.values():
                u_tot[node.tag] = u_tot[node.tag] + node.disp
                r_tot[node.tag] = r_tot[node.tag] + node.reaction
            for (nd, dof) in track:
                disp_out[(nd, dof)].append(float(u_tot[nd][dof]))
            for (nd, dof) in track_reactions:
                reac_out[(nd, dof)].append(float(r_tot[nd][dof]))
            for e in elems:
                ax_out[e.tag].append(N[e.tag])
                mo_out[e.tag].append(Mn[e.tag])

        return CreepFrameResult(
            times=t,
            disp={k: np.asarray(v) for k, v in disp_out.items()},
            reactions={k: np.asarray(v) for k, v in reac_out.items()},
            axial={k: np.asarray(v) for k, v in ax_out.items()},
            moment={k: np.asarray(v) for k, v in mo_out.items()},
        )


def restraint_force_relaxation(
    *,
    times,
    eps_restrained: float,
    A: float,
    E_c: float,
    phi: Callable[[float, float], float],
    shrinkage: Optional[Callable[[float], float]] = None,
) -> CreepHistory:
    """Time history of the **restraint force** in a member held at a
    constant total strain ``eps_restrained`` from ``times[0]``.

    The classic use: a shrinkage strain that would freely occur is
    fully prevented, so the member is held at zero length change while
    shrinkage tries to shorten it -- the induced tensile force then
    **relaxes** with creep. Pass ``eps_restrained = 0`` and a
    ``shrinkage`` function to get exactly that case; the returned
    ``stress`` is the restraint stress and ``stress · A`` the force.

    Returns
    -------
    CreepHistory (``stress`` is the restraint stress; multiply by ``A``
    for the force).
    """
    solver = StepByStepCreep(E_c=E_c, phi=phi, shrinkage=shrinkage)
    eps_hist = np.full(np.asarray(times, dtype=float).size,
                       float(eps_restrained))
    return solver.relaxation_history(times, eps_hist)
