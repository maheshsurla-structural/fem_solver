"""Performance-based seismic / capacity-design utilities.

Tools for the nonlinear-static design workflow common to FEMA 356,
ASCE 41, and EN 1998 (Eurocode 8):

1. Bilinearize a pushover capacity curve into an equivalent
   elastic-perfectly-plastic / strain-hardening bilinear.
2. Convert an MDOF capacity curve to its **equivalent SDOF** via
   the modal participation factor (the N2 method's foundation).
3. Compute the **target displacement** under a response-spectrum
   demand using either the N2 method (EC 8 Annex B) or the
   Coefficient Method (ASCE 41 / FEMA 356 nonlinear-static
   procedure).
4. Run a pushover analysis up to a prescribed roof drift using
   ``PushoverToTarget``.
5. Extract **story drifts** from a converged analysis and combine
   orthogonal-direction seismic responses with the 100-30 or SRSS
   rules.

Sign convention
---------------
"Drift" arrays are *roof displacement* in the analysis direction
(meters). "Force" arrays are *base shear* (newtons). Positive values
of both quantities indicate the direction of the pushover load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ============================================================ bilinearization

@dataclass
class BilinearCurve:
    """Equivalent bilinear capacity curve.

    Attributes
    ----------
    K_i      : initial stiffness (slope of the elastic branch).
    d_y, F_y : yield point (intersection of elastic and post-yield).
    d_u, F_u : ultimate point (end of the bilinear segment).
    alpha    : post-yield-to-initial stiffness ratio
               (so ``K_post = alpha * K_i``).
    """

    K_i: float
    d_y: float
    F_y: float
    d_u: float
    F_u: float
    alpha: float

    def force_at(self, d: float) -> float:
        """Bilinear force at drift ``d`` (extrapolates beyond ``d_u``)."""
        if d <= self.d_y:
            return self.K_i * d
        return self.F_y + self.K_i * self.alpha * (d - self.d_y)


def _initial_stiffness(drift: np.ndarray, force: np.ndarray,
                        secant_ratio: float = 0.6) -> float:
    """Secant stiffness from the origin to the first point where the
    force reaches ``secant_ratio * max(force)``. This is the
    standard FEMA 356 effective-elastic-stiffness definition."""
    peak = float(np.max(np.abs(force)))
    if peak <= 0.0:
        raise ValueError("force array has no positive values")
    target = secant_ratio * peak
    sign = 1.0 if np.max(force) > 0.0 else -1.0
    abs_force = np.abs(force)
    idx = int(np.searchsorted(abs_force, target))
    if idx <= 0:
        idx = 1
    if idx >= drift.size:
        idx = drift.size - 1
    # Linear interpolation
    f1, f2 = abs_force[idx - 1], abs_force[idx]
    d1, d2 = drift[idx - 1], drift[idx]
    if f2 == f1:
        d_sec = d2
        f_sec = f2
    else:
        t = (target - f1) / (f2 - f1)
        d_sec = d1 + t * (d2 - d1)
        f_sec = target * sign
    if d_sec == 0.0:
        return 0.0
    return abs(f_sec / d_sec)


def bilinearize_capacity_curve(
    drift: Sequence[float],
    force: Sequence[float],
    *,
    method: str = "equal_area",
    secant_ratio: float = 0.6,
    ultimate_drop_ratio: float = 0.8,
) -> BilinearCurve:
    """Compute the equivalent bilinear of a capacity curve.

    Parameters
    ----------
    drift, force : array-like
        Pushover capacity curve. Must be sorted by ``drift``.
    method : ``"equal_area"`` (default)
        Equal-area / equal-energy approach (FEMA 356, EC 8): the
        bilinear strikes the same area under the curve from 0 to the
        ultimate drift as the original capacity curve.
    secant_ratio : float, default 0.6
        For ``"equal_area"`` only: the initial stiffness is taken as
        the secant to the point where the force reaches this fraction
        of the peak (FEMA 356's 0.6 default).
    ultimate_drop_ratio : float, default 0.8
        Ultimate drift = the first drift past the peak where the
        force has dropped to this fraction of the peak (the "80%
        rule"). If the curve does not drop below this threshold, the
        ultimate point is the last data point.

    Returns
    -------
    BilinearCurve
    """
    drift = np.asarray(drift, dtype=float).ravel()
    force = np.asarray(force, dtype=float).ravel()
    if drift.size != force.size or drift.size < 2:
        raise ValueError("drift and force must be 1-D arrays of equal length >= 2")
    if np.any(np.diff(drift) < 0.0):
        raise ValueError("drift array must be sorted ascending")

    # Use absolute values so we work with the signed-positive branch
    # but preserve sign at the end.
    sign = 1.0 if np.max(force) >= -np.min(force) else -1.0
    abs_force = np.abs(force) if sign > 0 else -force.copy()

    # Initial stiffness (secant to 60% of peak)
    K_i = _initial_stiffness(drift, abs_force, secant_ratio=secant_ratio)
    if K_i <= 0.0:
        raise ValueError("initial stiffness must be positive")

    # Find ultimate point
    peak_idx = int(np.argmax(abs_force))
    peak = abs_force[peak_idx]
    if peak_idx == drift.size - 1:
        # No post-peak data -> ultimate is the last point
        d_u = drift[-1]
        F_u = abs_force[-1]
    else:
        target = ultimate_drop_ratio * peak
        idx = peak_idx
        while idx < drift.size - 1 and abs_force[idx + 1] >= target:
            idx += 1
        # idx is the last point at / above the threshold
        if idx == drift.size - 1:
            d_u = drift[-1]
            F_u = abs_force[-1]
        else:
            # Interpolate to the threshold crossing
            f1, f2 = abs_force[idx], abs_force[idx + 1]
            d1, d2 = drift[idx], drift[idx + 1]
            if f1 == f2:
                d_u = d2
                F_u = target
            else:
                t = (target - f1) / (f2 - f1)
                d_u = d1 + t * (d2 - d1)
                F_u = target

    # Compute area under the actual curve from 0 to d_u (trapezoidal)
    # Truncate the arrays to <= d_u
    mask = drift <= d_u + 1.0e-12
    d_trim = drift[mask]
    f_trim = abs_force[mask]
    if d_trim[-1] < d_u:
        d_trim = np.append(d_trim, d_u)
        f_trim = np.append(f_trim, F_u)
    area = float(np.trapezoid(f_trim, d_trim))

    # Equal area: F_y satisfying area = F_y * (d_u - F_y / (2 K_i))
    # Quadratic in F_y. Solve and take the smaller root.
    a_q = 0.5 / K_i
    b_q = -d_u
    c_q = area
    disc = b_q * b_q - 4.0 * a_q * c_q
    if disc < 0.0:
        # Area too small for any bilinear with this K_i; fall back to
        # peak as the yield force.
        F_y = peak
    else:
        sd = math.sqrt(disc)
        # Two roots: prefer the smaller positive root.
        root1 = (-b_q - sd) / (2.0 * a_q)
        root2 = (-b_q + sd) / (2.0 * a_q)
        # Both should be positive; we want the smaller (more yielded
        # bilinear).
        candidates = [r for r in (root1, root2) if 0.0 < r <= peak * 1.01]
        if not candidates:
            F_y = peak
        else:
            F_y = min(candidates)
    d_y = F_y / K_i
    if d_u <= d_y:
        # Degenerate: curve barely past yield
        alpha = 0.0
    else:
        alpha = (F_u - F_y) / (K_i * (d_u - d_y))
    return BilinearCurve(
        K_i=K_i * sign,
        d_y=d_y * sign,
        F_y=F_y * sign,
        d_u=d_u * sign,
        F_u=F_u * sign,
        alpha=alpha,
    )


# ============================================================ SDOF conversion

@dataclass
class EquivalentSDOF:
    """Equivalent SDOF system derived from an MDOF pushover.

    Attributes
    ----------
    Gamma     : modal participation factor.
    m_eff     : effective modal mass (kg).
    d_star    : SDOF capacity curve, displacement (m).
    F_star    : SDOF capacity curve, force (N).
    T_eff     : effective period at yield (s). Computed from
                ``T = 2 pi sqrt(m_eff / K_i_star)`` where ``K_i_star``
                is the initial slope of the SDOF curve.
    """

    Gamma: float
    m_eff: float
    d_star: np.ndarray
    F_star: np.ndarray
    T_eff: float


def equivalent_sdof(drift: Sequence[float], force: Sequence[float],
                     *, Gamma: float, m_eff: float) -> EquivalentSDOF:
    """Convert an MDOF capacity curve ``(d_top, V_base)`` to its
    equivalent SDOF curve ``(d*, F*)`` using the modal participation
    factor.

    Parameters
    ----------
    drift, force : array-like
        MDOF roof drift and base shear arrays.
    Gamma : float
        Modal participation factor (phi^T M iota / phi^T M phi for the
        chosen pushover mode -- usually mode 1).
    m_eff : float
        Effective modal mass = Gamma^2 * (phi^T M iota) (the standard
        N2 definition).

    Returns
    -------
    EquivalentSDOF
    """
    drift = np.asarray(drift, dtype=float).ravel()
    force = np.asarray(force, dtype=float).ravel()
    if Gamma == 0.0:
        raise ValueError("Gamma must be nonzero")
    d_star = drift / Gamma
    F_star = force / Gamma
    # Initial slope of SDOF curve
    if d_star.size >= 2 and d_star[1] != d_star[0]:
        K_i_star = (F_star[1] - F_star[0]) / (d_star[1] - d_star[0])
    else:
        # Find first nonzero d_star
        idx = int(np.searchsorted(np.abs(d_star), 1.0e-30))
        if idx < d_star.size and d_star[idx] != 0.0:
            K_i_star = F_star[idx] / d_star[idx]
        else:
            K_i_star = 0.0
    if K_i_star <= 0.0:
        T_eff = float("inf")
    else:
        T_eff = 2.0 * math.pi * math.sqrt(m_eff / K_i_star)
    return EquivalentSDOF(
        Gamma=float(Gamma),
        m_eff=float(m_eff),
        d_star=d_star,
        F_star=F_star,
        T_eff=T_eff,
    )


# ============================================================ N2 method

def n2_target_displacement(
    spectrum,
    sdof: EquivalentSDOF,
    bilinear: BilinearCurve,
    *,
    Tc: float = 0.5,
) -> dict:
    """N2-method (EC 8 Annex B) target displacement.

    Parameters
    ----------
    spectrum : ResponseSpectrum
        Elastic response spectrum (with ``Sa(T)`` method).
    sdof : EquivalentSDOF
        Equivalent-SDOF capacity curve from :func:`equivalent_sdof`.
    bilinear : BilinearCurve
        Bilinearized SDOF curve from
        :func:`bilinearize_capacity_curve` applied to ``(d_star, F_star)``.
    Tc : float, default 0.5
        Corner period separating the short-period and long-period
        ranges of the response spectrum. EC 8 default for typical soil
        is ~0.4-0.5 s.

    Returns
    -------
    dict with keys:

    * ``"T_star"`` : effective SDOF period (s).
    * ``"d_y_star"``, ``"F_y_star"`` : SDOF yield point.
    * ``"d_e_star"`` : elastic-spectrum SDOF demand at T_star.
    * ``"d_t_star"`` : inelastic SDOF target displacement.
    * ``"d_t_top"``  : MDOF target roof displacement (= Gamma * d_t_star).
    * ``"R"``        : strength-reduction factor at T_star.
    """
    T_star = sdof.T_eff
    d_y_star = abs(bilinear.d_y)
    F_y_star = abs(bilinear.F_y)
    # Elastic SDOF demand at T_star
    Sa_T = spectrum.Sa(T_star)
    d_e_star = Sa_T * (T_star / (2.0 * math.pi)) ** 2
    # Strength reduction factor at T_star
    if F_y_star <= 0.0:
        R = float("inf")
    else:
        R = Sa_T * sdof.m_eff / F_y_star
    # R-mu-T relations
    if T_star >= Tc:
        # Equal-displacement rule
        d_t_star = d_e_star
    else:
        # Short-period correction
        d_t_star = d_e_star * (1.0 + (R - 1.0) * Tc / T_star) / R
        # Limit: d_t* >= d_e* if no yielding (R <= 1)
        if R <= 1.0:
            d_t_star = d_e_star
    d_t_top = sdof.Gamma * d_t_star
    return {
        "T_star": T_star,
        "d_y_star": d_y_star,
        "F_y_star": F_y_star,
        "d_e_star": d_e_star,
        "d_t_star": d_t_star,
        "d_t_top": d_t_top,
        "R": R,
        "Sa_T_star": Sa_T,
    }


# ============================================================ Coefficient Method

def coefficient_method_target(
    spectrum,
    *,
    T_eff: float,
    C0: float = 1.3,
    C1: float = 1.0,
    C2: float = 1.0,
    g: float = 9.81,
) -> dict:
    """ASCE 41 / FEMA 356 Coefficient-Method target displacement.

    .. math::

        d_t = C_0 C_1 C_2 \\, S_a(T_e) \\, (T_e / 2 \\pi)^2 g

    Parameters
    ----------
    spectrum : ResponseSpectrum
    T_eff : float
        Effective period from the bilinearized SDOF curve.
    C0 : float, default 1.3
        Modification factor relating roof displacement to SDOF
        displacement. Typical values 1.0-1.5 depending on building
        height + first-mode shape (ASCE 41 Table 7-5).
    C1 : float, default 1.0
        Modification factor for inelastic-to-elastic SDOF displacement
        ratio. 1.0 in the equal-displacement range; can be > 1 for
        short-period structures.
    C2 : float, default 1.0
        Modification factor for hysteresis-pinching effects. 1.0 for
        typical force-controlled actions.
    g : float, default 9.81
        Gravity acceleration. ``Sa`` is assumed in units of ``m/s^2``
        already; if ``Sa`` is in ``g``, set ``g = 1.0`` and the
        formula returns ``d_t`` in the same length unit as ``T``.

    Returns
    -------
    dict with the same keys as :func:`n2_target_displacement` for
    consistency.
    """
    Sa = spectrum.Sa(T_eff)
    d_t = C0 * C1 * C2 * Sa * (T_eff / (2.0 * math.pi)) ** 2
    return {
        "T_eff": T_eff,
        "Sa_T_eff": Sa,
        "C0": C0,
        "C1": C1,
        "C2": C2,
        "d_t_top": d_t,
    }


# ============================================================ multi-axis

def seismic_combination(responses: dict, rule: str = "100-30") -> float:
    """Combine orthogonal-direction seismic responses.

    Parameters
    ----------
    responses : dict[str, float]
        Map of direction labels (e.g. ``"x"``, ``"y"``, ``"z"``) to
        the response magnitude in that direction at the same point of
        the structure.
    rule : ``"100-30"`` (default) or ``"SRSS"``

        * ``"100-30"`` -- ASCE 7 / EC 8 combination: 100% of one
          direction plus 30% of each of the others. Returns the
          maximum over the 3 permutations (or 2 in 2D).
        * ``"SRSS"`` -- Square-Root-of-Sum-of-Squares.

    Returns
    -------
    float
        Combined response magnitude.
    """
    vals = list(responses.values())
    abs_vals = [abs(v) for v in vals]
    if rule == "100-30":
        # Permute: 100% one direction + 30% of each other
        worst = 0.0
        for i in range(len(abs_vals)):
            total = abs_vals[i] + 0.3 * sum(
                abs_vals[j] for j in range(len(abs_vals)) if j != i
            )
            if total > worst:
                worst = total
        return worst
    if rule == "SRSS":
        return math.sqrt(sum(v * v for v in abs_vals))
    raise ValueError(f"unknown combination rule {rule!r}; "
                       f"expected '100-30' or 'SRSS'")


# ============================================================ pushover-to-target

class PushoverToTarget:
    """Run a displacement-controlled pushover until the tracked DOF
    reaches a target drift.

    Wraps :class:`NonlinearStaticAnalysis` with
    :class:`DisplacementControl`, recording the (drift, base shear)
    capacity curve along the way.

    Parameters
    ----------
    model : Model
        Model with the load pattern (proportional to the pushover
        invariant) already applied via ``add_nodal_load``.
    target_drift : float
        Roof drift (in the tracked DOF) at which to stop.
    track : tuple ``(node_tag, dof)``
        Tracked DOF -- usually the roof horizontal displacement.
    base_node_tags : sequence of int, optional
        Base nodes; the base shear is computed as the sum of their
        reactions in the tracked direction. If None the base shear
        is computed from element f_int (slower).
    num_steps : int, default 50
        Steps to march from 0 to ``target_drift``.
    tol : float, default 1e-6
    max_iter : int, default 50

    Notes
    -----
    Returns a result dictionary similar to NonlinearStaticAnalysis but
    augmented with the explicit capacity curve.
    """

    def __init__(self, model, *,
                 target_drift: float,
                 track: tuple,
                 base_node_tags: Sequence[int] | None = None,
                 num_steps: int = 50,
                 tol: float = 1.0e-6,
                 max_iter: int = 50):
        if target_drift == 0.0:
            raise ValueError("target_drift must be nonzero")
        self.model = model
        self.target_drift = float(target_drift)
        self.track = track
        self.base_node_tags = (list(base_node_tags)
                                  if base_node_tags is not None else None)
        self.num_steps = int(num_steps)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def run(self) -> dict:
        # Deferred imports to avoid circular references
        from femsolver.analysis.nonlinear_static import NonlinearStaticAnalysis
        from femsolver.analysis.static_integrator import DisplacementControl
        node_tag, dof = self.track
        d_target = self.target_drift
        dlambda = d_target / self.num_steps     # Δ (target disp per step)
        integrator = DisplacementControl(
            node_tag=node_tag, dof_index=dof, du_step=dlambda,
        )
        analysis = NonlinearStaticAnalysis(
            self.model, num_steps=self.num_steps,
            integrator=integrator, tol=self.tol, max_iter=self.max_iter,
            track=self.track,
        )
        # Build a per-step base-shear collector via a callback. We do
        # this by inspecting the model after run() (we record the
        # *final* state in res; per-step values come from res["lambdas"]
        # and res["tracked"] from NLSA).
        res = analysis.run()
        # res["tracked"] gives the displacement at each step; we still
        # need the corresponding base shears. The load pattern was set
        # by the user via add_nodal_load with magnitudes summing to
        # F_ref; lambda * F_ref is the applied force at each step.
        # For PUSHOVER, the base shear equals the applied load
        # (rigid-body equilibrium of the whole structure).
        F_ref_signed = 0.0
        for n in self.model.nodes.values():
            F_ref_signed += float(n._load[dof])
        drift = np.asarray(res["tracked"], dtype=float)
        force = np.asarray(res["lambdas"], dtype=float) * F_ref_signed
        return {
            "drift": drift,
            "force": force,
            "lambdas": res["lambdas"],
            "iter_counts": res["iter_counts"],
            "final_lambda": res["final_lambda"],
            "total_iterations": res["total_iterations"],
        }


# ============================================================ story drifts

def story_drifts(
    model,
    story_node_tags: Sequence[int],
    *,
    direction: int = 0,
    base_node_tag: int | None = None,
) -> dict:
    """Compute interstory drift ratios at each story.

    Parameters
    ----------
    model : Model
        Model with a converged displacement state (e.g. after a
        linear-static or nonlinear-static pushover run).
    story_node_tags : sequence of int
        Representative node tag at each story, ordered from lowest
        (just above the base) to highest (the roof). If a
        ``base_node_tag`` is provided it is treated as the base
        reference (drift relative to ground); otherwise the absolute
        zero is used.
    direction : int, default 0
        DOF index in which to measure drift (0 = x, 1 = y).
    base_node_tag : int, optional
        Node at the base used as the zero-drift reference. If None,
        the base level is at zero displacement (typical).

    Returns
    -------
    dict with:

    * ``"story"``         : 1..N story index.
    * ``"height_to_base"``: cumulative height to each story top.
    * ``"absolute_disp"`` : story displacement in ``direction``.
    * ``"interstory_drift"``: story-i disp - story-(i-1) disp.
    * ``"story_height"``   : height of each story.
    * ``"drift_ratio"``    : interstory drift / story height.
    """
    if not story_node_tags:
        raise ValueError("story_node_tags must be non-empty")
    coords = []
    disps = []
    base_disp = 0.0
    base_y = 0.0
    # Use the second coord (y in 2D / 3D) as the height direction.
    height_dim = 1
    if base_node_tag is not None:
        base_node = model.node(base_node_tag)
        base_disp = float(base_node.disp[direction])
        base_y = float(base_node.coords[height_dim])
    for tag in story_node_tags:
        n = model.node(tag)
        coords.append(float(n.coords[height_dim]))
        disps.append(float(n.disp[direction]))
    coords = np.array(coords)
    disps = np.array(disps)
    heights = coords - base_y
    # Story height = height_i - height_(i-1)
    story_heights = np.empty_like(heights)
    story_heights[0] = heights[0]
    story_heights[1:] = np.diff(heights)
    abs_disp = disps - base_disp
    interstory = np.empty_like(abs_disp)
    interstory[0] = abs_disp[0]
    interstory[1:] = np.diff(abs_disp)
    drift_ratio = np.where(story_heights > 0.0,
                              interstory / story_heights,
                              0.0)
    return {
        "story": np.arange(1, len(story_node_tags) + 1),
        "height_to_base": heights,
        "absolute_disp": abs_disp,
        "interstory_drift": interstory,
        "story_height": story_heights,
        "drift_ratio": drift_ratio,
    }
