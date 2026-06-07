"""High-level prestressing **tendon** with one-call ``apply_to(model)``.

This is the "define a tendon, then apply the prestress" workflow. You
describe the tendon once -- its type (pre-tension / post-tension), its
profile (eccentricity along the member), and its jacking force -- and
the engine:

1. computes the effective force ``P(x)`` after losses (friction,
   wobble, anchorage slip; reusing :mod:`femsolver.bridges.pt_tendon`),
2. lowers the tendon to a set of **equivalent nodal loads** on the host
   beam-column elements (the load-balancing / equivalent-load method),
3. applies them to the model (and can hand back a reusable
   :class:`~femsolver.analysis.load_combinations.LoadPattern`).

Equivalent-load method
----------------------
A prestressing tendon imposes on the concrete a self-equilibrated set
of forces: an axial compression ``P`` transferred at the anchors, a
primary moment ``P·e(x)`` from its eccentricity, and a transverse
"balancing" load ``w = P·κ(x)`` from its curvature. For a parabolic
drape ``a`` over a span ``L`` this is the classic upward
``w = 8 P a / L²`` (Lin's load balancing).

For a 2-node beam element of length ``L`` the work-equivalent nodal
load from the prestress section forces ``s₀ = [N₀, M₀(x)]`` (with
``N₀ = -P`` and ``M₀ = -P·e``) is, by virtual work
``f = ∫ Bᵀ s₀ dx`` in the element-local frame
``[u_i, w_i, θ_i, u_j, w_j, θ_j]``::

    f_axial = [ -N₀, 0, 0, +N₀, 0, 0 ]
    f_bend  = [ 0, (M_b - M_a)/L, -M_a, 0, (M_a - M_b)/L, M_b ]

where ``M_a = M₀(0)``, ``M_b = M₀(L)``. Summed over a finely meshed
parabolic tendon the bending shears assemble to the upward
``8 P a / L²``; for a straight eccentric tendon they vanish and the end
moments are the textbook ``± P·e``.

Applied to a statically **indeterminate** structure and solved, this
load set yields the *total* (primary + secondary) prestress effect
automatically; the **secondary / parasitic** moment is recovered as
``M_secondary = M_total - P·e`` (see :func:`tendon_secondary_moment`).

Scope (Phase B.4)
-----------------
* Element type: 2-D beam-column frames (``ndm = 2, ndf = 3``).
* Method: equivalent-load (works linear and nonlinear, bonded or
  unbonded -- the equivalent load is the same; bonding only matters for
  the tendon's own stress increment, recovered separately).
* 3-D beams and shell / solid hosts are planned follow-ups.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from femsolver.bridges.pt_tendon import (
    TendonProfile,
    anchorage_slip_loss,
    friction_loss,
)


# ============================================================ Tendon

@dataclass
class Tendon:
    """A prestressing tendon defined on a run of beam-column nodes.

    Parameters
    ----------
    nodes : Sequence[int]
        Ordered node tags the tendon passes through. Consecutive nodes
        must be joined by a beam-column element in the model.
    eccentricity : Sequence[float]
        Tendon eccentricity at each node (m), measured in the element
        local transverse (+y up) direction. **Negative = below the
        centroid** (the usual sagging drape that cambers the member
        up). Length must equal ``len(nodes)``.
    area : float
        Total tendon (strand) area (m²).
    jacking_force : float
        Jacking force ``P_0`` at the stressing anchor (N, > 0).
    tendon_type : {"post-tension", "pre-tension"}, default "post-tension"
        Selects the default loss set. Post-tension includes duct
        friction + wobble + anchorage slip; pre-tension omits duct
        friction (strand stressed against an abutment).
    bonded : bool, default True
        Recorded for downstream tendon-stress recovery; does not change
        the equivalent load itself.
    E_p : float, default 195e9
        Strand modulus (Pa), used for anchorage-slip length.
    mu : float, default 0.20
        Curvature friction coefficient (post-tension).
    wobble_k : float, default 0.0066
        Wobble coefficient (1/m, post-tension).
    anchor_slip : float, default 0.0
        Anchorage seating slip (m). 0 disables the slip calculation.
    jack_from : {"start", "end", "both"}, default "both"
        Stressing end(s). ``"both"`` takes the more-favourable (higher)
        force at each point.
    effective_force : float or Sequence[float], optional
        If given, used directly as ``P(x)`` (scalar = uniform, or one
        value per node) and **all loss calculations are skipped**. Use
        when the post-loss effective prestress is already known.
    name : str, default "tendon"
    """

    nodes: Sequence[int]
    eccentricity: Sequence[float]
    area: float
    jacking_force: float
    eccentricity_z: Optional[Sequence[float]] = None
    tendon_type: str = "post-tension"
    bonded: bool = True
    E_p: float = 195e9
    mu: float = 0.20
    wobble_k: float = 0.0066
    anchor_slip: float = 0.0
    jack_from: str = "both"
    effective_force: Optional[object] = None
    name: str = "tendon"

    def __post_init__(self) -> None:
        self.nodes = list(self.nodes)
        self.eccentricity = np.asarray(self.eccentricity, dtype=float).ravel()
        if len(self.nodes) < 2:
            raise ValueError("tendon needs at least 2 nodes")
        if self.eccentricity.size != len(self.nodes):
            raise ValueError("eccentricity length must equal nodes length")
        if self.eccentricity_z is None:
            self.eccentricity_z = np.zeros(len(self.nodes))
        else:
            self.eccentricity_z = np.asarray(
                self.eccentricity_z, dtype=float
            ).ravel()
            if self.eccentricity_z.size != len(self.nodes):
                raise ValueError(
                    "eccentricity_z length must equal nodes length"
                )
        if self.area <= 0:
            raise ValueError("area must be > 0")
        if self.jacking_force <= 0:
            raise ValueError("jacking_force must be > 0")
        if self.tendon_type not in ("post-tension", "pre-tension"):
            raise ValueError(
                "tendon_type must be 'post-tension' or 'pre-tension'"
            )
        if self.jack_from not in ("start", "end", "both"):
            raise ValueError("jack_from must be 'start', 'end', or 'both'")

    # ----------------------------------------------------------- geometry
    def _stations(self, model) -> np.ndarray:
        """Cumulative chord distance along the tendon path (m)."""
        coords = [np.asarray(model.node(t).coords, dtype=float)
                  for t in self.nodes]
        s = np.zeros(len(coords))
        for i in range(1, len(coords)):
            s[i] = s[i - 1] + float(np.linalg.norm(coords[i] - coords[i - 1]))
        return s

    # ----------------------------------------------------------- losses
    def force_profile(self, model) -> np.ndarray:
        """Effective prestress force ``P`` at each node (N).

        If :attr:`effective_force` was supplied it is returned directly;
        otherwise friction + wobble (post-tension only) and anchorage
        slip are applied to the jacking force.
        """
        n = len(self.nodes)
        if self.effective_force is not None:
            ef = np.asarray(self.effective_force, dtype=float).ravel()
            if ef.size == 1:
                return np.full(n, float(ef[0]))
            if ef.size != n:
                raise ValueError(
                    "effective_force must be scalar or one value per node"
                )
            return ef

        s = self._stations(model)
        P0 = self.jacking_force

        if self.tendon_type == "pre-tension":
            # No duct friction; uniform jacking force (immediate elastic /
            # time-dependent losses are applied separately by the caller).
            return np.full(n, P0)

        # Build the friction profile from the tendon's angle changes in
        # the (station, eccentricity) plane.
        seg_len = np.diff(s)
        de = np.diff(self.eccentricity)
        theta = np.arctan2(de, seg_len)
        dalpha = np.zeros_like(seg_len)
        dalpha[1:] = np.abs(np.diff(theta))
        dalpha[0] = abs(theta[0])
        profile = TendonProfile(segment_lengths=seg_len, segment_dalpha=dalpha)

        def _from_start() -> np.ndarray:
            if self.anchor_slip > 0.0:
                res = anchorage_slip_loss(
                    profile, P_0=P0, mu=self.mu, k=self.wobble_k,
                    slip=self.anchor_slip, E_s=self.E_p, A_ps=self.area,
                )
                return np.asarray(res.P_profile, dtype=float)
            fr = friction_loss(profile, mu=self.mu, k=self.wobble_k)
            return P0 * np.asarray(fr.P_over_P0, dtype=float)

        P_start = _from_start()
        if self.jack_from == "start":
            return P_start

        # Stress from the other end: reverse the profile, compute, flip back
        rev = TendonProfile(
            segment_lengths=seg_len[::-1].copy(),
            segment_dalpha=(np.r_[abs(theta[-1]),
                                   np.abs(np.diff(theta))[::-1]]),
        )
        if self.anchor_slip > 0.0:
            res = anchorage_slip_loss(
                rev, P_0=P0, mu=self.mu, k=self.wobble_k,
                slip=self.anchor_slip, E_s=self.E_p, A_ps=self.area,
            )
            P_end = np.asarray(res.P_profile, dtype=float)[::-1]
        else:
            fr = friction_loss(rev, mu=self.mu, k=self.wobble_k)
            P_end = (P0 * np.asarray(fr.P_over_P0, dtype=float))[::-1]

        if self.jack_from == "end":
            return P_end
        return np.maximum(P_start, P_end)   # jack_from == "both"

    # ----------------------------------------------------------- primary moment
    def primary_moment(self, model) -> dict:
        """Primary prestress moment ``M_p = P·e`` at each node (N·m),
        i.e. the moment from the tendon force at its eccentricity acting
        on the *released* (statically determinate) member."""
        P = self.force_profile(model)
        return {tag: float(P[i] * self.eccentricity[i])
                for i, tag in enumerate(self.nodes)}

    # ----------------------------------------------------------- apply
    def _host_element(self, model, na: int, nb: int):
        for e in model.elements.values():
            tags = tuple(e.node_tags)
            if tags == (na, nb) or tags == (nb, na):
                return e
        raise ValueError(
            f"tendon nodes {na}-{nb} are not joined by any element"
        )

    def apply_to(self, model, factor: float = 1.0) -> None:
        """Add the tendon's equivalent nodal loads to ``model``.

        Supports 2-D frames (``ndm=2, ndf=3``) and 3-D frames
        (``ndm=3, ndf=6``). For 3-D, the eccentricity in the section's
        local *z* direction is taken from :attr:`eccentricity_z`.

        Parameters
        ----------
        model : Model
        factor : float, default 1.0
            Scale on the whole prestress (for load combinations).
        """
        if model.ndm == 2 and model.ndf == 3:
            self._apply_2d(model, factor)
        elif model.ndm == 3 and model.ndf == 6:
            self._apply_3d(model, factor)
        else:
            raise NotImplementedError(
                "Tendon.apply_to supports 2-D frames (ndm=2, ndf=3) and "
                "3-D frames (ndm=3, ndf=6). Shell / solid hosts use the "
                "initial-stress eigenstrain path."
            )

    def _apply_2d(self, model, factor: float) -> None:
        P = self.force_profile(model) * factor
        ecc = self.eccentricity
        for i in range(len(self.nodes) - 1):
            na, nb = self.nodes[i], self.nodes[i + 1]
            elem = self._host_element(model, na, nb)
            Xa = np.asarray(model.node(na).coords, dtype=float)
            Xb = np.asarray(model.node(nb).coords, dtype=float)
            d = Xb - Xa
            L = float(np.hypot(d[0], d[1]))
            if L <= 0.0:
                continue
            c, s = d[0] / L, d[1] / L
            if tuple(elem.node_tags) == (na, nb):
                Pi, Pj, ei, ej = P[i], P[i + 1], ecc[i], ecc[i + 1]
            else:
                Pi, Pj, ei, ej = P[i + 1], P[i], ecc[i + 1], ecc[i]
                c, s = -c, -s
            # prestress section forces: axial compression N0 = -P, primary
            # moment M0 = P*e (verified by balanced-load + P*e checks).
            N0 = -0.5 * (Pi + Pj)
            Ma, Mb = Pi * ei, Pj * ej
            f_local = np.array([
                -N0, (Mb - Ma) / L, -Ma,
                N0, (Ma - Mb) / L, Mb,
            ])
            for k, ntag in ((0, na), (3, nb)):
                fx_l, fy_l, m_l = f_local[k], f_local[k + 1], f_local[k + 2]
                fx_g = c * fx_l - s * fy_l
                fy_g = s * fx_l + c * fy_l
                model.add_nodal_load(ntag, [fx_g, fy_g, m_l])

    def _apply_3d(self, model, factor: float) -> None:
        P = self.force_profile(model) * factor
        ey = np.asarray(self.eccentricity, dtype=float)
        ez = np.asarray(self.eccentricity_z, dtype=float)
        for i in range(len(self.nodes) - 1):
            na, nb = self.nodes[i], self.nodes[i + 1]
            elem = self._host_element(model, na, nb)
            L, _, _, _ = elem.length_and_axes()
            if L <= 0.0:
                continue
            # order the per-node quantities to match the element's i->j
            if tuple(elem.node_tags) == (na, nb):
                Pi, Pj = P[i], P[i + 1]
                eyi, eyj = ey[i], ey[i + 1]
                ezi, ezj = ez[i], ez[i + 1]
            else:
                Pi, Pj = P[i + 1], P[i]
                eyi, eyj = ey[i + 1], ey[i]
                ezi, ezj = ez[i + 1], ez[i]
            N0 = -0.5 * (Pi + Pj)
            # x-y plane: e_y -> M_z = P*e_y (identical to the 2-D plane)
            Mza, Mzb = Pi * eyi, Pj * eyj
            # x-z plane: e_z -> M_y = P*e_z. Bending about local y uses the
            # (w, theta_y) DOFs; theta_y = -dw/dx flips the rotation-entry
            # signs relative to the x-y plane (standard 3-D beam convention).
            Mya, Myb = Pi * ezi, Pj * ezj
            f_local = np.zeros(12)
            # node i (0:u,1:v,2:w,3:thx,4:thy,5:thz)
            f_local[0] = -N0                       # axial
            f_local[1] = (Mzb - Mza) / L           # v_i  (x-y shear)
            f_local[5] = -Mza                      # thz_i
            f_local[2] = (Myb - Mya) / L           # w_i  (x-z shear)
            f_local[4] = +Mya                      # thy_i  (sign-flipped)
            # node j (6:u,7:v,8:w,9:thx,10:thy,11:thz)
            f_local[6] = N0
            f_local[7] = (Mza - Mzb) / L
            f_local[11] = Mzb
            f_local[8] = (Mya - Myb) / L
            f_local[10] = -Myb                     # thy_j  (sign-flipped)
            # local -> global and scatter (6 DOF/node)
            f_global = elem.transform_matrix().T @ f_local
            model.add_nodal_load(na, f_global[0:6].tolist())
            model.add_nodal_load(nb, f_global[6:12].tolist())

    def load_pattern(self):
        """Return a :class:`~femsolver.analysis.load_combinations.LoadPattern` that
        applies this tendon (so it composes with load combinations)."""
        from femsolver.analysis.load_combinations import LoadPattern
        return LoadPattern(self.name, lambda m, f=1.0: self.apply_to(m, f))


# ============================================================ secondary moment

def tendon_secondary_moment(
    *,
    total_moment: float,
    P: float,
    e: float,
) -> float:
    """Secondary (parasitic) prestress moment at a section.

    In an indeterminate structure the analysis under the tendon's
    equivalent load gives the **total** prestress moment. The
    **secondary** (parasitic) moment, caused by the redundant supports
    restraining the prestress camber, is::

        M_secondary = M_total - M_primary = M_total - P·e

    Parameters
    ----------
    total_moment : float
        Bending moment from the equivalent-load analysis (N·m).
    P : float
        Effective tendon force at the section (N).
    e : float
        Tendon eccentricity at the section (m, same sign convention as
        :class:`Tendon`).
    """
    return float(total_moment - P * e)


def tendon_secondary_shear(*, total_shear: float, P: float, slope: float) -> float:
    """Secondary (parasitic) prestress shear at a section.

        V_secondary = V_total - V_primary = V_total - P · e'(x)

    The primary shear is the vertical component of the tendon force,
    ``P · slope`` where ``slope = de/dx`` is the tendon profile slope at
    the section. For a straight tendon (``slope = 0``) all of the total
    shear is secondary.
    """
    return float(total_shear - P * slope)


def tendon_secondary_forces(model, tendon, *, constraints: str = "transformation") -> dict:
    """Secondary (hyperstatic / parasitic) **reactions** from a
    prestress-only analysis.

    The tendon's equivalent load is a *self-equilibrated* set, so when
    it is the only load on the structure the support reactions that
    develop are precisely the secondary reactions: **zero for a
    statically determinate structure**, and the parasitic redundant
    reactions for an indeterminate one. Every secondary internal force
    (moment / shear diagram) is then the statics of these reactions on
    the released structure -- e.g. the secondary moment at a section
    also equals ``M_total - P·e`` (:func:`tendon_secondary_moment`).

    Parameters
    ----------
    model : Model
    tendon : Tendon
    constraints : str
        Constraint handler for the internal solve.

    Returns
    -------
    dict[int, np.ndarray]
        ``{node_tag: reaction_vector}`` at every restrained node (the
        secondary reactions).

    Notes
    -----
    This runs a **prestress-only** linear static analysis: it clears the
    model's loads, applies the tendon, and solves. Pre-existing nodal
    loads are restored; element distributed loads are cleared (re-apply
    your full load case afterwards). The model is left holding the
    prestress-only solved state.
    """
    from femsolver.analysis.linear_static import LinearStaticAnalysis

    snap = {n.tag: n._load.copy() for n in model.nodes.values()}
    model.clear_loads()
    tendon.apply_to(model)
    LinearStaticAnalysis(model, constraints=constraints).run()
    reactions = {
        n.tag: n.reaction.copy()
        for n in model.nodes.values()
        if bool(np.any(n.fixity))
    }
    for n in model.nodes.values():
        n._load[:] = snap[n.tag]
    return reactions
