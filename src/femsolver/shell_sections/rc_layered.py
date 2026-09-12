"""Reinforced-concrete **layered shell section** (Midas-style multi-layered
grid / "concrete shell").

This is the RC analogue of :class:`~femsolver.sections.response.fiber.FiberSection2D`
but for a *shell*: it maps generalized shell strains
``(eps_membrane, kappa)`` to generalized shell stress resultants
``(N, M)`` plus a tangent stiffness, integrating **nonlinear concrete**
through the thickness (Simpson sampling inside concrete zones) and
**smeared reinforcement sheets** at prescribed depths and orientations.
It is the direct implementation of what the Midas FEA NX *Define
Multi-layered Grid* / *Define Concrete Layer* dialogs describe.

Kinematics (shared with the shell elements)
-------------------------------------------
A shell reference/mid-surface carries a physical thickness. The in-plane
strain at through-thickness height ``z`` (measured from the reference
plane along the shell normal, ``+`` up) follows the Reissner-Mindlin /
Kirchhoff relation used everywhere in this package (see
``LayeredShellSection.ply_stresses`` and ``ShellMITC4.recover``)::

    eps(z) = eps_membrane + z * kappa

with ``eps_membrane = (eps_xx, eps_yy, gamma_xy)`` and
``kappa = (kappa_xx, kappa_yy, kappa_xy)``.

Constitutive model (documented assumption)
------------------------------------------
Concrete is evaluated with **one uniaxial law per in-plane direction**
(independent states along local x and local y) plus an elastic in-plane
shear modulus -- the classic *uniaxial layered* RC-shell model. This is
exactly the picture in the accompanying guide ("concrete is evaluated at
several points through the thickness ... the concrete material law
converts strain into stress") and is what makes the 1-D strip benchmark
hand-verifiable and identical to a 1 m-wide RC beam fibre section.

It is **not** a fully biaxial (total-strain rotating/fixed-crack)
plane-stress concrete -- that is what Midas FEA NX can additionally use
for strongly 2-D nonlinear runs. The section is deliberately structured
so a biaxial point-constitutive can be dropped in later at
``_concrete_point_response`` without touching the through-thickness
integration or the element interface. Reinforcement is always uniaxial
along the bar direction (steel carries no transverse or dowel action
here).

Resultant / strain ordering
---------------------------
::

    eps_membrane = (eps_xx, eps_yy, gamma_xy)
    kappa        = (kappa_xx, kappa_yy, kappa_xy)
    N            = (N_xx, N_yy, N_xy)     force  per unit width  [F/L]
    M            = (M_xx, M_yy, M_xy)     moment per unit width  [F]

and the tangent decomposes into the standard laminate matrices

    N = A eps_membrane + B kappa
    M = B eps_membrane + D kappa

with ``A = int Q dz``, ``B = int Q z dz``, ``D = int Q z^2 dz`` (concrete
Simpson-integrated, reinforcement added as discrete rank-1 sheets).

Drop-in compatibility
----------------------
:class:`ReinforcedConcreteShellSection` implements the
:class:`~femsolver.shell_sections.base.ShellSectionBase` contract, so it
can be handed straight to the existing linear shell elements
(``ShellMITC4(section=...)`` etc.). In that role the four ``D_*`` methods
return the **tangent evaluated at the committed strain state** -- the
gross-elastic (uncracked, transformed) stiffness when unstrained, which
is the correct initial shell stiffness including the reinforcement.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from femsolver.materials.uniaxial.base import UniaxialMaterial
from femsolver.shell_sections.base import ShellSectionBase


# ============================================================ Simpson rule

def _simpson_points(z_bot: float, z_top: float, n_points: int,
                    n_div: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite-Simpson sample heights and weights over ``[z_bot, z_top]``.

    The zone is first split into ``n_div`` equal cells (the Midas
    "Division"). Each cell is integrated with an ``n_points``-point
    Simpson rule (``n_points`` odd, ``>= 3``). Points are returned per
    cell (cell endpoints are duplicated between adjacent cells, each with
    an independent material state) so the weights stay the elementary
    Simpson weights and every sample is a genuine material-evaluation
    location.

    Returns
    -------
    z : (n_div * n_points,) array of sample heights.
    w : (n_div * n_points,) array of integration weights (length units);
        ``sum(w) == z_top - z_bot`` to machine precision.
    """
    if n_points < 3 or n_points % 2 == 0:
        raise ValueError(
            f"Simpson integration needs an odd number of points >= 3, "
            f"got {n_points}"
        )
    if n_div < 1:
        raise ValueError(f"n_div must be >= 1, got {n_div}")
    if z_top <= z_bot:
        raise ValueError(
            f"zone must have z_top > z_bot, got z_bot={z_bot}, z_top={z_top}"
        )
    # Elementary Simpson weights for n_points on a unit cell length.
    coeff = np.ones(n_points)
    coeff[1:-1:2] = 4.0
    coeff[2:-1:2] = 2.0
    cell_h = (z_top - z_bot) / n_div
    step = cell_h / (n_points - 1)
    base_w = coeff * step / 3.0                      # sums to cell_h
    zs: list[float] = []
    ws: list[float] = []
    for c in range(n_div):
        a = z_bot + c * cell_h
        for j in range(n_points):
            zs.append(a + j * step)
            ws.append(base_w[j])
    return np.asarray(zs), np.asarray(ws)


# ============================================================ layer specs

@dataclass
class ConcreteZone:
    """A through-thickness concrete zone (Midas "Concrete Layer").

    Parameters
    ----------
    z_bot, z_top : float
        Zone extents measured from the section reference plane along the
        shell normal (``+`` up). For a section centred on the mid-surface
        the full thickness spans ``[-t/2, +t/2]``.
    material : UniaxialMaterial
        Concrete constitutive law (e.g. :class:`ConcreteKentPark`,
        :class:`ConcreteParabolaRectangle`, optionally wrapped in
        :class:`ConcreteTensionStiffening`). Cloned per sample point and
        per in-plane direction so each carries independent history.
    n_simpson : int, default 3
        Simpson integration points through this zone's thickness (odd,
        ``>= 3``) -- the Midas "Simpson Integration Points".
    n_div : int, default 1
        Additional through-thickness subdivisions of the zone -- the
        Midas "Division". The zone is split into ``n_div`` equal cells,
        each Simpson-integrated.
    """
    z_bot: float
    z_top: float
    material: UniaxialMaterial
    n_simpson: int = 3
    n_div: int = 1


@dataclass
class RebarLayer:
    """A smeared reinforcement sheet (one row of the Midas grid).

    Reinforcement is represented as an equivalent steel area *per unit
    width* smeared into a thin sheet at a fixed depth and orientation --
    exactly the "Rebar Area / Spacing / Orientation Angle" input. The
    sheet follows the shell strain at its depth (perfect bond) and
    carries uniaxial stress along the bar direction only.

    Parameters
    ----------
    z : float
        Height of the reinforcement centroid from the reference plane
        (``+`` up). This is the *bar-centroid* location -- not the clear
        cover to the outside of the bar.
    area_per_width : float
        Steel area per unit width, ``A_s`` in [length^2 / length] (i.e.
        a length). For bars of area ``a`` at spacing ``s`` this is
        ``a / s`` (see :meth:`from_bars`).
    theta_deg : float, default 0.0
        Bar orientation, degrees CCW from local x (``0`` = local x /
        Grid X, ``90`` = local y / Grid Y).
    material : UniaxialMaterial
        Reinforcing-steel constitutive law (e.g.
        :class:`UniaxialReinforcingSteel`, :class:`UniaxialBilinear`,
        :class:`UniaxialMenegottoPinto`). Cloned so each sheet carries
        independent history.
    """
    z: float
    area_per_width: float
    material: UniaxialMaterial
    theta_deg: float = 0.0

    @classmethod
    def from_bars(cls, z: float, bar_area: float, spacing: float,
                  material: UniaxialMaterial, *,
                  theta_deg: float = 0.0) -> "RebarLayer":
        """Build a sheet from a discrete bar area and spacing.

        ``area_per_width = bar_area / spacing``. Example (guide values):
        ``bar_area = 78 mm^2`` at ``spacing = 100 mm`` gives
        ``0.78 mm^2/mm = 780 mm^2/m``.
        """
        if bar_area <= 0.0:
            raise ValueError(f"bar_area must be positive, got {bar_area}")
        if spacing <= 0.0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        return cls(z=z, area_per_width=bar_area / spacing,
                   material=material, theta_deg=theta_deg)


# ============================================================ response bundle

@dataclass
class ShellSectionResponse:
    """Full nonlinear through-thickness response at a strain state."""
    N: np.ndarray                       # (3,) membrane resultants  (F/L)
    M: np.ndarray                       # (3,) moment resultants     (F)
    Q: np.ndarray                       # (2,) transverse shear      (F/L)
    A: np.ndarray                       # (3, 3) membrane tangent
    B: np.ndarray                       # (3, 3) coupling tangent
    D: np.ndarray                       # (3, 3) bending tangent
    Ds: np.ndarray                      # (2, 2) transverse-shear tangent
    concrete_states: list[dict] = field(default_factory=list)
    rebar_states: list[dict] = field(default_factory=list)


# ============================================================ internal points

class _ConcretePoint:
    """One through-thickness concrete sample: height, weight, and the two
    independent uniaxial states (local x and local y)."""

    __slots__ = ("z", "w", "mat_x", "mat_y", "zone")

    def __init__(self, z: float, w: float, mat_x: UniaxialMaterial,
                 mat_y: UniaxialMaterial, zone: int):
        self.z = z
        self.w = w
        self.mat_x = mat_x
        self.mat_y = mat_y
        self.zone = zone


class _RebarPoint:
    """One smeared steel sheet expanded for evaluation."""

    __slots__ = ("z", "As", "c", "s", "m", "theta_deg", "material")

    def __init__(self, z: float, As: float, theta_deg: float,
                 material: UniaxialMaterial):
        self.z = z
        self.As = As
        self.theta_deg = theta_deg
        self.material = material
        th = math.radians(theta_deg)
        self.c = math.cos(th)
        self.s = math.sin(th)
        # Voigt projection m: eps_theta = m . (eps_xx, eps_yy, gamma_xy)
        self.m = np.array([self.c * self.c, self.s * self.s, self.c * self.s])


# ============================================================ the section

class ReinforcedConcreteShellSection(ShellSectionBase):
    """Multi-layered reinforced-concrete shell section.

    Parameters
    ----------
    thickness : float
        Total physical shell thickness.
    concrete_zones : sequence of :class:`ConcreteZone`
        Through-thickness concrete zones. Must cover the section without
        gaps or overlaps is *not* enforced (zones may be defined however
        the modeller likes), but typically they tile the full thickness.
    rebar_layers : sequence of :class:`RebarLayer`, optional
        Smeared reinforcement sheets. Default: none (plain concrete).
    nu_concrete : float, default 0.2
        Concrete Poisson ratio, used only to derive the shear modulus
        ``G = E0 / (2 (1 + nu))`` for the (elastic) in-plane and
        transverse shear response. In-plane normal Poisson coupling is
        **neglected** by the decoupled uniaxial constitutive (see module
        docstring).
    Ec : float, optional
        Concrete elastic modulus used for the shear moduli. Defaults to
        the ``E0`` of the first zone's material.
    k_shear : float, default 5/6
        Transverse-shear correction factor.
    rho_concrete : float, default 0.0
        Concrete mass density (per volume). The section ``density`` (mass
        per area) is ``rho_concrete * thickness``. Reinforcement mass is
        neglected.
    z_ref : float, default 0.0
        Reference-plane offset. Zone/rebar heights are taken relative to
        this. The default places the reference plane at ``z = 0``.
    """

    def __init__(
        self,
        thickness: float,
        concrete_zones: Sequence[ConcreteZone],
        rebar_layers: Sequence[RebarLayer] = (),
        *,
        nu_concrete: float = 0.2,
        Ec: Optional[float] = None,
        k_shear: float = 5.0 / 6.0,
        rho_concrete: float = 0.0,
        z_ref: float = 0.0,
    ):
        if thickness <= 0.0:
            raise ValueError(f"thickness must be positive, got {thickness}")
        if len(concrete_zones) == 0:
            raise ValueError("need at least one concrete zone")
        if not (0.0 < k_shear <= 1.0):
            raise ValueError(f"k_shear must be in (0, 1], got {k_shear}")
        if not (0.0 <= nu_concrete < 0.5):
            raise ValueError(
                f"nu_concrete must be in [0, 0.5), got {nu_concrete}"
            )
        self._thickness = float(thickness)
        self.k_shear = float(k_shear)
        self.nu_concrete = float(nu_concrete)
        self.rho_concrete = float(rho_concrete)
        self.z_ref = float(z_ref)

        # ----- expand concrete zones into independent sample points -----
        self._cpoints: list[_ConcretePoint] = []
        for iz, zone in enumerate(concrete_zones):
            zs, ws = _simpson_points(
                zone.z_bot - self.z_ref, zone.z_top - self.z_ref,
                zone.n_simpson, zone.n_div,
            )
            for z, w in zip(zs, ws):
                self._cpoints.append(_ConcretePoint(
                    z=float(z), w=float(w),
                    mat_x=zone.material.clone(),
                    mat_y=zone.material.clone(),
                    zone=iz,
                ))
        # ----- expand reinforcement sheets -----
        self._rpoints: list[_RebarPoint] = []
        for layer in rebar_layers:
            if layer.area_per_width <= 0.0:
                raise ValueError(
                    f"rebar area_per_width must be positive, got "
                    f"{layer.area_per_width}"
                )
            self._rpoints.append(_RebarPoint(
                z=float(layer.z) - self.z_ref,
                As=float(layer.area_per_width),
                theta_deg=float(layer.theta_deg),
                material=layer.material.clone(),
            ))

        # Concrete shear modulus (elastic). E0 from the first zone.
        E0 = Ec if Ec is not None else getattr(
            concrete_zones[0].material, "E0", None
        )
        if E0 is None:
            _, E0 = concrete_zones[0].material.get_response(0.0)
        self._Ec = float(E0)
        self._Gc = self._Ec / (2.0 * (1.0 + self.nu_concrete))

        # committed generalized strain (drives the drop-in D_* tangents)
        self._eps_committed = np.zeros(6)
        self._eps_trial = np.zeros(6)

    # --------------------------------------------------- ShellSectionBase
    @property
    def thickness(self) -> float:
        return self._thickness

    @property
    def density(self) -> float:
        return self.rho_concrete * self._thickness

    def D_membrane(self) -> np.ndarray:
        return self._assemble(self._eps_committed)[0]

    def D_coupling(self) -> np.ndarray:
        return self._assemble(self._eps_committed)[1]

    def D_bending(self) -> np.ndarray:
        return self._assemble(self._eps_committed)[2]

    def D_shear(self) -> np.ndarray:
        return self.k_shear * self._Gc * self._thickness * np.eye(2)

    # --------------------------------------------------- core assembly
    def _assemble(self, eps_gen: np.ndarray, *, want_stress: bool = False):
        """Through-thickness integration at generalized strain ``eps_gen``
        (6,) = ``(eps_xx, eps_yy, gamma_xy, kappa_xx, kappa_yy, kappa_xy)``.

        Returns ``(A, B, D)`` tangent blocks (each 3x3). When
        ``want_stress`` is true also returns ``(N, M, cstates, rstates)``.
        Updates each material's *trial* state as a side effect.
        """
        eps_m = eps_gen[:3]
        kappa = eps_gen[3:]
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        N = np.zeros(3)
        M = np.zeros(3)
        cstates: list[dict] = []
        rstates: list[dict] = []

        # ---- concrete Simpson points ----
        for p in self._cpoints:
            z = p.z
            ex = eps_m[0] + z * kappa[0]
            ey = eps_m[1] + z * kappa[1]
            gxy = eps_m[2] + z * kappa[2]
            sx, Etx = p.mat_x.get_response(ex)
            sy, Ety = p.mat_y.get_response(ey)
            sxy = self._Gc * gxy
            w = p.w
            # tangent diagonal (decoupled): (xx, yy, xy)
            dA = np.array([Etx, Ety, self._Gc]) * w
            A[0, 0] += dA[0]; A[1, 1] += dA[1]; A[2, 2] += dA[2]
            B[0, 0] += dA[0] * z; B[1, 1] += dA[1] * z; B[2, 2] += dA[2] * z
            D[0, 0] += dA[0] * z * z; D[1, 1] += dA[1] * z * z
            D[2, 2] += dA[2] * z * z
            if want_stress:
                sig = np.array([sx, sy, sxy])
                N += sig * w
                M += sig * w * z
                cstates.append({
                    "zone": p.zone, "z": z, "w": w,
                    "eps_xx": ex, "eps_yy": ey, "gamma_xy": gxy,
                    "sig_xx": sx, "sig_yy": sy, "sig_xy": sxy,
                    "cracked_xx": ex > 0.0 and abs(sx) < 1e-9,
                    "cracked_yy": ey > 0.0 and abs(sy) < 1e-9,
                })

        # ---- smeared reinforcement sheets ----
        for r in self._rpoints:
            z = r.z
            m = r.m
            eps_theta = float(m @ eps_m) + z * float(m @ kappa)
            sig_b, Ets = r.material.get_response(eps_theta)
            EA = Ets * r.As
            mmT = np.outer(m, m)
            A += EA * mmT
            B += EA * z * mmT
            D += EA * z * z * mmT
            if want_stress:
                f = r.As * sig_b
                N += f * m
                M += f * z * m
                eps_y = getattr(r.material, "eps_y", None)
                yielded = (eps_y is not None
                           and abs(eps_theta) >= eps_y - 1e-12)
                rstates.append({
                    "z": z, "As": r.As, "theta_deg": r.theta_deg,
                    "eps": eps_theta, "sigma": sig_b, "yielded": yielded,
                })

        if want_stress:
            return A, B, D, N, M, cstates, rstates
        return A, B, D

    # --------------------------------------------------- rich response
    def section_response(self, eps_membrane, kappa,
                         gamma_transverse=None) -> ShellSectionResponse:
        """Full nonlinear response at ``(eps_membrane, kappa)``.

        Parameters
        ----------
        eps_membrane : (3,) sequence -- ``(eps_xx, eps_yy, gamma_xy)``.
        kappa : (3,) sequence -- ``(kappa_xx, kappa_yy, kappa_xy)``.
        gamma_transverse : (2,) sequence, optional -- ``(gamma_xz,
            gamma_yz)`` for the transverse-shear resultant (elastic).

        Notes
        -----
        Updates trial material state. Call :meth:`commit_state` to accept
        the step or :meth:`revert_state` to discard it.
        """
        eps_m = np.asarray(eps_membrane, dtype=float).reshape(3)
        kap = np.asarray(kappa, dtype=float).reshape(3)
        eps_gen = np.concatenate([eps_m, kap])
        self._eps_trial = eps_gen.copy()
        A, B, D, N, M, cst, rst = self._assemble(eps_gen, want_stress=True)
        Ds = self.D_shear()
        if gamma_transverse is None:
            gamma = np.zeros(2)
        else:
            gamma = np.asarray(gamma_transverse, dtype=float).reshape(2)
        Q = Ds @ gamma
        return ShellSectionResponse(
            N=N, M=M, Q=Q, A=A, B=B, D=D, Ds=Ds,
            concrete_states=cst, rebar_states=rst,
        )

    def get_response(self, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """SectionBase-style interface: generalized strain ``e`` (6,) ->
        ``(s, ks)`` with ``s = (N, M)`` (6,) and ``ks`` the 6x6 tangent

            ks = [[A, B], [B^T, D]].

        Provided for a future nonlinear shell element / generic drivers;
        the transverse-shear block is kept separate (see
        :meth:`section_response`).
        """
        e = np.asarray(e, dtype=float).reshape(6)
        self._eps_trial = e.copy()
        A, B, D, N, M, _, _ = self._assemble(e, want_stress=True)
        s = np.concatenate([N, M])
        ks = np.zeros((6, 6))
        ks[:3, :3] = A
        ks[:3, 3:] = B
        ks[3:, :3] = B.T
        ks[3:, 3:] = D
        return s, ks

    # --------------------------------------------------- lifecycle
    def commit_state(self) -> None:
        for p in self._cpoints:
            p.mat_x.commit_state()
            p.mat_y.commit_state()
        for r in self._rpoints:
            r.material.commit_state()
        self._eps_committed = self._eps_trial.copy()

    def revert_state(self) -> None:
        for p in self._cpoints:
            p.mat_x.revert_state()
            p.mat_y.revert_state()
        for r in self._rpoints:
            r.material.revert_state()
        self._eps_trial = self._eps_committed.copy()

    def clone(self) -> "ReinforcedConcreteShellSection":
        """Deep copy with independent material state (one per integration
        point of a shell element)."""
        return copy.deepcopy(self)

    # --------------------------------------------------- introspection
    @property
    def n_concrete_points(self) -> int:
        return len(self._cpoints)

    @property
    def n_rebar_layers(self) -> int:
        return len(self._rpoints)

    @property
    def steel_area_per_width(self) -> float:
        """Total smeared steel area per unit width (all sheets)."""
        return float(sum(r.As for r in self._rpoints))

    def strain_at(self, eps_membrane, kappa, z: float) -> tuple:
        """Return ``(eps_xx, eps_yy, gamma_xy)`` at height ``z`` for the
        given generalized strain -- the kinematics ``eps(z) = eps_m + z
        kappa``. Pure geometry; no material evaluation."""
        eps_m = np.asarray(eps_membrane, dtype=float).reshape(3)
        kap = np.asarray(kappa, dtype=float).reshape(3)
        zz = z - self.z_ref
        return tuple(eps_m + zz * kap)

    def __repr__(self) -> str:
        return (
            f"ReinforcedConcreteShellSection(t={self._thickness:g}, "
            f"n_concrete_pts={self.n_concrete_points}, "
            f"n_rebar={self.n_rebar_layers}, "
            f"As_tot/width={self.steel_area_per_width:g})"
        )

    # --------------------------------------------------- convenience
    @classmethod
    def from_midas_grid(
        cls,
        thickness: float,
        concrete_material: UniaxialMaterial,
        rebar_layers: Sequence[RebarLayer] = (),
        *,
        n_simpson: int = 3,
        n_div: int = 4,
        nu_concrete: float = 0.2,
        Ec: Optional[float] = None,
        k_shear: float = 5.0 / 6.0,
        rho_concrete: float = 0.0,
    ) -> "ReinforcedConcreteShellSection":
        """Build a section the way the Midas dialog does: a single
        concrete material spanning the full thickness (centred on the
        mid-surface), sampled with ``n_div`` divisions x ``n_simpson``
        Simpson points, plus the given smeared reinforcement sheets.

        Reinforcement ``z`` values are offsets from the mid-surface, e.g.
        ``+100`` / ``-100`` mm for the guide's 300 mm section.
        """
        zone = ConcreteZone(
            z_bot=-0.5 * thickness, z_top=+0.5 * thickness,
            material=concrete_material, n_simpson=n_simpson, n_div=n_div,
        )
        return cls(
            thickness, [zone], rebar_layers,
            nu_concrete=nu_concrete, Ec=Ec, k_shear=k_shear,
            rho_concrete=rho_concrete,
        )


# ============================================================ M-phi driver

@dataclass
class ShellMomentCurvaturePoint:
    """One point on a shell-strip moment-curvature curve."""
    kappa: float                # curvature about the bending axis  [1/L]
    M: float                    # moment resultant per unit width    [F]
    N: float                    # membrane resultant per unit width  [F/L]
    eps_ref: float              # membrane strain at reference plane
    eps_conc_min: float         # most-compressive concrete strain (<= 0)
    eps_steel_max: float        # max tensile reinforcement strain
    converged: bool = True


@dataclass
class ShellMomentCurvatureResult:
    """Moment-curvature response of a shell strip in one bending
    direction, plus derived cracking / yield / ultimate quantities.

    ``M`` and ``N`` are **per unit width** (``F`` and ``F/L``): multiply
    by the strip width to get total actions.
    """
    points: list[ShellMomentCurvaturePoint]
    direction: str
    N_target: float
    M_cr: Optional[float] = None
    kappa_cr: Optional[float] = None
    M_y: Optional[float] = None
    kappa_y: Optional[float] = None
    M_u: Optional[float] = None
    kappa_u: Optional[float] = None
    failure_mode: str = ""

    @property
    def kappa_array(self) -> np.ndarray:
        return np.array([p.kappa for p in self.points])

    @property
    def M_array(self) -> np.ndarray:
        return np.array([p.M for p in self.points])


def shell_moment_curvature(
    section: ReinforcedConcreteShellSection,
    *,
    direction: str = "x",
    N_target: float = 0.0,
    kappa_max: float = 0.05,
    n_steps: int = 60,
    kappas: Optional[np.ndarray] = None,
    eps_cu_crush: float = 0.003,
    eps_steel_rupture: float = 0.05,
    max_newton: int = 60,
) -> ShellMomentCurvatureResult:
    """Sweep curvature at fixed membrane force to get the ``M(kappa)``
    response of the layered RC shell **in one bending direction** (a
    uniaxial-stress strip, ``N`` in the transverse direction left free).

    This is the guide's recommended benchmark: a strip bent about one
    axis, with the through-thickness strain ``eps(z) = eps_ref + z
    kappa``. Because the concrete constitutive is decoupled per
    direction, the transverse direction stays inactive and the result is
    directly comparable to a unit-width RC beam fibre section.

    Parameters
    ----------
    section : ReinforcedConcreteShellSection
        Cloned internally so the caller's material state is untouched.
    direction : {"x", "y"}
        Bending direction. ``"x"`` sweeps ``kappa_xx`` and reports
        ``M_xx`` / ``N_xx``; ``"y"`` uses the local-y components.
    N_target : float
        Membrane force per unit width held constant in the bending
        direction (default 0 -> pure bending). Tension positive.
    kappa_max, n_steps, kappas : curvature sweep controls.
    eps_cu_crush : float
        Concrete crushing strain magnitude; the sweep stops when the
        extreme compression sample reaches it.
    eps_steel_rupture : float
        Reinforcement rupture strain; the sweep stops when any sheet
        reaches it in tension.
    """
    if direction not in ("x", "y"):
        raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")
    if kappa_max <= 0.0:
        raise ValueError(f"kappa_max must be positive, got {kappa_max}")
    idx = 0 if direction == "x" else 1
    sec = section.clone()

    if kappas is None:
        kappas = np.linspace(0.0, kappa_max, n_steps + 1)
    else:
        kappas = np.asarray(kappas, dtype=float)

    def eval_state(eps_ref: float, kappa: float):
        eps_m = np.zeros(3)
        kap = np.zeros(3)
        eps_m[idx] = eps_ref
        kap[idx] = kappa
        return sec.section_response(eps_m, kap)

    eps_ref = 0.0
    points: list[ShellMomentCurvaturePoint] = []
    M_y = kappa_y = None
    M_cr = kappa_cr = None
    failure_mode = ""
    k0_uncracked = None          # initial (uncracked) tangent dM/dkappa

    for kappa in kappas:
        converged = False
        for _ in range(max_newton):
            resp = eval_state(eps_ref, kappa)
            N_now = resp.N[idx]
            residual = N_now - N_target
            tol = max(1.0, abs(N_target) * 1e-8)
            A_ii = resp.A[idx, idx]
            if abs(residual) < tol:
                converged = True
                break
            if abs(A_ii) < 1.0:
                A_ii = 1.0e6
            eps_ref -= residual / A_ii
        resp = eval_state(eps_ref, kappa)
        M_now = float(resp.M[idx])
        N_now = float(resp.N[idx])

        # extreme fibre strains
        eps_conc_min = 0.0
        for cs in resp.concrete_states:
            e_dir = cs["eps_xx"] if idx == 0 else cs["eps_yy"]
            eps_conc_min = min(eps_conc_min, e_dir)
        eps_steel_max = 0.0
        for rs in resp.rebar_states:
            eps_steel_max = max(eps_steel_max, rs["eps"])

        points.append(ShellMomentCurvaturePoint(
            kappa=float(kappa), M=M_now, N=N_now, eps_ref=float(eps_ref),
            eps_conc_min=float(eps_conc_min),
            eps_steel_max=float(eps_steel_max), converged=converged,
        ))

        # cracking: incremental stiffness dM/dkappa drops below half the
        # initial (uncracked) tangent -- the flexural-stiffness knee.
        if len(points) >= 2:
            dk = points[-1].kappa - points[-2].kappa
            if dk > 0.0:
                dM_dk = (points[-1].M - points[-2].M) / dk
                if k0_uncracked is None:
                    k0_uncracked = dM_dk
                elif (M_cr is None and dM_dk < 0.5 * k0_uncracked):
                    M_cr = points[-2].M
                    kappa_cr = points[-2].kappa

        # first yield of any reinforcement sheet
        if M_y is None and any(rs["yielded"] for rs in resp.rebar_states):
            M_y = M_now
            kappa_y = float(kappa)

        sec.commit_state()

        if -eps_conc_min >= eps_cu_crush:
            failure_mode = "concrete_crushing"
            break
        if eps_steel_max >= eps_steel_rupture:
            failure_mode = "steel_rupture"
            break

    M_arr = np.array([p.M for p in points])
    if M_arr.size:
        idx_peak = int(np.argmax(np.abs(M_arr)))
        M_u = float(M_arr[idx_peak])
        kappa_u = float(points[idx_peak].kappa)
        if not failure_mode:
            failure_mode = ("kappa_max_reached"
                            if idx_peak == len(points) - 1 else "M_peak")

    return ShellMomentCurvatureResult(
        points=points, direction=direction, N_target=float(N_target),
        M_cr=M_cr, kappa_cr=kappa_cr, M_y=M_y, kappa_y=kappa_y,
        M_u=M_u, kappa_u=kappa_u, failure_mode=failure_mode,
    )
