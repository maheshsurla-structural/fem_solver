"""Moment-curvature analysis for any RC section (Phase II.12).

Driver that takes a unified :class:`~femsolver.sections.Section` with
reinforcement and a fully nonlinear constitutive (any
:class:`UniaxialMaterial` for concrete and steel), then sweeps
curvature ``kappa`` at a fixed axial load ``P_target`` to produce
the full ``M(kappa)`` curve.

Output features
---------------
* ``points`` -- (kappa, M, axial_strain, eps_top_concrete,
  eps_max_steel) at each curvature step
* ``M_cr`` -- optional cracking moment (requires modulus of rupture)
* ``M_y, kappa_y`` -- first-yield of the extreme tension steel
* ``M_u, kappa_u`` -- ultimate (concrete crushing OR M peaks then
  drops; whichever comes first)
* ``mu_phi = kappa_u / kappa_y`` -- curvature ductility
* ``bilinear()`` -- equal-energy bilinear idealization for use as a
  hinge backbone in pushover / capacity analysis

Algorithm
---------
For each prescribed curvature ``kappa`` in the sweep, Newton-iterate
the axial strain at the section reference axis ``epsilon_0`` until
the fiber-section axial force matches ``N_target`` (= -P_target under
ACI compression-positive convention). Each fiber's strain is then
``eps_f = epsilon_0 - y · kappa`` (Bernoulli plane sections), and the
moment ``M`` falls out as the by-product of the same fiber
integration: ``M = -sum(y · sigma · A)`` (sagging-positive).

This is the standard fiber-section moment-curvature method used in
SAP2000 Section Designer, MIDAS PSC Section, STAAD Section Wizard,
and the OpenSees ``MomentCurvature.tcl`` example.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from femsolver.materials.uniaxial.base import UniaxialMaterial
from femsolver.materials.uniaxial.prestressed import PrestressedUniaxial
from femsolver.sections.response.fiber import Fiber, FiberSection2D
from femsolver.sections.section import (
    Section,
    _discretize_polygon_to_fibers,
)


# ============================================================ data types

@dataclass
class MomentCurvaturePoint:
    """One point on the M-kappa curve."""
    kappa: float                # 1/m
    M: float                    # N·m (sagging positive)
    P: float                    # N (compression positive)
    axial_strain: float         # epsilon_0 at reference axis (tension positive)
    eps_top_concrete: float     # extreme compression fibre strain (negative)
    eps_max_steel: float        # max tensile strain in any rebar
    converged: bool = True      # Newton on axial converged?


@dataclass
class MomentCurvatureResult:
    """Full M-kappa curve plus derived design quantities."""
    points: list[MomentCurvaturePoint]
    P_target: float             # N (compression positive)

    M_cr: Optional[float] = None        # cracking moment (if f_rupture given)
    kappa_cr: Optional[float] = None
    M_y: Optional[float] = None         # first-yield moment
    kappa_y: Optional[float] = None
    M_u: Optional[float] = None         # ultimate moment
    kappa_u: Optional[float] = None
    mu_phi: Optional[float] = None      # curvature ductility = kappa_u/kappa_y
    failure_mode: str = ""              # "concrete_crushing" | "steel_rupture" | "M_peak" | ""

    @property
    def kappa_array(self) -> np.ndarray:
        return np.array([p.kappa for p in self.points])

    @property
    def M_array(self) -> np.ndarray:
        return np.array([p.M for p in self.points])

    # ------------------------------------------------------- bilinear
    def bilinear(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Equal-energy bilinear idealization of the M-kappa curve.

        Returns ((kappa_y_bi, M_y_bi), (kappa_u, M_u)). The yield
        point is chosen so that the area under the bilinear (0 ->
        yield -> ultimate) equals the area under the actual M-kappa
        curve up to ``kappa_u``. The post-yield branch is a straight
        line from yield to (kappa_u, M_u).
        """
        if self.M_u is None or self.kappa_u is None:
            raise ValueError("no ultimate point in result")
        if not self.points:
            raise ValueError("no points")
        kappas = self.kappa_array
        Ms = self.M_array
        # Truncate at ultimate
        mask = kappas <= self.kappa_u + 1e-12
        k = kappas[mask]
        M = Ms[mask]
        # Area under actual M-kappa
        area = float(np.trapezoid(M, k))
        # Bilinear with secant stiffness fixed by the actual yield
        # point: choose M_y_bi = M_y_actual, then solve kappa_y_bi
        # from equal-area condition (gives the well-known
        # idealization).
        if self.M_y is None or self.kappa_y is None:
            # Fall back to (M_u, kappa_u)
            return (self.kappa_u, self.M_u), (self.kappa_u, self.M_u)
        # Equal-area: 0.5 * M_y_bi * kappa_y_bi
        #             + 0.5 * (M_u + M_y_bi) * (kappa_u - kappa_y_bi) = area
        # Pick M_y_bi = M_u (idealized "equivalent" plastic plateau
        # passing through the actual ultimate moment level).
        M_y_bi = self.M_u
        denom = (M_y_bi + self.M_u) / 2.0
        if denom <= 0:
            return (self.kappa_y, self.M_y), (self.kappa_u, self.M_u)
        kappa_y_bi = (
            self.kappa_u - (area - 0.5 * M_y_bi * self.kappa_u) /
            (denom - 0.5 * M_y_bi)
        )
        if kappa_y_bi <= 0 or kappa_y_bi >= self.kappa_u:
            return (self.kappa_y, self.M_y), (self.kappa_u, self.M_u)
        return (kappa_y_bi, M_y_bi), (self.kappa_u, self.M_u)


# ============================================================ driver

def moment_curvature(
    section: Section,
    *,
    P_target: float = 0.0,
    concrete_uniaxial: UniaxialMaterial,
    steel_uniaxial: UniaxialMaterial,
    kappa_max: float = 0.10,
    n_steps: int = 60,
    n_z: int = 16,
    n_y: int = 40,
    f_y: float = 420e6,
    E_s: float = 200e9,
    f_rupture: Optional[float] = None,
    eps_cu_crush: float = 0.003,
    eps_steel_rupture: float = 0.05,
    P_convention: str = "compression",
) -> MomentCurvatureResult:
    """Sweep ``kappa`` at fixed axial load to produce the M-kappa curve.

    Parameters
    ----------
    section : Section
        Unified Section with reinforcement layout.
    P_target : float
        Axial load (N). Sign per ``P_convention`` (default
        compression-positive).
    concrete_uniaxial : UniaxialMaterial
        Constitutive law for concrete fibres (e.g.
        :class:`ConcreteKentPark`, :class:`ConcreteMander`).
    steel_uniaxial : UniaxialMaterial
        Constitutive law for rebar (e.g.
        :class:`UniaxialBilinear`, :class:`UniaxialMenegottoPinto`).
    kappa_max : float
        Maximum curvature to sweep to (1/m).
    n_steps : int
        Number of curvature increments.
    n_z, n_y : int
        Polygon grid resolution for concrete fibres.
    f_y, E_s : float
        Steel yield (Pa) and modulus -- used to detect first yield.
    f_rupture : float, optional
        Modulus of rupture for concrete tension (Pa, positive). If
        given, the cracking moment is computed from the uncracked
        elastic section.
    eps_cu_crush : float
        Concrete crushing strain magnitude. Curve terminates when
        extreme compression fibre exceeds this in compression.
    eps_steel_rupture : float
        Steel rupture strain. Curve terminates if any rebar exceeds
        this in tension.
    P_convention : {"compression", "tension"}
        Sign convention for ``P_target`` (default "compression").
    """
    has_rebar = (section.reinforcement is not None
                  and section.reinforcement.bars)
    has_prestress = (section.prestress is not None
                     and section.prestress.tendons)
    if not has_rebar and not has_prestress:
        raise ValueError(
            "moment-curvature needs reinforcement and/or prestress"
        )
    if kappa_max <= 0:
        raise ValueError(f"kappa_max must be positive, got {kappa_max}")
    if P_convention not in ("compression", "tension"):
        raise ValueError(
            f"P_convention must be 'compression' or 'tension', "
            f"got {P_convention!r}"
        )

    # tension-positive N for the fiber section
    N_target = -P_target if P_convention == "compression" else P_target

    # Build fiber section explicitly so we control concrete vs steel
    # vs prestress material assignment.
    polygon = section.geometry.polygon
    concrete_fibers = _discretize_polygon_to_fibers(
        polygon, concrete_uniaxial, n_z=n_z, n_y=n_y,
    )
    rebar_fibers = []
    if has_rebar:
        for bar in section.reinforcement.bars:
            rebar_fibers.append(Fiber(
                y=bar.y, z=bar.z, area=bar.area,
                material=steel_uniaxial.clone(),
            ))
    tendon_fibers = []
    if has_prestress:
        for tendon in section.prestress.tendons:
            if not tendon.bonded:
                # Unbonded tendons don't follow section strain
                # compatibility; skip with a note in the docstring.
                continue
            if tendon.material is None:
                raise ValueError(
                    f"tendon at (z={tendon.z}, y={tendon.y}) has no "
                    f"material; cannot build prestressed fiber"
                )
            # Determine E_p from initial tangent of strand material
            _, E_p = tendon.material.get_response(0.0)
            if E_p <= 0:
                E_p = 195e9     # AASHTO default for Grade 270 strand
            eps_pe = tendon.f_pe / E_p
            wrapped = PrestressedUniaxial(tendon.material.clone(), eps_pe)
            tendon_fibers.append(Fiber(
                y=tendon.y, z=tendon.z, area=tendon.area,
                material=wrapped,
            ))
    fs = FiberSection2D(concrete_fibers + rebar_fibers + tendon_fibers)

    # Geometry extremes for strain extraction
    minz, miny, maxz, maxy = polygon.bounds
    y_top = maxy   # extreme +y fibre -> most compressed under positive kappa

    # Cracking moment via elastic uncracked section
    # PSC formula: M_cr = (P_pe/A + P_pe*e*y_t/I + P_ext/A + f_r) * I/y_t
    # The prestress contributes both axial compression (P_pe/A_g) and a
    # bending precompression at the tension face from its eccentricity
    # (P_pe*|e|*y_t/I).
    M_cr = None
    kappa_cr = None
    if f_rupture is not None and f_rupture > 0:
        A_g = section.geometry.area
        I_g = section.geometry.I_zz
        y_tens = abs(miny)         # bottom fibre distance from centroid
        P_comp = P_target if P_convention == "compression" else -P_target
        sigma_from_P_ext = P_comp / max(A_g, 1e-30)
        # Prestress contribution
        sigma_from_prestress = 0.0
        if has_prestress:
            P_pe_total = 0.0
            P_pe_moment = 0.0  # = sum(A_p * f_pe * |y_p|), tendons below assumed
            for tendon in section.prestress.tendons:
                if not tendon.bonded:
                    continue
                P_pe_total += tendon.area * tendon.f_pe
                # Eccentricity: tendons below centroid (y_p < 0) put
                # compression at the bottom face via the equivalent
                # hogging moment from prestress. Use the absolute value
                # for the favourable contribution at the cracking face.
                P_pe_moment += tendon.area * tendon.f_pe * abs(tendon.y)
            # sigma at tension face from prestress (compression-positive)
            sigma_from_prestress = (
                P_pe_total / max(A_g, 1e-30)
                + P_pe_moment * y_tens / max(I_g, 1e-30)
            )
        M_cr = (
            (f_rupture + sigma_from_P_ext + sigma_from_prestress)
            * I_g / max(y_tens, 1e-30)
        )
        # Estimate kappa_cr from elastic stiffness
        if hasattr(concrete_uniaxial, "E0"):
            E_c = float(concrete_uniaxial.E0)
        else:
            _, Et0 = concrete_uniaxial.get_response(0.0)
            E_c = max(Et0, 20e9)
        kappa_cr = M_cr / (E_c * I_g)

    # Sweep kappa
    eps_y_steel = f_y / E_s
    eps_0 = 0.0
    M_y = None
    kappa_y = None
    failure_mode = ""

    kappas = np.linspace(0.0, kappa_max, n_steps + 1)
    points: list[MomentCurvaturePoint] = []

    for kappa in kappas:
        # Newton on eps_0 to satisfy axial: N(eps_0, kappa) == N_target
        converged = False
        for _ in range(60):
            s, ks = fs.get_response(np.array([eps_0, kappa]))
            N_now = s[0]
            residual = N_now - N_target
            tol = max(1.0, abs(N_target) * 1e-8, 100.0)
            if abs(residual) < tol:
                converged = True
                break
            dN_de = ks[0, 0]
            if abs(dN_de) < 1e3:
                dN_de = 1e6   # stability floor
            eps_0 -= residual / dN_de

        # Evaluate at converged state
        s, ks = fs.get_response(np.array([eps_0, kappa]))
        N_now = s[0]
        M_now = s[1]
        P_now = (-N_now) if P_convention == "compression" else N_now

        eps_top = eps_0 - y_top * kappa   # most negative (compression)

        # Max tensile rebar strain (relative to concrete)
        eps_max_steel = -math.inf
        if has_rebar:
            for bar in section.reinforcement.bars:
                eps_bar = eps_0 - bar.y * kappa
                if eps_bar > eps_max_steel:
                    eps_max_steel = eps_bar
        if eps_max_steel == -math.inf:
            eps_max_steel = 0.0

        pt = MomentCurvaturePoint(
            kappa=float(kappa), M=float(M_now), P=float(P_now),
            axial_strain=float(eps_0),
            eps_top_concrete=float(eps_top),
            eps_max_steel=float(eps_max_steel),
            converged=converged,
        )
        points.append(pt)

        # Detect first yield
        if M_y is None and eps_max_steel >= eps_y_steel and kappa > 0:
            M_y = M_now
            kappa_y = float(kappa)

        # Commit fiber materials (Karsan-Jirsa cyclic stuff doesn't
        # matter for monotonic, but harmless)
        fs.commit_state()

        # Failure detection
        if -eps_top >= eps_cu_crush:
            failure_mode = "concrete_crushing"
            break
        if eps_max_steel >= eps_steel_rupture:
            failure_mode = "steel_rupture"
            break

    # Ultimate at peak M (if no clear failure mode triggered)
    M_arr = np.array([p.M for p in points])
    idx_peak = int(np.argmax(M_arr))
    M_u = float(M_arr[idx_peak])
    kappa_u = float(points[idx_peak].kappa)
    if not failure_mode:
        # Did we terminate at the peak then descend? Or just hit kappa_max?
        if idx_peak == len(points) - 1:
            failure_mode = "kappa_max_reached"
        else:
            failure_mode = "M_peak"

    mu_phi = None
    if kappa_y is not None and kappa_y > 0:
        mu_phi = kappa_u / kappa_y

    return MomentCurvatureResult(
        points=points,
        P_target=float(P_target),
        M_cr=M_cr, kappa_cr=kappa_cr,
        M_y=M_y, kappa_y=kappa_y,
        M_u=M_u, kappa_u=kappa_u,
        mu_phi=mu_phi,
        failure_mode=failure_mode,
    )
