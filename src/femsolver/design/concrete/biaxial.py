"""Biaxial P-Mz-My interaction surface for any RC section (Theme II.10).

Builds a 3-D nominal + design strength surface for a unified
:class:`~femsolver.sections.Section` with reinforcement, using
**analytical polygon clipping** of the Whitney compression block for
concrete plus discrete rebar fibers. Works for arbitrary polygon
shapes (rectangular, L, T, channel, hollow box, custom) -- not just
the rectangular Whitney-block formulation in
:mod:`femsolver.design.concrete.column`.

Algorithm
---------
For each neutral-axis angle ``theta`` (measured CCW from the section's
local z-axis) and each neutral-axis depth ``c`` (measured from the
extreme compression fiber along the perpendicular to the NA):

1. The rotated coordinate ``y' = -z·sin(theta) + y·cos(theta)``
   measures the signed distance perpendicular to the NA line;
   higher ``y'`` = farther into compression.

2. ``y'_max`` is the largest ``y'`` over the section -- the extreme
   compression fiber. ``y'_NA = y'_max - c`` defines the NA.

3. Whitney rectangular stress block (ACI 318-19 §22.2.2.4):
   ``a = beta_1·c`` (capped at section depth in the y' direction).
   The Whitney block region is ``{(z, y) : y' > y'_max - a}``, a
   half-plane perpendicular to the NA. The compression-block sub-
   polygon is computed by intersecting the section polygon with this
   half-plane using shapely. From it:
   - ``A_block`` = area of the intersection
   - ``(z_c, y_c)`` = centroid of the intersection
   The concrete contribution is then exactly:
   - ``F_c = 0.85·f_c'·A_block``
   - ``M_c,z = F_c·y_c``
   - ``M_c,y = -F_c·z_c`` (compression-positive convention)

4. Steel: each rebar contributes ``epsilon = epsilon_cu·(y' - y'_NA)/c``;
   ``f_s = clip(E_s·epsilon, -f_y, +f_y)``. Bars inside the Whitney
   block displace concrete; effective stress is ``f_s - 0.85·f_c'``.

5. ``phi`` from extreme tension steel strain per ACI Table 21.2.2.

Sweeping ``theta`` over 0..360° and ``c`` over the meaningful range
traces the full P-Mz-My surface.

Because the Whitney block is integrated **analytically** (not via a
fiber grid), the rectangular case matches the closed-form
:func:`~femsolver.design.concrete.column.column_interaction_point` to
within numerical round-off. For non-rectangular polygons, the same
exactness applies -- shapely computes the polygon-half-plane
intersection to machine precision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from femsolver.design.concrete.section import (
    EPSILON_CU,
    E_STEEL,
    beta_1_aci,
    phi_for_strain,
)
from femsolver.sections.geometry.base import Geometry
from femsolver.sections.section import Section


# ============================================================ stress block parameters

@dataclass(frozen=True)
class StressBlockParams:
    """Code-specific rectangular stress block + strain limits for
    biaxial P-M-M integration.

    All three design codes (ACI 318, EC2, IS 456) reduce to the same
    fiber integration algorithm with different constants:

    +-----------+---------------+------------------+-----------+--------------+
    | Code      | block depth   | block stress     | eps_cu    | f_yd         |
    +-----------+---------------+------------------+-----------+--------------+
    | ACI 318   | beta_1 * c    | 0.85 * f_c'      | 0.003     | f_y          |
    | EC2       | lambda * c    | eta*alpha_cc*    | 0.0035    | f_yk / 1.15  |
    |           | (=0.8 typ)    | f_ck/gamma_c     |           |              |
    | IS 456    | 0.84 * c      | 0.4286 * f_ck    | 0.0035    | f_y / 1.15   |
    +-----------+---------------+------------------+-----------+--------------+

    ACI applies phi separately via Table 21.2; EC2 and IS 456 build the
    partial safety factors gamma_c / gamma_s / gamma_m into the stress
    block itself, so ``apply_phi_table = False`` for them.
    """
    code: str
    beta_1: float
    sigma_block: float    # Pa, compression-positive
    eps_cu: float
    f_yd: float           # design yield for rebar (Pa)
    E_s: float
    apply_phi_table: bool   # ACI Table 21.2 phi interpolation
    spiral: bool = False


def aci_params(
    f_c_prime: float, f_y: float, *,
    E_s: float = E_STEEL, spiral: bool = False,
) -> StressBlockParams:
    """ACI 318-19 §22.2.2.4 Whitney block + Table 21.2.2 phi interp."""
    return StressBlockParams(
        code="ACI318",
        beta_1=beta_1_aci(f_c_prime),
        sigma_block=0.85 * f_c_prime,
        eps_cu=0.003,
        f_yd=f_y,
        E_s=E_s,
        apply_phi_table=True,
        spiral=spiral,
    )


def ec2_params(
    f_ck: float, f_yk: float, *,
    E_s: float = E_STEEL, alpha_cc: float = 0.85,
    gamma_c: float = 1.5, gamma_s: float = 1.15,
) -> StressBlockParams:
    """EC2 §3.1.7 rectangular stress block with partial safety
    factors built in (no separate phi).

    For ``f_ck <= 50 MPa``: eta = 1.0, lambda = 0.80, eps_cu3 = 0.0035.
    For higher strength, eta and lambda decrease per EC2 (3.21), (3.22).
    """
    if f_ck <= 50e6:
        eta = 1.0
        lam = 0.8
        eps_cu3 = 0.0035
    else:
        # EC2 high-strength reduction
        eta = max(0.8, 1.0 - (f_ck - 50e6) / 200e6)
        lam = max(0.7, 0.8 - (f_ck - 50e6) / 400e6)
        eps_cu3 = max(0.0026, 0.0026 + 0.035 * ((90e6 - f_ck) / 100e6) ** 4)
    sigma_block = eta * alpha_cc * f_ck / gamma_c
    return StressBlockParams(
        code="EC2",
        beta_1=lam,
        sigma_block=sigma_block,
        eps_cu=eps_cu3,
        f_yd=f_yk / gamma_s,
        E_s=E_s,
        apply_phi_table=False,
    )


def is456_params(
    f_ck: float, f_y: float, *,
    E_s: float = E_STEEL, gamma_m_steel: float = 1.15,
) -> StressBlockParams:
    """IS 456:2000 Annex G parabolic-rectangular curve as Whitney
    equivalent. Total concrete force = 0.36 * f_ck * b * x_u with
    centroid at 0.42 * x_u from the extreme compression fibre. The
    Whitney-equivalent rectangle has depth 0.84 * x_u and stress
    0.36 * f_ck / 0.84 = 0.4286 * f_ck. Steel design stress is
    f_yd = f_y / gamma_m_steel = 0.87 * f_y for gamma_m = 1.15.
    """
    return StressBlockParams(
        code="IS456",
        beta_1=0.84,
        sigma_block=0.36 * f_ck / 0.84,    # = 0.4286 * f_ck
        eps_cu=0.0035,
        f_yd=f_y / gamma_m_steel,
        E_s=E_s,
        apply_phi_table=False,
    )


# ============================================================ data types

@dataclass
class BiaxialPMMPoint:
    """One nominal point on the 3-D interaction surface."""
    theta_rad: float                # NA angle (CCW from z-axis)
    c: float                        # NA depth from extreme compression fibre (m)
    P_n: float                      # Pa·m^2 = N, compression-positive
    M_nz: float                     # N·m
    M_ny: float                     # N·m
    phi: float                      # ACI strength-reduction factor
    epsilon_t: float                # extreme tension steel strain (positive=tension)
    section_type: str               # "compression-controlled" / "transition" /
                                    # "tension-controlled" / "pure-compression-cap"

    @property
    def phi_P_n(self) -> float:
        return self.phi * self.P_n

    @property
    def phi_M_nz(self) -> float:
        return self.phi * self.M_nz

    @property
    def phi_M_ny(self) -> float:
        return self.phi * self.M_ny

    @property
    def M_n_resultant(self) -> float:
        """Magnitude of (M_nz, M_ny)."""
        return float(math.hypot(self.M_nz, self.M_ny))


@dataclass
class BiaxialPMMSurface:
    """The full biaxial nominal + design strength surface."""
    points: list[BiaxialPMMPoint]
    P_o: float                      # pure compression P_o
    P_n_max: float                  # capped pure compression (ACI 22.4.2.1)
    P_pure_tension: float           # negative number
    f_c_prime: float
    f_y: float
    spiral: bool
    n_angles: int
    n_depths: int

    # ----------------------------------------------------- slices
    def slice_at_theta(
        self, theta_rad: float, *, atol: float = 1e-4,
    ) -> list[BiaxialPMMPoint]:
        """All points at the given NA angle (within ``atol`` radians)."""
        return [
            p for p in self.points
            if abs(_angle_diff(p.theta_rad, theta_rad)) < atol
        ]

    def slice_uniaxial_z(self) -> list[BiaxialPMMPoint]:
        """P-Mz curve at theta=0 (NA horizontal -> bending about z)."""
        return self.slice_at_theta(0.0)

    def slice_uniaxial_y(self) -> list[BiaxialPMMPoint]:
        """P-My curve at theta=pi/2 (NA vertical -> bending about y)."""
        return self.slice_at_theta(math.pi / 2)

    # ----------------------------------------------------- utility
    def as_arrays(
        self, *, design: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (P, Mz, My) as flat numpy arrays.

        With ``design=True``, phi-reduced values."""
        if design:
            P = np.array([p.phi_P_n for p in self.points])
            Mz = np.array([p.phi_M_nz for p in self.points])
            My = np.array([p.phi_M_ny for p in self.points])
        else:
            P = np.array([p.P_n for p in self.points])
            Mz = np.array([p.M_nz for p in self.points])
            My = np.array([p.M_ny for p in self.points])
        return P, Mz, My

    # ----------------------------------------------------- plotting
    def plot_3d(self, *, design: bool = True, ax=None,
                  style: str = "radial"):
        """Render the surface in 3-D matplotlib axes.

        Parameters
        ----------
        design : bool
            ``True`` for phi-reduced design surface; ``False`` for nominal.
        ax : matplotlib axes, optional
            Existing 3-D axes to plot into.
        style : {"radial", "wireframe", "scatter"}
            * ``"radial"`` (default) -- one P-M curve per neutral-axis
              angle, drawn as separate lines. Cleanest visual; matches
              the way SAP2000 Section Designer / MIDAS Gen / STAAD
              show biaxial interaction surfaces.
            * ``"wireframe"`` -- full fishnet via matplotlib
              ``plot_wireframe``. Shows iso-c rings too but matplotlib
              doesn't hide back-side lines; can look busy.
            * ``"scatter"`` -- raw scatter of all (P, Mz, My) points.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D       # noqa: F401

        n_a = self.n_angles
        per_angle = len(self.points) // n_a
        P, Mz, My = self.as_arrays(design=design)
        P = P / 1e3        # kN
        Mz = Mz / 1e3      # kN.m
        My = My / 1e3

        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")

        if style == "scatter" or per_angle * n_a != len(self.points):
            ax.scatter(Mz, My, P, s=8, color="C0")
        elif style == "wireframe":
            P2 = P.reshape(n_a, per_angle)
            Mz2 = Mz.reshape(n_a, per_angle)
            My2 = My.reshape(n_a, per_angle)
            ax.plot_wireframe(Mz2, My2, P2, color="C0", linewidth=0.5)
        else:  # "radial"
            # One P-M curve per neutral-axis angle. This is the
            # standard rendering used by commercial section designers.
            P2 = P.reshape(n_a, per_angle)
            Mz2 = Mz.reshape(n_a, per_angle)
            My2 = My.reshape(n_a, per_angle)
            for i in range(n_a):
                ax.plot(Mz2[i], My2[i], P2[i],
                         color="C0", linewidth=0.8, alpha=0.7)

        kind = "design (phi)" if design else "nominal"
        ax.set_xlabel("M_z (kN·m)")
        ax.set_ylabel("M_y (kN·m)")
        ax.set_zlabel("P (kN, +compression)")
        ax.set_title(f"Biaxial P-M-M surface ({kind})")
        return ax

    def plot_slice(
        self, theta_rad: float, *, design: bool = True, ax=None,
    ):
        """Render a 2-D P-M slice at one NA angle."""
        import matplotlib.pyplot as plt
        pts = self.slice_at_theta(theta_rad)
        if not pts:
            raise ValueError(f"no points at theta={math.degrees(theta_rad):.1f} deg")
        M_resultant = np.array([
            (p.phi_M_nz if design else p.M_nz) *
            math.cos(theta_rad) +
            (p.phi_M_ny if design else p.M_ny) *
            math.sin(theta_rad)
            for p in pts
        ])
        P = np.array([(p.phi_P_n if design else p.P_n) for p in pts])
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(np.abs(M_resultant) / 1e3, P / 1e3, "o-", markersize=4)
        kind = "design" if design else "nominal"
        ax.set_xlabel("|M| (kN·m)")
        ax.set_ylabel("P (kN, +compression)")
        ax.set_title(
            f"P-M slice at theta = {math.degrees(theta_rad):.0f} deg ({kind})"
        )
        ax.grid(True, alpha=0.3)
        return ax


# ============================================================ analytical Whitney block

def _rebar_fibers(section: Section) -> list[tuple]:
    """Extract (z, y, area) for each rebar."""
    if not section.reinforcement or not section.reinforcement.bars:
        return []
    return [(b.z, b.y, b.area) for b in section.reinforcement.bars]


def _tendon_fibers(section: Section) -> list[tuple]:
    """Extract (z, y, area, f_pe, E_p, f_pu) for each bonded tendon.

    Each tendon enters the integration as a discrete fiber with an
    initial pre-strain ``eps_pe = f_pe / E_p``. The strand stress is
    ``f_pt = clip(E_p * (eps_pe + eps_concrete), -f_pu, +f_pu)``,
    matching the PrestressedUniaxial wrapper used in moment-curvature.
    """
    if not section.prestress or not section.prestress.tendons:
        return []
    out = []
    for t in section.prestress.tendons:
        if not t.bonded:
            continue
        if t.material is None:
            E_p = 195e9
            f_pu = 1860e6
        else:
            _, E_p = t.material.get_response(0.0)
            if E_p <= 0:
                E_p = 195e9
            # Probe the material at a strain past yield to estimate f_pu
            f_pu = abs(t.material.get_response(0.02)[0])
            if f_pu <= 0:
                f_pu = 1860e6
        out.append((t.z, t.y, t.area, t.f_pe, E_p, f_pu))
    return out


def _whitney_half_plane(theta_rad: float, y_block_lower: float,
                          bounds: tuple) -> "Polygon":
    """Build the half-plane ``y' > y_block_lower`` as a large polygon
    in the original (z, y) frame, sized to cover the section.

    The half-plane in the rotated frame is the rectangle
    z' in [-big, big], y' in [y_block_lower, big]. Mapping back:
        z = z' cos(theta) - y' sin(theta)
        y = z' sin(theta) + y' cos(theta)
    """
    from shapely.geometry import Polygon
    minz, miny, maxz, maxy = bounds
    span = max(maxz - minz, maxy - miny)
    big = 10.0 * (span + abs(y_block_lower) + 1.0)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    def _xy(zp, yp):
        return (zp * cos_t - yp * sin_t, zp * sin_t + yp * cos_t)

    corners = [
        _xy(-big, y_block_lower),
        _xy(+big, y_block_lower),
        _xy(+big, +big),
        _xy(-big, +big),
    ]
    return Polygon(corners)


def _whitney_block_contribution(
    polygon, theta_rad: float, y_block_lower: float,
    sigma_block: float,
) -> tuple[float, float, float]:
    """Analytical Whitney-block contribution to (P, M_z, M_y) using
    polygon-half-plane intersection.

    Returns ``(F_c, M_cz, M_cy)`` where:
        F_c   = 0.85·f_c' · A_block         (compression positive)
        M_cz  = F_c · y_centroid_of_block
        M_cy  = -F_c · z_centroid_of_block
    """
    bounds = polygon.bounds
    half_plane = _whitney_half_plane(theta_rad, y_block_lower, bounds)
    block = polygon.intersection(half_plane)
    if block.is_empty or block.area <= 0:
        return 0.0, 0.0, 0.0
    A_b = float(block.area)
    c = block.centroid
    z_c = float(c.x)
    y_c = float(c.y)
    F_c = sigma_block * A_b
    M_cz = F_c * y_c
    M_cy = -F_c * z_c
    return F_c, M_cz, M_cy


# ============================================================ point evaluation

def _biaxial_pmm_engine(
    section: Section,
    theta_rad: float,
    c: float,
    params: StressBlockParams,
    *,
    _cached_rebar: Optional[list[tuple]] = None,
    _cached_tendons: Optional[list[tuple]] = None,
) -> BiaxialPMMPoint:
    """Code-agnostic biaxial P-M-M point evaluation.

    Internal engine -- callers use the code-specific wrappers:
    :func:`biaxial_pmm_point` (ACI), :func:`biaxial_pmm_point_ec2`,
    :func:`biaxial_pmm_point_is456`.
    """
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")

    beta_1 = params.beta_1
    sigma_block = params.sigma_block
    eps_cu = params.eps_cu
    f_yd = params.f_yd
    E_s = params.E_s
    sin_t = math.sin(theta_rad)
    cos_t = math.cos(theta_rad)
    polygon = section.geometry.polygon

    if _cached_rebar is None:
        rebar = _rebar_fibers(section)
    else:
        rebar = _cached_rebar
    if _cached_tendons is None:
        tendons = _tendon_fibers(section)
    else:
        tendons = _cached_tendons

    ext_coords = list(polygon.exterior.coords)
    yps_poly = [(-zz * sin_t + yy * cos_t) for (zz, yy) in ext_coords]
    yp_max = max(yps_poly)
    yp_min = min(yps_poly)
    section_depth_y = yp_max - yp_min

    a = min(beta_1 * c, section_depth_y)
    yp_NA = yp_max - c
    yp_block_lower = yp_max - a

    F_c, M_cz, M_cy = _whitney_block_contribution(
        polygon, theta_rad, yp_block_lower, sigma_block,
    )

    P_s = 0.0
    M_sz = 0.0
    M_sy = 0.0
    eps_t_signed = math.inf
    for (z, y, A) in rebar:
        yp = -z * sin_t + y * cos_t
        eps_i = eps_cu * (yp - yp_NA) / c
        f_si = max(-f_yd, min(f_yd, E_s * eps_i))
        if yp > yp_block_lower:
            f_si_eff = f_si - sigma_block
        else:
            f_si_eff = f_si
        P_s += f_si_eff * A
        M_sz += f_si_eff * y * A
        M_sy += -f_si_eff * z * A
        if eps_i < eps_t_signed:
            eps_t_signed = eps_i

    P_pt = 0.0
    M_ptz = 0.0
    M_pty = 0.0
    for (z, y, A, f_pe, E_p, f_pu) in tendons:
        yp = -z * sin_t + y * cos_t
        eps_concrete = eps_cu * (yp - yp_NA) / c
        eps_pe = f_pe / max(E_p, 1.0)
        eps_total = eps_pe + eps_concrete
        f_pt = max(-f_pu, min(f_pu, E_p * eps_total))
        if yp > yp_block_lower:
            f_pt_eff = f_pt - sigma_block
        else:
            f_pt_eff = f_pt
        P_pt += f_pt_eff * A
        M_ptz += f_pt_eff * y * A
        M_pty += -f_pt_eff * z * A

    P = F_c + P_s + P_pt
    Mz = M_cz + M_sz + M_ptz
    My = M_cy + M_sy + M_pty

    eps_t = -eps_t_signed if eps_t_signed < 0.0 else 0.0
    eps_ty = f_yd / E_s
    # ACI applies the Table 21.2 phi-interp; EC2 and IS 456 already
    # have the partial safety factors baked into the stress block so
    # we report phi = 1.0 (caller can apply additional reductions).
    if params.apply_phi_table:
        phi = phi_for_strain(eps_t, eps_ty, spiral=params.spiral)
    else:
        phi = 1.0

    if eps_t >= 0.005:
        section_type = "tension-controlled"
    elif eps_t <= eps_ty:
        section_type = "compression-controlled"
    else:
        section_type = "transition"

    return BiaxialPMMPoint(
        theta_rad=theta_rad, c=c,
        P_n=P, M_nz=Mz, M_ny=My,
        phi=phi, epsilon_t=eps_t,
        section_type=section_type,
    )


def biaxial_pmm_point(
    section: Section,
    theta_rad: float,
    c: float,
    *,
    f_c_prime: float,
    f_y: float,
    E_s: float = E_STEEL,
    spiral: bool = False,
    _cached_rebar: Optional[list[tuple]] = None,
    _cached_tendons: Optional[list[tuple]] = None,
) -> BiaxialPMMPoint:
    """Evaluate one point ``(P_n, M_nz, M_ny)`` for a given NA angle
    and depth.

    Parameters
    ----------
    section : Section
        Unified Section with reinforcement.
    theta_rad : float
        Neutral-axis angle (CCW from local +z), radians.
    c : float
        NA depth from extreme compression fiber along the perpendicular
        to the NA (m). ``c > 0``.
    f_c_prime, f_y : float
        Concrete strength and steel yield, Pa (positive numbers).
    E_s : float
        Steel modulus.
    spiral : bool
        Spiral reinforcement -> φ uses 0.75 instead of 0.65 in the
        compression-controlled limit.
    """
    if f_c_prime <= 0 or f_y <= 0 or E_s <= 0:
        raise ValueError("f_c_prime, f_y, E_s must be positive")
    params = aci_params(f_c_prime, f_y, E_s=E_s, spiral=spiral)
    return _biaxial_pmm_engine(
        section, theta_rad, c, params,
        _cached_rebar=_cached_rebar, _cached_tendons=_cached_tendons,
    )


def biaxial_pmm_point_ec2(
    section: Section,
    theta_rad: float,
    c: float,
    *,
    f_ck: float, f_yk: float,
    alpha_cc: float = 0.85,
    gamma_c: float = 1.5, gamma_s: float = 1.15,
    E_s: float = E_STEEL,
    _cached_rebar: Optional[list[tuple]] = None,
    _cached_tendons: Optional[list[tuple]] = None,
) -> BiaxialPMMPoint:
    """Eurocode 2 §3.1.7 biaxial P-M-M point. Partial safety factors
    gamma_c and gamma_s are built into the stress block; no additional
    phi reduction is applied."""
    if f_ck <= 0 or f_yk <= 0 or E_s <= 0:
        raise ValueError("f_ck, f_yk, E_s must be positive")
    params = ec2_params(
        f_ck, f_yk, E_s=E_s,
        alpha_cc=alpha_cc, gamma_c=gamma_c, gamma_s=gamma_s,
    )
    return _biaxial_pmm_engine(
        section, theta_rad, c, params,
        _cached_rebar=_cached_rebar, _cached_tendons=_cached_tendons,
    )


def biaxial_pmm_point_is456(
    section: Section,
    theta_rad: float,
    c: float,
    *,
    f_ck: float, f_y: float,
    gamma_m_steel: float = 1.15,
    E_s: float = E_STEEL,
    _cached_rebar: Optional[list[tuple]] = None,
    _cached_tendons: Optional[list[tuple]] = None,
) -> BiaxialPMMPoint:
    """IS 456:2000 Annex G biaxial P-M-M point. The partial safety
    factors gamma_m_concrete = 1.5 (built into the 0.36 factor) and
    gamma_m_steel = 1.15 (giving f_yd = 0.87*f_y) are applied here;
    no additional phi reduction."""
    if f_ck <= 0 or f_y <= 0 or E_s <= 0:
        raise ValueError("f_ck, f_y, E_s must be positive")
    params = is456_params(
        f_ck, f_y, E_s=E_s, gamma_m_steel=gamma_m_steel,
    )
    return _biaxial_pmm_engine(
        section, theta_rad, c, params,
        _cached_rebar=_cached_rebar, _cached_tendons=_cached_tendons,
    )


# ============================================================ surface builder

def _biaxial_pmm_surface_engine(
    section: Section,
    params: StressBlockParams,
    *,
    n_angles: int = 24,
    n_depths: int = 24,
) -> BiaxialPMMSurface:
    """Code-agnostic biaxial P-M-M surface builder. Used by the three
    code-specific public wrappers."""
    has_rebar = (section.reinforcement is not None
                  and section.reinforcement.bars)
    has_prestress = (section.prestress is not None
                     and section.prestress.tendons)
    if not has_rebar and not has_prestress:
        raise ValueError(
            "section has no reinforcement and no prestress; "
            "P-M-M surface needs at least one"
        )

    rebar = _rebar_fibers(section)
    tendons = _tendon_fibers(section)

    # Smooth single geomspace c-distribution. Earlier we tried a
    # 3-segment scheme to concentrate samples around the transition
    # zone, but the abrupt spacing jumps at segment boundaries
    # produced visible ridges in the 3-D plot. A single geomspace
    # with adequate n_depths gives the cleanest surface; the engine
    # is fast enough that 30+ samples cost nothing.
    minz, miny, maxz, maxy = section.geometry.polygon.bounds
    diag = math.hypot(maxz - minz, maxy - miny)
    c_min = 0.005 * diag
    c_max = diag / max(params.beta_1, 0.1)
    cs = np.geomspace(c_min, c_max, n_depths)

    thetas = np.linspace(0.0, 2 * math.pi, n_angles, endpoint=False)
    points: list[BiaxialPMMPoint] = []
    for theta in thetas:
        for c in cs:
            pt = _biaxial_pmm_engine(
                section, float(theta), float(c), params,
                _cached_rebar=rebar, _cached_tendons=tendons,
            )
            points.append(pt)

    A_st = sum(A for (_, _, A) in rebar)
    A_pt = sum(A for (_, _, A, _, _, _) in tendons)
    A_g = section.geometry.area
    fpt_for_Po = 0.0
    for (_, _, A, f_pe, E_p, f_pu) in tendons:
        fpt_for_Po += A * f_pu
    # Pure compression: concrete at sigma_block + rebar at f_yd
    # + tendons at f_pu, minus concrete displaced by rebar / tendons.
    P_o = (params.sigma_block * (A_g - A_st - A_pt)
           + params.f_yd * A_st + fpt_for_Po)
    cap = 0.85 if params.spiral else 0.80
    P_n_max = cap * P_o
    P_pure_tension = -(A_st * params.f_yd + fpt_for_Po)

    # Plastic-centroid moments at pure compression. For a doubly-
    # symmetric section centred at origin these are zero; for an
    # asymmetric section (L, T, channel, or any polygon whose
    # geometric centroid is not at the moment-reference origin) they
    # are non-zero. The cap surface must preserve these offsets, NOT
    # force everything to (0, 0, P_n_max) -- doing so produces the
    # "teepee" artefact for asymmetric sections.
    cz_g, cy_g = section.geometry.centroid
    # Concrete contribution to moments (about origin)
    M_z_concrete = params.sigma_block * A_g * cy_g
    M_y_concrete = -params.sigma_block * A_g * cz_g
    # Rebar contribution (each bar at +f_yd minus displaced concrete)
    M_z_rebar = 0.0
    M_y_rebar = 0.0
    for (z, y, A) in rebar:
        f_eff = params.f_yd - params.sigma_block   # concrete displacement
        M_z_rebar += f_eff * y * A
        M_y_rebar += -f_eff * z * A
    # Tendon contribution
    M_z_tendon = 0.0
    M_y_tendon = 0.0
    for (z, y, A, f_pe, E_p, f_pu) in tendons:
        f_eff = f_pu - params.sigma_block
        M_z_tendon += f_eff * y * A
        M_y_tendon += -f_eff * z * A
    M_z_at_Po = M_z_concrete + M_z_rebar + M_z_tendon
    M_y_at_Po = M_y_concrete + M_y_rebar + M_y_tendon
    # The cap point (P_n_max) sits at the same moment-axis offset,
    # scaled by the cap factor (since concrete & steel stresses
    # scale linearly under uniform compression with strain).
    cap_factor = P_n_max / max(P_o, 1.0)
    M_z_cap = M_z_at_Po * cap_factor
    M_y_cap = M_y_at_Po * cap_factor

    # Cap any computed P_n above P_n_max per ACI 22.4.2.1. Above the
    # cap the section is at pure axial compression; the resultant
    # passes through the plastic centroid, so the moments about the
    # origin are NOT zero unless the section is doubly symmetric.
    # Clip to (P_n_max, M_z_cap, M_y_cap) where the cap moments come
    # from the plastic-centroid analysis above.
    for p in points:
        if p.P_n > P_n_max:
            p.P_n = P_n_max
            p.M_nz = M_z_cap
            p.M_ny = M_y_cap
            p.section_type = "pure-compression-cap"

    return BiaxialPMMSurface(
        points=points,
        P_o=float(P_o),
        P_n_max=float(P_n_max),
        P_pure_tension=float(P_pure_tension),
        f_c_prime=float(params.sigma_block / 0.85),  # back-out f_c for legacy field
        f_y=float(params.f_yd),
        spiral=bool(params.spiral),
        n_angles=int(n_angles),
        n_depths=int(n_depths),
    )


def biaxial_pmm_surface(
    section: Section, *,
    f_c_prime: float, f_y: float,
    E_s: float = E_STEEL,
    n_angles: int = 24, n_depths: int = 24,
    spiral: bool = False,
) -> BiaxialPMMSurface:
    """ACI 318-19 biaxial P-M-M surface (default).

    See :class:`StressBlockParams` for the stress block constants.
    """
    if f_c_prime <= 0 or f_y <= 0 or E_s <= 0:
        raise ValueError("f_c_prime, f_y, E_s must be positive")
    params = aci_params(f_c_prime, f_y, E_s=E_s, spiral=spiral)
    return _biaxial_pmm_surface_engine(
        section, params, n_angles=n_angles, n_depths=n_depths,
    )


def biaxial_pmm_surface_ec2(
    section: Section, *,
    f_ck: float, f_yk: float,
    alpha_cc: float = 0.85,
    gamma_c: float = 1.5, gamma_s: float = 1.15,
    E_s: float = E_STEEL,
    n_angles: int = 24, n_depths: int = 24,
) -> BiaxialPMMSurface:
    """Eurocode 2 §3.1.7 biaxial P-M-M surface.

    Uses the rectangular stress block with eta·alpha_cc·f_ck/gamma_c
    over depth lambda·c (lambda = 0.80 for f_ck <= 50 MPa), and
    f_yd = f_yk/gamma_s. Partial safety factors are built into the
    stress block; no additional phi reduction is applied (the returned
    surface has phi=1.0 at every point).

    Defaults match EC2 §3.1.6: alpha_cc=0.85 (sustained load reduction
    factor; check your National Annex), gamma_c=1.5, gamma_s=1.15.
    """
    if f_ck <= 0 or f_yk <= 0 or E_s <= 0:
        raise ValueError("f_ck, f_yk, E_s must be positive")
    params = ec2_params(
        f_ck, f_yk, E_s=E_s,
        alpha_cc=alpha_cc, gamma_c=gamma_c, gamma_s=gamma_s,
    )
    return _biaxial_pmm_surface_engine(
        section, params, n_angles=n_angles, n_depths=n_depths,
    )


def biaxial_pmm_surface_is456(
    section: Section, *,
    f_ck: float, f_y: float,
    gamma_m_steel: float = 1.15,
    E_s: float = E_STEEL,
    n_angles: int = 24, n_depths: int = 24,
) -> BiaxialPMMSurface:
    """IS 456:2000 Annex G biaxial P-M-M surface.

    Uses the parabolic-rectangular concrete curve as a Whitney
    equivalent: stress = 0.36*f_ck/0.84 = 0.4286*f_ck, block depth
    = 0.84*c. Steel design stress = f_y/gamma_m_steel (= 0.87*f_y by
    default). Partial safety factors built in; no extra phi.
    """
    if f_ck <= 0 or f_y <= 0 or E_s <= 0:
        raise ValueError("f_ck, f_y, E_s must be positive")
    params = is456_params(
        f_ck, f_y, E_s=E_s, gamma_m_steel=gamma_m_steel,
    )
    return _biaxial_pmm_surface_engine(
        section, params, n_angles=n_angles, n_depths=n_depths,
    )


# ============================================================ helpers

def _angle_diff(a: float, b: float) -> float:
    """Smallest signed angular difference between two angles."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d
