"""Stress / strain field query for any RC section (Phase II.15).

Given a unified :class:`~femsolver.sections.Section` and a loading
state ``(P, M_z, M_y)``, compute the strain and stress at every fibre
in the section -- and at any specific ``(z, y)`` point. Useful for:

* Stress-state verification (top concrete sigma_c, extreme tension
  rebar sigma_s, tendon delta-stress)
* Crack-pattern visualization: render the section SVG with concrete
  fibres coloured by tensile strain (cracked = where eps > f_r/E_c)
* Hand-checkable spot queries at any (z, y) without re-discretizing

Algorithm
---------
Three unknowns (eps_0, kappa_z, kappa_y) and three equations
(N = -P, M_z = target, M_y = target). Solve via Newton iteration on
the 3x3 axial-bending block of the fiber section's tangent matrix.

The fiber section is built using the unified Section's adapter
(:meth:`Section.fiber_section_3d`), so tendons with pre-strain are
automatically included.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from femsolver.materials.uniaxial.base import UniaxialMaterial
from femsolver.materials.uniaxial.prestressed import PrestressedUniaxial
from femsolver.sections.response.fiber import Fiber, FiberSection3D
from femsolver.sections.section import (
    Section,
    _discretize_polygon_to_fibers,
)


# ============================================================ data types

@dataclass
class FiberState:
    """Strain + stress at one fiber location."""
    z: float        # m
    y: float        # m
    area: float     # m^2
    eps: float      # tension-positive
    sigma: float    # tension-positive (Pa)
    kind: str       # "concrete" | "rebar" | "tendon"


@dataclass
class SectionStressField:
    """Full strain / stress distribution across a section at one
    loading state."""
    # Loading state (compression-positive P, sagging-positive M_z)
    P: float
    M_z: float
    M_y: float

    # Solution
    eps_0: float    # axial strain at section centroid (tension-positive)
    kappa_z: float  # curvature about z (positive -> compression at +y)
    kappa_y: float  # curvature about y

    # Fibers
    fibers: list[FiberState] = field(default_factory=list)

    # Convergence
    converged: bool = True
    iterations: int = 0
    residual_norm: float = 0.0

    # ----------------------------------------------------- query
    def stress_at(self, z: float, y: float) -> float:
        """Concrete stress at a point (z, y) under the plane-sections
        kinematic. Uses the elastic modulus of the *closest* concrete
        fiber (so the constitutive nonlinearity is preserved)."""
        eps = self.eps_0 - y * self.kappa_z + z * self.kappa_y
        # Find nearest concrete fiber and use its tangent
        concrete = [f for f in self.fibers if f.kind == "concrete"]
        if not concrete:
            return 0.0
        nearest = min(concrete, key=lambda f: (f.z - z) ** 2 + (f.y - y) ** 2)
        # Use the same epsilon -> stress mapping the nearest fiber used
        # (assumes monotonic): interpolate via sigma/eps slope
        if abs(nearest.eps) > 1e-12:
            E_local = nearest.sigma / nearest.eps
        else:
            E_local = 0.0
        return E_local * eps

    def strain_at(self, z: float, y: float) -> float:
        """Strain at point (z, y) under plane-sections (linear)."""
        return self.eps_0 - y * self.kappa_z + z * self.kappa_y

    # ----------------------------------------------------- summary
    def extreme_compression_strain(self) -> float:
        """Most-compressed concrete fiber strain (most negative)."""
        concrete = [f.eps for f in self.fibers if f.kind == "concrete"]
        return float(min(concrete)) if concrete else 0.0

    def extreme_tension_strain(self) -> float:
        """Most-tensile fiber strain (most positive), any kind."""
        return float(max(f.eps for f in self.fibers)) if self.fibers else 0.0

    def cracked_fibers(self, eps_crack: float) -> list[FiberState]:
        """Concrete fibers with tensile strain exceeding ``eps_crack``."""
        return [f for f in self.fibers
                if f.kind == "concrete" and f.eps >= eps_crack]


# ============================================================ solver

def stress_field(
    section: Section,
    *,
    P: float = 0.0,
    M_z: float = 0.0,
    M_y: float = 0.0,
    concrete_uniaxial: UniaxialMaterial,
    steel_uniaxial: UniaxialMaterial,
    n_z: int = 16,
    n_y: int = 24,
    P_convention: str = "compression",
    max_iter: int = 60,
    tol: float = 1e-4,
) -> SectionStressField:
    """Solve for the strain / stress field under prescribed (P, M_z, M_y).

    Parameters
    ----------
    section : Section
        Unified Section with rebar and/or tendons.
    P : float
        Axial load (default convention: compression-positive).
    M_z, M_y : float
        Moments about the section's z and y axes (sagging-positive).
    concrete_uniaxial, steel_uniaxial : UniaxialMaterial
        Constitutive laws.
    n_z, n_y : int
        Polygon discretization grid.
    P_convention : {"compression", "tension"}
        Default "compression": P > 0 means compression.
    max_iter : int
        Newton iteration cap.
    tol : float
        Residual tolerance (relative to P_target + 100 N).
    """
    if P_convention not in ("compression", "tension"):
        raise ValueError(
            f"P_convention must be 'compression' or 'tension', "
            f"got {P_convention!r}"
        )

    has_rebar = (section.reinforcement is not None
                  and section.reinforcement.bars)
    has_prestress = (section.prestress is not None
                     and section.prestress.tendons)
    if not has_rebar and not has_prestress:
        # Plain concrete section is allowed; just no steel fibers
        pass

    # Convert P to tension-positive N for the fiber section
    N_target = -P if P_convention == "compression" else P

    # Build fiber section (concrete + rebar + tendons with pre-strain)
    polygon = section.geometry.polygon
    concrete_fibers = _discretize_polygon_to_fibers(
        polygon, concrete_uniaxial, n_z=n_z, n_y=n_y,
    )
    rebar_fibers = []
    if has_rebar:
        for bar in section.reinforcement.bars:
            mat = bar.material if bar.material is not None else steel_uniaxial
            rebar_fibers.append(Fiber(
                y=bar.y, z=bar.z, area=bar.area,
                material=mat.clone(),
            ))
    tendon_fibers = []
    tendon_info: list[tuple] = []   # (z, y, area, eps_pe) for later post-processing
    if has_prestress:
        for tendon in section.prestress.tendons:
            if not tendon.bonded or tendon.material is None:
                continue
            _, E_p = tendon.material.get_response(0.0)
            if E_p <= 0:
                E_p = 195e9
            eps_pe = tendon.f_pe / E_p
            wrapped = PrestressedUniaxial(tendon.material.clone(), eps_pe)
            tendon_fibers.append(Fiber(
                y=tendon.y, z=tendon.z, area=tendon.area,
                material=wrapped,
            ))
            tendon_info.append((tendon.z, tendon.y, tendon.area, eps_pe))

    all_fibers = concrete_fibers + rebar_fibers + tendon_fibers
    fs = FiberSection3D(all_fibers, GJ=1.0)   # GJ irrelevant for axial-bending only

    # Initial guess
    e = np.zeros(4)
    target = np.array([N_target, M_z, M_y, 0.0])
    tol_use = max(tol * (abs(N_target) + abs(M_z) + abs(M_y)), 100.0)

    converged = False
    res_norm = math.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        s, ks = fs.get_response(e)
        res = s - target
        res_norm = float(np.linalg.norm(res[:3]))
        if res_norm < tol_use:
            converged = True
            break
        # Solve 3x3 axial-bending sub-block
        J = ks[:3, :3].copy()
        # Regularize to avoid singular matrix near unloaded states
        for i in range(3):
            if abs(J[i, i]) < 1.0:
                J[i, i] = 1.0
        try:
            step = np.linalg.solve(J, res[:3])
        except np.linalg.LinAlgError:
            break
        e[:3] -= step

    eps_0 = float(e[0])
    kappa_z = float(e[1])
    kappa_y = float(e[2])

    # Extract per-fiber strain/stress
    fiber_states: list[FiberState] = []
    n_concrete = len(concrete_fibers)
    n_rebar = len(rebar_fibers)
    for i, f in enumerate(all_fibers):
        eps_f = eps_0 - f.y * kappa_z + f.z * kappa_y
        sigma_f, _ = f.material.get_response(eps_f)
        if i < n_concrete:
            kind = "concrete"
        elif i < n_concrete + n_rebar:
            kind = "rebar"
        else:
            kind = "tendon"
        fiber_states.append(FiberState(
            z=f.z, y=f.y, area=f.area,
            eps=float(eps_f), sigma=float(sigma_f),
            kind=kind,
        ))

    return SectionStressField(
        P=P, M_z=M_z, M_y=M_y,
        eps_0=eps_0, kappa_z=kappa_z, kappa_y=kappa_y,
        fibers=fiber_states,
        converged=converged,
        iterations=n_iter,
        residual_norm=res_norm,
    )


def stress_at_point(
    section: Section,
    z: float, y: float,
    *,
    P: float = 0.0, M_z: float = 0.0, M_y: float = 0.0,
    concrete_uniaxial: UniaxialMaterial,
    steel_uniaxial: UniaxialMaterial,
    **kwargs,
) -> tuple[float, float]:
    """Convenience: return ``(strain, stress)`` at a single ``(z, y)``
    point under the given loading state. Builds the stress field
    internally; for many queries on the same loading state, call
    :func:`stress_field` once and use its :meth:`stress_at` method."""
    sf = stress_field(
        section, P=P, M_z=M_z, M_y=M_y,
        concrete_uniaxial=concrete_uniaxial,
        steel_uniaxial=steel_uniaxial,
        **kwargs,
    )
    return sf.strain_at(z, y), sf.stress_at(z, y)


# ============================================================ SVG overlay

def stress_field_to_svg(
    section: Section,
    sf: SectionStressField,
    *,
    width_px: int = 400,
    pad_px: int = 40,
    eps_crack: float = 1.5e-4,
    show_rebar: bool = True,
    title: Optional[str] = None,
) -> str:
    """Render the section with a per-fibre stress / strain overlay.

    Color mapping (default):
        - concrete in compression (eps < 0): blue gradient
        - concrete in tension (eps > 0, below crack): light orange
        - concrete cracked (eps >= eps_crack): dark red
        - rebar: green dots scaled by strain magnitude
        - tendon: magenta dots
    """
    polygon = section.geometry.polygon
    minz, miny, maxz, maxy = polygon.bounds
    geo_w = maxz - minz
    geo_h = maxy - miny

    inner_w = width_px - 2 * pad_px
    inner_h = inner_w * (geo_h / geo_w)
    height_px = int(inner_h + 2 * pad_px) + 30   # extra for title/legend
    scale = inner_w / geo_w

    def _x(z): return pad_px + (z - minz) * scale
    def _y(y): return pad_px + (maxy - y) * scale + 20   # leave room for title

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {width_px} {height_px}">',
        '<g font-family="Helvetica, sans-serif" font-size="11" fill="#222">',
    ]

    # Title
    title_str = title or (
        f"P={sf.P/1e3:.0f} kN, M_z={sf.M_z/1e3:.0f} kN.m, "
        f"M_y={sf.M_y/1e3:.0f} kN.m"
    )
    parts.append(
        f'<text x="{pad_px:.1f}" y="14" font-weight="bold">'
        f'{_escape(title_str)}</text>'
    )

    # Background polygon outline
    ext = list(polygon.exterior.coords)
    if len(ext) > 1 and ext[0] == ext[-1]:
        ext = ext[:-1]
    path_d = []
    if ext:
        path_d.append(f"M {_x(ext[0][0]):.2f},{_y(ext[0][1]):.2f}")
        for (z, y) in ext[1:]:
            path_d.append(f"L {_x(z):.2f},{_y(y):.2f}")
        path_d.append("Z")
    parts.append(
        f'<path d="{" ".join(path_d)}" '
        f'fill="white" stroke="#1f4f73" stroke-width="1.5"/>'
    )

    # Concrete fiber overlay
    concrete = [f for f in sf.fibers if f.kind == "concrete"]
    if concrete:
        eps_min = min(f.eps for f in concrete)
        eps_max = max(f.eps for f in concrete)
        for f in concrete:
            color = _eps_to_color(f.eps, eps_min, eps_max, eps_crack)
            # Estimate cell size: rectangle around fiber centroid
            cell_size = math.sqrt(f.area)
            half = cell_size / 2
            x0 = _x(f.z - half)
            y0 = _y(f.y + half)
            x1 = _x(f.z + half)
            y1 = _y(f.y - half)
            parts.append(
                f'<rect x="{min(x0,x1):.2f}" y="{min(y0,y1):.2f}" '
                f'width="{abs(x1-x0):.2f}" height="{abs(y1-y0):.2f}" '
                f'fill="{color}" fill-opacity="0.7" stroke="none"/>'
            )

    # Rebar dots
    if show_rebar:
        rebar = [f for f in sf.fibers if f.kind == "rebar"]
        for f in rebar:
            r_px = max(3.0, math.sqrt(f.area / math.pi) * scale)
            # Green if tension, blue if compression
            color = "#0a8030" if f.eps > 0 else "#0a5080"
            parts.append(
                f'<circle cx="{_x(f.z):.2f}" cy="{_y(f.y):.2f}" '
                f'r="{r_px:.2f}" fill="{color}" '
                f'stroke="#000" stroke-width="0.5"/>'
            )

        tendons = [f for f in sf.fibers if f.kind == "tendon"]
        for f in tendons:
            r_px = max(3.0, math.sqrt(f.area / math.pi) * scale)
            parts.append(
                f'<circle cx="{_x(f.z):.2f}" cy="{_y(f.y):.2f}" '
                f'r="{r_px:.2f}" fill="#c000a0" '
                f'stroke="#000" stroke-width="0.5"/>'
            )

    # Annotate cracking
    cracked = sf.cracked_fibers(eps_crack)
    if cracked:
        parts.append(
            f'<text x="{pad_px:.1f}" y="{height_px - 6:.1f}" '
            f'font-size="10" fill="#900">'
            f'{len(cracked)} concrete fibers cracked '
            f'(eps &gt; {eps_crack*1e3:.2f}e-3)</text>'
        )

    parts.append('</g></svg>')
    return "\n".join(parts)


def _eps_to_color(eps: float, eps_min: float, eps_max: float,
                    eps_crack: float) -> str:
    """Color map: compression -> blue, tension below crack -> light
    orange, cracked -> dark red."""
    if eps >= eps_crack:
        return "#8b0000"   # cracked: dark red
    if eps > 0:
        # tension below crack: light orange to orange
        return "#fdb863"
    # compression: lighter blue near zero, darker as more compressed
    if eps_min == 0:
        return "#cfe8ff"
    t = eps / eps_min   # in [0, 1]
    # Interpolate between light blue and dark blue
    r = int(207 - 130 * t)
    g = int(232 - 100 * t)
    b = int(255 - 40 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
              .replace(">", "&gt;").replace('"', "&quot;"))
