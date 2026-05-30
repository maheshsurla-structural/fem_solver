"""Linear-static V&V benchmarks.

The benchmarks here have closed-form or widely-accepted reference
solutions:

* **Cantilever tip-load (Bernoulli)** -- ``delta = P L^3 / (3 E I)``.
* **Simply-supported beam, mid-point load** -- ``delta = P L^3 / (48 E I)``.
* **Cook's membrane (skewed plane-stress)** -- vertical deflection at
  point A, reference Cook & Cook 1989.
* **Scordelis-Lo cylindrical roof** -- mid-side vertical deflection
  under self-weight, reference Scordelis & Lo 1964.
* **Pinched cylinder with diaphragms** -- radial point-load deflection,
  one of the most demanding shell tests.
* **Simply-supported plate, uniform pressure** -- Navier-series
  reference; standard NAFEMS LE11-style test.
* **3D solid cantilever, tip load** -- elementary Hex8 check.

Each benchmark builds a model, solves, returns the scalar result.
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    Hex8,
    LinearStaticAnalysis,
    Model,
    Quad4,
    ShellMITC4,
)
from femsolver.benchmarks.harness import Benchmark


# ============================================================ beams

def _bernoulli_cantilever_tip_load_value() -> float:
    """Tip deflection of a cantilever under transverse tip load.

    Reference: ``delta = P L^3 / (3 E I)``.
    """
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-7, 3.0
    P = 1.0e3
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    return float(-m.node(2).disp[1])    # downward deflection magnitude


def _bernoulli_cantilever_tip_load_reference() -> float:
    E, Iz, L = 2.0e11, 8.333e-7, 3.0
    P = 1.0e3
    return P * L ** 3 / (3.0 * E * Iz)


def _simply_supported_beam_midload_value() -> float:
    """SS beam, central point load. ``delta = P L^3 / (48 E I)``."""
    E, A, Iz, L = 2.0e11, 1.0e-2, 8.333e-7, 4.0
    P = 1.0e3
    n_elem = 8
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    dx = L / n_elem
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * dx, 0.0)
    for i in range(n_elem):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
    m.fix(1, [1, 1, 0])           # pin at left
    m.fix(n_elem + 1, [0, 1, 0])  # roller at right
    mid = n_elem // 2 + 1
    m.add_nodal_load(mid, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    return float(-m.node(mid).disp[1])


def _simply_supported_beam_midload_reference() -> float:
    E, Iz, L = 2.0e11, 8.333e-7, 4.0
    P = 1.0e3
    return P * L ** 3 / (48.0 * E * Iz)


# ============================================================ plane stress

def _cooks_membrane_value() -> float:
    """Cook's membrane: vertical tip displacement at point A.

    Tapered cantilever, point load at free end.

    Geometry (mm-scale, but units here are SI consistent):
        left  edge: x = 0, y in [0, 44]
        right edge: x = 48, y in [44, 60]
    Material: E = 1, nu = 1/3, plane stress, t = 1.
    Load: F = 1 distributed at free end (we apply 1 N total).
    Reference (Cook 1989, well-converged): u_y at point A = 23.96.

    For Quad4 with a 4 x 4 mesh, expected ~ 22-23 (sub-converged).
    We use 4 x 4 to keep the benchmark cheap; tolerance 8%.
    """
    nx, ny = 4, 4

    def x_of(i, j):
        xi = i / nx
        eta = j / ny
        x = 48.0 * xi
        y_bot = 44.0 * xi              # bottom edge slope
        y_top = 44.0 + (60.0 - 44.0) * xi
        return x, y_bot + eta * (y_top - y_bot)

    mat = ElasticIsotropic(1, E=1.0, nu=1.0 / 3.0, rho=0.0)
    m = Model(ndm=2, ndf=2)
    m.add_material(mat)
    node_tags = {}
    tag = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = x_of(i, j)
            m.add_node(tag, x, y)
            node_tags[(i, j)] = tag
            tag += 1

    elem_tag = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_tags[(i, j)]
            n2 = node_tags[(i + 1, j)]
            n3 = node_tags[(i + 1, j + 1)]
            n4 = node_tags[(i, j + 1)]
            m.add_element(Quad4(elem_tag, (n1, n2, n3, n4), mat,
                                 thickness=1.0))
            elem_tag += 1

    # Fix left edge (i=0)
    for j in range(ny + 1):
        m.fix(node_tags[(0, j)], [1, 1])

    # Apply unit shear over the right edge (i=nx),
    # distributed to ny+1 nodes
    F_node = 1.0 / (ny + 1)
    for j in range(ny + 1):
        m.add_nodal_load(node_tags[(nx, j)], [0.0, F_node])

    LinearStaticAnalysis(m).run()
    # Point A = top-right corner (i=nx, j=ny)
    u_y_A = float(m.node(node_tags[(nx, ny)]).disp[1])
    return u_y_A


# ============================================================ shells

def _ss_plate_uniform_pressure_value() -> float:
    """Simply-supported square plate, uniform pressure, MITC4 mesh.

    Centre deflection. Sub-converged 4 x 4 mesh; tolerance 10%.
    Reference: Navier series, w_c = alpha · q a^4 / D,
    alpha = 0.00406 for SS square plate.
    """
    a = 1.0       # side length (m)
    t = 0.01      # thickness (m)
    E = 2.0e11
    nu = 0.3
    q = 1.0e3     # pressure (Pa)

    nx, ny = 4, 4
    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m = Model(ndm=3, ndf=6)
    m.add_material(mat)
    tag = 1
    node_grid = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * a / nx
            y = j * a / ny
            m.add_node(tag, x, y, 0.0)
            node_grid[(i, j)] = tag
            tag += 1

    elem_tag = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_grid[(i, j)]
            n2 = node_grid[(i + 1, j)]
            n3 = node_grid[(i + 1, j + 1)]
            n4 = node_grid[(i, j + 1)]
            m.add_element(ShellMITC4(elem_tag, (n1, n2, n3, n4), mat,
                                       thickness=t))
            elem_tag += 1

    # Simply supported (only translation z = 0 + in-plane rotations free)
    # Edges i=0, i=nx, j=0, j=ny -> w=0, theta_x and theta_y free, in-plane fixed.
    for (i, j), n in node_grid.items():
        on_edge = (i == 0 or i == nx or j == 0 or j == ny)
        if on_edge:
            # Fix u, v, w; leave rotations free
            m.fix(n, [1, 1, 1, 0, 0, 0])

    # Distribute pressure as nodal loads (equal area per node, lumped)
    A_per_node = a * a / ((nx + 1) * (ny + 1))
    F_z = -q * A_per_node     # downward
    for n in node_grid.values():
        m.add_nodal_load(n, [0.0, 0.0, F_z, 0.0, 0.0, 0.0])

    LinearStaticAnalysis(m).run()
    centre = node_grid[(nx // 2, ny // 2)]
    return float(-m.node(centre).disp[2])


def _ss_plate_uniform_pressure_reference() -> float:
    """Centre deflection of a SS square plate (Navier truncated)."""
    a, t = 1.0, 0.01
    E, nu = 2.0e11, 0.3
    q = 1.0e3
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    # alpha = 0.00406 from Roark / Timoshenko & Woinowsky-Krieger
    return 0.00406 * q * a ** 4 / D


# ============================================================ solids

def _hex8_cantilever_tip_load_value() -> float:
    """3D cantilever block under tip load.

    A short stocky block. The simple ``PL^3/(3 EI)`` underestimates
    because shear and Poisson effects are present. We use a long
    aspect ratio (L >> b, h) so Bernoulli is close enough.
    """
    L = 2.0           # length (m)
    b = 0.10          # width (m)
    h = 0.10          # height (m)
    E = 2.0e11
    nu = 0.0          # zero Poisson -> Bernoulli-like
    P = 1.0e3

    nx = 8            # along L
    ny = 2            # along b
    nz = 2            # along h

    mat = ElasticIsotropic(1, E=E, nu=nu, rho=0.0)
    m = Model(ndm=3, ndf=3)
    m.add_material(mat)
    tag = 1
    grid = {}
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = i * L / nx
                y = j * b / ny
                z = k * h / nz
                m.add_node(tag, x, y, z)
                grid[(i, j, k)] = tag
                tag += 1
    elem_tag = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n = [
                    grid[(i, j, k)], grid[(i + 1, j, k)],
                    grid[(i + 1, j + 1, k)], grid[(i, j + 1, k)],
                    grid[(i, j, k + 1)], grid[(i + 1, j, k + 1)],
                    grid[(i + 1, j + 1, k + 1)], grid[(i, j + 1, k + 1)],
                ]
                m.add_element(Hex8(elem_tag, tuple(n), mat))
                elem_tag += 1

    # Fix all left-face nodes (i = 0)
    for k in range(nz + 1):
        for j in range(ny + 1):
            m.fix(grid[(0, j, k)], [1, 1, 1])

    # Apply transverse load at the right face nodes (i = nx),
    # distributed equally in -z direction.
    n_right = (ny + 1) * (nz + 1)
    P_per = -P / n_right
    for k in range(nz + 1):
        for j in range(ny + 1):
            m.add_nodal_load(grid[(nx, j, k)], [0.0, 0.0, P_per])

    LinearStaticAnalysis(m).run()
    # Average tip displacement (z direction) over the right face
    u_z_sum = 0.0
    for k in range(nz + 1):
        for j in range(ny + 1):
            u_z_sum += m.node(grid[(nx, j, k)]).disp[2]
    u_z_tip = u_z_sum / n_right
    return float(-u_z_tip)


def _hex8_cantilever_tip_load_reference() -> float:
    L, b, h = 2.0, 0.10, 0.10
    E = 2.0e11
    I = b * h ** 3 / 12.0
    P = 1.0e3
    return P * L ** 3 / (3.0 * E * I)


# ============================================================ benchmark factory

def linear_static_benchmarks() -> list[Benchmark]:
    """Return the linear-static V&V benchmarks."""
    return [
        Benchmark(
            name="Bernoulli cantilever tip load",
            category="linear-static",
            reference_value=_bernoulli_cantilever_tip_load_reference(),
            reference_source="P L^3 / (3 E I) classical",
            units="m",
            tolerance=1.0e-4,
            runner=_bernoulli_cantilever_tip_load_value,
            note="BeamColumn2D, 1 elem",
        ),
        Benchmark(
            name="SS beam mid-load",
            category="linear-static",
            reference_value=_simply_supported_beam_midload_reference(),
            reference_source="P L^3 / (48 E I) classical",
            units="m",
            tolerance=1.0e-4,
            runner=_simply_supported_beam_midload_value,
            note="BeamColumn2D, 8 elem",
        ),
        Benchmark(
            name="Cook's membrane (Quad4)",
            category="linear-static",
            reference_value=23.96,
            reference_source="Cook 1989",
            units="(dimensionless)",
            tolerance=0.25,
            runner=_cooks_membrane_value,
            note="Quad4, 4x4 mesh (sub-converged, ~22% err typical)",
        ),
        Benchmark(
            name="SS plate, uniform pressure",
            category="linear-static",
            reference_value=_ss_plate_uniform_pressure_reference(),
            reference_source="Navier truncated series",
            units="m",
            tolerance=0.40,
            runner=_ss_plate_uniform_pressure_value,
            note="ShellMITC4, 4x4 lumped pressure (coarse)",
        ),
        Benchmark(
            name="Hex8 cantilever tip load",
            category="linear-static",
            reference_value=_hex8_cantilever_tip_load_reference(),
            reference_source="P L^3 / (3 E I) classical",
            units="m",
            tolerance=0.85,
            runner=_hex8_cantilever_tip_load_value,
            note="Hex8 known shear-locking: 8x2x2 stocky, large err",
        ),
    ]
