"""Self-weight load case — ``desktop/project.py``.

A ``LoadCase`` with a non-zero ``self_weight_factor`` applies the structure's
own gravity weight: members lumped to their end nodes (ρ·A·L·g, global-down)
and 3-D shell areas as a ρ·t·g surface load. Covers the lumped nodal values,
scaling, the buckling ``resolved_loads`` path, combinations and equilibrium
(reactions sum to the total weight) for frames and slabs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

import numpy as np  # noqa: E402

from femsolver import LinearStaticAnalysis  # noqa: E402
from project import (GRAVITY, Area, LoadCase, LoadCombination,  # noqa: E402
                     Material, Member, Node, Project, Section, ShellSection)


def _beam(rho=7850.0, A=0.01, L=4.0, sw=1.0):
    p = Project(ndm=2, ndf=3)
    p.materials.append(Material(id=1, name="Steel", E=200e9, nu=0.3, rho=rho))
    p.sections.append(Section(id=1, name="S", A=A, Iz=1.0e-4))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, supports=(1, 1, 0)),
                    Node(id=2, x=L, y=0.0, supports=(0, 1, 0))])
    p.members.append(Member(id=1, n1=1, n2=2, section=1, material=1))
    p.load_cases = [LoadCase(id=1, name="SW", nature="dead",
                             self_weight_factor=sw)]
    return p, L, A, rho


def test_lumped_nodal_self_weight_2d():
    p, L, A, rho = _beam(sw=1.0)
    m = p.build_model(with_loads=True)
    half = 0.5 * rho * A * L * GRAVITY
    assert np.isclose(m.node(1).load[1], -half)
    assert np.isclose(m.node(2).load[1], -half)
    assert np.isclose(m.node(1).load[0], 0.0)      # gravity is global-down only


def test_no_self_weight_when_factor_zero():
    p, *_ = _beam(sw=0.0)
    m = p.build_model(with_loads=True)
    assert np.allclose(m.node(1).load, 0.0)
    assert np.allclose(m.node(2).load, 0.0)


def test_self_weight_scales_with_factor():
    p, L, A, rho = _beam(sw=0.5)
    m = p.build_model(with_loads=True)
    assert np.isclose(m.node(1).load[1], -0.25 * rho * A * L * GRAVITY)


def test_no_self_weight_without_density():
    p, L, A, rho = _beam(rho=0.0, sw=1.0)
    m = p.build_model(with_loads=True)
    assert np.allclose(m.node(1).load, 0.0)


def test_resolved_loads_includes_self_weight():
    p, L, A, rho = _beam(sw=1.0)
    nodal, _member = p.resolved_loads(("all", None))
    half = 0.5 * rho * A * L * GRAVITY
    assert np.isclose(nodal[1][1], -half)
    assert np.isclose(nodal[2][1], -half)


def test_self_weight_via_combination():
    p, L, A, rho = _beam(sw=1.0)
    p.combinations.append(LoadCombination(id=1, name="1.4D", factors={1: 1.4}))
    m = p.build_model(with_loads=False)
    p.apply_loads(m, ("combination", 1))
    half = 0.5 * rho * A * L * GRAVITY
    assert np.isclose(m.node(1).load[1], -1.4 * half)


def test_equilibrium_reactions_sum_to_weight_2d():
    p, L, A, rho = _beam(sw=1.0)
    m = p.build_model(with_loads=True)
    LinearStaticAnalysis(m).run()
    total = rho * A * L * GRAVITY
    ry = m.node(1).reaction[1] + m.node(2).reaction[1]
    assert np.isclose(ry, total, rtol=1e-9)


def test_area_self_weight_3d_equilibrium():
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C30", E=30e9, nu=0.2, rho=2500.0))
    p.shell_sections.append(ShellSection(id=1, name="T200", thickness=0.20))
    Lx, Ly = 4.0, 3.0
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0, supports=(1, 1, 1, 1, 1, 1)),
        Node(id=2, x=Lx, y=0.0, z=0.0, supports=(1, 1, 1, 1, 1, 1)),
        Node(id=3, x=Lx, y=Ly, z=0.0, supports=(1, 1, 1, 1, 1, 1)),
        Node(id=4, x=0.0, y=Ly, z=0.0, supports=(1, 1, 1, 1, 1, 1)),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=(2, 2)))
    p.load_cases = [LoadCase(id=1, name="SW", nature="dead",
                             self_weight_factor=1.0)]
    m = p.build_model(with_loads=True)
    LinearStaticAnalysis(m).run()
    weight = 2500.0 * 0.20 * (Lx * Ly) * GRAVITY
    rz = sum(m.node(t).reaction[2] for t in m.nodes)
    assert np.isclose(rz, weight, rtol=1e-6)
