"""Slab-modeling S1 — build_model emits shell elements for Areas.

Covers the compile step: a 3-D ``Project`` with ``Area`` objects produces the
right engine surface elements (MITC4 quad / Tri3 / DKMQ4 plate), the 3-D gate
(2-D projects have no surfaces), graceful skipping of dangling references, the
SAP-style stiffness-modifier wrapper, and an end-to-end solve that responds to
a bending modifier.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from femsolver import ShellDKMQ4, ShellMITC4, ShellTri3  # noqa: E402
from femsolver.analysis.linear_static import LinearStaticAnalysis  # noqa: E402

from project import (  # noqa: E402
    AREA_TAG_BASE,
    Area,
    Material,
    Node,
    Project,
    ShellSection,
    _ModifiedShellSection,
)


def _quad_project(kind="shell-thin", ndm=3, modifiers=None):
    ndf = 6 if ndm == 3 else 3
    p = Project(ndm=ndm, ndf=ndf)
    p.materials.append(Material(id=1, name="C30", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(
        ShellSection(id=1, name="SLAB200", thickness=0.20, kind=kind,
                     modifiers=modifiers or {})
    )
    z = 0.0
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=z),
        Node(id=2, x=4.0, y=0.0, z=z),
        Node(id=3, x=4.0, y=3.0, z=z),
        Node(id=4, x=0.0, y=3.0, z=z),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1))
    return p


def _elements(model):
    return list(model.elements.values())


# --------------------------------------------------------- element mapping

def test_build_emits_shell_quad_mitc4():
    m = _quad_project(kind="shell-thin").build_model(with_loads=False)
    els = _elements(m)
    assert len(els) == 1
    e = els[0]
    assert isinstance(e, ShellMITC4)
    assert e.tag == AREA_TAG_BASE + 1
    assert tuple(e.node_tags) == (1, 2, 3, 4)
    assert e.thickness == pytest.approx(0.20)


def test_build_emits_shell_tri3():
    p = _quad_project()
    p.areas[0] = Area(id=1, nodes=[1, 2, 3], shell_section=1, material=1)
    els = _elements(p.build_model(with_loads=False))
    assert len(els) == 1
    assert isinstance(els[0], ShellTri3)


def test_build_plate_kind_uses_dkmq4():
    m = _quad_project(kind="plate-thin").build_model(with_loads=False)
    els = _elements(m)
    assert isinstance(els[0], ShellDKMQ4)


# ---------------------------------------------------------------- 3-D gate

def test_2d_project_skips_areas():
    p = _quad_project(ndm=2)
    # a 2-D model can't carry shells; the area is silently skipped
    assert _elements(p.build_model(with_loads=False)) == []


# --------------------------------------------------------- dangling refs

def test_missing_shell_section_skipped():
    p = _quad_project()
    p.areas[0].shell_section = 999          # no such section
    assert _elements(p.build_model(with_loads=False)) == []


def test_missing_material_skipped():
    p = _quad_project()
    p.areas[0].material = 999               # no such material
    assert _elements(p.build_model(with_loads=False)) == []


# ---------------------------------------------------- modifier wrapper

def test_modifier_wrapper_uniform_bending_scale():
    from femsolver import ElasticIsotropic, ElasticShellSection
    mat = ElasticIsotropic(1, E=30e9, nu=0.2)
    base = ElasticShellSection(mat, 0.2)
    wrapped = _ModifiedShellSection(base, {"m11": 0.5, "m22": 0.5, "m12": 0.5})
    # uniform 0.5 on all three bending components -> exactly 0.5 * D_bending
    assert np.allclose(wrapped.D_bending(), 0.5 * base.D_bending())
    # membrane untouched (all f* default to 1.0)
    assert np.allclose(wrapped.D_membrane(), base.D_membrane())
    # thickness / k_shear pass through
    assert wrapped.thickness == pytest.approx(0.2)


def test_modifier_wrapper_symmetric_offdiagonal():
    from femsolver import ElasticIsotropic, ElasticShellSection
    mat = ElasticIsotropic(1, E=30e9, nu=0.2)
    base = ElasticShellSection(mat, 0.2)
    wrapped = _ModifiedShellSection(base, {"f11": 0.25, "f22": 1.0, "f12": 1.0})
    D = wrapped.D_membrane()
    assert np.allclose(D, D.T)                  # stays symmetric
    b = base.D_membrane()
    assert D[0, 0] == pytest.approx(0.25 * b[0, 0])    # sqrt(.25*.25)=.25
    assert D[0, 1] == pytest.approx(0.5 * b[0, 1])     # sqrt(.25*1)=.5


# ---------------------------------------------------------- end-to-end solve

def _clamped_corner_load(p, modifier_factor=None):
    """Build the quad, clamp 3 corners, push the 4th down 10 kN, solve,
    return the free-corner vertical deflection (negative = down)."""
    if modifier_factor is not None:
        p.shell_sections[0].modifiers = {
            "m11": modifier_factor, "m22": modifier_factor,
            "m12": modifier_factor,
        }
    m = p.build_model(with_loads=False)
    for tag in (1, 2, 3):
        m.fix(tag, [1, 1, 1, 1, 1, 1])
    m.add_nodal_load(4, [0, 0, -1.0e4, 0, 0, 0])
    LinearStaticAnalysis(m).run()
    return m.node(4).disp[2]


def test_area_slab_solves_and_deflects_down():
    w = _clamped_corner_load(_quad_project())
    assert np.isfinite(w)
    assert w < 0.0            # pushed down -> deflects down


def test_softer_bending_modifier_increases_deflection():
    w_full = abs(_clamped_corner_load(_quad_project()))
    w_soft = abs(_clamped_corner_load(_quad_project(), modifier_factor=0.25))
    assert w_soft > w_full     # a softer plate deflects more
