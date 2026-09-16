"""Slab-modeling S7-lite — nodal displacement contour on area elements.

Headless coverage of ``model_geometry.areas_contour_mesh``: after a solve the
filled-face mesh carries the right per-node scalar (magnitude / component), and
it is ``None`` when the model has no surfaces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from femsolver.analysis.linear_static import LinearStaticAnalysis  # noqa: E402

import model_geometry as mg  # noqa: E402
from project import (AREA_TAG_BASE, Area, AreaLoad, Material,  # noqa: E402
                     Node, Project, ShellSection)


def _solved_slab():
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C30", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="SLAB200", thickness=0.20))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=4.0, y=3.0, z=0.0),
        Node(id=4, x=0.0, y=3.0, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1))
    p.area_loads.append(AreaLoad(area=1, w=5000.0, kind="gravity", case=1))
    m = p.build_model(with_loads=True)
    for tag in (1, 2, 3):
        m.fix(tag, [1, 1, 1, 1, 1, 1])
    LinearStaticAnalysis(m).run()
    return m


def test_contour_none_without_areas():
    p = Project(ndm=3, ndf=6)
    p.nodes.append(Node(id=1, x=0.0, y=0.0, z=0.0))
    assert mg.areas_contour_mesh(p.build_model(with_loads=False)) is None


def test_contour_magnitude_scalar_present():
    m = _solved_slab()
    poly = mg.areas_contour_mesh(m, "Umag")
    assert poly is not None
    vals = poly.point_data["value"]
    assert vals.shape[0] == poly.n_points
    assert np.all(vals >= 0.0)                 # magnitude
    assert float(vals.max()) > 0.0             # something moved


def test_contour_uz_free_corner_is_negative_and_extreme():
    m = _solved_slab()
    poly = mg.areas_contour_mesh(m, "Uz")
    vals = poly.point_data["value"]
    # node order is model insertion order (1,2,3,4); node 4 is the free corner
    assert vals[3] < 0.0                        # free corner sagged
    assert abs(vals[3]) == max(abs(v) for v in vals)


def test_contour_matches_solved_disp():
    m = _solved_slab()
    poly = mg.areas_contour_mesh(m, "Uz")
    uz4 = m.node(4).disp[2]
    assert poly.point_data["value"][3] == uz4
