"""Tests for Phase 21 I/O extensions:

* VTK output for shells (MITC4 / Tri3) and solids (Hex8 / Tet4).
* VTK extra fields (reactions, mode shapes, custom point/cell data).
* JSON serialization for shells and solids.
* Beam force diagrams against analytical cantilever values.
* Extraction utilities: time-history, mode-shape table, capacity curve.
"""
import json
import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    EigenAnalysis,
    ElasticIsotropic,
    Hex8,
    LinearStaticAnalysis,
    Model,
    NonlinearStaticAnalysis,
    Quad4,
    ShellMITC4,
    ShellTri3,
    Tet4,
    Truss2D,
)
from femsolver.io import (
    beam_force_diagram,
    capacity_curve,
    gather_node_history,
    load_model_json,
    mode_shape_table,
    plot_beam_diagrams,
    save_model_json,
    write_vtk,
    write_vtk_unstructured,
)


# ====================================================== VTK -- new cell types

def test_vtk_writes_shell_quad_cell_type(tmp_path):
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    m.add_node(1, 0, 0, 0); m.add_node(2, 1, 0, 0)
    m.add_node(3, 1, 1, 0); m.add_node(4, 0, 1, 0)
    m.add_element(ShellMITC4(1, (1, 2, 3, 4), mat, thickness=0.01))
    f = tmp_path / "shell.vtk"
    write_vtk(m, str(f))
    text = f.read_text()
    # VTK_QUAD = 9 is the only cell type
    assert "CELL_TYPES 1" in text
    assert "\n9\n" in text


def test_vtk_writes_shell_tri_cell_type(tmp_path):
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    m.add_node(1, 0, 0, 0); m.add_node(2, 1, 0, 0); m.add_node(3, 0, 1, 0)
    m.add_element(ShellTri3(1, (1, 2, 3), mat, thickness=0.01))
    f = tmp_path / "tri.vtk"
    write_vtk(m, str(f))
    text = f.read_text()
    # VTK_TRIANGLE = 5
    assert "\n5\n" in text


def test_vtk_writes_hex8_cell_type(tmp_path):
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    coords = [(0,0,0), (1,0,0), (1,1,0), (0,1,0),
              (0,0,1), (1,0,1), (1,1,1), (0,1,1)]
    for i, (x, y, z) in enumerate(coords):
        m.add_node(i + 1, x, y, z)
    m.add_element(Hex8(1, tuple(range(1, 9)), mat))
    f = tmp_path / "hex.vtk"
    write_vtk(m, str(f))
    text = f.read_text()
    # VTK_HEXAHEDRON = 12
    assert "\n12\n" in text


def test_vtk_writes_tet4_cell_type(tmp_path):
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0, 0); m.add_node(2, 1, 0, 0)
    m.add_node(3, 0, 1, 0); m.add_node(4, 0, 0, 1)
    m.add_element(Tet4(1, (1, 2, 3, 4), mat))
    f = tmp_path / "tet.vtk"
    write_vtk(m, str(f))
    text = f.read_text()
    # VTK_TETRA = 10
    assert "\n10\n" in text


# ====================================================== VTK -- extra fields

def test_vtk_includes_reactions(tmp_path):
    """Run a linear-static problem (with all DOFs supported) and verify
    the reactions vector appears in the VTK file."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=7850.0)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    # Single truss with both ends fixed -> trivial reactions.
    m.add_node(1, 0, 0); m.add_node(2, 1, 0)
    m.add_element(Truss2D(1, (1, 2), mat, area=1e-4))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.add_nodal_load(2, [1e3, 0])
    LinearStaticAnalysis(m).run()
    f = tmp_path / "react.vtk"
    write_vtk(m, str(f), include_reactions=True)
    text = f.read_text()
    assert "VECTORS reaction float" in text


def test_vtk_includes_custom_point_data(tmp_path):
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 1, 0); m.add_node(3, 0, 1)
    m.add_element(Truss2D(1, (1, 2), mat, area=1e-4))
    f = tmp_path / "custom.vtk"
    write_vtk(m, str(f), point_data={"temperature": np.array([20.0, 25.0, 22.5])})
    text = f.read_text()
    assert "SCALARS temperature float 1" in text


def test_vtk_includes_custom_cell_data(tmp_path):
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 1, 0); m.add_node(3, 0, 1)
    m.add_element(Truss2D(1, (1, 2), mat, area=1e-4))
    m.add_element(Truss2D(2, (1, 3), mat, area=1e-4))
    f = tmp_path / "cell.vtk"
    write_vtk(m, str(f),
                cell_data={"axial_stress": np.array([100e6, 200e6])})
    text = f.read_text()
    assert "CELL_DATA 2" in text
    assert "SCALARS axial_stress float 1" in text


def test_vtk_includes_mode_shapes(tmp_path):
    """Run eigen, then write the first two modes as separate vectors."""
    mat = ElasticIsotropic(1, E=2e10, nu=0.3, rho=7850.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 3); m.add_node(3, 0, 6)
    m.add_element(BeamColumn2D(1, (1, 2), mat, area=1e-2, Iz=1e-5))
    m.add_element(BeamColumn2D(2, (2, 3), mat, area=1e-2, Iz=1e-5))
    m.fix(1, [1, 1, 1])
    EigenAnalysis(m, num_modes=2).run()
    f = tmp_path / "modes.vtk"
    write_vtk(m, str(f), include_mode_shapes=True)
    text = f.read_text()
    assert "VECTORS mode_1 float" in text
    assert "VECTORS mode_2 float" in text


def test_vtk_unstructured_backward_compat(tmp_path):
    """The legacy entry point write_vtk_unstructured still works."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 1, 0)
    m.add_element(Truss2D(1, (1, 2), mat, area=1e-4))
    f = tmp_path / "legacy.vtk"
    write_vtk_unstructured(m, str(f), deformation_scale=10.0)
    assert f.exists()


# ====================================================== JSON -- new elements

def test_json_roundtrip_with_shell_mitc4(tmp_path):
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    m.add_node(1, 0, 0, 0); m.add_node(2, 1, 0, 0)
    m.add_node(3, 1, 1, 0); m.add_node(4, 0, 1, 0)
    m.add_element(ShellMITC4(1, (1, 2, 3, 4), mat, thickness=0.05))
    p = tmp_path / "shell.json"
    save_model_json(m, p)
    m2 = load_model_json(p)
    e = m2.element(1)
    assert e.thickness == 0.05
    assert isinstance(e, ShellMITC4)


def test_json_roundtrip_with_shell_tri3(tmp_path):
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    m.add_node(1, 0, 0, 0); m.add_node(2, 1, 0, 0); m.add_node(3, 0, 1, 0)
    m.add_element(ShellTri3(1, (1, 2, 3), mat, thickness=0.03))
    p = tmp_path / "tri.json"
    save_model_json(m, p)
    m2 = load_model_json(p)
    e = m2.element(1)
    assert e.thickness == 0.03
    assert isinstance(e, ShellTri3)


def test_json_roundtrip_with_hex8(tmp_path):
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    coords = [(0,0,0), (1,0,0), (1,1,0), (0,1,0),
              (0,0,1), (1,0,1), (1,1,1), (0,1,1)]
    for i, (x, y, z) in enumerate(coords):
        m.add_node(i + 1, x, y, z)
    m.add_element(Hex8(1, tuple(range(1, 9)), mat))
    p = tmp_path / "hex.json"
    save_model_json(m, p)
    m2 = load_model_json(p)
    assert isinstance(m2.element(1), Hex8)


def test_json_roundtrip_with_tet4(tmp_path):
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0, 0); m.add_node(2, 1, 0, 0)
    m.add_node(3, 0, 1, 0); m.add_node(4, 0, 0, 1)
    m.add_element(Tet4(1, (1, 2, 3, 4), mat))
    p = tmp_path / "tet.json"
    save_model_json(m, p)
    m2 = load_model_json(p)
    assert isinstance(m2.element(1), Tet4)


# ====================================================== Force diagrams

def test_beam_diagram_cantilever_tip_vertical_load():
    """For a 2D cantilever with tip vertical load P (downward), the
    moment at the root is P * L in magnitude, zero at the tip."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    L = 2.0
    P = -150.0
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, L, 0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, area=1e-2, Iz=1e-4))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0, P, 0])
    LinearStaticAnalysis(m).run()
    fd = beam_force_diagram(m.element(1), n_points=21)
    # M at the root has magnitude |P * L|
    assert abs(fd["M"][0]) == pytest.approx(abs(P * L), rel=1e-6)
    # M at the tip is zero
    assert abs(fd["M"][-1]) == pytest.approx(0.0, abs=1e-6)
    # N is identically zero
    assert np.max(np.abs(fd["N"])) < 1e-6
    # V is constant
    assert np.max(np.abs(fd["V"] - fd["V"][0])) < 1e-6


def test_beam_diagram_axial_load():
    """Pure axial cantilever: N = -P (compression), V = M = 0."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    L = 1.0
    P = -1000.0     # compression
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, L, 0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, area=1e-2, Iz=1e-4))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [P, 0, 0])
    LinearStaticAnalysis(m).run()
    fd = beam_force_diagram(m.element(1), n_points=10)
    # N along the beam equals P (compression -> negative)
    assert np.allclose(fd["N"], P, rtol=1e-6)
    assert np.max(np.abs(fd["V"])) < 1e-6
    assert np.max(np.abs(fd["M"])) < 1e-6


def test_beam_diagram_truss_only_axial():
    """A truss element gives an axial diagram with V = M = 0."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 1.0, 0)
    m.add_element(Truss2D(1, (1, 2), mat, area=1e-4))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.add_nodal_load(2, [500.0, 0])
    LinearStaticAnalysis(m).run()
    fd = beam_force_diagram(m.element(1), n_points=10)
    assert np.allclose(fd["N"], 500.0, rtol=1e-6)
    assert np.max(np.abs(fd["V"])) < 1e-12
    assert np.max(np.abs(fd["M"])) < 1e-12


# ====================================================== Extraction utilities

def test_gather_node_history_simple():
    """The history dict round-trips ``times`` and ``tracked_disp``."""
    results = {
        "times": [0.0, 0.1, 0.2, 0.3],
        "tracked_disp": [0.0, 0.001, 0.002, 0.001],
        "tracked_node": 5,
        "tracked_dof": 1,
    }
    out = gather_node_history(None, results, node_tag=5, dof=1)
    assert out["node"] == 5
    assert out["dof"] == 1
    np.testing.assert_allclose(out["times"], [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(out["disp"], [0.0, 0.001, 0.002, 0.001])


def test_gather_node_history_rejects_non_transient_dict():
    with pytest.raises(ValueError, match="times"):
        gather_node_history(None, {"foo": "bar"}, node_tag=1, dof=0)


def test_mode_shape_table_from_eigen():
    """Run a real eigen analysis and verify the table contains the
    right columns."""
    mat = ElasticIsotropic(1, E=2e10, nu=0.3, rho=7850.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 0, 3); m.add_node(3, 0, 6)
    m.add_element(BeamColumn2D(1, (1, 2), mat, area=1e-2, Iz=1e-5))
    m.add_element(BeamColumn2D(2, (2, 3), mat, area=1e-2, Iz=1e-5))
    m.fix(1, [1, 1, 1])
    eig = EigenAnalysis(m, num_modes=3).run()
    tbl = mode_shape_table(eig)
    assert tbl["period_s"].size == 3
    np.testing.assert_array_equal(tbl["mode"], [1, 2, 3])
    # Periods should be monotonically decreasing
    assert tbl["period_s"][0] > tbl["period_s"][1] > tbl["period_s"][2]


def test_capacity_curve_extraction():
    """Pull a (drift, force) capacity curve from a nonlinear-static
    result."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 1.0, 0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, area=1e-2, Iz=1e-4))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0, -1.0e3, 0])
    res = NonlinearStaticAnalysis(
        m, num_steps=10, dlambda=1.0 / 10, tol=1e-6,
        track=(2, 1),
    ).run()
    # NonlinearStaticAnalysis returns 'tracked' (drift) and 'lambdas'
    # (load factors); capacity_curve defaults to these.
    cc = capacity_curve(res)
    assert "drift" in cc
    assert "force" in cc
    assert cc["drift"].size == cc["force"].size
    assert cc["drift"].size == 10
