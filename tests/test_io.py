"""I/O tests: JSON round-trip + VTK output."""
import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    EqualDOF,
    LinearStaticAnalysis,
    MPConstraint,
    Model,
    Quad4,
    RigidDiaphragm,
    RigidLink,
    Truss2D,
)
from femsolver.results import save_model_json, load_model_json, write_vtk_unstructured


def test_json_roundtrip_truss(tmp_path):
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.add_node(3, 1.0, 1.0)
    mat = ElasticIsotropic(1, E=2e11, nu=0.3, rho=7850.0)
    m.add_material(mat)
    m.add_element(Truss2D(1, (1, 2), mat, 1e-4))
    m.add_element(Truss2D(2, (2, 3), mat, 2e-4))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.add_nodal_load(3, [1e3, 0.0])

    p = tmp_path / "model.json"
    save_model_json(m, p)
    m2 = load_model_json(p)

    assert m2.ndm == 2 and m2.ndf == 2
    assert len(m2.nodes) == 3 and len(m2.elements) == 2
    assert m2.element(2).area == 2e-4
    assert m2.material(1).rho == 7850.0
    np.testing.assert_array_equal(m2.node(1).fixity, [True, True])
    np.testing.assert_allclose(m2.node(3)._load, [1e3, 0.0])


def test_json_roundtrip_full(tmp_path):
    """Round-trip a model containing one of every supported element, run the
    analysis on both, and confirm equal results."""
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 2.0, 0.0)
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m.add_material(mat)
    m.add_element(BeamColumn2D(1, (1, 2), mat, 1e-2, 1e-5))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -1e3, 0.0])
    LinearStaticAnalysis(m).run()
    d1 = m.node(2).disp.copy()

    p = tmp_path / "beam.json"
    save_model_json(m, p)
    m2 = load_model_json(p)
    LinearStaticAnalysis(m2).run()
    np.testing.assert_allclose(m2.node(2).disp, d1, rtol=1e-12)


def test_json_roundtrip_constraints(tmp_path):
    """All four constraint types round-trip through JSON, and the loaded
    model produces identical analysis results."""
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m.add_material(mat)
    for i, (x, y) in enumerate([(0.0, 0.0), (2.0, 0.0), (2.0, 0.5), (4.0, 0.0)]):
        m.add_node(i + 1, x, y)
    m.fix(1, [1, 1, 1])
    m.fix(3, [0, 0, 1])
    m.fix(4, [0, 1, 1])
    m.add_element(BeamColumn2D(1, (1, 2), mat, 1e-2, 1e-5))
    # Cover EqualDOF, RigidLink (bar), and a general MPConstraint
    m.equal_dof(retained=2, constrained=4, dofs=[0])
    m.rigid_link(retained=2, constrained=3, kind="bar")
    m.add_mp_constraint(MPConstraint(constrained=(4, 0), retained=[(2, 0, 1.0)]))
    m.add_nodal_load(2, [0.0, -1e3, 0.0])

    # the second add (MPConstraint on node 4 dof 0) duplicates EqualDOF — drop it
    m._mp_constraints.pop()  # keep round-trip non-duplicate

    LinearStaticAnalysis(m).run()
    d2 = m.node(2).disp.copy()
    d3 = m.node(3).disp.copy()

    p = tmp_path / "with_constraints.json"
    save_model_json(m, p)
    m2 = load_model_json(p)

    # constraint set survives the round-trip
    assert len(m2.mp_constraints) == 2
    types = sorted(type(c).__name__ for c in m2.mp_constraints)
    assert types == ["EqualDOF", "RigidLink"]
    rl = next(c for c in m2.mp_constraints if isinstance(c, RigidLink))
    assert rl.kind == "bar"

    # rerun and confirm physics is identical
    LinearStaticAnalysis(m2).run()
    np.testing.assert_allclose(m2.node(2).disp, d2, rtol=1e-12)
    np.testing.assert_allclose(m2.node(3).disp, d3, rtol=1e-12)


def test_json_roundtrip_rigid_diaphragm_and_mp(tmp_path):
    """3D model with RigidDiaphragm and a generic MPConstraint with g != 0."""
    m = Model(ndm=3, ndf=6)
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0, 0.0)
    m.add_node(3, 2.0, 0.0, 0.0)
    m.add_mp_constraint(RigidDiaphragm(master=1, slaves=[2, 3], perp_dir=2))
    m.add_mp_constraint(
        MPConstraint(constrained=(2, 5), retained=[(1, 5, 0.5)], g=0.001)
    )

    p = tmp_path / "diaphragm.json"
    save_model_json(m, p)
    m2 = load_model_json(p)
    assert len(m2.mp_constraints) == 2
    diaph = next(c for c in m2.mp_constraints if isinstance(c, RigidDiaphragm))
    assert diaph.slaves == [2, 3]
    assert diaph.perp_dir == 2
    mp = next(c for c in m2.mp_constraints if isinstance(c, MPConstraint))
    assert mp.g == 0.001
    assert mp.r_terms == [(1, 5, 0.5)]


def test_json_old_version_loads_without_constraints(tmp_path):
    """Files written by the v0.1 schema (no 'constraints' key) still load."""
    import json as _json
    payload = {
        "version": "0.1",
        "ndm": 2,
        "ndf": 2,
        "nodes": [
            {"tag": 1, "coords": [0.0, 0.0], "fixity": [1, 1], "load": [0.0, 0.0]},
            {"tag": 2, "coords": [1.0, 0.0], "fixity": [0, 1], "load": [10.0, 0.0]},
        ],
        "materials": [
            {"type": "ElasticIsotropic",
             "params": {"tag": 1, "E": 1e7, "nu": 0.3, "rho": 0.0}}
        ],
        "elements": [
            {"type": "Truss2D",
             "params": {"tag": 1, "nodes": [1, 2], "material_tag": 1, "area": 1.0}}
        ],
    }
    p = tmp_path / "v01.json"
    p.write_text(_json.dumps(payload))
    m = load_model_json(p)
    assert len(m.mp_constraints) == 0
    LinearStaticAnalysis(m).run()  # works end-to-end


def test_vtk_output(tmp_path):
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    m.add_node(3, 1.0, 1.0)
    m.add_node(4, 0.0, 1.0)
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m.add_material(mat)
    m.add_element(Quad4(1, (1, 2, 3, 4), mat, thickness=0.1))
    m.fix(1, [1, 1])
    m.fix(4, [1, 0])
    m.add_nodal_load(2, [1e3, 0.0])
    m.add_nodal_load(3, [1e3, 0.0])
    LinearStaticAnalysis(m).run()
    p = tmp_path / "out.vtk"
    write_vtk_unstructured(m, p, deformation_scale=1.0)
    text = p.read_text()
    assert "DATASET UNSTRUCTURED_GRID" in text
    assert "POINTS 4 float" in text
    assert "CELLS 1 5" in text
    assert "VECTORS displacement float" in text
