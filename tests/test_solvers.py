"""Tests for the Phase 20 solver infrastructure:

* ``DirectSparseSolver`` vs ``IterativeSolver`` give identical answers
  on canonical problems.
* Sparse vs dense ``LinearBucklingAnalysis`` give consistent results.
* Iterative solver works with and without ILU preconditioning.
"""
import math

import numpy as np
import pytest
import scipy.sparse as sp

from femsolver import (
    BeamColumn2D,
    BeamColumn2DCorotational,
    DirectSparseSolver,
    ElasticIsotropic,
    IterativeSolver,
    LinearStaticAnalysis,
    Model,
    Truss2D,
)
from femsolver.analysis.buckling import LinearBucklingAnalysis


# ====================================================== solver basics

def test_direct_solver_solves_small_system():
    """Tiny SPD system: K = [[2, -1], [-1, 2]], b = [1, 0], x = (2, 1) / 3."""
    K = sp.csc_matrix([[2.0, -1.0], [-1.0, 2.0]])
    b = np.array([1.0, 0.0])
    s = DirectSparseSolver()
    x = s.solve(K, b)
    np.testing.assert_allclose(x, np.array([2.0, 1.0]) / 3.0, rtol=1e-12)


def test_direct_solver_singular_raises():
    """A singular matrix should raise with a helpful message. The
    underlying ``spsolve`` warns via MatrixRankWarning before our code
    raises RuntimeError -- the warning is expected and silenced
    here so the test output stays clean."""
    import warnings
    from scipy.sparse.linalg import MatrixRankWarning
    K = sp.csc_matrix([[1.0, 1.0], [1.0, 1.0]])  # rank 1
    b = np.array([1.0, 1.0])
    s = DirectSparseSolver()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MatrixRankWarning)
        with pytest.raises(RuntimeError,
                              match="singular|non-finite|sparse solve failed"):
            s.solve(K, b)


def test_iterative_solver_construction_validates_method():
    with pytest.raises(ValueError, match="method"):
        IterativeSolver(method="bicgstab")


def test_iterative_solver_construction_validates_preconditioner():
    with pytest.raises(ValueError, match="preconditioner"):
        IterativeSolver(preconditioner="amg")


def test_iterative_solver_cg_matches_direct():
    """For an SPD system, CG converges to the same answer as direct."""
    np.random.seed(42)
    n = 50
    A = sp.random(n, n, density=0.1, format="csr")
    A = (A + A.T + n * sp.eye(n)).tocsc()  # SPD
    b = np.random.rand(n)
    x_direct = DirectSparseSolver().solve(A, b)
    x_iter = IterativeSolver(method="cg", tol=1e-12).solve(A, b)
    np.testing.assert_allclose(x_iter, x_direct, rtol=1e-8)


def test_iterative_solver_gmres_matches_direct_on_nonsymmetric():
    """Non-symmetric system: CG would fail; GMRES converges."""
    np.random.seed(42)
    n = 30
    A_dense = np.random.rand(n, n) + n * np.eye(n)  # diag-dominant
    A = sp.csc_matrix(A_dense)
    b = np.random.rand(n)
    x_direct = DirectSparseSolver().solve(A, b)
    x_iter = IterativeSolver(method="gmres", tol=1e-12).solve(A, b)
    np.testing.assert_allclose(x_iter, x_direct, rtol=1e-6)


def test_iterative_solver_reports_iteration_count():
    np.random.seed(0)
    n = 20
    A = sp.random(n, n, density=0.2, format="csc")
    A = (A + A.T + n * sp.eye(n)).tocsc()
    b = np.random.rand(n)
    s = IterativeSolver(method="cg", tol=1e-10)
    s.solve(A, b)
    assert s.last_iterations > 0


def test_iterative_solver_no_preconditioner_works():
    """Without ILU, the iterative solver still converges (just slower)."""
    np.random.seed(0)
    n = 20
    A = sp.random(n, n, density=0.2, format="csc")
    A = (A + A.T + n * sp.eye(n)).tocsc()
    b = np.random.rand(n)
    s = IterativeSolver(method="cg", tol=1e-10, preconditioner="none")
    x = s.solve(A, b)
    x_ref = DirectSparseSolver().solve(A, b)
    np.testing.assert_allclose(x, x_ref, rtol=1e-7)


# ====================================================== LinearStatic with solvers

def test_linear_static_with_direct_solver():
    """Default direct solver: existing behaviour."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m = Model(ndm=2, ndf=2); m.add_material(mat)
    m.add_node(1, 0, 0); m.add_node(2, 1, 0); m.add_node(3, 0, 1)
    m.add_element(Truss2D(1, (1, 2), mat, area=1e-4))
    m.add_element(Truss2D(2, (1, 3), mat, area=1e-4))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.fix(3, [1, 0])
    m.add_nodal_load(2, [1000.0, 0])
    LinearStaticAnalysis(m).run()
    assert m.node(2).disp[0] > 0


def test_linear_static_with_iterative_solver_matches_direct():
    """Iterative solver gives the same answer as direct."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)

    def _build():
        m = Model(ndm=2, ndf=2); m.add_material(mat)
        # 10-element 1-D truss row -- a single chain of axial-only springs
        # so all y DOFs need restraint.
        n_nodes = 11
        for i in range(n_nodes):
            m.add_node(i + 1, float(i), 0.0)
        for i in range(n_nodes - 1):
            m.add_element(Truss2D(i + 1, (i + 1, i + 2), mat, area=1e-4))
        m.fix(1, [1, 1])
        # Restrain y at every node (1-D truss row)
        for i in range(2, n_nodes + 1):
            m.fix(i, [0, 1])
        m.add_nodal_load(6, [1000.0, 0])
        return m

    m_d = _build()
    LinearStaticAnalysis(m_d).run()
    m_i = _build()
    LinearStaticAnalysis(
        m_i,
        solver=IterativeSolver(method="cg", tol=1e-14),
    ).run()
    for tag in range(1, 12):
        np.testing.assert_allclose(
            m_d.node(tag).disp, m_i.node(tag).disp, rtol=1e-7,
        )


# ====================================================== Buckling sparse/dense

def _build_pin_pin_column(n_elem: int = 4, *,
                            E: float = 2e11, L: float = 1.0,
                            A: float = 1e-4, I: float = 1e-8) -> Model:
    """Pin-pin corotational column ready for buckling. Returns the
    model with a unit compressive reference load applied at the top."""
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, 0.0, i * L / n_elem)
    for i in range(n_elem):
        m.add_element(BeamColumn2DCorotational(
            i + 1, (i + 1, i + 2), mat, area=A, Iz=I,
        ))
    m.fix(1, [1, 1, 0])
    m.fix(n_elem + 1, [1, 0, 0])
    m.add_nodal_load(n_elem + 1, [0.0, -1.0, 0.0])    # unit compression
    return m


def test_buckling_dense_and_sparse_agree():
    """The new sparse path gives the same critical load factor as
    the original dense path (within numerical tolerance)."""
    m_d = _build_pin_pin_column(n_elem=6)
    res_d = LinearBucklingAnalysis(m_d, num_modes=1, mode="dense").run()
    m_s = _build_pin_pin_column(n_elem=6)
    res_s = LinearBucklingAnalysis(m_s, num_modes=1, mode="sparse").run()
    assert res_s["critical_load_factor"] == pytest.approx(
        res_d["critical_load_factor"], rel=1e-6
    )


def test_buckling_sparse_default_for_larger_models():
    """Default mode is 'sparse'; verify it still finds the critical
    load factor on a moderately-sized model."""
    m = _build_pin_pin_column(n_elem=20)
    res = LinearBucklingAnalysis(m, num_modes=1).run()    # default sparse
    # Pin-pin Euler load: P_cr = pi^2 EI / L^2
    EI = 2e11 * 1e-8
    P_cr_euler = math.pi ** 2 * EI / 1.0 ** 2
    assert res["critical_load_factor"] == pytest.approx(P_cr_euler, rel=2e-2)


def test_buckling_falls_back_to_dense_for_small_neq():
    """When neq is too small for eigsh's Lanczos space, the sparse
    request falls back to dense automatically rather than failing."""
    # 3-element pin-pin: small enough that sparse with k = 2*1+2 = 4 may
    # be too many for the available neq. The fallback to dense still
    # finds the critical load factor.
    m = _build_pin_pin_column(n_elem=3)
    res = LinearBucklingAnalysis(m, num_modes=1, mode="sparse").run()
    EI = 2e11 * 1e-8
    P_cr_euler = math.pi ** 2 * EI / 1.0 ** 2
    assert res["critical_load_factor"] == pytest.approx(P_cr_euler, rel=0.1)


def test_buckling_construction_validates_mode():
    m = _build_pin_pin_column(n_elem=3)
    with pytest.raises(ValueError, match="mode"):
        LinearBucklingAnalysis(m, num_modes=1, mode="iterative")
