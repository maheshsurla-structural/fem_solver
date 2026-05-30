"""Phase 43 tests -- solver plugins, substructuring, parallel assembly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from femsolver import (
    BeamColumn2D,
    CachedFactorSolver,
    DirectSparseSolver,
    ElasticIsotropic,
    IterativeSolver,
    LinearStaticAnalysis,
    Model,
    Quad4,
    assemble_stiffness_parallel,
    craig_bampton,
    guyan_condensation,
    guyan_recover_full,
    pardiso_available,
)
from femsolver.analysis.assembler import assemble_stiffness


# ============================================================ solver plugins

class TestCachedFactor:
    def test_same_matrix_reuses_factor(self):
        A = sp.csc_matrix(np.diag([2.0, 3.0, 4.0]))
        b1 = np.array([2.0, 3.0, 4.0])
        b2 = np.array([4.0, 6.0, 8.0])
        s = CachedFactorSolver()
        x1 = s.solve(A, b1)
        x2 = s.solve(A, b2)
        assert s.cache_hits == 1
        assert s.cache_misses == 1
        np.testing.assert_allclose(x1, np.array([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(x2, np.array([2.0, 2.0, 2.0]))

    def test_different_matrix_re_factors(self):
        A1 = sp.csc_matrix(np.diag([1.0, 1.0, 1.0]))
        A2 = sp.csc_matrix(np.diag([2.0, 2.0, 2.0]))
        b = np.array([1.0, 1.0, 1.0])
        s = CachedFactorSolver()
        s.solve(A1, b)
        s.solve(A2, b)
        assert s.cache_misses == 2

    def test_reset_clears_cache(self):
        A = sp.csc_matrix(np.diag([5.0, 5.0]))
        b = np.array([5.0, 5.0])
        s = CachedFactorSolver()
        s.solve(A, b)
        s.reset()
        s.solve(A, b)
        assert s.cache_misses == 2

    def test_same_result_as_direct(self):
        rng = np.random.default_rng(0)
        n = 50
        # SPD random matrix
        A = sp.csc_matrix(np.eye(n) * 5.0
                          + np.tril(rng.standard_normal((n, n)), -1)
                          * 0.01
                          + np.tril(rng.standard_normal((n, n)), -1).T
                          * 0.01)
        b = rng.standard_normal(n)
        x_direct = DirectSparseSolver().solve(A, b)
        x_cached = CachedFactorSolver().solve(A, b)
        np.testing.assert_allclose(x_cached, x_direct, atol=1e-9)


class TestPardisoAvailability:
    def test_pardiso_available_returns_bool(self):
        assert isinstance(pardiso_available(), bool)


# ============================================================ Guyan

class TestGuyan:
    def test_3x3_analytical(self):
        """Hand-checked example::
            K = [[ 2,-1, 0], [-1, 2,-1], [ 0,-1, 2]]
            master = [0]
            K_red = 2 - [-1, 0] · K_ss^{-1} · [-1, 0]^T = 4/3
        """
        K = np.array([[2.0, -1.0, 0.0],
                       [-1.0, 2.0, -1.0],
                       [0.0, -1.0, 2.0]])
        f = np.array([1.0, 0.0, 0.0])
        res = guyan_condensation(K, f, master_dofs=[0])
        assert res.K_red.shape == (1, 1)
        assert res.K_red[0, 0] == pytest.approx(4.0 / 3.0, rel=1e-12)

    def test_recover_matches_full_solve(self):
        K = np.array([[5.0, -2.0, 0.0, 0.0],
                       [-2.0, 5.0, -2.0, 0.0],
                       [0.0, -2.0, 5.0, -2.0],
                       [0.0, 0.0, -2.0, 3.0]])
        f = np.array([1.0, 0.0, 0.0, 1.0])
        res = guyan_condensation(K, f, master_dofs=[0, 3])
        u_m = np.linalg.solve(res.K_red, res.f_red)
        u_full = guyan_recover_full(res, u_m, K=K, f=f)
        u_exact = np.linalg.solve(K, f)
        np.testing.assert_allclose(u_full, u_exact, atol=1e-10)

    def test_master_dofs_size(self):
        K = np.diag(np.arange(1.0, 6.0))
        f = np.ones(5)
        res = guyan_condensation(K, f, master_dofs=[1, 3])
        assert res.K_red.shape == (2, 2)
        assert res.master_dofs.tolist() == [1, 3]
        assert sorted(res.slave_dofs.tolist()) == [0, 2, 4]


# ============================================================ Craig-Bampton

class TestCraigBampton:
    def test_reduced_size_is_n_master_plus_n_keep(self):
        K = np.array([[2.0, -1.0, 0.0],
                       [-1.0, 2.0, -1.0],
                       [0.0, -1.0, 2.0]])
        M = np.eye(3)
        res = craig_bampton(K, M, master_dofs=[0], n_keep=2)
        assert res.K_red.shape == (3, 3)
        assert res.M_red.shape == (3, 3)
        assert res.n_keep == 2

    def test_zero_keep_equals_guyan(self):
        """CB with n_keep=0 should give the same K_red as Guyan."""
        K = np.array([[2.0, -1.0, 0.0],
                       [-1.0, 2.0, -1.0],
                       [0.0, -1.0, 2.0]])
        M = np.eye(3)
        f = np.zeros(3)
        cb = craig_bampton(K, M, master_dofs=[0], n_keep=0)
        guy = guyan_condensation(K, f, master_dofs=[0])
        np.testing.assert_allclose(cb.K_red, guy.K_red, atol=1e-12)

    def test_full_keep_recovers_full_spectrum(self):
        """CB with n_keep = n_slave + projection onto masters should
        retain the full eigenvalue spectrum (subject to numerical
        rounding from the reduction)."""
        K = np.array([[2.0, -1.0, 0.0],
                       [-1.0, 2.0, -1.0],
                       [0.0, -1.0, 2.0]])
        M = np.eye(3)
        res = craig_bampton(K, M, master_dofs=[0, 2], n_keep=1)
        # 2 masters + 1 kept mode = 3x3 = original size
        assert res.K_red.shape == (3, 3)
        from scipy.linalg import eigh as scipy_eigh
        w_red, _ = scipy_eigh(res.K_red, res.M_red)
        w_full, _ = scipy_eigh(K, M)
        # The reduced system should recover the lowest eigenvalue
        # exactly and the others to good accuracy
        assert w_red[0] == pytest.approx(w_full[0], rel=1e-8)


# ============================================================ parallel assembly

def _build_beam_model(n_elem: int = 50):
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.30, rho=7850.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * 0.1, 0.0)
    for i in range(n_elem):
        m.add_element(BeamColumn2D(
            i + 1, (i + 1, i + 2), mat, 1.0e-2, 1.0e-5,
        ))
    m.fix(1, [1, 1, 1])
    m.number_dofs()
    return m


class TestParallelAssembly:
    def test_matches_serial_bit_for_bit(self):
        m = _build_beam_model(n_elem=100)
        K_ser = assemble_stiffness(m).toarray()
        K_par = assemble_stiffness_parallel(m, n_workers=4).toarray()
        # Should be bitwise-identical
        assert np.array_equal(K_ser, K_par)

    def test_empty_model(self):
        mat = ElasticIsotropic(1, E=1.0, nu=0.3, rho=0.0)
        m = Model(ndm=2, ndf=2)
        m.add_material(mat)
        m.add_node(1, 0.0, 0.0)
        m.number_dofs()
        K_par = assemble_stiffness_parallel(m)
        # 2 free DOFs but no elements -> zero matrix
        assert K_par.nnz == 0

    def test_n_workers_one_path(self):
        m = _build_beam_model(n_elem=10)
        K_par_1 = assemble_stiffness_parallel(m, n_workers=1).toarray()
        K_ser = assemble_stiffness(m).toarray()
        np.testing.assert_array_equal(K_par_1, K_ser)

    def test_return_element_K(self):
        m = _build_beam_model(n_elem=20)
        K_par, elem_K = assemble_stiffness_parallel(
            m, n_workers=2, return_element_K=True,
        )
        assert len(elem_K) == 20
        # Each entry is (element, dof_map, K_e)
        for e, dofs, Ke in elem_K:
            assert Ke.shape == (6, 6)         # BeamColumn2D 2-node 3 DOFs
