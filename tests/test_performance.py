"""Performance and renumbering tests.

These tests exercise the vectorized assembler, K_e cache, and the
opt-in RCM DOF numbering. The wall-clock bound is generous to avoid
flakiness on slow CI runners — its purpose is to catch a future
regression that re-introduces a Python-level inner loop, not to assert
a precise speed.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp

from femsolver import (
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
    Quad4,
)
from femsolver.analysis.assembler import (
    assemble_force,
    assemble_reactions,
    assemble_stiffness,
)
from femsolver.numerics.dof_numbering import rcm_renumber


def _plate(n: int) -> Model:
    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    m.add_material(mat)
    tag = 1
    for j in range(n + 1):
        for i in range(n + 1):
            m.add_node(tag, float(i), float(j))
            tag += 1

    def nidx(i, j):
        return j * (n + 1) + i + 1

    etag = 1
    for j in range(n):
        for i in range(n):
            m.add_element(
                Quad4(etag, (nidx(i, j), nidx(i + 1, j), nidx(i + 1, j + 1), nidx(i, j + 1)),
                      mat, thickness=0.01)
            )
            etag += 1
    for j in range(n + 1):
        m.fix(nidx(0, j), [1, 1])
    for j in range(n + 1):
        scale = 0.5 if (j == 0 or j == n) else 1.0
        m.add_nodal_load(nidx(n, j), [1.0e3 * scale, 0.0])
    return m


# ---------------------------------------------------------------------------
# vectorized assembler


def test_assemble_returns_element_K_cache():
    m = _plate(4)
    K, cache = assemble_stiffness(m, return_element_K=True)
    assert isinstance(K, sp.csc_matrix)
    assert len(cache) == len(m.elements)
    for (e, dofs, Ke) in cache:
        assert dofs.shape == (e.n_dof,)
        assert Ke.shape == (e.n_dof, e.n_dof)


def test_assemble_with_and_without_cache_match():
    m1 = _plate(8)
    m2 = _plate(8)
    K_a = assemble_stiffness(m1)
    K_b, cache = assemble_stiffness(m2, return_element_K=True)
    np.testing.assert_allclose(K_a.toarray(), K_b.toarray(), rtol=1e-14, atol=1e-14)

    F_a = assemble_force(m1)
    F_b = assemble_force(m2, elem_K_list=cache)
    np.testing.assert_allclose(F_a, F_b, rtol=1e-14, atol=1e-14)


def test_cached_reactions_match_legacy():
    m1 = _plate(6)
    m2 = _plate(6)
    LinearStaticAnalysis(m1).run()  # cached path internally

    # legacy: rerun without cache for reactions
    m2.number_dofs()
    K, cache = assemble_stiffness(m2, return_element_K=True)
    F = assemble_force(m2, elem_K_list=cache)
    from scipy.sparse.linalg import spsolve
    u = spsolve(K, F)
    for n in m2.nodes.values():
        for i in range(n.ndf):
            eq = n.eqn[i]
            n.disp[i] = u[eq] if eq >= 0 else 0.0
    for e in m2.elements.values():
        e.recover()
    assemble_reactions(m2)  # legacy path

    for tag in m1.nodes:
        np.testing.assert_allclose(
            m1.node(tag).reaction, m2.node(tag).reaction, rtol=1e-12, atol=1e-12
        )


# ---------------------------------------------------------------------------
# RCM numbering


def test_rcm_renumber_assigns_unique_eqns():
    m = _plate(5)
    rcm_renumber(m)
    eqns = []
    for n in m.nodes.values():
        for i in range(n.ndf):
            if n.eqn[i] >= 0:
                eqns.append(int(n.eqn[i]))
    assert sorted(eqns) == list(range(m.neq))


def test_rcm_produces_same_physical_result():
    """Default and RCM numbering must give identical nodal displacements
    (the relabelling is a permutation, not a different problem)."""
    m_default = _plate(6)
    m_rcm = _plate(6)
    LinearStaticAnalysis(m_default, numberer="default").run()
    LinearStaticAnalysis(m_rcm, numberer="rcm").run()
    for tag in m_default.nodes:
        np.testing.assert_allclose(
            m_default.node(tag).disp,
            m_rcm.node(tag).disp,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            m_default.node(tag).reaction,
            m_rcm.node(tag).reaction,
            rtol=1e-10,
            atol=1e-9,
        )


def test_rcm_handles_no_elements():
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 1.0, 0.0)
    rcm_renumber(m)  # must not crash
    assert m.neq == 4


def test_unknown_numberer_raises():
    m = _plate(2)
    with pytest.raises(ValueError, match="numberer"):
        LinearStaticAnalysis(m, numberer="foo")


# ---------------------------------------------------------------------------
# regression bound — catches a future re-introduction of Python inner loops


def test_assembly_perf_bound_50x50_plate():
    """50x50 plate (2500 elements, 5202 DOFs) should assemble + solve in
    well under 5 s on any reasonable dev machine. The vectorized assembler
    typically does it in < 1 s; the bound is loose enough to survive CI
    variance but tight enough to flag a regression to per-entry Python
    appends.
    """
    m = _plate(50)
    t0 = time.perf_counter()
    LinearStaticAnalysis(m).run()
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"50x50 analysis took {elapsed:.2f} s (expected < 5 s)"
