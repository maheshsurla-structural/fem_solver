"""Tests for :class:`LinearBucklingAnalysis`.

Validation against the two canonical Euler-buckling cases:

* **Pin-pin column** — analytical critical load ``P_cr = pi^2 EI / L^2``.
* **Cantilever column** — effective length ``L_eff = 2 L``, so
  ``P_cr = pi^2 EI / (4 L^2)``.

Plus tests for the higher-mode series ``P_n = n^2 pi^2 EI / L^2`` of a
pin-pin column, mesh convergence (error decreases with refinement),
and a few input-validation paths.
"""
import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    BeamColumn2DCorotational,
    ElasticIsotropic,
    LinearBucklingAnalysis,
    Model,
)


# =============================================================== helpers ==

def _build_column(*, n_elem: int, boundary: str,
                  E: float = 2.0e11, A: float = 1.0e-3, Iz: float = 1.0e-7,
                  L: float = 5.0):
    """Build a uniform column of ``n_elem`` corotational beam elements.

    ``boundary`` is one of:

    * ``"pinned"`` — pin at one end, roller at the other (the standard
      Euler pin-pin column with effective length L).
    * ``"cantilever"`` — fully fixed at node 1, free at the far end
      (effective length 2L).
    * ``"fixed-fixed"`` — fully fixed at both ends (effective length L/2).
    """
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    for i in range(n_elem + 1):
        m.add_node(i + 1, i * L / n_elem, 0.0)
    for i in range(n_elem):
        m.add_element(
            BeamColumn2DCorotational(i + 1, (i + 1, i + 2), mat, A, Iz)
        )
    if boundary == "pinned":
        m.fix(1, [1, 1, 0])              # pin: u, v fixed; theta free
        m.fix(n_elem + 1, [0, 1, 0])     # roller: v fixed; u, theta free
    elif boundary == "cantilever":
        m.fix(1, [1, 1, 1])
    elif boundary == "fixed-fixed":
        m.fix(1, [1, 1, 1])
        m.fix(n_elem + 1, [0, 1, 1])     # axial free, rotation/transverse fixed
    else:
        raise ValueError(boundary)
    # Compressive reference load at the free end (or roller).
    m.add_nodal_load(n_elem + 1, [-1.0, 0.0, 0.0])
    return m, dict(E=E, A=A, Iz=Iz, L=L)


def _euler_load(*, E, Iz, L, effective_length_factor=1.0):
    """``P_cr = pi^2 EI / (K L)^2`` with K the effective-length factor."""
    return math.pi ** 2 * E * Iz / (effective_length_factor * L) ** 2


# ====================================================== pin-pin column ==

@pytest.mark.parametrize("n_elem", [8, 16, 32])
def test_pin_pin_column_buckles_at_euler_load(n_elem):
    """First buckling load of a pin-pin column equals
    ``pi^2 EI / L^2`` up to mesh-discretisation error."""
    m, cn = _build_column(n_elem=n_elem, boundary="pinned")
    P_euler = _euler_load(E=cn["E"], Iz=cn["Iz"], L=cn["L"])
    res = LinearBucklingAnalysis(m, num_modes=3).run()
    P_fe = res["critical_load_factor"]
    # Error tolerance scales with mesh size; loosen for coarser meshes.
    tolerance = 5.0e-2 if n_elem <= 8 else 2.0e-2
    assert P_fe == pytest.approx(P_euler, rel=tolerance)


def test_pin_pin_column_higher_modes_follow_n_squared():
    """Higher buckling modes of a pin-pin column have critical loads
    ``P_n = n^2 pi^2 EI / L^2``. Accuracy degrades with higher
    modes (more curvature, more discretisation error)."""
    m, cn = _build_column(n_elem=20, boundary="pinned")
    P_1 = _euler_load(E=cn["E"], Iz=cn["Iz"], L=cn["L"])
    res = LinearBucklingAnalysis(m, num_modes=3).run()
    lambdas = res["load_factors"]
    # P_1 to within 1%, P_2 to within 5%, P_3 to within 15%
    assert lambdas[0] == pytest.approx(P_1, rel=1.0e-2)
    assert lambdas[1] == pytest.approx(4.0 * P_1, rel=5.0e-2)
    assert lambdas[2] == pytest.approx(9.0 * P_1, rel=1.5e-1)


def test_pin_pin_column_error_decreases_with_mesh_refinement():
    """h-refinement: error decreases monotonically as the column is
    discretised into more elements."""
    errors = []
    for n in (4, 8, 16, 32):
        m, cn = _build_column(n_elem=n, boundary="pinned")
        P_euler = _euler_load(E=cn["E"], Iz=cn["Iz"], L=cn["L"])
        res = LinearBucklingAnalysis(m, num_modes=1).run()
        errors.append(abs(res["critical_load_factor"] - P_euler) / P_euler)
    # Each refinement should at most double the previous error (in
    # practice errors fall like 1/n^2 for cubic elements). A
    # monotonic decrease is the test.
    for k in range(1, len(errors)):
        assert errors[k] < errors[k - 1]


# ====================================================== cantilever ==

def test_cantilever_buckles_at_euler_effective_length_2L():
    """Cantilever buckles at ``pi^2 EI / (2L)^2`` — effective-length
    factor K = 2."""
    n_elem = 16
    m, cn = _build_column(n_elem=n_elem, boundary="cantilever")
    P_euler = _euler_load(E=cn["E"], Iz=cn["Iz"], L=cn["L"],
                          effective_length_factor=2.0)
    res = LinearBucklingAnalysis(m, num_modes=2).run()
    assert res["critical_load_factor"] == pytest.approx(P_euler, rel=2.0e-2)


# ====================================================== fixed-fixed ==

def test_fixed_fixed_column_buckles_at_effective_length_0p5L():
    """Both ends rotationally fixed (with one allowing axial slide):
    effective-length factor K = 0.5 (Euler case III)."""
    n_elem = 16
    m, cn = _build_column(n_elem=n_elem, boundary="fixed-fixed")
    P_euler = _euler_load(E=cn["E"], Iz=cn["Iz"], L=cn["L"],
                          effective_length_factor=0.5)
    res = LinearBucklingAnalysis(m, num_modes=1).run()
    # 5% tolerance: the higher buckling load needs more mesh resolution.
    assert res["critical_load_factor"] == pytest.approx(P_euler, rel=5.0e-2)


# ====================================================== mode shapes ==

def test_mode_shape_resembles_first_half_sine():
    """The first buckling mode of a pin-pin column is a half-sine
    transverse displacement. We check this by sampling the lateral
    displacements along the column and confirming all entries have the
    same sign (i.e., one half-wave) and the maximum is at midspan."""
    n_elem = 20
    m, _ = _build_column(n_elem=n_elem, boundary="pinned")
    LinearBucklingAnalysis(m, num_modes=2).run()
    # Lateral displacement at interior nodes (skipping ends)
    v_along = []
    for tag in range(2, n_elem + 1):
        v_along.append(m.node(tag).mode_disp[1, 0])
    v_along = np.array(v_along)
    # Mode is non-trivial
    assert np.max(np.abs(v_along)) > 0.0
    # All same sign (or normalise: flip sign if first negative)
    if v_along[0] < 0:
        v_along = -v_along
    assert np.all(v_along > -1.0e-12)
    # Maximum at midspan
    midspan_idx = (len(v_along) - 1) // 2
    assert np.argmax(v_along) == pytest.approx(midspan_idx, abs=1)


# ====================================================== guard rails ==

def test_buckling_rejects_invalid_num_modes():
    with pytest.raises(ValueError):
        LinearBucklingAnalysis(model=None, num_modes=0)


def test_buckling_rejects_unknown_numberer():
    with pytest.raises(ValueError, match="numberer"):
        LinearBucklingAnalysis(model=None, num_modes=1, numberer="banana")


def test_buckling_raises_when_no_corotational_elements():
    """A model built entirely from linear (non-corotational) elements
    has K_T = K, so K_g is zero — buckling is undefined. The analysis
    must raise a helpful error explaining the cause."""
    E, A, Iz, L = 2.0e11, 1.0e-3, 1.0e-7, 5.0
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    m.add_node(1, 0.0, 0.0); m.add_node(2, L, 0.0)
    # *Linear* beam, not corotational. No geometric stiffness.
    m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    m.fix(2, [0, 1, 0])
    m.add_nodal_load(2, [-1.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="geometric stiffness"):
        LinearBucklingAnalysis(m, num_modes=1).run()


def test_buckling_raises_when_no_compression():
    """If the reference load is *tensile* (positive axial), no
    buckling occurs — all generalized eigenvalues nu are non-negative
    and lambda = -1/nu is non-positive. The analysis must raise an
    informative error."""
    n_elem = 8
    m, cn = _build_column(n_elem=n_elem, boundary="pinned")
    # Flip the sign of the applied load — now it's tensile.
    m.nodes[n_elem + 1]._load[0] = +1.0
    with pytest.raises(RuntimeError, match="no buckling modes"):
        LinearBucklingAnalysis(m, num_modes=1).run()
