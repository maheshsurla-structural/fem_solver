"""Tests for 1-D quadrature rules.

For an n-point rule on [-1, 1]:

* Gauss-Legendre is exact for polynomials of degree up to ``2n - 1``.
* Gauss-Lobatto is exact for polynomials of degree up to ``2n - 3``
  (because two of the n abscissae are pinned at the endpoints).
"""
import numpy as np
import pytest

from femsolver.numerics import gauss_legendre_1d, gauss_lobatto_1d


def _exact_int_xk(k: int) -> float:
    """Exact value of int_{-1}^{1} x^k dx."""
    if k % 2 == 1:
        return 0.0
    return 2.0 / (k + 1)


# -------------------------------------------------------------- Gauss-Lobatto

@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_lobatto_includes_endpoints(n):
    xi, _ = gauss_lobatto_1d(n)
    assert xi[0] == pytest.approx(-1.0)
    assert xi[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_lobatto_weights_sum_to_two(n):
    _, w = gauss_lobatto_1d(n)
    assert w.sum() == pytest.approx(2.0, rel=1e-14)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_lobatto_exact_up_to_2n_minus_3(n):
    """n-point Gauss-Lobatto integrates x^k exactly for k <= 2n - 3."""
    xi, w = gauss_lobatto_1d(n)
    max_exact_degree = 2 * n - 3
    for k in range(max_exact_degree + 1):
        approx = float(np.sum(w * xi ** k))
        exact = _exact_int_xk(k)
        assert approx == pytest.approx(exact, abs=1e-13), (
            f"n={n}, degree {k}: got {approx}, expected {exact}"
        )


@pytest.mark.parametrize("n", [3, 4, 5])
def test_lobatto_higher_order_fallback_matches_table(n):
    """The numpy-based fallback path must agree with the closed-form tables.

    We exercise the fallback by re-deriving points/weights via the
    Legendre-derivative approach and checking they match the cached table.
    """
    xi_tab, w_tab = gauss_lobatto_1d(n)
    coeffs = np.zeros(n)
    coeffs[n - 1] = 1.0
    deriv = np.polynomial.legendre.legder(coeffs)
    interior = np.sort(np.polynomial.legendre.legroots(deriv))
    xi_num = np.concatenate(([-1.0], interior, [1.0]))
    pn_minus_1 = np.polynomial.legendre.legval(xi_num, coeffs)
    w_num = 2.0 / (n * (n - 1) * pn_minus_1 ** 2)
    np.testing.assert_allclose(xi_tab, xi_num, atol=1e-13)
    np.testing.assert_allclose(w_tab, w_num, atol=1e-13)


def test_lobatto_n_too_small_raises():
    with pytest.raises(ValueError):
        gauss_lobatto_1d(1)


def test_lobatto_high_order_fallback_runs():
    """n > 6 hits the numpy fallback path. Sanity-check exactness."""
    n = 8
    xi, w = gauss_lobatto_1d(n)
    assert xi[0] == pytest.approx(-1.0)
    assert xi[-1] == pytest.approx(1.0)
    assert w.sum() == pytest.approx(2.0, rel=1e-14)
    # exact for x^(2n-3) = x^13
    for k in range(2 * n - 2):
        assert float(np.sum(w * xi ** k)) == pytest.approx(_exact_int_xk(k), abs=1e-12)


# -------------------------------------------------------------- Gauss-Legendre

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_legendre_exact_up_to_2n_minus_1(n):
    """Sanity check that Gauss-Legendre still works after our edits."""
    xi, w = gauss_legendre_1d(n)
    for k in range(2 * n):
        assert float(np.sum(w * xi ** k)) == pytest.approx(_exact_int_xk(k), abs=1e-13)
