"""Quadrature points and weights on reference domains.

Two families are available:

* Gauss-Legendre — interior points only. Exact for polynomials up to
  degree ``2n - 1`` with ``n`` points. Used for area/volume integration of
  isoparametric elements (e.g. :class:`Quad4`).
* Gauss-Lobatto — includes the endpoints :math:`\\pm 1`. Exact up to degree
  ``2n - 3``. The natural choice for beam-column elements with distributed
  plasticity, where yielding concentrates at the element ends and we want
  integration points to land exactly there.
"""
from __future__ import annotations

import numpy as np


_GL_TABLES = {
    1: (np.array([0.0]), np.array([2.0])),
    2: (
        np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]),
        np.array([1.0, 1.0]),
    ),
    3: (
        np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]),
        np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]),
    ),
    4: (
        np.array([
            -np.sqrt((3.0 + 2.0 * np.sqrt(6.0 / 5.0)) / 7.0),
            -np.sqrt((3.0 - 2.0 * np.sqrt(6.0 / 5.0)) / 7.0),
            np.sqrt((3.0 - 2.0 * np.sqrt(6.0 / 5.0)) / 7.0),
            np.sqrt((3.0 + 2.0 * np.sqrt(6.0 / 5.0)) / 7.0),
        ]),
        np.array([
            (18.0 - np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 + np.sqrt(30.0)) / 36.0,
            (18.0 - np.sqrt(30.0)) / 36.0,
        ]),
    ),
}


def gauss_legendre_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """1-D Gauss-Legendre points and weights on [-1, 1]."""
    if n in _GL_TABLES:
        xi, w = _GL_TABLES[n]
        return xi.copy(), w.copy()
    # fall back to numpy for higher orders
    xi, w = np.polynomial.legendre.leggauss(n)
    return xi, w


def gauss_legendre_2d_quad(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tensor-product Gauss-Legendre quadrature on the bi-unit square.

    Returns (xi, eta, w) flattened arrays of length n*n.
    """
    xi1, w1 = gauss_legendre_1d(n)
    XI, ETA = np.meshgrid(xi1, xi1, indexing="ij")
    W = np.outer(w1, w1)
    return XI.ravel(), ETA.ravel(), W.ravel()


def gauss_legendre_3d_hex(n: int) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    """Tensor-product Gauss-Legendre quadrature on the bi-unit cube.

    Returns (xi, eta, zeta, w) flattened arrays of length n*n*n.
    """
    xi1, w1 = gauss_legendre_1d(n)
    XI, ETA, ZETA = np.meshgrid(xi1, xi1, xi1, indexing="ij")
    # Triple outer product for the weights
    W = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
    return XI.ravel(), ETA.ravel(), ZETA.ravel(), W.ravel()


# ---------------------------------------------------------------- Gauss-Lobatto
#
# The Gauss-Lobatto-Legendre rule for n points on [-1, 1] uses the endpoints
# x = -1 and x = +1 plus the (n-2) interior roots of P'_{n-1}(x), where
# P_{n-1} is the Legendre polynomial of degree n-1. The endpoint weights are
# 2 / (n (n-1)); interior weights are 2 / (n (n-1) [P_{n-1}(x_i)]^2). The rule
# is exact for polynomials up to degree 2n-3.
#
# Closed-form tables for n = 2..6 are stored below; higher n falls back to a
# numerical solve via NumPy's Legendre-polynomial roots.

_LOB_TABLES = {
    2: (
        np.array([-1.0, 1.0]),
        np.array([1.0, 1.0]),
    ),
    3: (
        np.array([-1.0, 0.0, 1.0]),
        np.array([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]),
    ),
    4: (
        np.array([-1.0, -np.sqrt(1.0 / 5.0), np.sqrt(1.0 / 5.0), 1.0]),
        np.array([1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0]),
    ),
    5: (
        np.array([
            -1.0,
            -np.sqrt(3.0 / 7.0),
            0.0,
            np.sqrt(3.0 / 7.0),
            1.0,
        ]),
        np.array([
            1.0 / 10.0,
            49.0 / 90.0,
            32.0 / 45.0,
            49.0 / 90.0,
            1.0 / 10.0,
        ]),
    ),
    6: (
        np.array([
            -1.0,
            -np.sqrt((7.0 + 2.0 * np.sqrt(7.0)) / 21.0),
            -np.sqrt((7.0 - 2.0 * np.sqrt(7.0)) / 21.0),
            np.sqrt((7.0 - 2.0 * np.sqrt(7.0)) / 21.0),
            np.sqrt((7.0 + 2.0 * np.sqrt(7.0)) / 21.0),
            1.0,
        ]),
        np.array([
            1.0 / 15.0,
            (14.0 - np.sqrt(7.0)) / 30.0,
            (14.0 + np.sqrt(7.0)) / 30.0,
            (14.0 + np.sqrt(7.0)) / 30.0,
            (14.0 - np.sqrt(7.0)) / 30.0,
            1.0 / 15.0,
        ]),
    ),
}


def gauss_lobatto_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """1-D Gauss-Lobatto-Legendre points and weights on [-1, 1].

    Includes the endpoints :math:`\\pm 1`. With ``n`` points the rule is
    exact for polynomials of degree ``2n - 3``. Requires ``n >= 2``.
    """
    if n < 2:
        raise ValueError(f"Gauss-Lobatto requires n >= 2, got {n}")
    if n in _LOB_TABLES:
        xi, w = _LOB_TABLES[n]
        return xi.copy(), w.copy()
    # Higher-order fallback: interior points are roots of P'_{n-1}.
    # P'_{n-1}(x) coefficients via numpy.polynomial.legendre.
    coeffs = np.zeros(n)
    coeffs[n - 1] = 1.0
    deriv = np.polynomial.legendre.legder(coeffs)
    interior = np.sort(np.polynomial.legendre.legroots(deriv))
    xi = np.concatenate(([-1.0], interior, [1.0]))
    # weights: w_i = 2 / (n (n-1) [P_{n-1}(x_i)]^2)
    pn_minus_1 = np.polynomial.legendre.legval(xi, coeffs)
    w = 2.0 / (n * (n - 1) * pn_minus_1 ** 2)
    return xi, w
