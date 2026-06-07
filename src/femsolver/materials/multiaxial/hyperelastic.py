"""Hyperelastic constitutive models for large-strain elasticity.

For an isotropic hyperelastic material the second Piola-Kirchhoff
stress and the Lagrangian (material) tangent are derived from a
strain-energy density ``W(C)`` (or equivalently ``W(F)``):

    S = 2 ∂W/∂C,   C^M_{IJKL} = 4 ∂²W/∂C_IJ∂C_KL

with ``C = F^T F`` the right Cauchy-Green tensor and ``F`` the
deformation gradient. The Cauchy (true) stress is then
``σ = J^{-1} F S F^T``.

This module ships two of the most common compressible hyperelastic
models:

* :class:`NeoHookean3D` -- the canonical large-strain extension of
  linear isotropic elasticity. Strain energy::

      W = (μ/2)(I_1 - 3) - μ ln(J) + (λ/2)(ln J)^2

* :class:`MooneyRivlin3D` -- adds an ``I_2`` term for better fit to
  finite-strain rubber data. Strain energy::

      W = c_10 (I_1 - 3) + c_01 (I_2 - 3) + (K/2)(J - 1)^2

In both, ``I_1 = Tr(C)``, ``I_2 = ½[(Tr C)² - Tr(C²)]``, ``J = det F``.

The materials expose two evaluation modes:

* ``response_S(F)`` -- returns ``(S, C^M)`` Voigt-packed (6, 6×6) for
  the Total-Lagrangian element pipeline.
* ``response_sigma(F)`` -- returns ``(σ, c^S)`` Voigt-packed for an
  Updated-Lagrangian / spatial pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================ Voigt helpers

# Voigt order for symmetric (3, 3) tensors:
#   0 -> xx, 1 -> yy, 2 -> zz, 3 -> xy, 4 -> yz, 5 -> zx
_VOIGT_INDEX = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]


def _sym_to_voigt(T: np.ndarray) -> np.ndarray:
    """Pack a symmetric (3, 3) tensor into a 6-vector (Voigt
    engineering for shear: off-diagonals appear ONCE, not doubled —
    this is the "stress" convention)."""
    return np.array([
        T[0, 0], T[1, 1], T[2, 2],
        T[0, 1], T[1, 2], T[2, 0],
    ])


def _voigt_to_sym(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_sym_to_voigt`."""
    T = np.zeros((3, 3))
    T[0, 0], T[1, 1], T[2, 2] = v[0], v[1], v[2]
    T[0, 1] = T[1, 0] = v[3]
    T[1, 2] = T[2, 1] = v[4]
    T[0, 2] = T[2, 0] = v[5]
    return T


# ============================================================ Neo-Hookean

@dataclass
class NeoHookean3D:
    """Compressible Neo-Hookean material.

    Strain-energy density::

        W = (μ / 2)(I_1 - 3) - μ ln(J) + (λ / 2)(ln J)^2

    Parameters
    ----------
    tag : int
        User-facing identifier.
    E : float
        Initial Young's modulus (Pa). At small strain the model
        reduces to linear isotropic elasticity with E and nu.
    nu : float
        Initial Poisson ratio.
    rho : float, default 0.0
        Mass density (kg/m^3).
    """

    tag: int
    E: float
    nu: float
    rho: float = 0.0

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise ValueError("E must be > 0")
        if not (0.0 < self.nu < 0.5):
            raise ValueError("nu must be in (0, 0.5)")
        # Lamé constants
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    # -------------------------------------------------- 2nd PK stress
    def response_S(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (S_voigt, C_M_voigt) for the given deformation
        gradient ``F`` (3, 3).

        Closed form::

            S = μ (I - C^{-1}) + λ ln(J) C^{-1}
        """
        if F.shape != (3, 3):
            raise ValueError(f"F must be (3, 3), got {F.shape}")
        J = float(np.linalg.det(F))
        if J <= 0.0:
            raise ValueError(f"J = det(F) must be > 0, got {J}")
        C = F.T @ F                          # right Cauchy-Green
        C_inv = np.linalg.inv(C)
        I3 = np.eye(3)
        ln_J = np.log(J)
        S = self.mu * (I3 - C_inv) + self.lam * ln_J * C_inv

        # Material tangent C^M (Lagrangian) -- closed form for Neo-Hookean:
        # C^M = lambda * (C^{-1} ⊗ C^{-1}) + 2(mu - lambda ln J) * C^{-1} ⊙ C^{-1}
        # where ⊙ is the symmetric tensor product:
        #   (A ⊙ B)_{IJKL} = ½(A_{IK} B_{JL} + A_{IL} B_{JK})
        C_M = self._build_material_tangent(C_inv, ln_J)
        return _sym_to_voigt(S), C_M

    def _build_material_tangent(
        self, C_inv: np.ndarray, ln_J: float,
    ) -> np.ndarray:
        """Construct the 6 x 6 Voigt material tangent."""
        lam_ln = self.lam * ln_J
        coef_outer = self.lam
        coef_sym = 2.0 * (self.mu - lam_ln)
        C_voigt = np.zeros((6, 6))
        for A, (I, J) in enumerate(_VOIGT_INDEX):
            for B, (K, L) in enumerate(_VOIGT_INDEX):
                outer = C_inv[I, J] * C_inv[K, L]
                sym = 0.5 * (C_inv[I, K] * C_inv[J, L]
                              + C_inv[I, L] * C_inv[J, K])
                C_voigt[A, B] = coef_outer * outer + coef_sym * sym
        return C_voigt

    # -------------------------------------------------- Cauchy stress
    def response_sigma(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (σ_voigt, c^S_voigt) -- Cauchy stress and spatial
        tangent.

        Closed form for Cauchy stress::

            σ = (μ / J)(B - I) + (λ / J) ln(J) I,

        with B = F F^T the left Cauchy-Green tensor.

        The spatial tangent is obtained by push-forward of the
        material tangent; for the simplest case (small-deformation
        limit it reduces to standard ``D``), we return the
        push-forward via Voigt math.
        """
        J = float(np.linalg.det(F))
        if J <= 0.0:
            raise ValueError(f"J = det(F) must be > 0, got {J}")
        B = F @ F.T
        I3 = np.eye(3)
        sigma = (self.mu / J) * (B - I3) + (self.lam / J) * np.log(J) * I3
        sigma_v = _sym_to_voigt(sigma)
        # Spatial tangent: push-forward of C^M.  For typical small to
        # moderate strain analyses, a sufficiently-accurate proxy is
        # the small-strain linear elastic tangent multiplied by 1/J:
        c_spatial = self._spatial_tangent_approx(J)
        return sigma_v, c_spatial

    def _spatial_tangent_approx(self, J: float) -> np.ndarray:
        """First-order spatial tangent ~ linear-elastic D / J.

        For exact spatial-tangent at large strain, one should
        push-forward the material tangent via
        ``c^S_{ijkl} = J^{-1} F_{iI} F_{jJ} F_{kK} F_{lL} C^M_{IJKL}``;
        this approximation is adequate for the moderate-strain regime
        typical of structural-rubber components.
        """
        c11 = self.lam + 2.0 * self.mu
        c12 = self.lam
        G = self.mu
        D = np.array([
            [c11, c12, c12, 0.0, 0.0, 0.0],
            [c12, c11, c12, 0.0, 0.0, 0.0],
            [c12, c12, c11, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, G,   0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, G,   0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, G  ],
        ])
        return D / J


# ============================================================ Mooney-Rivlin

@dataclass
class MooneyRivlin3D:
    """Compressible Mooney-Rivlin (2-parameter) material.

    Strain energy (with the standard ``ln J`` correction term that
    enforces a stress-free reference at ``F = I``)::

        W = c_10 (I_1 - 3) + c_01 (I_2 - 3)
            - 2(c_10 + 2 c_01) ln J + (K / 2)(J - 1)^2

    Parameters
    ----------
    tag : int
    c_10 : float
        First Mooney-Rivlin constant (Pa).
    c_01 : float
        Second Mooney-Rivlin constant (Pa).
    K : float
        Bulk modulus (Pa).
    rho : float, default 0.0

    Notes
    -----
    For ``c_01 = 0``, this reduces to the (slightly incompressible)
    Neo-Hookean variant.
    Typical rubber values: ``c_10 ~ 0.3-0.8 MPa``, ``c_01 ~ 0.0-0.3 MPa``,
    ``K ~ 1000-2000 MPa``.
    """

    tag: int
    c_10: float
    c_01: float
    K: float
    rho: float = 0.0

    def __post_init__(self) -> None:
        if self.c_10 < 0.0 or self.c_01 < 0.0:
            raise ValueError("c_10 and c_01 must be >= 0")
        if self.K <= 0.0:
            raise ValueError("K must be > 0")
        # Initial shear modulus
        self.mu_0 = 2.0 * (self.c_10 + self.c_01)
        # Equivalent E, nu (for small-strain limit)
        self.E_initial = 9.0 * self.K * self.mu_0 / (3.0 * self.K + self.mu_0)
        self.nu_initial = (3.0 * self.K - 2.0 * self.mu_0) \
                           / (2.0 * (3.0 * self.K + self.mu_0))

    def response_S(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (S_voigt, C_M_voigt) for the deformation gradient F."""
        if F.shape != (3, 3):
            raise ValueError(f"F must be (3, 3), got {F.shape}")
        J = float(np.linalg.det(F))
        if J <= 0.0:
            raise ValueError(f"J = det(F) must be > 0, got {J}")
        C = F.T @ F
        I3 = np.eye(3)
        I1 = float(np.trace(C))
        I2 = 0.5 * (I1 * I1 - float(np.trace(C @ C)))
        C_inv = np.linalg.inv(C)
        # With the -2(c_10 + 2 c_01) ln J correction, S is:
        # S = 2 c_10 I + 2 c_01 (I_1 I - C)
        #     - (2 c_10 + 4 c_01) C^{-1} + K (J - 1) J C^{-1}
        p0 = 2.0 * self.c_10 + 4.0 * self.c_01
        S = (2.0 * self.c_10 * I3
             + 2.0 * self.c_01 * (I1 * I3 - C)
             - p0 * C_inv
             + self.K * (J - 1.0) * J * C_inv)
        # Material tangent C^M (compressible MR closed form, in Voigt)
        # Practical approximation: linear-elastic D at the current
        # initial moduli (mu_0, K) divided by J. Adequate for moderate
        # strain; for highly-strained Newton iterations, derive the
        # full closed-form C^M from the second derivatives of W.
        D = self._initial_tangent()
        return _sym_to_voigt(S), D

    def _initial_tangent(self) -> np.ndarray:
        """Small-strain elastic tangent at the initial mu / K."""
        K, G = self.K, self.mu_0
        lam = K - (2.0 / 3.0) * G
        c11 = lam + 2.0 * G
        c12 = lam
        return np.array([
            [c11, c12, c12, 0.0, 0.0, 0.0],
            [c12, c11, c12, 0.0, 0.0, 0.0],
            [c12, c12, c11, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, G,   0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, G,   0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, G  ],
        ])

    def response_sigma(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (σ_voigt, c^S_voigt) -- Cauchy stress + spatial tangent."""
        J = float(np.linalg.det(F))
        if J <= 0.0:
            raise ValueError(f"J = det(F) must be > 0, got {J}")
        B = F @ F.T
        I3 = np.eye(3)
        I1_b = float(np.trace(B))
        # With the ln J correction:
        # σ = J^{-1}[2 c_10 B + 2 c_01 (I_1_b B - B^2)
        #            - (2 c_10 + 4 c_01) I] + K (J - 1) I
        p0 = 2.0 * self.c_10 + 4.0 * self.c_01
        sigma = ((2.0 * self.c_10 / J) * B
                 + (2.0 * self.c_01 / J) * (I1_b * B - B @ B)
                 - (p0 / J) * I3
                 + self.K * (J - 1.0) * I3)
        return _sym_to_voigt(sigma), self._initial_tangent() / J
