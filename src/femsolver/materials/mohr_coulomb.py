"""Mohr-Coulomb plasticity in 3-D.

The Mohr-Coulomb (MC) yield surface in principal-stress space is

    f = (sigma_1 - sigma_3) + (sigma_1 + sigma_3) sin(phi) - 2 c cos(phi)

with ``sigma_1 >= sigma_2 >= sigma_3`` (tension positive throughout
this library). The surface is an *irregular hexagonal cone* about
the hydrostatic axis: six smooth-face sections meeting at six edges,
all converging at the *apex* ``sigma_1 = sigma_2 = sigma_3 = c
cot(phi)``.

This implementation follows the principal-stress return mapping of
de Souza Neto / Peric / Owen "Computational Methods for Plasticity"
(Chapter 8). The four return-mapping regions are:

* **Main face** -- the standard smooth-surface return; sliding on a
  single MC face.
* **Right edge** (``sigma_1 = sigma_2``) -- two faces active, two
  consistency conditions.
* **Left edge** (``sigma_2 = sigma_3``) -- the symmetric case.
* **Apex** -- all three principal stresses collapse to ``c cot(phi)``.

Non-associated flow is supported via a separate dilation angle
``psi``; setting ``psi = phi`` gives associated flow (volumetric
expansion on shear yield) and ``psi < phi`` gives the engineering
choice with reduced dilation.

Sign convention
---------------
Tension positive for both stress and strain. Voigt order matches the
library: ``(xx, yy, zz, xy, yz, zx)`` with engineering shear strain.
"""
from __future__ import annotations

import copy
import math

import numpy as np


# Voigt <-> tensor helpers (reused across this module)

def _voigt_to_tensor(sigma: np.ndarray) -> np.ndarray:
    """Convert a (6,) Voigt stress to a (3, 3) symmetric tensor."""
    return np.array([
        [sigma[0], sigma[3], sigma[5]],
        [sigma[3], sigma[1], sigma[4]],
        [sigma[5], sigma[4], sigma[2]],
    ])


def _tensor_to_voigt(T: np.ndarray) -> np.ndarray:
    return np.array([T[0, 0], T[1, 1], T[2, 2],
                      T[0, 1], T[1, 2], T[0, 2]])


def _principal_decomposition(sigma: np.ndarray):
    """Return descending-sorted principal stresses + eigenvector
    matrix Q such that ``sigma_tensor = Q diag(s) Q^T``."""
    T = _voigt_to_tensor(sigma)
    w, V = np.linalg.eigh(T)
    # eigh returns ascending. Reverse so s_1 >= s_2 >= s_3.
    idx = np.argsort(w)[::-1]
    s = w[idx]
    Q = V[:, idx]
    return s, Q


def _principal_to_voigt(s: np.ndarray, Q: np.ndarray) -> np.ndarray:
    T = Q @ np.diag(s) @ Q.T
    return _tensor_to_voigt(T)


class MohrCoulomb3D:
    """3-D Mohr-Coulomb perfectly-plastic material.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    cohesion : float
        Cohesion ``c`` (positive). For purely frictional soil set to a
        very small positive value (e.g., 1 Pa) so the apex projection
        is well-defined.
    phi_deg : float
        Friction angle ``phi`` in degrees; ``0 <= phi < 90``.
    psi_deg : float, optional
        Dilation angle ``psi`` in degrees; ``0 <= psi <= phi``. If
        omitted, ``psi = phi`` (associated flow).
    """

    def __init__(
        self,
        E: float,
        nu: float,
        cohesion: float,
        phi_deg: float,
        psi_deg: float | None = None,
    ):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"nu must be in (-1, 0.5), got {nu}")
        if cohesion <= 0.0:
            raise ValueError(f"cohesion must be positive, got {cohesion}")
        if not (0.0 <= phi_deg < 90.0):
            raise ValueError(f"phi_deg must be in [0, 90), got {phi_deg}")
        if psi_deg is None:
            psi_deg = phi_deg
        if not (0.0 <= psi_deg <= phi_deg):
            raise ValueError(
                f"psi_deg must be in [0, phi_deg]={phi_deg}, got {psi_deg}"
            )
        self.E = float(E)
        self.nu = float(nu)
        self.cohesion = float(cohesion)
        self.phi_deg = float(phi_deg)
        self.psi_deg = float(psi_deg)
        phi = math.radians(phi_deg)
        psi = math.radians(psi_deg)
        self._sin_phi = math.sin(phi)
        self._cos_phi = math.cos(phi)
        self._sin_psi = math.sin(psi)
        self._cos_psi = math.cos(psi)
        # Apex of the cone (hydrostatic stress where surface collapses)
        self._apex = (
            cohesion * self._cos_phi / max(self._sin_phi, 1.0e-12)
        )
        # Elastic constants
        self.G = E / (2.0 * (1.0 + nu))
        self.K_bulk = E / (3.0 * (1.0 - 2.0 * nu))
        self._lambda_lame = (
            E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        )
        self._D_elastic = self._build_D_elastic()
        # ---- state ----
        self.eps_p_committed = np.zeros(6)
        self.sigma_committed = np.zeros(6)
        self.eps_p_trial = np.zeros(6)
        self.sigma_trial = np.zeros(6)

    def _build_D_elastic(self) -> np.ndarray:
        lam = self._lambda_lame
        mu = self.G
        D = np.zeros((6, 6))
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        return D

    def D_elastic(self) -> np.ndarray:
        return self._D_elastic.copy()

    # ----------------------------------------------------- yield
    def yield_function(self, sigma: np.ndarray) -> float:
        """``f = (s1 - s3) + (s1 + s3) sin(phi) - 2c cos(phi)``."""
        s, _ = _principal_decomposition(np.asarray(sigma, dtype=float))
        s1, s3 = float(s[0]), float(s[2])
        return (s1 - s3) + (s1 + s3) * self._sin_phi \
               - 2.0 * self.cohesion * self._cos_phi

    # ----------------------------------------------------- response
    def get_response(self, eps_voigt) -> tuple[np.ndarray, np.ndarray]:
        """Mohr-Coulomb return mapping with full 4-region handling.

        Returns ``(sigma, D_elastic)``. The plastic update follows the
        de Souza Neto / Peric / Owen Chapter 8 algorithm:

        1. Trial elastic stress; check yield.
        2. Try **Region I** (main face): single plastic multiplier,
           closed-form return. Accept if ordering preserved.
        3. If ordering violated: try **Region II** (right edge,
           sigma_1 = sigma_2) or **Region III** (left edge, sigma_2 =
           sigma_3) -- 2x2 closed-form return.
        4. Apex projection if even the edge return is infeasible.
        """
        eps = np.asarray(eps_voigt, dtype=float).reshape(6)
        eps_e_trial = eps - self.eps_p_committed
        sigma_trial = self._D_elastic @ eps_e_trial
        s_trial, Q = _principal_decomposition(sigma_trial)
        s1, s2, s3 = float(s_trial[0]), float(s_trial[1]), float(s_trial[2])
        sp = self._sin_phi
        cp_c = self.cohesion * self._cos_phi
        sps = self._sin_psi
        f_trial = (s1 - s3) + (s1 + s3) * sp - 2.0 * cp_c
        if f_trial <= 0.0:
            self.sigma_trial = sigma_trial
            self.eps_p_trial = self.eps_p_committed.copy()
            return sigma_trial.copy(), self._D_elastic.copy()

        # 4-region return mapping
        s_new = self._return_map_principal(s1, s2, s3)

        sigma_new = _principal_to_voigt(s_new, Q)
        # Plastic strain increment via D_e^{-1} (sigma_trial - sigma_new)
        eps_e_new = self._invert_D_e_voigt(sigma_new)
        eps_p_inc = (eps - eps_e_new) - self.eps_p_committed
        self.eps_p_trial = self.eps_p_committed + eps_p_inc
        self.sigma_trial = sigma_new
        return sigma_new.copy(), self._D_elastic.copy()

    # ----------------------------------------------------- 4-region return
    def _return_map_principal(self, s1, s2, s3) -> np.ndarray:
        """Sequential 4-region return mapping in principal-stress space.

        Mathematical derivation
        -----------------------
        For the yield function

            f = (s1 - s3) + (s1 + s3) sin(phi) - 2 c cos(phi)

        and non-associated flow potential

            g = (s1 - s3) + (s1 + s3) sin(psi)

        the elastic-plastic stress update in principal coordinates
        with bulk modulus ``K`` and shear modulus ``G`` is:

            s1 = s1_tr - dl * (2G + 2(K + G/3) sin psi)
            s2 = s2_tr - dl * (2(K - 2G/3) sin psi)
            s3 = s3_tr + dl * (2G - 2(K + G/3) sin psi)

        Substituting into ``f = 0`` and solving for dl gives the
        **Region I (main face)** scalar return:

            dl = f_trial / A,   with A = 4G + 4(K + G/3) sin phi sin psi

        If the returned stresses violate ordering, the edge cases are
        handled by the 2x2 systems below (Region II for right edge,
        Region III for left edge). The cross-coupling coefficient is

            B = -2G(1 - sin psi) + [2G + (2G/3 - 4K) sin psi] sin phi

        and the 2x2 linear system in (dl_a, dl_b) is

            [ A  -B ] [dl_a]   [f_a_trial]
            [ -B  A ] [dl_b] = [f_b_trial]

        with solution via Cramer's rule. The apex projection sigma_1
        = sigma_2 = sigma_3 = c cot(phi) closes the algorithm when
        even an edge return is infeasible.
        """
        sp = self._sin_phi
        sps = self._sin_psi
        cp_c = self.cohesion * self._cos_phi
        K = self.K_bulk
        G = self.G
        # Coefficients shared across regions
        A = 4.0 * G + 4.0 * (K + G / 3.0) * sp * sps
        B = -2.0 * G * (1.0 - sps) \
            + (2.0 * G + (2.0 * G / 3.0 - 4.0 * K) * sps) * sp

        # ============================================================ Region I
        f_a_trial = (s1 - s3) + (s1 + s3) * sp - 2.0 * cp_c
        dl = f_a_trial / A
        # Stress increments per unit dl
        c1 = 2.0 * G + 2.0 * (K + G / 3.0) * sps
        c2 = 2.0 * (K - 2.0 * G / 3.0) * sps
        c3 = 2.0 * G - 2.0 * (K + G / 3.0) * sps      # +ve sign as derived
        s1_new = s1 - dl * c1
        s2_new = s2 - dl * c2
        s3_new = s3 + dl * c3
        # NOTE: s3 has + dl * c3 because the d eps_p_3 = -dl (1 - sin psi)
        # (negative), and the (K - 2G/3) volumetric contribution to s3 is
        # the same as to s1, s2 (subtracts).
        tol = 1.0e-12 * max(abs(s1), abs(s3), 1.0)
        if s1_new + tol >= s2_new >= s3_new - tol and dl >= -tol:
            return np.array([s1_new, s2_new, s3_new])

        # ============================================================ Region II / III
        # Determine which edge: if s1_new < s2_new -> right edge
        # (sigma_1 = sigma_2 must be enforced); if s2_new < s3_new ->
        # left edge (sigma_2 = sigma_3 must be enforced).
        det = A * A - B * B
        if abs(det) < 1.0e-30:
            return self._apex_array()

        if s1_new < s2_new:
            # Right edge: sigma_1 = sigma_2
            f_b_trial = (s2 - s3) + (s2 + s3) * sp - 2.0 * cp_c
            dl_a = (A * f_a_trial + B * f_b_trial) / det
            dl_b = (B * f_a_trial + A * f_b_trial) / det
            if dl_a >= -tol and dl_b >= -tol:
                # Updated stresses (right edge: dl_a active on face A,
                # dl_b on the σ_2/σ_3 face). Using formulas derived above:
                s1_n = s1 - 2.0 * G * dl_a * (1.0 + sps) \
                       - (K - 2.0 * G / 3.0) * (dl_a + dl_b) * 2.0 * sps
                s2_n = s2 - 2.0 * G * dl_b * (1.0 + sps) \
                       - (K - 2.0 * G / 3.0) * (dl_a + dl_b) * 2.0 * sps
                s3_n = s3 + 2.0 * G * (dl_a + dl_b) * (1.0 - sps) \
                       - (K - 2.0 * G / 3.0) * (dl_a + dl_b) * 2.0 * sps
                if s2_n >= s3_n - tol:
                    return np.array([s1_n, s2_n, s3_n])
        else:
            # Left edge: sigma_2 = sigma_3. Use f_c on face (σ_1, σ_2).
            f_c_trial = (s1 - s2) + (s1 + s2) * sp - 2.0 * cp_c
            dl_a = (A * f_a_trial + B * f_c_trial) / det
            dl_c = (B * f_a_trial + A * f_c_trial) / det
            if dl_a >= -tol and dl_c >= -tol:
                # Left-edge return: flow on σ_3 (a) and σ_2 (c).
                s1_n = s1 - 2.0 * G * (dl_a + dl_c) * (1.0 + sps) \
                       - (K - 2.0 * G / 3.0) * (dl_a + dl_c) * 2.0 * sps
                s2_n = s2 + 2.0 * G * dl_c * (1.0 - sps) \
                       - (K - 2.0 * G / 3.0) * (dl_a + dl_c) * 2.0 * sps
                s3_n = s3 + 2.0 * G * dl_a * (1.0 - sps) \
                       - (K - 2.0 * G / 3.0) * (dl_a + dl_c) * 2.0 * sps
                if s1_n >= s2_n - tol:
                    return np.array([s1_n, s2_n, s3_n])

        # ============================================================ Apex
        return self._apex_array()

    def _apex_array(self) -> np.ndarray:
        a = self._apex
        return np.array([a, a, a])

    def _invert_D_e_voigt(self, sigma: np.ndarray) -> np.ndarray:
        """Return ``eps = D_e^{-1} sigma`` for a (6,) Voigt stress with
        engineering shear strain convention."""
        E, nu = self.E, self.nu
        G = self.G
        eps = np.empty(6)
        eps[0] = (sigma[0] - nu * (sigma[1] + sigma[2])) / E
        eps[1] = (sigma[1] - nu * (sigma[0] + sigma[2])) / E
        eps[2] = (sigma[2] - nu * (sigma[0] + sigma[1])) / E
        eps[3] = sigma[3] / G
        eps[4] = sigma[4] / G
        eps[5] = sigma[5] / G
        return eps

    # ----------------------------------------------------- state
    def commit_state(self) -> None:
        self.eps_p_committed = self.eps_p_trial.copy()
        self.sigma_committed = self.sigma_trial.copy()

    def revert_state(self) -> None:
        self.eps_p_trial = self.eps_p_committed.copy()
        self.sigma_trial = self.sigma_committed.copy()

    def clone(self) -> "MohrCoulomb3D":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"MohrCoulomb3D(E={self.E:g}, nu={self.nu:g}, "
            f"c={self.cohesion:g}, phi={self.phi_deg:g}°, "
            f"psi={self.psi_deg:g}°)"
        )
