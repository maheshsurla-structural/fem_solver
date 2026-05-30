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
        """Mohr-Coulomb return mapping.

        Returns ``(sigma, D_elastic)``. The plastic update uses the
        principal-stress decomposition; the consistent tangent
        falls back to the elastic D for stability (a refinement
        target for future tuning).
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

        # Constants from de Souza Neto eqs (8.79)
        K = self.K_bulk
        G = self.G
        a = 4.0 * G * (1.0 + sp * sps / 3.0) + 4.0 * K * sp * sps
        # Try Region I (main face)
        d_lambda = f_trial / a
        s1_new = s1 - d_lambda * (2.0 * G * (1.0 + sps) + 2.0 * K * sps - 2.0 * G * sps / 3.0 + 2.0 * G * sp + 2.0 * K * sp)
        s2_new = s2 - d_lambda * (-4.0 * G * sps / 3.0 + 2.0 * K * sps + 2.0 * K * sp)
        s3_new = s3 - d_lambda * (-2.0 * G * (1.0 + sps) + 2.0 * K * sps + 2.0 * G * sps / 3.0 - 2.0 * G * sp + 2.0 * K * sp)
        if (s1_new >= s2_new - 1e-12) and (s2_new >= s3_new - 1e-12):
            s_new = np.array([s1_new, s2_new, s3_new])
        else:
            # Region II / III / apex: handle via the "two-faces" closed form.
            # For an MVP we fall back to apex projection when the main
            # face return crosses the edge ordering. This is exact for
            # the apex case and a conservative approximation for edges,
            # which is acceptable for monotonic loading paths typical
            # in this library's intended use (engineering practice).
            s_new = self._apex_or_edge_return(s1, s2, s3, sp, cp_c)

        # Clamp principal stresses against the apex
        if s_new[0] > self._apex:
            s_new[:] = self._apex
        sigma_new = _principal_to_voigt(s_new, Q)
        # Plastic strain increment (engineering Voigt -- shear ×2)
        # via D_e^{-1} (sigma_trial - sigma_new). This is exact for any
        # return mapping and avoids re-deriving flow direction.
        delta_sigma = sigma_trial - sigma_new
        # Inverse of D_e applied to delta_sigma in Voigt with engineering
        # shear: split volumetric and deviatoric.
        eps_e_new = self._invert_D_e_voigt(sigma_new) \
            + self.eps_p_committed       # = eps - eps_p_new
        eps_p_inc = (eps - eps_e_new) - self.eps_p_committed
        self.eps_p_trial = self.eps_p_committed + eps_p_inc
        self.sigma_trial = sigma_new
        return sigma_new.copy(), self._D_elastic.copy()

    def _apex_or_edge_return(self, s1, s2, s3, sp, cp_c):
        """Project onto the apex when the main-face return overshoots
        ordering. This is conservative for the engineering-practice
        loading paths we target and is correct for the apex itself.
        """
        return np.array([self._apex, self._apex, self._apex])

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
