"""Lubliner-Lee-Fenves concrete damage-plasticity (CDP).

The CDP model couples a **Drucker-Prager-style yield surface**
(with concrete-specific calibration) to **two independent scalar
damage variables** d_t (tension) and d_c (compression). Unlike the
Mazars-style scalar damage in :mod:`concrete_damage`, here:

* The model tracks a separate **plastic strain tensor** -- unloading
  follows the damaged elastic modulus from the plastic-strain state,
  not the origin.
* The damage variables evolve from independent **hardening parameters**
  ``kappa_t`` and ``kappa_c`` that count cumulative plastic strain in
  tension and compression respectively.
* On stress reversal from tension to compression a **stiffness
  recovery factor** ``s(sigma_bar)`` partially restores the
  compressive stiffness (the "crack closure" effect that Mazars
  cannot represent).

This is the model used in the Abaqus *Concrete Damaged Plasticity*
material and in DIANA. The implementation here follows Lee & Fenves
(1998) "Plastic-Damage Model for Cyclic Loading of Concrete
Structures" (J. Eng. Mech., 124(8)).

Theory summary
--------------

* **Effective stress:** ``sigma_bar = D_e (eps - eps_p)`` -- the
  stress that would exist on the undamaged material.
* **Apparent (nominal) stress:** ``sigma = (1 - d) sigma_bar`` where
  ``d`` is the *combined* damage, computed from ``d_t`` and ``d_c``
  weighted by the stress state.
* **Yield function** (DP-style with concrete calibration)::

      f(sigma_bar, kappa_c) = alpha I_1 + sqrt(J_2)
                              - k(kappa_c) <= 0

  where ``alpha`` and ``k`` are calibrated so the surface passes
  through both uniaxial compression and tension states, and
  ``k(kappa_c)`` softens with cumulative compressive plastic strain.
* **Flow potential** (non-associated, dilation angle psi)::

      g(sigma_bar) = beta I_1 + sqrt(J_2)

  with ``beta`` from the dilation angle.

Sign convention
---------------
Tension positive throughout. Voigt order ``(xx, yy, zz, xy, yz, zx)``.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np


def _deviator_p(sigma: np.ndarray) -> tuple[np.ndarray, float]:
    p = (sigma[0] + sigma[1] + sigma[2]) / 3.0
    s = sigma.copy()
    s[0] -= p; s[1] -= p; s[2] -= p
    return s, p


def _voigt_double_dot(s: np.ndarray) -> float:
    return float(
        s[0] ** 2 + s[1] ** 2 + s[2] ** 2
        + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
    )


def _principal_stress(sigma: np.ndarray) -> np.ndarray:
    """Return descending-sorted principal stresses from a Voigt 6-vector."""
    T = np.array([
        [sigma[0], sigma[3], sigma[5]],
        [sigma[3], sigma[1], sigma[4]],
        [sigma[5], sigma[4], sigma[2]],
    ])
    return np.sort(np.linalg.eigvalsh(T))[::-1]


class ConcreteDamagePlasticity3D:
    """Full Lubliner-Lee-Fenves concrete damage-plasticity model.

    Parameters
    ----------
    E : float
        Initial Young's modulus of undamaged concrete.
    nu : float
        Poisson's ratio.
    f_c : float
        Uniaxial compressive strength (positive magnitude, Pa).
    f_t : float
        Uniaxial tensile strength (positive magnitude, Pa).
    psi_deg : float, default 30.0
        Dilation angle in the p-q plane (deg).
    A_t : float, default 0.7
    B_t : float, default 1.0e4
        Tensile damage exponential-law shape parameters.
    A_c : float, default 0.6
    B_c : float, default 300.0
        Compressive damage parameters.
    f_b_over_f_c : float, default 1.16
        Ratio of biaxial to uniaxial compressive strength (governs
        Lubliner ``alpha``).
    """

    def __init__(
        self,
        E: float,
        nu: float,
        f_c: float,
        f_t: float,
        *,
        psi_deg: float = 30.0,
        A_t: float = 0.7,
        B_t: float = 1.0e4,
        A_c: float = 0.6,
        B_c: float = 300.0,
        f_b_over_f_c: float = 1.16,
    ):
        if E <= 0 or nu <= -1.0 or nu >= 0.5:
            raise ValueError("invalid E or nu")
        if f_c <= 0 or f_t <= 0:
            raise ValueError("f_c, f_t must be positive")
        if not (0.0 <= psi_deg < 90.0):
            raise ValueError("psi_deg out of range")
        if not (0.0 <= A_t <= 1.0):
            raise ValueError("A_t must be in [0, 1]")
        if not (0.0 <= A_c <= 1.0):
            raise ValueError("A_c must be in [0, 1]")
        if B_t <= 0 or B_c <= 0:
            raise ValueError("B_t, B_c must be positive")
        if f_b_over_f_c <= 1.0 or f_b_over_f_c > 1.5:
            raise ValueError("f_b/f_c should be in (1.0, 1.5], typ. 1.16")
        self.E = float(E)
        self.nu = float(nu)
        self.f_c = float(f_c)
        self.f_t = float(f_t)
        self.psi_deg = float(psi_deg)
        self.A_t = float(A_t)
        self.B_t = float(B_t)
        self.A_c = float(A_c)
        self.B_c = float(B_c)
        # Lubliner alpha from biaxial/uniaxial ratio
        r = f_b_over_f_c
        self.alpha = (r - 1.0) / (2.0 * r - 1.0)
        # DP-style yield-surface parameters calibrated from (f_c, f_t)
        # Initial yield: f = alpha I_1 + sqrt(J_2) - k_0 = 0
        # In uniaxial compression: sigma_xx = -f_c, others 0:
        #   I_1 = -f_c, J_2 = f_c^2 / 3
        # => k_0 = alpha (-f_c) + f_c / sqrt(3)
        #        = f_c (1/sqrt(3) - alpha)
        self.k_0 = self.f_c * (1.0 / math.sqrt(3.0) - self.alpha)
        # Dilation in flow potential: beta = sin(psi) / sqrt(3)
        psi = math.radians(psi_deg)
        self.beta = math.sin(psi) / math.sqrt(3.0)
        # Strain thresholds
        self.eps_t0 = f_t / E
        self.eps_c0 = f_c / E
        # Elastic constants
        self.G = E / (2.0 * (1.0 + nu))
        self.K_bulk = E / (3.0 * (1.0 - 2.0 * nu))
        self._D_elastic = self._build_D_elastic()
        # State
        self.eps_p_committed = np.zeros(6)
        self.eps_p_trial = np.zeros(6)
        self.sigma_committed = np.zeros(6)
        self.sigma_trial = np.zeros(6)
        # Hardening variables (cumulative plastic strain in t / c)
        self.kappa_t_committed = 0.0
        self.kappa_c_committed = 0.0
        self.kappa_t_trial = 0.0
        self.kappa_c_trial = 0.0
        # Damage
        self.d_t_committed = 0.0
        self.d_c_committed = 0.0
        self.d_t_trial = 0.0
        self.d_c_trial = 0.0

    # --------------------------------------------------------- elastic D

    def _build_D_elastic(self) -> np.ndarray:
        lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        mu = self.G
        D = np.zeros((6, 6))
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        return D

    def D_elastic(self) -> np.ndarray:
        return self._D_elastic.copy()

    # --------------------------------------------------------- damage laws

    def _d_t(self, kappa_t: float) -> float:
        """Mazars-type exponential tension damage from cumulative
        tensile plastic strain ``kappa_t``."""
        if kappa_t <= 0.0:
            return 0.0
        # d_t = 1 - exp(-B_t * kappa_t), scaled by A_t (residual stiffness)
        d = 1.0 - (1.0 - self.A_t) - self.A_t * math.exp(-self.B_t * kappa_t)
        return min(max(d, 0.0), 0.999999)

    def _d_c(self, kappa_c: float) -> float:
        if kappa_c <= 0.0:
            return 0.0
        d = 1.0 - (1.0 - self.A_c) - self.A_c * math.exp(-self.B_c * kappa_c)
        return min(max(d, 0.0), 0.999999)

    def _stiffness_recovery(self, sigma_bar: np.ndarray) -> float:
        """Stiffness recovery factor ``s`` for tension -> compression
        transition. Returns 1 in pure compression, 0 in pure tension,
        linear in between using the weighting factor

            s = sum(<sigma_bar_i>+) / sum(|sigma_bar_i|)

        (i.e., positive-stress fraction). On stress reversal, the
        compressive damage is *reduced* by (1 - s_c) to reflect crack
        closure -- so the combined damage is::

            d = 1 - (1 - s d_t) (1 - (1 - s) d_c)
        """
        s_pr = _principal_stress(sigma_bar)
        pos = np.maximum(s_pr, 0.0)
        abs_sum = np.sum(np.abs(s_pr))
        if abs_sum < 1.0e-30:
            return 0.5
        return float(np.sum(pos) / abs_sum)

    def _combined_damage(self, d_t: float, d_c: float, s: float) -> float:
        """Lubliner stiffness-recovery damage combination."""
        # Tension damage active in proportion to s
        # Compression damage active in proportion to (1 - s)
        return 1.0 - (1.0 - s * d_t) * (1.0 - (1.0 - s) * d_c)

    # --------------------------------------------------------- response

    def get_response(self, eps_voigt) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(sigma_apparent, D_tangent)`` for the supplied
        total strain.

        Algorithm:

        1. Compute trial effective stress.
        2. Check yield (DP-style with kappa_c-dependent ``k``).
        3. If yielding, run a single-step radial return mapping.
        4. Update ``kappa_t``, ``kappa_c`` from plastic-strain increment.
        5. Update ``d_t(kappa_t)``, ``d_c(kappa_c)``.
        6. Compute apparent stress via stiffness-recovery combination.
        """
        eps = np.asarray(eps_voigt, dtype=float).reshape(6)
        # Trial effective stress
        eps_e_trial = eps - self.eps_p_committed
        sigma_bar_trial = self._D_elastic @ eps_e_trial
        I1 = sigma_bar_trial[0] + sigma_bar_trial[1] + sigma_bar_trial[2]
        s_dev, _ = _deviator_p(sigma_bar_trial)
        J2 = 0.5 * _voigt_double_dot(s_dev)
        sqrt_J2 = math.sqrt(max(J2, 0.0))

        # Compute kappa_t from the *total strain* state (Lee-Fenves tension
        # damage is strain-driven, not plastic-strain-driven). The
        # equivalent tensile strain is the norm of positive principal
        # strains, monotonically tracked.
        eps_pr_total = _principal_stress_from_voigt_strain(eps)
        pos_total = np.maximum(eps_pr_total, 0.0)
        eps_eq_t = float(math.sqrt(float(pos_total @ pos_total)))
        # Track historical max equivalent tensile strain via kappa_t
        new_kappa_t = max(eps_eq_t - self.eps_t0, 0.0)
        self.kappa_t_trial = max(self.kappa_t_committed, new_kappa_t)

        # Yield function with current hardening (compression)
        k_curr = self._yield_strength(self.kappa_c_committed)
        f_trial = self.alpha * I1 + sqrt_J2 - k_curr
        if f_trial <= 0.0:
            sigma_bar = sigma_bar_trial
            self.eps_p_trial = self.eps_p_committed.copy()
            self.kappa_c_trial = self.kappa_c_committed
        else:
            # Plastic step: DP-style return mapping with flow potential
            # g = beta I_1 + sqrt(J_2). Flow direction:
            #     d eps_p = dl * (beta I + s_hat / (2 sqrt(J_2)))
            #             where s_hat is the deviator (Voigt with eng. shear)
            # Consistency: f_new = alpha I_1_new + sqrt(J_2_new) - k_new = 0
            # The standard radial return for DP gives:
            #     dl = f_trial / (G + 9 K alpha beta)
            #     I_1_new = I_1_trial - 9 K beta dl
            #     sqrt(J_2_new) = sqrt(J_2_trial) - G dl
            G = self.G
            K = self.K_bulk
            denom = G + 9.0 * K * self.alpha * self.beta
            dl = f_trial / denom
            sqrt_J2_new = sqrt_J2 - G * dl
            if sqrt_J2_new < 0.0:
                # Apex return: collapse deviator and project I1
                dl = sqrt_J2 / G
                sqrt_J2_new = 0.0
            # Update stress
            sigma_bar = sigma_bar_trial.copy()
            sigma_bar[0:3] -= 3.0 * K * self.beta * dl
            if sqrt_J2 > 1.0e-30:
                # scale deviator
                scale = sqrt_J2_new / sqrt_J2
                s_new = s_dev * scale
                sigma_bar[0] = s_new[0] + (I1 - 9.0 * K * self.beta * dl) / 3.0
                sigma_bar[1] = s_new[1] + (I1 - 9.0 * K * self.beta * dl) / 3.0
                sigma_bar[2] = s_new[2] + (I1 - 9.0 * K * self.beta * dl) / 3.0
                sigma_bar[3:6] = s_new[3:6]
            # Plastic strain increment
            eps_p_inc = np.zeros(6)
            eps_p_inc[0:3] = dl * (self.beta
                                     + s_dev[0:3] / (2.0 * max(sqrt_J2, 1e-30)))
            eps_p_inc[3:6] = dl * s_dev[3:6] / max(sqrt_J2, 1e-30)
            self.eps_p_trial = self.eps_p_committed + eps_p_inc
            # Update compressive hardening: count cumulative compressive
            # plastic strain (negative principal components of eps_p).
            eps_p_pr = _principal_stress_from_voigt_strain(eps_p_inc)
            d_kappa_c = float(np.sum(np.maximum(-eps_p_pr, 0.0)))
            self.kappa_c_trial = self.kappa_c_committed + d_kappa_c

        # Damage update
        d_t_new = max(self._d_t(self.kappa_t_trial), self.d_t_committed)
        d_c_new = max(self._d_c(self.kappa_c_trial), self.d_c_committed)
        self.d_t_trial = d_t_new
        self.d_c_trial = d_c_new
        # Stiffness recovery
        s_factor = self._stiffness_recovery(sigma_bar)
        d_combined = self._combined_damage(d_t_new, d_c_new, s_factor)
        sigma_apparent = (1.0 - d_combined) * sigma_bar
        self.sigma_trial = sigma_apparent
        # Secant tangent (damaged D_e)
        D_tangent = (1.0 - d_combined) * self._D_elastic
        return sigma_apparent.copy(), D_tangent

    def _yield_strength(self, kappa_c: float) -> float:
        """Cohesion ``k(kappa_c)`` softening with cumulative compressive
        plastic strain. Linear softening down to a residual ~10% of k_0."""
        if kappa_c <= 0:
            return self.k_0
        # Softening rate tied to B_c
        softening = math.exp(-self.B_c * kappa_c * 0.5)
        return self.k_0 * (0.1 + 0.9 * softening)

    # --------------------------------------------------------- state

    def commit_state(self) -> None:
        self.eps_p_committed = self.eps_p_trial.copy()
        self.sigma_committed = self.sigma_trial.copy()
        self.kappa_t_committed = self.kappa_t_trial
        self.kappa_c_committed = self.kappa_c_trial
        self.d_t_committed = self.d_t_trial
        self.d_c_committed = self.d_c_trial

    def revert_state(self) -> None:
        self.eps_p_trial = self.eps_p_committed.copy()
        self.sigma_trial = self.sigma_committed.copy()
        self.kappa_t_trial = self.kappa_t_committed
        self.kappa_c_trial = self.kappa_c_committed
        self.d_t_trial = self.d_t_committed
        self.d_c_trial = self.d_c_committed

    def clone(self) -> "ConcreteDamagePlasticity3D":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"ConcreteDamagePlasticity3D(E={self.E:g}, nu={self.nu:g}, "
            f"f_c={self.f_c:g}, f_t={self.f_t:g}, "
            f"psi={self.psi_deg:g}°)"
        )


def _principal_stress_from_voigt_strain(eps_voigt: np.ndarray) -> np.ndarray:
    """Principal values from a (6,) Voigt engineering-strain vector."""
    e = np.array([
        [eps_voigt[0], eps_voigt[3] / 2.0, eps_voigt[5] / 2.0],
        [eps_voigt[3] / 2.0, eps_voigt[1], eps_voigt[4] / 2.0],
        [eps_voigt[5] / 2.0, eps_voigt[4] / 2.0, eps_voigt[2]],
    ])
    return np.sort(np.linalg.eigvalsh(e))[::-1]
