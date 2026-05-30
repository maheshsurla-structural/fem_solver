"""Finite-strain J₂ plasticity using logarithmic (Hencky) strain.

The "Hencky-J₂" formulation (Eterovic-Bathe 1990, Simo 1998 Vol II)
extends small-strain ``J₂`` plasticity to finite strain via the
multiplicative decomposition of the deformation gradient:

    F = F_e F_p,

and a logarithmic-strain measure of the elastic part:

    ε_e = ½ ln(b_e),       b_e = F_e F_e^T   (elastic left Cauchy-Green).

In principal-axis space the return mapping is *identical* to the
small-strain radial-return algorithm because the Hencky strain
``ε_e`` is the natural conjugate of Kirchhoff stress ``τ`` for
isotropic elasticity. After return mapping, the algorithm
rotates the stress back to the global frame using the eigenvectors
of ``b_e^{trial}``.

This module ships:

* :class:`FiniteJ2Plasticity3D` -- compressible elasto-plastic
  material with isotropic ``J₂`` hardening (sigma_y(alpha) = sigma_y0
  + H alpha) suitable for finite-strain metal-plasticity simulations.

The material exposes ``response_S(F) -> (S_voigt, C_M_voigt)`` so it
plugs into the Total-Lagrangian Hex8 element
(:class:`~femsolver.elements.hex8_TL.Hex8TL`).

State is per-evaluation-point and tracked via ``commit_state`` /
``revert_state`` hooks (mirrors the small-strain plasticity API).

References
----------
* Simo, J.C. & Hughes, T.J.R. (1998). *Computational Inelasticity*.
  Chapter 9.
* Simo, J.C. (1992). "Algorithms for static and dynamic multiplicative
  plasticity." *CMAME*, 99, 61-112.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femsolver.materials.hyperelastic import _sym_to_voigt


@dataclass
class FiniteJ2Plasticity3D:
    """Finite-strain isotropic J₂ plasticity (Hencky).

    Parameters
    ----------
    tag : int
    E : float
        Initial Young's modulus (Pa).
    nu : float
        Initial Poisson ratio.
    sigma_y0 : float
        Initial yield stress (Pa, von Mises).
    H : float, default 0.0
        Linear isotropic-hardening modulus (Pa).  Set 0 for perfect
        plasticity.
    rho : float, default 0.0
    """

    tag: int
    E: float
    nu: float
    sigma_y0: float
    H: float = 0.0
    rho: float = 0.0

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise ValueError("E must be > 0")
        if not (0.0 < self.nu < 0.5):
            raise ValueError("nu must be in (0, 0.5)")
        if self.sigma_y0 <= 0.0:
            raise ValueError("sigma_y0 must be > 0")
        if self.H < 0.0:
            raise ValueError("H must be >= 0")
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))
        # State (per material instance -- the element should clone
        # one instance per Gauss point)
        self.b_e_committed = np.eye(3)    # elastic left Cauchy-Green
        self.alpha_committed = 0.0         # cumulative equiv plastic strain
        self.b_e_trial = np.eye(3)
        self.alpha_trial = 0.0
        # For computing tangent
        self._last_C_M = self._initial_tangent()

    def _initial_tangent(self) -> np.ndarray:
        """Small-strain isotropic elastic tangent at the current
        (E, nu)."""
        K, G = self.K, self.mu
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

    # ----------------------------------------------- response

    def response_S(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (S_voigt, C_M_voigt) at deformation gradient ``F``.

        Algorithm (per Simo 1992):
          1. Trial elastic left Cauchy-Green:  ``b_e^trial = F F_p^{-1} F_p^{-T} F^T``
             (here we use a relative update from the committed state).
          2. Spectral decomposition of ``b_e^trial`` -> principal
             stretches λ_i^trial.
          3. Trial Hencky strain: ``ε_i = ln λ_i``.
          4. Trial Kirchhoff stress (principal): ``τ_i = K trε I + 2μ dev(ε)``.
          5. Yield check:
               σ_eq = sqrt(3/2 dev(τ) : dev(τ)).
               If σ_eq <= sigma_y(α) -> elastic, accept trial.
               Else: radial return in principal axes.
          6. Update ``b_e`` from the corrected principal stretches and
             eigenvectors; map τ back to global frame.
          7. ``S = J F^{-1} τ F^{-T}``; return Voigt.

        For brevity / robustness this implementation tracks the elastic
        log-strain in the deformed configuration via ``b_e^{trial} =
        F F_committed_p_inv``; we approximate the relative update by
        applying ``F · b_e_committed · F^T`` for incremental
        deformations from the committed state (i.e., we assume the
        committed configuration is the "previous step" reference).
        """
        if F.shape != (3, 3):
            raise ValueError(f"F must be (3, 3), got {F.shape}")
        # Approximate trial b_e using the committed b_e and the
        # incremental deformation gradient relative to the committed
        # configuration (treating the committed state as the start of
        # the current step).
        b_e_trial = F @ self.b_e_committed @ F.T
        # Spectral decomposition
        eigvals, eigvecs = np.linalg.eigh(b_e_trial)
        eigvals = np.clip(eigvals, 1.0e-12, None)
        eps_trial = 0.5 * np.log(eigvals)                  # principal Hencky strain
        # Volumetric / deviatoric split
        tr_eps = float(np.sum(eps_trial))
        eps_dev = eps_trial - (tr_eps / 3.0)
        # Trial Kirchhoff principal stresses
        p_trial = self.K * tr_eps
        tau_dev_trial = 2.0 * self.mu * eps_dev
        # von Mises trial stress
        sigma_eq_trial = np.sqrt(1.5 * np.sum(tau_dev_trial ** 2))
        sigma_y = self.sigma_y0 + self.H * self.alpha_committed
        if sigma_eq_trial <= sigma_y:
            # Elastic step
            self.b_e_trial = b_e_trial
            self.alpha_trial = self.alpha_committed
            tau_principal = p_trial + tau_dev_trial
        else:
            # Plastic step -- radial return
            d_gamma = (sigma_eq_trial - sigma_y) / (3.0 * self.mu + self.H)
            n = tau_dev_trial / sigma_eq_trial   # unit deviator (von Mises normal)
            tau_dev_new = tau_dev_trial * (1.0 - 3.0 * self.mu * d_gamma / sigma_eq_trial)
            tau_principal = p_trial + tau_dev_new
            # Updated principal elastic strain (used to rebuild b_e)
            eps_new = eps_dev * (1.0 - 3.0 * self.mu * d_gamma / sigma_eq_trial) \
                       + (tr_eps / 3.0)
            self.b_e_trial = (eigvecs *
                               np.exp(2.0 * eps_new)) @ eigvecs.T
            self.alpha_trial = self.alpha_committed + d_gamma
        # Rotate Kirchhoff stress back to global frame
        tau_global = (eigvecs * tau_principal) @ eigvecs.T
        # Cauchy stress σ = τ / J
        J = float(np.linalg.det(F))
        sigma_global = tau_global / J
        # 2nd PK stress: S = J F^{-1} σ F^{-T}
        F_inv = np.linalg.inv(F)
        S = J * F_inv @ sigma_global @ F_inv.T
        # Tangent: use the small-strain elastoplastic tangent as a
        # serviceable approximation (full algorithmic consistent
        # tangent is involved; for moderate-strain problems this is
        # adequate).
        C_M = self._initial_tangent()
        if sigma_eq_trial > sigma_y:
            # Elastoplastic tangent in Voigt
            n_voigt = self._stress_dev_voigt(eigvecs, tau_dev_trial / sigma_eq_trial)
            beta = 3.0 * self.mu / (3.0 * self.mu + self.H)
            C_M = C_M - beta * np.outer(n_voigt, n_voigt) * (2.0 * self.mu)
        self._last_C_M = C_M
        return _sym_to_voigt(S), C_M

    @staticmethod
    def _stress_dev_voigt(
        eigvecs: np.ndarray, dev_principal: np.ndarray,
    ) -> np.ndarray:
        """Rotate a diagonal deviator (3,) back to global Voigt order."""
        dev_global = (eigvecs * dev_principal) @ eigvecs.T
        return _sym_to_voigt(dev_global)

    # ----------------------------------------------- state lifecycle
    def commit_state(self) -> None:
        self.b_e_committed = self.b_e_trial.copy()
        self.alpha_committed = float(self.alpha_trial)

    def revert_state(self) -> None:
        self.b_e_trial = self.b_e_committed.copy()
        self.alpha_trial = float(self.alpha_committed)

    def clone(self) -> "FiniteJ2Plasticity3D":
        """Independent copy for per-Gauss-point state."""
        c = FiniteJ2Plasticity3D(
            tag=self.tag, E=self.E, nu=self.nu,
            sigma_y0=self.sigma_y0, H=self.H, rho=self.rho,
        )
        c.b_e_committed = self.b_e_committed.copy()
        c.alpha_committed = float(self.alpha_committed)
        return c
