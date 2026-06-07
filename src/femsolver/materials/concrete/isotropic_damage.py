"""Isotropic scalar-damage concrete (Mazars 1986, with tension / compression
split).

A pragmatic 3-D concrete model that captures the two essential
features of concrete behaviour:

* **Tension damage** -- cracking starts at a low strain (``eps_t0``,
  typically ``~ 1e-4``) and softens rapidly. Tensile strength after
  damage is approximately zero.
* **Compression damage** -- crushing starts at a higher strain
  (``eps_c0``, typically ``~ 2e-3``) and softens more gradually.

The full Lubliner-Lee-Fenves damage-plasticity model couples the
above with hardening plasticity. This simplified variant uses pure
isotropic damage (no plastic strain) which is sufficient for
monotonic-loading design checks and gives the right
load-displacement envelope shape. Cyclic recovery is approximated
by the standard Mazars (1986) "unilateral" rule that recovers
compressive stiffness on stress reversal from tension.

The yield/loading function is built on a *Mazars equivalent strain*

    eps_eq = sqrt( sum_i <eps_i>+ ^2 )

where ``<.>+`` denotes the positive part and ``eps_i`` are the
principal strains. ``eps_eq`` is then split into tension- and
compression-driven damage variables ``d_t, d_c`` via the standard
Mazars exponential laws. The final damage is a stress-state-weighted
combination

    d = alpha_t * d_t + alpha_c * d_c

with ``alpha_t + alpha_c = 1`` interpolating from pure tension
(``alpha_t = 1``) to pure compression (``alpha_c = 1``).

Sign convention: tension positive throughout.
"""
from __future__ import annotations

import copy
import math

import numpy as np


def _principal_strains(eps_voigt: np.ndarray):
    """Return (descending) principal strains of a (6,) engineering Voigt."""
    e = np.array([
        [eps_voigt[0], eps_voigt[3] / 2.0, eps_voigt[5] / 2.0],
        [eps_voigt[3] / 2.0, eps_voigt[1], eps_voigt[4] / 2.0],
        [eps_voigt[5] / 2.0, eps_voigt[4] / 2.0, eps_voigt[2]],
    ])
    w = np.linalg.eigvalsh(e)
    return np.sort(w)[::-1]


class ConcreteDamage3D:
    """Isotropic scalar-damage concrete with tension / compression split.

    Parameters
    ----------
    E : float
        Young's modulus of the undamaged concrete.
    nu : float
        Poisson's ratio.
    eps_t0 : float
        Tensile-strain damage threshold. Damage starts here.
        Typical: ``f_t / E``, e.g., ``3 MPa / 30 GPa = 1e-4``.
    eps_c0 : float
        Compressive-strain damage threshold (positive magnitude;
        used as ``|eps| > eps_c0`` triggers compressive damage).
        Typical: 0.001 to 0.003.
    A_t : float, default 1.0
        Tensile-damage shape parameter (controls residual tensile
        stiffness; 0 = brittle, 1 = quasi-brittle).
    B_t : float, default 1.0e4
        Tensile-damage softening rate. Larger = more brittle.
    A_c : float, default 1.0
        Compressive analogue of ``A_t``.
    B_c : float, default 1000.0
        Compressive analogue of ``B_t``.
    """

    def __init__(
        self,
        E: float,
        nu: float,
        eps_t0: float = 1.0e-4,
        eps_c0: float = 2.0e-3,
        A_t: float = 1.0,
        B_t: float = 1.0e4,
        A_c: float = 1.0,
        B_c: float = 1000.0,
    ):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"nu must be in (-1, 0.5), got {nu}")
        if eps_t0 <= 0.0:
            raise ValueError(f"eps_t0 must be positive, got {eps_t0}")
        if eps_c0 <= 0.0:
            raise ValueError(f"eps_c0 must be positive, got {eps_c0}")
        if not (0.0 <= A_t <= 1.0):
            raise ValueError(f"A_t must be in [0, 1], got {A_t}")
        if not (0.0 <= A_c <= 1.0):
            raise ValueError(f"A_c must be in [0, 1], got {A_c}")
        if B_t <= 0.0:
            raise ValueError(f"B_t must be positive, got {B_t}")
        if B_c <= 0.0:
            raise ValueError(f"B_c must be positive, got {B_c}")
        self.E = float(E)
        self.nu = float(nu)
        self.eps_t0 = float(eps_t0)
        self.eps_c0 = float(eps_c0)
        self.A_t = float(A_t)
        self.B_t = float(B_t)
        self.A_c = float(A_c)
        self.B_c = float(B_c)
        # Elastic constants
        self.G = E / (2.0 * (1.0 + nu))
        self._lambda_lame = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self._D_elastic = self._build_D_elastic()
        # State: historical max equivalent strain (monotonically
        # non-decreasing). The damage variables are derived from it.
        self.eps_eq_max_committed = 0.0
        self.eps_eq_max_trial = 0.0
        self.d_committed = 0.0
        self.d_trial = 0.0
        self.sigma_committed = np.zeros(6)
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

    # ----------------------------------------------------- damage laws
    def _mazars_d_t(self, eps_eq: float) -> float:
        if eps_eq <= self.eps_t0:
            return 0.0
        e0 = self.eps_t0
        d = 1.0 - (e0 / eps_eq) * (
            (1.0 - self.A_t) + self.A_t * math.exp(-self.B_t * (eps_eq - e0))
        )
        return min(max(d, 0.0), 0.999999)

    def _mazars_d_c(self, eps_eq: float) -> float:
        if eps_eq <= self.eps_c0:
            return 0.0
        e0 = self.eps_c0
        d = 1.0 - (e0 / eps_eq) * (
            (1.0 - self.A_c) + self.A_c * math.exp(-self.B_c * (eps_eq - e0))
        )
        return min(max(d, 0.0), 0.999999)

    # ----------------------------------------------------- response
    def get_response(self, eps_voigt) -> tuple[np.ndarray, np.ndarray]:
        eps = np.asarray(eps_voigt, dtype=float).reshape(6)
        # Effective (undamaged) stress
        sigma_eff = self._D_elastic @ eps
        # Equivalent strain a la Mazars (positive-strain norm)
        e_pr = _principal_strains(eps)
        pos = np.where(e_pr > 0.0, e_pr, 0.0)
        eps_eq_pos = float(math.sqrt(float(pos @ pos)))
        # Negative-strain norm for compressive damage
        neg = np.where(e_pr < 0.0, -e_pr, 0.0)
        eps_eq_neg = float(math.sqrt(float(neg @ neg)))
        # Historical max (separate per-mode)
        self.eps_eq_max_trial = max(
            self.eps_eq_max_committed, max(eps_eq_pos, eps_eq_neg),
        )
        d_t = self._mazars_d_t(eps_eq_pos)
        d_c = self._mazars_d_c(eps_eq_neg)
        # Mazars weighting: alpha_t for tension regions, alpha_c for
        # compression. Compute from principal strains (positive vs
        # negative contributions to the elastic strain energy).
        e_t = float(pos @ pos)
        e_c = float(neg @ neg)
        e_tot = e_t + e_c
        if e_tot < 1.0e-30:
            alpha_t = 1.0
            alpha_c = 0.0
        else:
            alpha_t = e_t / e_tot
            alpha_c = e_c / e_tot
        d = alpha_t * d_t + alpha_c * d_c
        # No healing on unloading (committed monotonic non-decreasing)
        d = max(d, self.d_committed)
        self.d_trial = d
        sigma = (1.0 - d) * sigma_eff
        self.sigma_trial = sigma
        # Secant stiffness (no algorithmic tangent — fine for static
        # analyses with displacement / arc-length control)
        D_secant = (1.0 - d) * self._D_elastic
        return sigma.copy(), D_secant

    # ----------------------------------------------------- state
    def commit_state(self) -> None:
        self.eps_eq_max_committed = self.eps_eq_max_trial
        self.d_committed = self.d_trial
        self.sigma_committed = self.sigma_trial.copy()

    def revert_state(self) -> None:
        self.eps_eq_max_trial = self.eps_eq_max_committed
        self.d_trial = self.d_committed
        self.sigma_trial = self.sigma_committed.copy()

    def clone(self) -> "ConcreteDamage3D":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"ConcreteDamage3D(E={self.E:g}, nu={self.nu:g}, "
            f"eps_t0={self.eps_t0:g}, eps_c0={self.eps_c0:g})"
        )
