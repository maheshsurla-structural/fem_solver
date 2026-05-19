"""Damping models for transient and dynamic analysis.

Currently provides :class:`RayleighDamping` — the standard
``C = alpha_M * M + alpha_K * K`` decomposition used in essentially all
commercial structural-dynamics codes. The mass-proportional part
suppresses low-frequency modes; the stiffness-proportional part
suppresses high-frequency modes. Together they give the structural
engineer two knobs to tune two target modes' damping ratios.

Future damping models (modal damping with per-mode ratios, Caughey series,
hysteretic damping) can be added as additional classes alongside.
"""
from __future__ import annotations

from dataclasses import dataclass

import scipy.sparse as sp


@dataclass
class RayleighDamping:
    """Mass- and stiffness-proportional damping: ``C = aM M + aK K``.

    Parameters
    ----------
    alpha_M : float
        Mass-proportional coefficient. Adds damping that decreases with
        frequency (dominant on low modes).
    alpha_K : float
        Stiffness-proportional coefficient. Adds damping that increases
        with frequency (dominant on high modes).
    """

    alpha_M: float = 0.0
    alpha_K: float = 0.0

    def build(self, M, K) -> sp.csc_matrix:
        """Return ``C = alpha_M M + alpha_K K`` as a CSC sparse matrix."""
        C = self.alpha_M * M + self.alpha_K * K
        return C.tocsc() if sp.issparse(C) else C

    @classmethod
    def from_modes(
        cls,
        omega_1: float, zeta_1: float,
        omega_2: float, zeta_2: float,
    ) -> "RayleighDamping":
        """Construct from two target frequency / damping-ratio pairs.

        Solves the 2 x 2 system::

            zeta_i = alpha_M / (2 omega_i)  +  alpha_K omega_i / 2

        for ``i = 1, 2``. The resulting damping ratio at any other
        frequency follows the Rayleigh curve, so modes between
        ``omega_1`` and ``omega_2`` are damped *less* and modes outside
        that range are damped *more*. The standard recipe in
        commercial seismic analysis is to set ``omega_1, omega_2`` at
        the first and a higher significant mode, and ``zeta_1 = zeta_2 = 0.05``
        for 5 % structural damping.

        Parameters
        ----------
        omega_1, omega_2 : float
            Angular frequencies (rad/s). Must be positive and distinct.
        zeta_1, zeta_2 : float
            Damping ratios at the two frequencies.
        """
        if omega_1 <= 0.0 or omega_2 <= 0.0:
            raise ValueError("omegas must be positive")
        if omega_1 == omega_2:
            raise ValueError("omegas must be distinct")
        # Solve the 2x2 system:
        #   [1/(2 w1)     w1/2 ] [aM]   [z1]
        #   [1/(2 w2)     w2/2 ] [aK] = [z2]
        # Cramer's rule:
        det = 0.25 * (omega_2 - omega_1) * (omega_2 + omega_1) / (omega_1 * omega_2)
        # Equivalent: det = (w2^2 - w1^2) / (4 w1 w2)
        alpha_M = (zeta_1 * omega_2 / 2.0 - zeta_2 * omega_1 / 2.0) / det
        # For alpha_K the cofactor uses the FIRST column of the 2x2
        # system (the 1/(2 w_i) entries), so the omegas pair *opposite*
        # to the alpha_M case:
        #   alpha_K_num = z2/(2 w1) - z1/(2 w2)
        alpha_K = (zeta_2 / (2.0 * omega_1) - zeta_1 / (2.0 * omega_2)) / det
        return cls(alpha_M=alpha_M, alpha_K=alpha_K)

    def damping_ratio_at(self, omega: float) -> float:
        """Return the Rayleigh damping ratio at angular frequency ``omega``."""
        return 0.5 * (self.alpha_M / omega + self.alpha_K * omega)
