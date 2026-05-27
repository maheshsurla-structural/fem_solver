"""Ibarra-Medina-Krawinkler (IMK) deteriorating material.

The "modified IMK" model is the standard collapse-capable hysteretic
rule used by FEMA P695 for incremental dynamic analysis. The
defining feature is a **post-cap negative-stiffness branch** that
makes simulated structures collapse under sufficient cyclic excursion.

Backbone
--------
Five branches (positive side; symmetric on negative side):

::

      σ
      |          σ_cap o-----.
      |              /       \\\\
      |   σ_y o-----/         \\\\
      |        /              \\\\___ σ_res (plateau)
      |       /                          \\\\
      |      /                            \\\\___ fracture
      +-----+----+----+----+----+----+----+----- ε
            ε_y  ε_cap   ε_res      ε_ult

* **Elastic** (ε in [0, ε_y]): slope E
* **Hardening** (ε in [ε_y, ε_cap]): slope ``b · E``, reaches
  ``σ_cap = σ_y + b · E · (ε_cap - ε_y)``
* **Post-cap** (ε in [ε_cap, ε_res]): NEGATIVE slope
  ``α_pc · E`` (α_pc < 0) until strength drops to residual
  ``σ_res = r · σ_cap``
* **Residual plateau** (ε in [ε_res, ε_ult]): horizontal at σ_res
* **Fracture** (ε > ε_ult): σ = 0

The capping point ``(ε_cap, σ_cap)`` is the "peak load" beyond which
the system loses strength.

Cyclic memory (peak-oriented)
-----------------------------
This implementation uses the simplest peak-oriented memory:
* Loading toward an as-yet-unreached extreme follows the backbone.
* Once a peak ``(ε_peak, σ_peak)`` has been recorded on either side,
  unloading from it is elastic at slope ``E`` to the residual strain
  ``ε_residual = ε_peak - σ_peak / E``.
* Loading after a reversal follows a line toward the opposite-side
  peak (or the first-yield point if no opposite-side history).

Limitations / future work
-------------------------
* No cyclic strength deterioration (basic, post-cap, unloading-
  stiffness, or accelerated-reloading-stiffness) -- the Ibarra-
  Krawinkler hysteretic-energy-driven Lambda formulation is a
  natural extension and a future Phase 24.4.x.
* Symmetric backbone (asymmetric a future extension; users with
  asymmetric needs can pair two IMKs in opposed fibers).

What you get today
------------------
A working trilinear-backbone material that exhibits the defining
features of IMK -- particularly the post-cap negative slope that
drives simulated collapse in IDA and time-history analyses --
suitable for first-order collapse-capacity studies.
"""
from __future__ import annotations

from femsolver.materials.uniaxial.base import UniaxialMaterial


class UniaxialIMK(UniaxialMaterial):
    """Ibarra-Medina-Krawinkler deteriorating material.

    Parameters
    ----------
    E : float
        Initial elastic modulus.
    sigma_y : float
        Yield stress magnitude (positive, symmetric).
    b : float, default 0.03
        Strain-hardening ratio (post-yield slope / E), in ``[0, 1)``.
    eps_cap : float, default 8 ε_y
        Strain at peak load (where post-cap negative slope begins).
        Default is "8 times yield strain", typical for ductile steel
        beam-column plastic hinges.
    alpha_pc : float, default -0.1
        Post-cap-to-initial stiffness ratio (negative; e.g., -0.1 =
        post-cap slope is -10% of E).
    sigma_res_ratio : float, default 0.4
        Residual strength as a fraction of σ_cap (peak load).
        ``0.4`` means residual ≈ 40% of σ_cap (typical for steel
        beam-column hinges with moderate damage).
    eps_ult : float, default 30 ε_y
        Ultimate strain (fracture). Beyond this strain, σ drops
        abruptly to zero.
    """

    def __init__(self, E: float, sigma_y: float, *,
                 b: float = 0.03,
                 eps_cap: float | None = None,
                 alpha_pc: float = -0.1,
                 sigma_res_ratio: float = 0.4,
                 eps_ult: float | None = None):
        if E <= 0.0:
            raise ValueError(f"E must be positive, got {E}")
        if sigma_y <= 0.0:
            raise ValueError(f"sigma_y must be positive, got {sigma_y}")
        if not (0.0 <= b < 1.0):
            raise ValueError(f"b must be in [0, 1), got {b}")
        if alpha_pc >= 0.0:
            raise ValueError(
                f"alpha_pc (post-cap slope ratio) must be < 0, got {alpha_pc}"
            )
        if not (0.0 < sigma_res_ratio < 1.0):
            raise ValueError(
                f"sigma_res_ratio must be in (0, 1), got {sigma_res_ratio}"
            )
        self.E = float(E)
        self.sigma_y = float(sigma_y)
        self.b = float(b)
        self.alpha_pc = float(alpha_pc)
        self.sigma_res_ratio = float(sigma_res_ratio)
        self.eps_y = self.sigma_y / self.E
        self.eps_cap = (float(eps_cap) if eps_cap is not None
                          else 8.0 * self.eps_y)
        if self.eps_cap <= self.eps_y:
            raise ValueError(
                f"eps_cap ({self.eps_cap}) must exceed eps_y ({self.eps_y})"
            )
        # Peak strength
        self.sigma_cap = (self.sigma_y
                            + self.b * self.E
                            * (self.eps_cap - self.eps_y))
        # Residual strength
        self.sigma_res = self.sigma_res_ratio * self.sigma_cap
        # Strain at which negative slope reaches residual
        # σ_cap + α_pc · E · (ε_res - ε_cap) = σ_res
        # ε_res = ε_cap + (σ_res - σ_cap) / (α_pc · E)
        self.eps_res = (self.eps_cap
                          + (self.sigma_res - self.sigma_cap)
                          / (self.alpha_pc * self.E))
        # Ultimate strain
        self.eps_ult = (float(eps_ult) if eps_ult is not None
                          else 30.0 * self.eps_y)
        if self.eps_ult <= self.eps_res:
            raise ValueError(
                f"eps_ult ({self.eps_ult}) must exceed eps_res "
                f"({self.eps_res:.6g})"
            )
        # State (committed) -- start at the first-yield strains so
        # initial elastic loading produces the standard backbone.
        self.eps_max_pos_committed: float = self.eps_y
        self.eps_max_neg_committed: float = -self.eps_y
        self.last_eps_committed: float = 0.0
        self.eps_max_pos_trial: float = self.eps_y
        self.eps_max_neg_trial: float = -self.eps_y
        self.last_eps_trial: float = 0.0
        self.sigma_trial: float = 0.0
        self.Et: float = self.E

    # ----------------------------------------------------- backbone
    def _backbone_pos(self, eps: float) -> tuple[float, float]:
        """Positive-side trilinear backbone."""
        if eps <= 0.0:
            return self.E * eps, self.E
        if eps <= self.eps_y:
            return self.E * eps, self.E
        if eps <= self.eps_cap:
            return (self.sigma_y + self.b * self.E * (eps - self.eps_y),
                    self.b * self.E)
        if eps <= self.eps_res:
            return (self.sigma_cap + self.alpha_pc * self.E
                    * (eps - self.eps_cap),
                    self.alpha_pc * self.E)
        if eps <= self.eps_ult:
            return self.sigma_res, 0.0
        # Fracture
        return 0.0, 0.0

    def _backbone(self, eps: float) -> tuple[float, float]:
        """Symmetric trilinear envelope (positive + negative)."""
        if eps >= 0.0:
            return self._backbone_pos(eps)
        s, Et = self._backbone_pos(-eps)
        return -s, Et

    # ----------------------------------------------------- response
    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        em_pos = max(self.eps_max_pos_committed, eps)
        em_neg = min(self.eps_max_neg_committed, eps)
        self.eps_max_pos_trial = em_pos
        self.eps_max_neg_trial = em_neg
        self.last_eps_trial = eps

        # Past previous peak -> follow backbone
        if eps >= self.eps_max_pos_committed:
            sigma, Et = self._backbone(eps)
            self.sigma_trial = sigma
            self.Et = Et
            return sigma, Et
        if eps <= self.eps_max_neg_committed:
            sigma, Et = self._backbone(eps)
            self.sigma_trial = sigma
            self.Et = Et
            return sigma, Et

        # Interior -- peak-oriented memory: elastic unload to residual
        # strain, then aim at the opposite peak.
        ep_pos = self.eps_max_pos_committed
        ep_neg = self.eps_max_neg_committed
        sg_pos, _ = self._backbone(ep_pos)
        sg_neg, _ = self._backbone(ep_neg)
        # Elastic unload residuals
        er_pos = ep_pos - sg_pos / self.E
        er_neg = ep_neg - sg_neg / self.E
        deps = eps - self.last_eps_committed
        if deps >= 0.0:
            # Loading toward +max
            if eps <= er_neg:
                # On the elastic unload-from-(-) line, slope E
                sigma = sg_neg + self.E * (eps - ep_neg)
                Et = self.E
            else:
                # Reload-to-+peak line: from (er_neg, 0) to (ep_pos, sg_pos)
                slope = sg_pos / (ep_pos - er_neg)
                sigma = slope * (eps - er_neg)
                Et = slope
        else:
            # Unloading from + or loading toward -max
            if eps >= er_pos:
                # Elastic unload from +peak, slope E
                sigma = sg_pos - self.E * (ep_pos - eps)
                Et = self.E
            else:
                # Reload-to--peak line: from (er_pos, 0) to (ep_neg, sg_neg)
                slope = sg_neg / (ep_neg - er_pos)
                sigma = slope * (eps - er_pos)
                Et = slope

        self.sigma_trial = sigma
        self.Et = Et
        return sigma, Et

    def commit_state(self) -> None:
        self.eps_max_pos_committed = self.eps_max_pos_trial
        self.eps_max_neg_committed = self.eps_max_neg_trial
        self.last_eps_committed = self.last_eps_trial

    def revert_state(self) -> None:
        self.eps_max_pos_trial = self.eps_max_pos_committed
        self.eps_max_neg_trial = self.eps_max_neg_committed
        self.last_eps_trial = self.last_eps_committed

    def __repr__(self) -> str:
        return (
            f"UniaxialIMK(E={self.E:g}, sigma_y={self.sigma_y:g}, "
            f"b={self.b:g}, eps_cap={self.eps_cap:g}, "
            f"alpha_pc={self.alpha_pc:g})"
        )
