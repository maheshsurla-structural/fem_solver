"""FEMA P-58 component-level damage and loss assessment.

The FEMA P-58 methodology assesses building seismic performance at
the **component** level (drywall partitions, ceiling tiles, suspended
mechanical equipment, structural beam-column connections, ...) rather
than at the system level. For each component:

1. A **demand parameter** (PSD, PFA, residual drift, ...) is computed
   from the structural response.
2. A set of **damage states** (DS0=undamaged, DS1, DS2, ..., DS_N) is
   defined, each with a **fragility curve**::

       P(DS >= ds_i | EDP) = Phi( ln(EDP / theta_i) / beta_i )

   i.e., a lognormal CDF anchored at median ``theta_i`` and log-stddev
   ``beta_i``.
3. Each damage state carries a **consequence** -- here we model
   *repair cost* as a lognormal RV with median ``c_i`` and log-stddev
   ``sigma_cost,i``. (Other consequences: repair time, casualties,
   recovery quantity -- the framework generalises.)
4. The **expected loss** for a component group of ``Q`` units at a
   single EDP is::

       E[L] = Q · sum_{i=1..N} P(DS = ds_i | EDP) · E[cost_i]

   where ``P(DS = ds_i) = P(DS >= ds_i) - P(DS >= ds_{i+1})``
   (sequential damage states).
5. **Monte-Carlo** assessment: per realisation, sample the damage
   state from the discrete distribution, sample the consequence from
   the lognormal, sum across all component groups to get a total-loss
   realisation. The empirical CDF of these realisations is the
   *building loss curve*.

References
----------
* FEMA P-58 (2018) "Seismic Performance Assessment of Buildings."
  Vol 1 (Methodology) and Vol 2 (Implementation Guide).
* Porter, K.A. (2003) "An Overview of PEER's Performance-Based
  Earthquake Engineering Methodology." *Proc. 9th ICASP*, San
  Francisco.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ============================================================ Damage states

@dataclass
class DamageState:
    """One damage state with its fragility + repair-cost consequence.

    Attributes
    ----------
    name : str
        Short identifier, e.g. ``"DS1"``, ``"DS2"``.
    fragility_theta : float
        Median EDP at which this damage state is reached
        (P(DS >= this | EDP = theta) = 0.5). Must be > 0.
    fragility_beta : float
        Log-stddev of the fragility lognormal CDF. Typical 0.3-0.6.
    cost_median : float
        Median repair cost (USD or any consistent unit) PER UNIT of
        component. Must be > 0.
    cost_beta : float, default 0.3
        Log-stddev of repair cost. Typical 0.2-0.4 for well-defined
        damage states.
    """

    name: str
    fragility_theta: float
    fragility_beta: float
    cost_median: float
    cost_beta: float = 0.3

    def __post_init__(self) -> None:
        if self.fragility_theta <= 0.0:
            raise ValueError("fragility_theta must be > 0")
        if self.fragility_beta <= 0.0:
            raise ValueError("fragility_beta must be > 0")
        if self.cost_median <= 0.0:
            raise ValueError("cost_median must be > 0")
        if self.cost_beta < 0.0:
            raise ValueError("cost_beta must be >= 0")

    def prob_exceeds(self, edp: float) -> float:
        """``P(DS >= this | EDP)`` -- the lognormal CDF."""
        if edp <= 0.0:
            return 0.0
        z = math.log(edp / self.fragility_theta) / self.fragility_beta
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def expected_cost(self) -> float:
        """E[cost] = median · exp(beta^2 / 2)  (lognormal mean)."""
        return self.cost_median * math.exp(0.5 * self.cost_beta ** 2)

    def sample_cost(self, rng: np.random.Generator) -> float:
        """Single sample of repair cost from the lognormal."""
        if self.cost_beta == 0.0:
            return self.cost_median
        # ln(cost) ~ Normal(ln median, cost_beta)
        return float(self.cost_median * math.exp(
            rng.normal(0.0, self.cost_beta)
        ))


# ============================================================ Component

@dataclass
class ComponentFragility:
    """A component type with its ordered list of damage states.

    Damage states are interpreted **sequentially**: ``DS1`` is the
    first level of damage, ``DS2`` is more severe than ``DS1``, etc.
    The fragility ``theta`` values must therefore be monotonically
    increasing across the list. ``cost_median`` values are also
    typically increasing.

    Attributes
    ----------
    name : str
        Component identifier, e.g. ``"B1041.001 drywall partition"``.
    edp_type : str
        EDP driver: ``"PSD"`` (peak story drift), ``"PFA"`` (peak
        floor acceleration), or ``"RES"`` (residual drift).
    damage_states : list of DamageState
        Ordered increasing in fragility_theta.
    """

    name: str
    edp_type: str
    damage_states: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.damage_states:
            raise ValueError("at least one damage state required")
        thetas = [ds.fragility_theta for ds in self.damage_states]
        if any(thetas[i + 1] <= thetas[i]
               for i in range(len(thetas) - 1)):
            raise ValueError(
                "damage states must have strictly increasing "
                f"fragility_theta; got {thetas}"
            )

    def damage_state_probs(self, edp: float) -> np.ndarray:
        """Return ``P(DS = ds_i | EDP)`` for ``i = 0, 1, ..., N``.

        ``DS0`` is the undamaged state; index ``i = 0`` of the output.
        Output sums to 1 (within numerical tolerance).
        """
        # P(DS >= ds_i) for i = 1..N
        cum = np.array([ds.prob_exceeds(edp)
                          for ds in self.damage_states])
        # P(DS = DS0) = 1 - P(DS >= DS1)
        # P(DS = ds_i) = P(DS >= ds_i) - P(DS >= ds_{i+1}) for i < N
        # P(DS = ds_N) = P(DS >= ds_N)
        N = len(self.damage_states)
        probs = np.zeros(N + 1)
        probs[0] = 1.0 - cum[0]
        for i in range(N - 1):
            probs[i + 1] = cum[i] - cum[i + 1]
        probs[N] = cum[N - 1]
        # Clip to avoid numerical negatives
        return np.maximum(probs, 0.0)


# ============================================================ Component group

@dataclass
class ComponentGroup:
    """A quantity of identical components seeing the same EDP.

    Attributes
    ----------
    component : ComponentFragility
    quantity : float
        Number of units (e.g., 100 m of drywall, 8 ceiling-tile units).
    edp_value : float
        The driving EDP for this group (e.g., PSD = 0.012 = 1.2 percent).
    location : str
        Free-text label, e.g. ``"Floor 3, partition"``.
    """

    component: ComponentFragility
    quantity: float
    edp_value: float
    location: str = ""

    def expected_loss(self) -> float:
        """Closed-form expected repair cost for this group."""
        probs = self.component.damage_state_probs(self.edp_value)
        loss = 0.0
        for i, ds in enumerate(self.component.damage_states):
            # probs[0] is DS0 (no damage, no cost)
            loss += probs[i + 1] * ds.expected_cost()
        return float(self.quantity * loss)

    def sample_loss(self, rng: np.random.Generator) -> float:
        """One Monte-Carlo loss realisation for this group.

        Damage state is sampled from ``damage_state_probs`` and the
        repair cost is sampled from the lognormal for that damage
        state. The total quantity ``Q`` is treated as a single
        aggregate (which assumes all Q units are correlated --
        appropriate when they are co-located).
        """
        probs = self.component.damage_state_probs(self.edp_value)
        idx = int(rng.choice(len(probs), p=probs / probs.sum()))
        if idx == 0:
            return 0.0
        ds = self.component.damage_states[idx - 1]
        return float(self.quantity * ds.sample_cost(rng))


# ============================================================ Assessment

@dataclass
class P58AssessmentResult:
    """Monte-Carlo loss-distribution result.

    Attributes
    ----------
    realisation_losses : np.ndarray
        Per-realisation total building loss.
    expected_loss : float
        Closed-form expected loss (analytical, no MC noise).
    mean_loss : float
        Sample mean from MC.
    median_loss : float
    p84_loss : float
        84-th percentile (mean + 1 sigma in log-space, roughly).
    p95_loss : float
    n_realisations : int
    """

    realisation_losses: np.ndarray
    expected_loss: float
    mean_loss: float
    median_loss: float
    p84_loss: float
    p95_loss: float
    n_realisations: int


class ComponentDamageAssessment:
    """Building-level damage/loss assessment over many component groups.

    Parameters
    ----------
    groups : list of ComponentGroup
    """

    def __init__(self, groups: list[ComponentGroup]):
        if not groups:
            raise ValueError("need at least one component group")
        self.groups = list(groups)

    def expected_loss(self) -> float:
        """Closed-form expected total loss = sum of per-group expected losses."""
        return float(sum(g.expected_loss() for g in self.groups))

    def monte_carlo(
        self,
        n_realisations: int = 1000,
        *,
        seed: int | None = None,
    ) -> P58AssessmentResult:
        """Run Monte-Carlo loss sampling.

        Parameters
        ----------
        n_realisations : int, default 1000
        seed : int, optional
            Random-number-generator seed.

        Returns
        -------
        P58AssessmentResult
        """
        if n_realisations < 1:
            raise ValueError(
                f"n_realisations must be >= 1, got {n_realisations}"
            )
        rng = np.random.default_rng(seed)
        losses = np.zeros(n_realisations)
        for r in range(n_realisations):
            total = 0.0
            for g in self.groups:
                total += g.sample_loss(rng)
            losses[r] = total
        return P58AssessmentResult(
            realisation_losses=losses,
            expected_loss=self.expected_loss(),
            mean_loss=float(np.mean(losses)),
            median_loss=float(np.median(losses)),
            p84_loss=float(np.percentile(losses, 84)),
            p95_loss=float(np.percentile(losses, 95)),
            n_realisations=n_realisations,
        )
