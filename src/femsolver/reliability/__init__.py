"""Reliability / probabilistic structural analysis.

Submodules
----------
* :mod:`rv`           -- Random-variable distributions (Normal,
                          Lognormal, Uniform, Gumbel, Weibull) +
                          :class:`RandomVariableVector` with the
                          Nataf/Rosenblatt transformation.
* :mod:`form`         -- First-Order Reliability Method (HLRF).
* :mod:`sorm`         -- Second-Order Reliability Method (Breitung).
* :mod:`monte_carlo`  -- Crude MC, Latin-hypercube, importance
                          sampling.

The library composes cleanly with :mod:`femsolver.performance.p58`:
the FORM design point can drive importance-sampling around the
collapse-fragility curve; FORM/SORM provide design-point estimates
for component-level fragilities.
"""
from femsolver.reliability.form import (
    FORMResult,
    form_hlrf,
)
from femsolver.reliability.monte_carlo import (
    MonteCarloResult,
    crude_monte_carlo,
    importance_sampling_around_u_star,
    latin_hypercube_monte_carlo,
)
from femsolver.reliability.rv import (
    Gumbel,
    Lognormal,
    Normal,
    RandomVariable,
    RandomVariableVector,
    Uniform,
    Weibull,
)
from femsolver.reliability.sorm import (
    SORMResult,
    sorm_breitung,
)


__all__ = [
    # rv
    "RandomVariable",
    "RandomVariableVector",
    "Normal", "Lognormal", "Uniform", "Gumbel", "Weibull",
    # FORM
    "FORMResult", "form_hlrf",
    # SORM
    "SORMResult", "sorm_breitung",
    # MC
    "MonteCarloResult",
    "crude_monte_carlo",
    "latin_hypercube_monte_carlo",
    "importance_sampling_around_u_star",
]
