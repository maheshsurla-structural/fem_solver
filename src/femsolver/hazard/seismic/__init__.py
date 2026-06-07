"""Site-specific seismic hazard, ground motion prediction, and 1-D
site response (Phase 54 / Theme U).

Submodules
----------

* :mod:`gmpe`       -- Ground Motion Prediction Equations (median Sa
  + sigma_lnSa from magnitude / distance / Vs30).
* :mod:`psha`       -- Probabilistic Seismic Hazard Analysis: hazard
  curve, Uniform Hazard Spectrum.
* :mod:`deaggregation` -- Identify which (M, R, eps) scenarios
  dominate a given hazard level.
* :mod:`site_response` -- Linear 1-D wave propagation through a
  layered soil profile (frequency-domain transfer function).
* :mod:`risk_targeted` -- ASCE 7-22 Ch 21 risk-targeted maximum
  considered earthquake (MCE_R).
"""
from femsolver.hazard.seismic.gmpe import (
    GmpeResult,
    BooreAtkinsonLike,
    bssa14,
)
from femsolver.hazard.seismic.bssa14 import (
    Bssa14Coefficients,
    bssa14_at_period,
    bssa14_available_periods,
)
from femsolver.hazard.seismic.psha import (
    PointSource,
    AreaSource,
    GutenbergRichterMFD,
    HazardCurve,
    UniformHazardSpectrum,
    compute_hazard_curve,
    return_period_to_im,
    compute_uhs,
)
from femsolver.hazard.seismic.deaggregation import (
    DeaggregationResult,
    deaggregate,
)
from femsolver.hazard.seismic.site_response import (
    SoilLayer,
    SoilProfile,
    site_amplification_spectrum,
    transfer_function,
)
from femsolver.hazard.seismic.equivalent_linear import (
    NonlinearSoilCurves,
    EquivalentLinearResult,
    equivalent_linear_iterate,
    vucetic_dobry_curves,
)
from femsolver.hazard.seismic.risk_targeted import (
    risk_targeted_im,
    annual_collapse_rate,
)

__all__ = [
    "GmpeResult", "BooreAtkinsonLike", "bssa14",
    "Bssa14Coefficients", "bssa14_at_period", "bssa14_available_periods",
    "PointSource", "AreaSource",
    "GutenbergRichterMFD",
    "HazardCurve", "UniformHazardSpectrum",
    "compute_hazard_curve", "return_period_to_im", "compute_uhs",
    "DeaggregationResult", "deaggregate",
    "SoilLayer", "SoilProfile",
    "site_amplification_spectrum", "transfer_function",
    "NonlinearSoilCurves", "EquivalentLinearResult",
    "equivalent_linear_iterate", "vucetic_dobry_curves",
    "risk_targeted_im", "annual_collapse_rate",
]
