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
from femsolver.seismic.gmpe import (
    GmpeResult,
    BooreAtkinsonLike,
)
from femsolver.seismic.psha import (
    PointSource,
    AreaSource,
    GutenbergRichterMFD,
    HazardCurve,
    UniformHazardSpectrum,
    compute_hazard_curve,
    return_period_to_im,
    compute_uhs,
)
from femsolver.seismic.deaggregation import (
    DeaggregationResult,
    deaggregate,
)
from femsolver.seismic.site_response import (
    SoilLayer,
    SoilProfile,
    site_amplification_spectrum,
    transfer_function,
)
from femsolver.seismic.risk_targeted import (
    risk_targeted_im,
    annual_collapse_rate,
)

__all__ = [
    "GmpeResult", "BooreAtkinsonLike",
    "PointSource", "AreaSource",
    "GutenbergRichterMFD",
    "HazardCurve", "UniformHazardSpectrum",
    "compute_hazard_curve", "return_period_to_im", "compute_uhs",
    "DeaggregationResult", "deaggregate",
    "SoilLayer", "SoilProfile",
    "site_amplification_spectrum", "transfer_function",
    "risk_targeted_im", "annual_collapse_rate",
]
