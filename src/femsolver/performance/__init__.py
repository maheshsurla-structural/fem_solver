"""Performance-based earthquake engineering (PBE).

Seismic performance assessment that builds on top of the analysis drivers:
nonlinear-static (pushover) target displacement, story-drift checks,
incremental dynamic analysis (IDA), collapse fragility, ground-motion
record selection/scaling, conditional mean spectra, and FEMA P-58 loss
assessment. Carved out of the former ``analysis`` catch-all so the
assessment pipeline reads as one domain.

Submodules
----------
* :mod:`capacity_design` -- pushover bilinearization, N2 / ASCE-41
  coefficient-method target displacement, equivalent SDOF, story drifts.
* :mod:`drift_check`     -- ASCE 7 story-drift acceptance checks.
* :mod:`ida`             -- incremental dynamic analysis driver.
* :mod:`ida_collapse`    -- collapse detection + multi-record IDA.
* :mod:`fragility`       -- lognormal fragility fitting (MLE / MoM).
* :mod:`record_scaling`  -- amplitude scaling + response-spectrum tools.
* :mod:`cms`             -- conditional mean spectrum (Baker-Jayaram).
* :mod:`p58`             -- FEMA P-58 component damage / loss assessment.
"""
from femsolver.performance.capacity_design import (
    BilinearCurve,
    EquivalentSDOF,
    PushoverToTarget,
    bilinearize_capacity_curve,
    coefficient_method_target,
    equivalent_sdof,
    n2_target_displacement,
    seismic_combination,
    story_drifts,
)
from femsolver.performance.drift_check import (
    DriftCheck,
    drift_check,
    drift_check_worst_combo,
)
from femsolver.performance.ida import (
    IDADriver,
    IDAPoint,
    IDARecord,
    max_drift_edp,
    pga_scale_factor,
)
from femsolver.performance.ida_collapse import (
    CollapseResult,
    IDASummary,
    detect_collapse,
    multi_record_ida,
)
from femsolver.performance.fragility import (
    FragilityFit,
    fit_collapse_fragility,
    fit_lognormal_mle,
    fit_lognormal_method_of_moments,
)
from femsolver.performance.record_scaling import (
    SuiteScalingResult,
    amplitude_scale_factor,
    compute_sdof_response_spectrum,
    period_range_mask,
    record_response_spectrum,
    scale_record_suite,
)
from femsolver.performance.cms import (
    baker_jayaram_correlation,
    compute_epsilon,
    conditional_mean_spectrum,
    conditional_spectrum_variance,
)
from femsolver.performance.p58 import (
    ComponentDamageAssessment,
    ComponentFragility,
    ComponentGroup,
    DamageState,
    P58AssessmentResult,
)

__all__ = [
    # capacity_design
    "BilinearCurve", "EquivalentSDOF", "PushoverToTarget",
    "bilinearize_capacity_curve", "coefficient_method_target",
    "equivalent_sdof", "n2_target_displacement", "seismic_combination",
    "story_drifts",
    # drift_check
    "DriftCheck", "drift_check", "drift_check_worst_combo",
    # ida
    "IDADriver", "IDAPoint", "IDARecord", "max_drift_edp", "pga_scale_factor",
    # ida_collapse
    "CollapseResult", "IDASummary", "detect_collapse", "multi_record_ida",
    # fragility
    "FragilityFit", "fit_collapse_fragility", "fit_lognormal_mle",
    "fit_lognormal_method_of_moments",
    # record_scaling
    "SuiteScalingResult", "amplitude_scale_factor",
    "compute_sdof_response_spectrum", "period_range_mask",
    "record_response_spectrum", "scale_record_suite",
    # cms
    "baker_jayaram_correlation", "compute_epsilon",
    "conditional_mean_spectrum", "conditional_spectrum_variance",
    # p58
    "ComponentDamageAssessment", "ComponentFragility", "ComponentGroup",
    "DamageState", "P58AssessmentResult",
]
