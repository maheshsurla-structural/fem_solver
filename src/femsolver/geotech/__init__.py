"""Geotechnical / soil-structure-interaction models.

Foundation and soil response that sits alongside the structural solver:
elastic foundations, pile groups, nonlinear soil springs (p-y / t-z / q-z),
static and dynamic footing impedance, and liquefaction triggering. Carved
out of the former ``analysis`` catch-all so the soil/foundation domain has
an obvious home.

Submodules
----------
* :mod:`winkler`         -- beam on elastic (Winkler) foundation.
* :mod:`pile_group`      -- pile-group efficiency, p-multipliers, settlement.
* :mod:`soil_springs`    -- API p-y / t-z / q-z nonlinear soil springs.
* :mod:`ssi`             -- Gazetas static surface-footing impedance.
* :mod:`dynamic_gazetas` -- frequency-dependent (dynamic) footing impedance.
* :mod:`liquefaction`    -- simplified liquefaction-triggering procedure.
"""
from femsolver.geotech.winkler import (
    BeamOnWinklerFoundation2D,
    HetenyiInfiniteBeamResult,
    hetenyi_characteristic_length,
    hetenyi_infinite_beam_point_load,
    subgrade_modulus_table,
)
from femsolver.geotech.pile_group import (
    GroupSettlementResult,
    group_efficiency_converse_labarre,
    group_p_multipliers,
    group_settlement_elastic,
    p_multiplier,
)
from femsolver.geotech.soil_springs import (
    SoilSpringBackbone,
    py_curve_sand,
    py_curve_soft_clay,
    qz_curve,
    tz_curve_clay,
    tz_curve_sand,
)
from femsolver.geotech.ssi import (
    FootingImpedance,
    HalfspaceSoil,
    embedment_correction,
    gazetas_surface_footing,
)
from femsolver.geotech.dynamic_gazetas import (
    DynamicFootingImpedance,
    DynamicImpedanceCoefficients,
    dimensionless_frequency,
    dynamic_footing_impedance,
    gazetas_dynamic_coefficients,
)
from femsolver.geotech.liquefaction import (
    LiquefactionTriggeringResult,
    CRR_from_N1_60cs,
    cyclic_stress_ratio,
    evaluate_liquefaction,
    fines_content_correction,
    K_sigma,
    magnitude_scaling_factor,
    stress_reduction_coefficient,
)

__all__ = [
    # winkler
    "BeamOnWinklerFoundation2D", "HetenyiInfiniteBeamResult",
    "hetenyi_characteristic_length", "hetenyi_infinite_beam_point_load",
    "subgrade_modulus_table",
    # pile_group
    "GroupSettlementResult", "group_efficiency_converse_labarre",
    "group_p_multipliers", "group_settlement_elastic", "p_multiplier",
    # soil_springs
    "SoilSpringBackbone", "py_curve_sand", "py_curve_soft_clay",
    "qz_curve", "tz_curve_clay", "tz_curve_sand",
    # ssi
    "FootingImpedance", "HalfspaceSoil", "embedment_correction",
    "gazetas_surface_footing",
    # dynamic_gazetas
    "DynamicFootingImpedance", "DynamicImpedanceCoefficients",
    "dimensionless_frequency", "dynamic_footing_impedance",
    "gazetas_dynamic_coefficients",
    # liquefaction
    "LiquefactionTriggeringResult", "CRR_from_N1_60cs", "cyclic_stress_ratio",
    "evaluate_liquefaction", "fines_content_correction", "K_sigma",
    "magnitude_scaling_factor", "stress_reduction_coefficient",
]
