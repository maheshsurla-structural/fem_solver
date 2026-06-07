"""Thermal & fire analysis.

Heat-transfer analysis and fire engineering, carved out of the former
``analysis`` catch-all so the temperature domain has an obvious home.

Submodules
----------
* :mod:`heat_conduction` -- steady and transient finite-element heat
  conduction (the thermal field analysis).
* :mod:`fire`            -- standard/parametric fire temperature curves
  (ISO 834, ASTM E119, hydrocarbon, EC1) and EC2/EC3 material
  strength/modulus reduction at temperature.

Note: thermal *strain* (imposed-temperature load on a structural model)
lives with the other imposed-strain loads; see
``femsolver.apply_thermal_load``.
"""
from femsolver.thermal.heat_conduction import (
    SteadyHeatAnalysis,
    SteadyHeatResult,
    TransientHeatAnalysis,
    TransientHeatResult,
)
from femsolver.thermal.fire import (
    astm_e119_temperature,
    concrete_strength_reduction_ec2,
    ec1_parametric_temperature,
    hydrocarbon_temperature,
    iso_834_temperature,
    steel_critical_temperature,
    steel_modulus_reduction_ec3,
    steel_strength_reduction_ec3,
)

__all__ = [
    "SteadyHeatAnalysis", "SteadyHeatResult",
    "TransientHeatAnalysis", "TransientHeatResult",
    "astm_e119_temperature", "concrete_strength_reduction_ec2",
    "ec1_parametric_temperature", "hydrocarbon_temperature",
    "iso_834_temperature", "steel_critical_temperature",
    "steel_modulus_reduction_ec3", "steel_strength_reduction_ec3",
]
