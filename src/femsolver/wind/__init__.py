"""Wind engineering (Phase 52 / Theme S).

Subpackages and modules:

* :mod:`asce7`         -- ASCE 7-22 velocity pressure, pressure
  coefficients, gust factor, MWFRS design pressures.
* :mod:`is875`         -- IS 875 Part 3 (2015) basic wind pressure
  and design-pressure helpers.
* :mod:`eurocode`      -- EN 1991-1-4 (EC1) basic-wind-velocity to
  peak-velocity-pressure path.
* :mod:`vortex`        -- Vortex shedding (Strouhal, Scruton, lock-in).
"""
from femsolver.wind.asce7 import (
    Asce7VelocityPressure,
    asce7_velocity_pressure,
    asce7_exposure_constants,
    asce7_Kz,
    Asce7WallPressureCoefficients,
    asce7_wall_Cp,
    asce7_roof_Cp_flat,
    Asce7MWFRSResult,
    asce7_mwfrs_design_pressures,
    asce7_gust_factor_rigid,
    asce7_gust_factor_flexible,
)
from femsolver.wind.is875 import (
    Is875DesignWindPressure,
    is875_design_wind_pressure,
    is875_terrain_category_factor,
)
from femsolver.wind.eurocode import (
    Ec1PeakVelocityPressure,
    ec1_peak_velocity_pressure,
    ec1_roughness_factor,
)
from femsolver.wind.vortex import (
    StrouhalResult,
    vortex_shedding_frequency,
    scruton_number,
    is_lock_in_risk,
)

__all__ = [
    # ASCE 7
    "Asce7VelocityPressure",
    "asce7_velocity_pressure",
    "asce7_exposure_constants",
    "asce7_Kz",
    "Asce7WallPressureCoefficients",
    "asce7_wall_Cp",
    "asce7_roof_Cp_flat",
    "Asce7MWFRSResult",
    "asce7_mwfrs_design_pressures",
    "asce7_gust_factor_rigid",
    "asce7_gust_factor_flexible",
    # IS 875
    "Is875DesignWindPressure",
    "is875_design_wind_pressure",
    "is875_terrain_category_factor",
    # EC1
    "Ec1PeakVelocityPressure",
    "ec1_peak_velocity_pressure",
    "ec1_roughness_factor",
    # Vortex
    "StrouhalResult",
    "vortex_shedding_frequency",
    "scruton_number",
    "is_lock_in_risk",
]
