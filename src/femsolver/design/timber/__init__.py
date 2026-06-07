"""Timber design per NDS-2024 (Phase D.1.3+).

NDS (National Design Specification for Wood Construction) is the
US/Canadian timber design standard. This module implements:

* :mod:`factors` -- the C-factor framework (load duration, wet service,
  temperature, size, lateral stability, column stability, etc.)
* :mod:`nds` -- member-level design checks (bending, tension,
  compression, shear, combined H-interaction)

Design philosophy
-----------------
NDS uses the allowable-stress-design (ASD) framework. Each reference
strength ``F`` (e.g. ``F_b``, ``F_c``) is multiplied by a set of
applicable C-factors to give the adjusted allowable ``F'``:

    F' = F · C_D · C_M · C_t · C_F · C_L · C_P · C_r · C_fu · C_b · C_i

Different factors apply to different design checks (e.g. C_L applies
to bending only, C_P to compression only). The design check is then
simply ``f ≤ F'`` for the relevant stress.
"""
from femsolver.design.timber.factors import (
    NDSFactors,
    C_D_load_duration,
    C_F_size_factor,
    C_L_lateral_stability,
    C_M_wet_service,
    C_P_column_stability,
    C_r_repetitive_member,
)
from femsolver.design.timber.nds import (
    NDSBendingCheck,
    NDSCombinedCheck,
    NDSCompressionCheck,
    NDSShearCheck,
    NDSTensionCheck,
    nds_bending_check,
    nds_combined_check,
    nds_compression_check,
    nds_shear_check,
    nds_tension_check,
)
from femsolver.design.timber.ec5 import (
    EC5BendingCheck,
    EC5CombinedCheck,
    EC5CompressionCheck,
    EC5Factors,
    EC5ShearCheck,
    EC5TensionCheck,
    ec5_bending_check,
    ec5_combined_check,
    ec5_compression_check,
    ec5_shear_check,
    ec5_tension_check,
    gamma_M_partial_factor,
    k_c_column_stability,
    k_crit_lateral_stability,
    k_h_glulam,
    k_h_solid,
    k_mod_factor,
)
from femsolver.design.timber.clt_design import (
    CLTDeflectionCheck,
    CLTRollingShearCheck,
    CLTTwoWayBendingCheck,
    CLTVibrationCheck,
    clt_deflection_check,
    clt_rolling_shear_check,
    clt_two_way_bending_check,
    clt_vibration_check,
    k_def_factor,
    k_sys_clt,
)

__all__ = [
    "NDSFactors",
    "C_D_load_duration",
    "C_F_size_factor",
    "C_L_lateral_stability",
    "C_M_wet_service",
    "C_P_column_stability",
    "C_r_repetitive_member",
    "NDSBendingCheck",
    "NDSCompressionCheck",
    "NDSTensionCheck",
    "NDSShearCheck",
    "NDSCombinedCheck",
    "nds_bending_check",
    "nds_compression_check",
    "nds_tension_check",
    "nds_shear_check",
    "nds_combined_check",
    "EC5Factors",
    "k_mod_factor",
    "gamma_M_partial_factor",
    "k_h_solid",
    "k_h_glulam",
    "k_crit_lateral_stability",
    "k_c_column_stability",
    "EC5BendingCheck",
    "EC5TensionCheck",
    "EC5CompressionCheck",
    "EC5ShearCheck",
    "EC5CombinedCheck",
    "ec5_bending_check",
    "ec5_tension_check",
    "ec5_compression_check",
    "ec5_shear_check",
    "ec5_combined_check",
    # CLT-specific (D.1.5)
    "k_sys_clt",
    "k_def_factor",
    "CLTRollingShearCheck",
    "CLTTwoWayBendingCheck",
    "CLTDeflectionCheck",
    "CLTVibrationCheck",
    "clt_rolling_shear_check",
    "clt_two_way_bending_check",
    "clt_deflection_check",
    "clt_vibration_check",
]
