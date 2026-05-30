"""Seismic capacity-design detailing.

Phase 32 modules:

* :mod:`scwb` -- Strong-Column-Weak-Beam check (ACI 18.7.3 /
  AISC 341 E3.4a)
* :mod:`capacity_shear` -- Capacity-design beam shear from probable
  moments (ACI 18.6.5)
* :mod:`confinement` -- Confined-concrete reinforcement detailing
  (ACI 18.7.5)
"""
from femsolver.design.seismic.scwb import (
    SCWBCheck,
    SCWB_RATIO_ACI_SMF,
    SCWB_RATIO_AISC_SMF,
    scwb_check,
)
from femsolver.design.seismic.capacity_shear import (
    CapacityShearCheck,
    capacity_design_shear,
    probable_moment,
)
from femsolver.design.seismic.confinement import (
    ConfinementDetail,
    confined_concrete_detailing,
)

__all__ = [
    "SCWBCheck",
    "SCWB_RATIO_ACI_SMF",
    "SCWB_RATIO_AISC_SMF",
    "scwb_check",
    "CapacityShearCheck",
    "capacity_design_shear",
    "probable_moment",
    "ConfinementDetail",
    "confined_concrete_detailing",
]
