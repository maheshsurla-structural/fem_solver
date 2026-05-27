"""Design code modules.

Phase 29-33: code-based member design (ACI 318, AISC 360, ASCE 7).

Submodules
----------
* :mod:`concrete` -- ACI 318-19 reinforced concrete (Phase 29, 32)
* future: ``steel`` -- AISC 360-22 (Phase 30, 32)
* future: ``combos`` -- ASCE 7-22 load combinations + drift (Phase 31)
* future: ``reports`` -- HTML / CSV design reports (Phase 33)
"""
from femsolver.design import concrete

__all__ = ["concrete"]
