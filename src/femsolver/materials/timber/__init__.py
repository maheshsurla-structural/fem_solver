"""Timber material constitutive models + reference databases
(Phase D.1.1+).

Timber is **orthotropic** with the longitudinal (grain) direction
having dramatically higher modulus and strength than the two
perpendicular directions. The first cut of femsolver timber support
provides:

* :class:`TimberMaterial` -- canonical orthotropic timber material
  carrying both elastic constants and characteristic strengths,
  matching how NDS, EC5, and IS 883 describe wood for design.
* Reference databases:
    - :mod:`nds` -- NDS-2024 reference design values (Douglas Fir-Larch,
      Southern Pine, Spruce-Pine-Fir; glulam combinations)
    - :mod:`ec5` -- EN 338 strength classes (C-classes for solid timber,
      GL-classes for glulam, T-classes for tension-graded)
    - :mod:`is883` -- IS 883:2016 grouping (Class I / II / III timber)

The orthotropic / characteristic-value description is the bridge
between fiber-section nonlinear analysis (which needs E, G, sigma_y
or a full constitutive curve) and code-based design checks (which
need f_b, f_t, f_c, f_v at characteristic or design level).
"""
from femsolver.materials.timber.material import TimberMaterial
from femsolver.materials.timber.nds import (
    NDS_SAWN_LUMBER,
    NDS_GLULAM,
    get_nds_timber,
    list_nds_species,
)
from femsolver.materials.timber.ec5 import (
    EC5_C_CLASS,
    EC5_GL_CLASS,
    EC5_T_CLASS,
    get_ec5_class,
    list_ec5_classes,
)
from femsolver.materials.timber.is883 import (
    IS883_CLASSES,
    get_is883_class,
)

__all__ = [
    "TimberMaterial",
    "NDS_SAWN_LUMBER",
    "NDS_GLULAM",
    "get_nds_timber",
    "list_nds_species",
    "EC5_C_CLASS",
    "EC5_GL_CLASS",
    "EC5_T_CLASS",
    "get_ec5_class",
    "list_ec5_classes",
    "IS883_CLASSES",
    "get_is883_class",
]
