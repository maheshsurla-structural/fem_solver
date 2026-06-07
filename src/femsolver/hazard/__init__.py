"""Environmental hazard / demand models.

Groups the code that quantifies the *demand* the environment places on a
structure -- earthquake ground motion and wind -- as opposed to the
structural response (analysis/) or the code design checks (design/).

Sub-packages
------------
* :mod:`seismic` -- seismic *hazard*: GMPEs, PSHA hazard curves / UHS,
  deaggregation, 1-D site response, risk-targeted MCE_R. (Distinct from
  ``femsolver.design.seismic``, which is code seismic *design*.)
* :mod:`wind`    -- wind loading: ASCE 7, IS 875, EC1 pressures, gust
  effect, components-and-cladding, vortex shedding.
"""
from femsolver.hazard import seismic, wind

__all__ = ["seismic", "wind"]
