"""Multiaxial (2-D/3-D continuum) constitutive models.

Full stress-tensor material models for solid and shell elements:
metal plasticity, soil/rock plasticity, hyperelasticity, and orthotropic
elasticity. (1-D fiber materials live in :mod:`femsolver.materials.uniaxial`.)

Submodules
----------
* :mod:`j2_plasticity` -- von Mises (J2) plasticity (small strain).
* :mod:`j2_finite_strain`     -- finite-strain J2 plasticity.
* :mod:`drucker_prager`-- Drucker-Prager (soils, pressure-dependent).
* :mod:`mohr_coulomb`  -- Mohr-Coulomb (full 4-region return mapping).
* :mod:`cam_clay`      -- Modified Cam-Clay (critical-state soil).
* :mod:`hyperelastic`  -- Neo-Hookean / Mooney-Rivlin large-strain.
* :mod:`orthotropic`   -- orthotropic lamina (composites).
"""
