Sections
========

Cross-section response models for beam-column and shell elements.

Beam / column sections (1-D)
----------------------------

The user-facing :class:`femsolver.sections.Section` and the solver-facing
response layer in :mod:`femsolver.sections.response`.

.. automodule:: femsolver.sections.response.elastic
   :members:

.. automodule:: femsolver.sections.response.fiber
   :members:

.. automodule:: femsolver.sections.response.hinges
   :members:

Wall sections
-------------

.. automodule:: femsolver.sections.response.wall
   :members:

.. automodule:: femsolver.sections.response.wall_shear
   :members:

Shell / plate sections (2-D)
----------------------------

The 2-D / surface-element section family lives in
:mod:`femsolver.shell_sections`.

.. automodule:: femsolver.shell_sections.base
   :members:

.. automodule:: femsolver.shell_sections.layered
   :members:

.. automodule:: femsolver.shell_sections.ply_failure
   :members:

.. automodule:: femsolver.shell_sections.clt
   :members:
