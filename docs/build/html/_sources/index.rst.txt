femsolver — a Python finite-element solver for structural analysis
====================================================================

.. image:: https://img.shields.io/badge/tests-1198_passing-brightgreen
   :alt: Tests passing

.. image:: https://img.shields.io/badge/python-3.11+-blue
   :alt: Python 3.11+

**femsolver** is a Python finite-element library targeted at structural
engineering. It covers the full pipeline from linear-elastic analysis
through full performance-based earthquake engineering (PBE), with
member-level design checks to ACI 318, AISC 360, IS 456, IS 800, and
seismic-detailing rules per ACI 18, AISC 341, IS 1893, and IS 13920.

Architecture
------------

The library is organised in layers:

.. list-table::
   :widths: 22 78
   :header-rows: 1

   * - Layer
     - Contents
   * - **Model**
     - :class:`~femsolver.Model`, :class:`~femsolver.Node`, element + material registries
   * - **Materials**
     - Elastic, J2/Drucker-Prager plasticity, uniaxial library (Bilinear, IMK, BRB, Takeda, Pivot, Mander, Kent-Park, …)
   * - **Sections**
     - Elastic 2D/3D, Fiber 2D/3D, Layered Shell, hinges, Wall fiber sections (T/L/U/I)
   * - **Elements**
     - Truss/Beam (corotational + force-based + fiber + hinged), Quad4, MITC4/9, DKMQ4, Tri3, DKT3, Hex8, Tet4
   * - **Analysis**
     - Linear/nonlinear static, transient (Newmark/HHT/Gen-α/CD), eigen, buckling, response spectrum, MPA, IDA
   * - **Constraints & solvers**
     - Rigid diaphragm, equal-DOF, rigid link, MPC; direct sparse + iterative
   * - **PBE pipeline**
     - IDA → collapse → fragility → CMS-based record selection → P-58 component damage
   * - **Design**
     - ACI 318, AISC 360, IS 456, IS 800, ASCE 7/41, IS 1893/13920, AISC 341 detailing
   * - **Bridges**
     - Influence lines + HL-93/IRC, PT tendons + losses, CEB-FIP creep/shrinkage, composite sections
   * - **Connections**
     - Krawinkler panel zone, AISC 358 RBS, Richard-Abbott PR, bolts/welds (AISC + IS 800)
   * - **V&V**
     - 11+ benchmarks with closed-form references (NAFEMS, Scordelis-Lo, Euler columns, EPP shape factor)

Quick links
-----------

.. toctree::
   :maxdepth: 1
   :caption: User guide

   guide/getting_started
   guide/modeling_primer
   guide/analysis_types
   guide/materials_catalog
   guide/design_workflows
   guide/pbe_pipeline
   guide/bridges
   guide/connections
   guide/vv_benchmarks

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/core
   api/materials
   api/sections
   api/elements
   api/analysis
   api/design
   api/bridges
   api/benchmarks
   api/io

.. toctree::
   :maxdepth: 1
   :caption: Examples

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Theory & verification

   theory/overview
   theory/vv_report

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
