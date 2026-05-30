Tutorial examples
=================

The ``examples/`` directory of the source tree ships 50+ self-contained
scripts, organised below by topic. Each script is runnable from the
repo root:

.. code-block:: bash

   python examples/01_simple_truss.py

The list is curated; the script numbers reflect the historical phase
ordering (Phase 1 → Phase 38), not difficulty.

Beginner — linear, static
-------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - What it demonstrates
   * - ``01_simple_truss.py``
     - 3-bar pin/roller truss; first model from scratch
   * - ``02_cantilever_beam.py``
     - Simple cantilever under tip load (Bernoulli)
   * - ``03_portal_frame.py``
     - 2D portal frame, lateral + gravity
   * - ``04_plane_stress_plate.py``
     - Plate under tension via Quad4
   * - ``05_rigid_diaphragm_frame.py``
     - 3D frame with rigid floor diaphragm

Modal, dynamics, response spectrum
----------------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``06_cantilever_modal.py``
     - Modal periods + shapes of a cantilever beam
   * - ``14_sdof_damped_response.py``
     - SDOF under impulse, damping ratio sweep
   * - ``15_cantilever_free_vibration.py``
     - Free-vibration transient, Newmark integration
   * - ``22_response_spectrum_vs_time_history.py``
     - SRSS / CQC response spectrum vs direct integration
   * - ``29_advanced_dynamics_integrators.py``
     - HHT-α vs Generalised-α vs central difference

Nonlinear: pushover, plasticity
-------------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``07_mises_truss_snapthrough.py``
     - Classic snap-through with arc-length
   * - ``08_hinged_cantilever_pushover.py``
     - Lumped plastic hinge
   * - ``09_fiber_cantilever_pushover.py``
     - Fiber-section cantilever
   * - ``10_corotational_column_p_delta.py``
     - Corotational beam-column with P-Delta
   * - ``11_corotational_fiber_column.py``
     - Corotational + fiber section
   * - ``12_displacement_control_epp_plateau.py``
     - Displacement-control past the plateau
   * - ``13_arc_length_mises_snap_through.py``
     - Crisfield arc-length
   * - ``16_nonlinear_cyclic_hysteresis.py``
     - Bilinear hinge under cyclic load
   * - ``17_buckling_mesh_convergence.py``
     - Euler P_cr mesh convergence
   * - ``18_force_based_vs_displacement_based.py``
     - Force-based vs DB fiber column

3D + advanced sections
----------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``19_3d_fiber_column_biaxial.py``
     - Biaxial fiber column
   * - ``20_3d_corotational_p_delta.py``
     - 3D corotational beam-column
   * - ``21_3d_corot_fiber_pushover.py``
     - Combined 3D corotational + fiber
   * - ``27_3d_solid_cantilever.py``
     - Hex8 cantilever
   * - ``29_hex8_j2_pushover.py``
     - Hex8 with J₂ plasticity

Shells + composites
-------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``23_shell_ss_plate_navier.py``
     - MITC4 simply-supported plate vs Navier series
   * - ``24_shell_buckling.py``
     - Shell linear-buckling analysis
   * - ``25_shell_tri3_vs_mitc4.py``
     - Tri3 vs MITC4 convergence comparison
   * - ``26_layered_shell_sandwich.py``
     - Layered shell sandwich section
   * - ``34_composite_laminate.py``
     - Composite laminate analysis
   * - ``35_composite_failure.py``
     - Tsai-Wu / Tsai-Hill / max-stress
   * - ``37_shell_mitc9_convergence.py``
     - MITC9 convergence study
   * - ``38_dkmq4_thin_plate.py``
     - DKMQ4 on a thin plate

Materials catalogue
-------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``28_concrete_confined_vs_unconfined.py``
     - Kent-Park vs Mander concrete
   * - ``39_csi_hysteresis_catalog.py``
     - σ-ε loops of all CSI hysteresis types
   * - ``44_fiber_section_csi_hysteresis.py``
     - M-κ loops of the same materials in a fiber section

Capacity design + post-processing
---------------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``30_post_processing_workflow.py``
     - VTK + JSON + force diagrams
   * - ``31_capacity_design_workflow.py``
     - Pushover-to-target, bilinearisation, N2
   * - ``32_base_isolation.py``
     - Lead-rubber bearing under ground motion
   * - ``33_solver_benchmark.py``
     - Solver timing on a benchmark model
   * - ``36_modal_pushover.py``
     - Modal pushover analysis vs first-mode pushover

PBE pipeline
------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``40_ida_collapse_fragility.py``
     - IDA + collapse detection + lognormal fragility fit
   * - ``41_record_selection_to_ida.py``
     - CMS + ASCE 7 scaling + IDA
   * - ``42_pile_py_pushover.py``
     - Pile-soil interaction with API p-y springs
   * - ``43_pbe_full_workflow.py``
     - **Capstone** PBE: IDA → EDPs → P-58 loss curve

Shear walls
-----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``45_coupled_wall_pushover.py``
     - Two walls + coupling beams; linked vs unlinked

V&V
---

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``46_vnv_report.py``
     - Full benchmark suite, CSV export, pass/fail report

Design — code-based
-------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``47_indian_codes_frame_design.py``
     - 4-storey RC frame to IS 456 / IS 1893 / IS 13920
   * - ``48_connection_capstone.py``
     - Panel zone + RBS + PR + bolts/welds
   * - ``49_rc_frame_design.py``
     - RC frame to ACI 318
   * - ``50_steel_frame_design.py``
     - Steel frame to AISC 360
   * - ``51_envelope_drift_check.py``
     - Load-combo envelope + ASCE 7 drift
   * - ``52_smrf_capacity_design.py``
     - Special moment frame seismic detailing
   * - ``53_full_design_report.py``
     - HTML + CSV design report

Bridges
-------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Script
     - Demonstrates
   * - ``49_psc_girder_bridge.py``
     - **Capstone** PSC girder: composite section, HL-93,
       PT losses, CEB-FIP creep / shrinkage / relaxation

Capstone walkthroughs
---------------------

These scripts compose features end-to-end and make good "what can
femsolver do" demos:

* ``43_pbe_full_workflow.py`` — full PBE pipeline
* ``45_coupled_wall_pushover.py`` — coupled-wall system
* ``46_vnv_report.py`` — V&V suite output
* ``47_indian_codes_frame_design.py`` — IS-codes design pipeline
* ``48_connection_capstone.py`` — connection mechanics
* ``49_psc_girder_bridge.py`` — bridge engineering
* ``52_smrf_capacity_design.py`` — SMRF capacity design
* ``53_full_design_report.py`` — full design report
