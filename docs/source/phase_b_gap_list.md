# Phase B -- Consolidation gap list (driven by the claims matrix)

The audit identified **9 documented limitations** classified as Beta
or Limited that should be closed before any new feature work.
Closing these means every Production claim becomes defensible
against vendor scrutiny.

Phase B is one theme: **Theme HH -- Solver Consolidation**.

## Sub-phase order (by visibility × effort)

| Sub-phase | Item | Where it lives now | What "done" means |
|---|---|---|---|
| **HH.1** | Mohr-Coulomb full 4-region return mapping | `materials/mohr_coulomb.py` | Edges (sigma_1 = sigma_2 and sigma_2 = sigma_3) handled exactly via 2x2 closed-form return; chatter in triaxial figure gone; documented in tests with a "passes 4 corners" check |
| **HH.2** | Modified Cam-Clay stress-dependent K | `materials/cam_clay.py` | K = (1+e)/kappa * p_eff per MCC theory; cyclic loading produces correct hysteresis loops |
| **HH.3** | Concrete damage -- full Lubliner-Lee-Fenves | `materials/concrete_damage.py` | Separate plastic strain + damage variables; cyclic test under tension-compression-tension shows proper unloading stiffness recovery |
| **HH.4** | PSHA -- period-dependent GMPE coefficients | `seismic/gmpe.py` | BSSA14 coefficient tables for PGA, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0 s; UHS no longer flat across periods |
| **HH.5** | Site response -- equivalent-linear iteration | `seismic/site_response.py` | Vucetic-Dobry / Darendeli G/G_max + damping vs strain curves; iterative SHAKE-style solver |
| **HH.6** | ASCE 7 -- components-and-cladding (C&C) zones | `wind/asce7.py` | C&C pressure coefficients Figure 30.5-1 (zones 1, 1', 2, 3, 4, 5); partially-enclosed building case |
| **HH.7** | IS 875 -- dynamic response factor C_dyn | `wind/is875.py` | Annex C gust factor for tall buildings; background + resonant + size reduction |
| **HH.8** | Punching shear reinforcement design | `design/punching.py` | Stud rail / stirrup design per ACI 318 8.5.5 + EC2 6.4.5; minimum/maximum spacing |
| **HH.9** | Theme HH tests + capstone | -- | Updated capstones from Themes R, S, U, W rerun with no caveats; comparison plots before/after |

## Estimated effort

At our ~150-300 lines per sub-phase + tests cadence, Theme HH is **~8-9 sub-phases** = on par with prior themes. Probably one extended session per sub-phase.

## What changes after Theme HH

Every "Beta" entry in the claims matrix flips to **Production**. The
documented caveats throughout the codebase get cleaned up. The
triaxial test figure in `examples/60_theme_r_capstone.py` regenerates
without chattering. The site-specific UHS in
`examples/63_theme_u_capstone.py` shows a properly curved rock
spectrum.

After HH, Phase C (vendor V&V) becomes the next theme.

## Phases C-E preview (post-consolidation)

| Phase | Theme | Subject |
|---|---|---|
| **C** | Theme JJ -- Vendor V&V | CSI SAP2000 + MIDAS Civil + Abaqus benchmark cases with documented reference values |
| **D.1** | Theme KK -- Timber | NDS 2018 + EC5 + IS 883 |
| **D.2** | Theme LL -- Cold-formed steel | AISI S100-22 + EC3-1-3; CFS section catalogue |
| **D.3** | Theme MM -- Masonry | TMS 402-22 + EC6 + IS 1905; URM, RM, infilled frame |
| **D.4** | Theme NN -- Aluminum / glass | Aluminum Design Manual + ASTM E1300 |
| **D.5** | Theme OO -- Membrane / tensile | Form-finding + dynamic relaxation |
| **E.1** | Theme PP -- Tunnels | TBM staged excavation + ring lining + ground convergence |
| **E.2** | Theme QQ -- Storage tanks | API 650 sloshing + cylindrical shell |
| **E.3** | Theme RR -- Wave loading | Morison + sea-state generation |
| **E.4** | Theme SS -- Slope stability + 2-D SOG | FE slope analysis + 2-D Winkler |

This gives ~10 more themes after HH before the solver is genuinely
"complete" by our claims-matrix definition.
