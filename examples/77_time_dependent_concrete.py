"""Phase B.11 -- Time-dependent concrete: strength-gain curve, creep &
shrinkage properties and their structural effects.

Brings together the time-dependent concrete capability:

1. **Strength / modulus / tensile gain curve** f_cm(t), E_cm(t),
   f_ctm(t) per EN 1992-1-1 §3.1.2 (and the ACI 209 alternative) --
   the "compression strength graph".
2. **Creep & shrinkage** from CEB-FIP MC 2010 (phi(t,t0), eps_cs(t)).
3. **Structural effects**: creep long-term deflection (AAEM), a
   shrinkage restraint force in an axially-held member, and shrinkage
   imposed on a finite-element model as an eigenstrain (restraint
   reactions on a restrained block; free shortening on a free one).

Run::

    python examples/77_time_dependent_concrete.py
"""
from __future__ import annotations

import numpy as np

from femsolver.materials.concrete_time import (
    en1992_strength_gain, aci209_strength_gain, strength_gain_curve,
)
from femsolver.bridges.creep_shrinkage import (
    cebfip_creep_coefficient, cebfip_shrinkage,
)
from femsolver.analysis.time_dependent import (
    age_adjusted_modulus, creep_deflection, shrinkage_axial_force,
    apply_shrinkage_load, StepByStepCreep, StepByStepCreepFE,
)
from femsolver.elements.plane import Quad4
from femsolver.core.model import Model
from femsolver.elements.solid import Hex8
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.analysis.linear_static import LinearStaticAnalysis


def main() -> None:
    f_ck28 = 40e6        # C40/50

    print("=" * 66)
    print(" Time-dependent concrete  (C40/50, normal cement)")
    print("=" * 66)

    # ---- (1) strength / modulus / tensile gain curve -----------------
    print("\n EN 1992 strength-gain curve:")
    print(f"   {'age (d)':>8}{'beta_cc':>9}{'f_cm':>8}{'f_ck':>8}"
          f"{'f_ctm':>8}{'E_cm':>8}")
    print(f"   {'':>8}{'':>9}{'(MPa)':>8}{'(MPa)':>8}{'(MPa)':>8}{'(GPa)':>8}")
    for t in (3, 7, 28, 90, 365, 3650):
        p = en1992_strength_gain(t, f_ck_28=f_ck28, cement_class="N")
        print(f"   {t:>8}{p.beta_cc:>9.3f}{p.f_cm/1e6:>8.1f}{p.f_ck/1e6:>8.1f}"
              f"{p.f_ctm/1e6:>8.2f}{p.E_cm/1e9:>8.1f}")
    aci = aci209_strength_gain(7, f_c_28=f_ck28, cement_type="III", curing="moist")
    print(f"   ACI 209 (Type III, moist) f_c(7d) = {aci.f_cm/1e6:.1f} MPa "
          f"(rapid early gain)")

    # ---- (2) creep + shrinkage (CEB-FIP MC 2010) ----------------------
    f_cm = f_ck28 + 8e6
    cr = cebfip_creep_coefficient(t_days=3650, t0_days=28, f_cm=f_cm,
                                  RH=70.0, h_0=0.30)
    sh = cebfip_shrinkage(t_days=3650, t_s_days=7, f_cm=f_cm,
                          RH=70.0, h_0=0.30)
    print(f"\n CEB-FIP creep  phi(10yr, t0=28d) = {cr.phi:.2f}")
    print(f" CEB-FIP shrink eps_cs(10yr)      = {sh.eps_cs*1e6:.0f} micro-strain")

    # ---- (3) structural effects ---------------------------------------
    E28 = en1992_strength_gain(28, f_ck_28=f_ck28).E_cm
    E_eff = age_adjusted_modulus(E_c=E28, phi=cr.phi, chi=0.8)
    print(f"\n Age-adjusted effective modulus E_eff = {E_eff/1e9:.1f} GPa "
          f"(from E_28 = {E28/1e9:.1f} GPa, phi = {cr.phi:.2f})")

    inst = 12.0   # mm instantaneous deflection under sustained load
    print(f" Sustained-load deflection: {inst:.1f} mm inst -> "
          f"{creep_deflection(instantaneous=inst, phi=cr.phi):.1f} mm long-term")

    A = 0.6 * 0.4
    N = shrinkage_axial_force(E_eff=E_eff, A=A, eps_sh=sh.eps_cs)
    print(f" Restrained shrinkage axial force (A={A:.2f} m^2): "
          f"{N/1e3:+.0f} kN (tension)")

    # shrinkage on a restrained FE block -> restraint reactions
    mat = ElasticIsotropic(1, E=E28, nu=0.2, rho=0.0)
    m = Model(ndm=3, ndf=3); m.add_material(mat)
    pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    for i, (x, y, z) in enumerate(pts):
        m.add_node(i + 1, x, y, z)
    m.add_element(Hex8(1, (1, 2, 3, 4, 5, 6, 7, 8), mat))
    for nd in (1, 4, 5, 8):
        m.fix(nd, [1, 1, 1])
    for nd in (2, 3, 6, 7):
        m.fix(nd, [1, 0, 0])
    apply_shrinkage_load(m, eps_sh=sh.eps_cs)
    LinearStaticAnalysis(m).run()
    Rx = sum(m.node(nd).reaction[0] for nd in (1, 4, 5, 8))
    print(f" FE shrinkage restraint reaction on a held 1m cube face = "
          f"{Rx/1e3:.0f} kN")

    # ---- (4) step-by-step relaxation (Volterra superposition) ---------
    def phi_t(t, t0):
        if t <= t0:
            return 0.0
        return cebfip_creep_coefficient(t_days=t, t0_days=t0, f_cm=f_cm,
                                        RH=70.0, h_0=0.30).phi
    scc = StepByStepCreep(E_c=E28, phi=phi_t)
    eps0 = 1.0e-3
    grid = np.unique(np.concatenate([[28.0], 28.0 + np.logspace(-1, 4, 60)]))
    rel = scc.relaxation_history(grid, np.full_like(grid, eps0))
    R_over_E = rel.stress[-1] / (E28 * eps0)
    phi_inf = phi_t(grid[-1], 28.0)
    chi = 1.0 / (1.0 - R_over_E) - 1.0 / phi_inf
    print(f"\n Step-by-step relaxation (held strain from t0=28 d):")
    print(f"   sigma relaxes to {R_over_E*100:.0f}% of elastic over "
          f"{(grid[-1]-28)/365:.0f} yr")
    print(f"   implied ageing coefficient chi = {chi:.2f} "
          f"(Trost/Bazant chi(inf,28) ~ 0.80)")
    print("   (this is the rigorous Volterra superposition -- exact creep,")
    print("    true relaxation -- the engine for step-by-step time analysis)")

    # ---- (5) per-element FE creep time-march --------------------------
    matc = ElasticIsotropic(2, E=E28, nu=0.0, rho=0.0)
    mc = Model(ndm=2, ndf=2); mc.add_material(matc)
    mc.add_node(1, 0, 0); mc.add_node(2, 1, 0)
    mc.add_node(3, 1, 1); mc.add_node(4, 0, 1)
    mc.add_element(Quad4(1, (1, 2, 3, 4), matc, thickness=1.0))
    mc.fix(1, [1, 1]); mc.fix(4, [1, 0])
    grid2 = np.array([28, 40, 100, 400, 1000, 3650], float)
    march = StepByStepCreepFE(mc, E_c=E28, phi=phi_t)
    res = march.run(grid2,
                    sustained_loads=lambda model: (
                        model.add_nodal_load(2, [5e5, 0]),
                        model.add_nodal_load(3, [5e5, 0])),
                    track=[(2, 0)])
    u = res.disp[(2, 0)]
    print(f"\n FE creep time-march (sustained axial load, determinate bar):")
    print(f"   elongation grows {u[0]*1e6:.1f} -> {u[-1]*1e6:.1f} um "
          f"(x{u[-1]/u[0]:.2f} = 1+phi); a determinate member just creeps,")
    print(f"   an indeterminate one would redistribute its internal forces.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
