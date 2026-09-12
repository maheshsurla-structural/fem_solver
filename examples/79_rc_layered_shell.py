"""Reinforced-concrete layered shell section -- the Midas FEA NX
"multi-layered grid / concrete shell", reproduced and verified.

This builds the section from the FEA NX dialog screenshot:

* thickness = 300 mm
* concrete fc' = 30 MPa, sampled through the thickness with Simpson points
* reinforcement sheets at z = +100 mm and -100 mm (bar centroids), each
  78 mm^2 @ 100 mm  (= 780 mm^2/m), in two orthogonal grid directions
  (Grid X at 0 deg, Grid Y at 90 deg)

and runs the guide's recommended verification programme on a 1 m-wide,
300 mm-thick strip:

1. Elastic (uncracked, transformed) A / B / D stiffness vs hand values.
2. Axial case  -- pure membrane strain -> N resultant.
3. Pure-bending case -- through-thickness strain at z = +100 / 0 / -100 mm
   checked against ``eps(z) = eps_ref + z * kappa``.
4. Full moment-curvature response (cracking -> yield -> ultimate).

Run::

    python examples/79_rc_layered_shell.py
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    ConcreteKentPark,
    RebarLayer,
    ReinforcedConcreteShellSection,
    shell_moment_curvature,
)
from femsolver.materials.uniaxial import (
    ConcreteTensionStiffening,
    UniaxialReinforcingSteel,
)

# ------------------------------------------------------------------ inputs
T = 0.300                       # shell thickness [m]
FC = 30.0e6                     # concrete cylinder strength [Pa]
FY = 500.0e6                    # reinforcement yield [Pa]
ES = 200.0e9                    # reinforcement modulus [Pa]
AS_BAR = 78.0e-6                # one bar area [m^2]  (~10 mm bar)
SPACING = 0.100                 # bar spacing [m]
Z_TOP, Z_BOT = +0.100, -0.100   # reinforcement centroid offsets [m]


def _concrete(tension_stiffening: bool):
    base = ConcreteKentPark(fpc=FC, eps_c0=0.002, fpcu=0.2 * FC, eps_cu=0.005)
    if not tension_stiffening:
        return base
    f_ct = 0.62 * np.sqrt(FC / 1e6) * 1e6           # modulus of rupture
    return ConcreteTensionStiffening(base, f_ct=f_ct, E_ct=base.E0,
                                     eps_decay=3e-4)


def _steel():
    return UniaxialReinforcingSteel(ES, FY, 1.2 * FY, eps_sh=0.01, eps_su=0.08)


def build_section(tension_stiffening: bool = True):
    """The screenshot section: 300 mm, two grid directions of steel."""
    rebar = [
        # Grid X (0 deg) top & bottom
        RebarLayer.from_bars(Z_TOP, AS_BAR, SPACING, _steel(), theta_deg=0.0),
        RebarLayer.from_bars(Z_BOT, AS_BAR, SPACING, _steel(), theta_deg=0.0),
        # Grid Y (90 deg) top & bottom
        RebarLayer.from_bars(Z_TOP, AS_BAR, SPACING, _steel(), theta_deg=90.0),
        RebarLayer.from_bars(Z_BOT, AS_BAR, SPACING, _steel(), theta_deg=90.0),
    ]
    return ReinforcedConcreteShellSection.from_midas_grid(
        T, _concrete(tension_stiffening), rebar,
        n_simpson=3, n_div=6,
    )


def main() -> None:
    sec = build_section(tension_stiffening=True)
    Ec = _concrete(False).E0
    print("=" * 68)
    print("RC LAYERED SHELL  (Midas multi-layered grid)")
    print("=" * 68)
    print(sec)
    print(f"  concrete Ec        = {Ec/1e9:.1f} GPa")
    print(f"  steel As per width = {sec.steel_area_per_width*1e6:.0f} mm^2/m "
          f"total  ({AS_BAR*1e6:.0f} mm^2 @ {SPACING*1e3:.0f} mm x 4 sheets)")

    # -- 1. elastic stiffness vs hand values --------------------------------
    A = sec.D_membrane()
    D = sec.D_bending()
    B = sec.D_coupling()
    As = AS_BAR / SPACING
    print("\n1) ELASTIC (uncracked, transformed) STIFFNESS  [per unit width]")
    print(f"   A_xx = {A[0,0]/1e6:10.1f} MN/m   "
          f"(hand Ec*t + 2 As Es = {(Ec*T + 2*As*ES)/1e6:.1f})")
    print(f"   D_xx = {D[0,0]/1e3:10.1f} kN.m    "
          f"(hand Ec t^3/12 + 2 As Es z^2 = "
          f"{(Ec*T**3/12 + 2*As*ES*Z_TOP**2)/1e3:.1f})")
    print(f"   B_xx = {B[0,0]:10.3e}      (symmetric steel -> ~0)")

    # -- 2. axial (pure membrane) case --------------------------------------
    eps_axial = -0.0002        # 0.2 permille compression
    resp = sec.section_response([eps_axial, 0.0, 0.0], [0.0, 0.0, 0.0])
    print("\n2) AXIAL CASE   eps_membrane_xx = -2.0e-4 (compression)")
    print(f"   N_xx = {resp.N[0]/1e3:8.1f} kN/m")
    print(f"   strain at z=+100 / 0 / -100 mm = "
          f"{sec.strain_at([eps_axial,0,0],[0,0,0],+0.1)[0]:.2e} / "
          f"{sec.strain_at([eps_axial,0,0],[0,0,0],0.0)[0]:.2e} / "
          f"{sec.strain_at([eps_axial,0,0],[0,0,0],-0.1)[0]:.2e}  (uniform)")
    sec.revert_state()

    # -- 3. pure-bending kinematics check -----------------------------------
    kappa = 0.02               # 1/m  (= 2e-5 /mm, the guide's example)
    e_top = sec.strain_at([0, 0, 0], [kappa, 0, 0], +0.1)[0]
    e_mid = sec.strain_at([0, 0, 0], [kappa, 0, 0], 0.0)[0]
    e_bot = sec.strain_at([0, 0, 0], [kappa, 0, 0], -0.1)[0]
    print("\n3) PURE-BENDING KINEMATICS   kappa_xx = 0.02 /m  (2e-5 /mm)")
    print(f"   eps(z) = eps_ref + z*kappa :  z=+100mm -> {e_top:+.4f}   "
          f"z=0 -> {e_mid:+.4f}   z=-100mm -> {e_bot:+.4f}")
    print(f"   hand: z*kappa = 0.1*0.02 = {0.1*kappa:+.4f}  (guide value 0.002)")

    # -- 4. moment-curvature (cracking -> yield -> ultimate) ----------------
    res = shell_moment_curvature(sec, direction="x", N_target=0.0,
                                 kappa_max=0.06, n_steps=120)
    print("\n4) MOMENT-CURVATURE  (1 m strip, pure bending about local x)")
    print(f"   M_cr ~ {(res.M_cr or 0)/1e3:7.1f} kN.m/m  at kappa "
          f"{res.kappa_cr or 0:.4f}")
    print(f"   M_y  ~ {(res.M_y or 0)/1e3:7.1f} kN.m/m  at kappa "
          f"{res.kappa_y or 0:.4f}")
    print(f"   M_u  ~ {(res.M_u or 0)/1e3:7.1f} kN.m/m  at kappa "
          f"{res.kappa_u or 0:.4f}   (mode: {res.failure_mode})")
    print("\n   kappa[1/m]   M[kN.m/m]   eps_conc_min   eps_steel_max")
    for p in res.points[::12]:
        print(f"   {p.kappa:8.4f}   {p.M/1e3:9.2f}   {p.eps_conc_min:12.2e}"
              f"   {p.eps_steel_max:12.2e}")
    print("=" * 68)


if __name__ == "__main__":
    main()
