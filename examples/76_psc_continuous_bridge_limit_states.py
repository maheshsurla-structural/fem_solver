"""Phase B.8 -- PSC continuous-bridge limit-state checks end to end.

The full prestressed-concrete bridge pipeline this session has built:

    define a Tendon  ->  apply_to(model)  ->  analyse
        ->  primary (P·e) + secondary (hyperstatic) moments
        ->  AASHTO LRFD / EN 1992 limit-state stress checks

A two-span continuous PT box girder is post-tensioned by a parabolic
tendon. We extract the **secondary** prestress moment over the interior
pier (which a single-span analysis can never see) and feed it -- with
the primary `P·e` and the external service moment -- into the service
stress checks of both codes.

Run::

    python examples/76_psc_continuous_bridge_limit_states.py
"""
from __future__ import annotations

import numpy as np

from femsolver.core.model import Model
from femsolver.elements.beam import BeamColumn2D
from femsolver.materials.elastic import ElasticIsotropic
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.results.diagrams import beam_force_diagram
from femsolver.bridges.tendon import Tendon, tendon_secondary_moment
from femsolver.design.psc import (
    PscSection, aashto_service_check, ec2_service_check,
    ec2_decompression_check, psc_factored_moment,
    aashto_flexure_capacity, psc_flexure_check,
)


def main() -> None:
    Lspan, nps = 30.0, 24
    # solid-rectangular idealisation of the box: 1.5 m wide x 1.6 m deep
    b, h = 1.5, 1.6
    A = b * h
    I = b * h ** 3 / 12.0
    P = 12.0e6          # effective prestress after losses (N)
    drape = 0.55        # parabolic drape per span (m), below centroid
    f_c = 45e6          # 28-day strength
    f_ck = 45e6

    mat = ElasticIsotropic(1, E=34e9, nu=0.2, rho=0.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    nn = 2 * nps + 1
    for i in range(nn):
        m.add_node(i + 1, i * Lspan / nps, 0.0)
    for i in range(nn - 1):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    m.fix(1, [1, 1, 0]); m.fix(nps + 1, [0, 1, 0]); m.fix(nn, [0, 1, 0])

    # parabolic tendon: -drape at mid-span, 0 over the pier (eccentricity
    # below the centroid is negative in the Tendon local +y convention)
    ecc = np.zeros(nn)
    for i in range(nn):
        xs = (i * Lspan / nps) % Lspan
        ecc[i] = -4.0 * drape * xs * (Lspan - xs) / Lspan ** 2

    tendon = Tendon(nodes=list(range(1, nn + 1)), eccentricity=ecc,
                    area=0.012, jacking_force=P, effective_force=P, name="PT")
    tendon.apply_to(m)
    LinearStaticAnalysis(m).run()

    # ---- primary / secondary at the interior pier ---------------------
    e_pier = ecc[nps]                       # ~0 over the pier
    M_total = beam_force_diagram(m.element(nps))["M"][-1]
    M_sec = tendon_secondary_moment(total_moment=M_total, P=P, e=e_pier)

    print("=" * 66)
    print(" PSC two-span continuous bridge -- limit-state checks")
    print(f"   2 x {Lspan:.0f} m, P = {P/1e6:.0f} MN, drape = {drape:.2f} m, "
          f"f_c = f_ck = {f_c/1e6:.0f} MPa")
    print("=" * 66)
    print(f"\n Over the interior pier:")
    print(f"   primary moment  P.e        = {P*e_pier/1e3:+8.1f} kN.m")
    print(f"   secondary (hyperstatic)    = {M_sec/1e3:+8.1f} kN.m  "
          f"(sagging -> relieves the hogging pier moment)")

    # ---- service stress checks consuming primary + secondary ----------
    sec = PscSection(A=A, I=I, y_top=h / 2, y_bot=h / 2)
    # over an interior pier the external service moment is HOGGING (-ve)
    M_service_ext = -4500e3
    e_mag = abs(e_pier)

    print(f"\n Service stress check at the pier (M_ext = "
          f"{M_service_ext/1e3:.0f} kN.m hogging, incl. secondary):")
    a = aashto_service_check(sec, P=P, e=e_mag, M_external=M_service_ext,
                             M_secondary=M_sec, f_c=f_c, prestress_class="U")
    e2 = ec2_service_check(sec, P=P, e=e_mag, M_external=M_service_ext,
                           M_secondary=M_sec, f_ck=f_ck,
                           combination="characteristic")
    dec = ec2_decompression_check(sec, P=P, e=e_mag,
                                  M_external=M_service_ext, M_secondary=M_sec)
    for name, c in [("AASHTO U  ", a), ("EN1992 char", e2)]:
        print(f"   {name}: f_top={c.f_top/1e6:6.2f}  f_bot={c.f_bot/1e6:6.2f} MPa"
              f"  | comp_lim {c.comp_limit/1e6:.1f}  tens_lim {c.tens_limit/1e6:.2f}"
              f"  | DCR={c.DCR:.3f}  {'OK' if c.passes else 'FAIL'}")
    print(f"   EN1992 decompression: f_top={dec.f_top/1e6:6.2f} MPa  "
          f"{'OK (compression)' if dec.passes else 'FAIL (tension)'}")

    # ---- ULS factored demand including secondary ----------------------
    Mu = psc_factored_moment(
        factored_external=-(1.25 * 4000e3 + 1.5 * 3000e3),  # hogging DC+LL
        M_secondary=M_sec,                                   # at 1.0
    )
    print(f"\n ULS factored moment M_Ed (1.25*DC + 1.5*LL + 1.0*M_sec) = "
          f"{Mu/1e3:.0f} kN.m")
    print("   (the secondary moment is carried into the strength demand at")
    print("    load factor 1.0 -- AASHTO 3.4.1 / EN 1992 gamma_P = 1.0)")

    # ---- ULS flexural capacity vs demand (one-call M_u <= phiM_n) ------
    cap = aashto_flexure_capacity(
        A_ps=0.012, f_pu=1860e6, f_py=1674e6, d_p=h / 2 + abs(e_pier) + 0.55,
        b=b, f_c=f_c,
    )
    flex = psc_flexure_check(M_u=abs(Mu), capacity=cap)
    print(f"\n ULS flexure (AASHTO 5.6.3): phiM_n = {cap.phi_M_n/1e3:.0f} kN.m "
          f"(phi={cap.phi:.2f}, {cap.controlled})")
    print(f"   M_Ed / phiM_n = {flex.DCR:.3f}  -> {'OK' if flex.passes else 'FAIL'}")


if __name__ == "__main__":
    main()
