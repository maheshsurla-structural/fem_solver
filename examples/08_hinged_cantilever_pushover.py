"""Pushover analysis of a cantilever with a plastic hinge at the base.

A 3 m cantilever is loaded transversely at its tip. A bilinear plastic
hinge at the fixed-end joint accumulates plastic rotation once the
moment at the support reaches ``M_y``, producing a clear bilinear
force-displacement curve.

Two key references for the analytical comparison:

* **Yield force**: with the hinge at the fixed end, the force that
  brings the support moment to ``M_y`` is ``P_y = M_y / L``.
* **Elastic stiffness**: the cantilever sees the spring in series with
  the beam's own bending flexibility, so

      K_elastic = 1 / (L^3/(3 EI) + L^2 / K_h)

  With ``K_h = 4 EI / L`` (chosen here) the spring contributes a
  noticeable fraction of the total flexibility — about 25 % at the
  elastic level and a much larger fraction post-yield.

Run::

    python examples/08_hinged_cantilever_pushover.py
"""
from __future__ import annotations

from femsolver import (
    BilinearMomentRotationSpring,
    ElasticIsotropic,
    HingedBeamColumn2D,
    Model,
    NonlinearStaticAnalysis,
)


def main() -> None:
    # --------------------------------------------------------------- inputs
    E = 2.0e11
    A = 1.0e-2
    Iz = 8.333e-6                 # 100x100 mm rectangular
    L = 3.0
    K_h = 4.0 * E * Iz / L        # comparable to the beam's 4 EI/L
    My = 5.0e3                     # yield moment (N.m)
    b = 0.05                       # post-yield slope ratio (kinematic hardening)

    P_yield = My / L
    P_max = 1.6 * P_yield          # push past yield by 60 percent
    n_steps = 40

    # --------------------------------------------------------------- build
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    model = Model(ndm=2, ndf=3)
    model.add_material(mat)
    model.add_node(1, 0.0, 0.0)
    model.add_node(2, L, 0.0)
    hinge = BilinearMomentRotationSpring(K0=K_h, My=My, b=b)
    elem = HingedBeamColumn2D(1, (1, 2), mat, A, Iz, hinge_i=hinge)
    model.add_element(elem)
    model.fix(1, [1, 1, 1])
    model.add_nodal_load(2, [0.0, -P_max, 0.0])

    # --------------------------------------------------------------- solve
    analysis = NonlinearStaticAnalysis(
        model,
        num_steps=n_steps,
        dlambda=1.0 / n_steps,
        track=(2, 1),
        tol=1e-8,
        max_iter=30,
    )
    analysis.run()

    # --------------------------------------------------------------- print
    print(f"\nCantilever pushover with bilinear hinge at base")
    print(f"  L = {L} m, EI = {E * Iz:.3g} N.m^2")
    print(f"  K_h = {K_h:.3g} N.m/rad,  M_y = {My} N.m,  b = {b}")
    print(f"  P_y = M_y / L = {P_yield:.3g} N\n")
    print(f"  step  lambda      P (N)         v_tip (m)        M_h (N.m)    "
          f"theta_p (rad)")
    for i, (lmbd, v_tip) in enumerate(zip(analysis.lambdas, analysis.tracked), 1):
        # peek at the spring state at this step (committed values)
        # NOTE: hinge_i state is whatever was committed at the LAST converged
        # step, so we can only fully report for the last step.
        if i == n_steps:
            theta_p = elem.hinge_i.theta_p_committed
            M_h = elem.hinge_i.M_trial
            print(f"  {i:4d}  {lmbd:.3f}  {lmbd * P_max:9.2f}  "
                  f"{v_tip:14.6e}  {M_h:9.2f}  {theta_p:.4e}")
        elif i % 5 == 0 or i == 1:
            print(f"  {i:4d}  {lmbd:.3f}  {lmbd * P_max:9.2f}  "
                  f"{v_tip:14.6e}")
    print()
    print(f"  Final hinge moment: {elem.hinge_i.M_trial:.4g} N.m  "
          f"(My = {My})")
    print(f"  Plastic rotation:   {elem.hinge_i.theta_p_committed:.4e} rad")
    if elem.hinge_i.theta_p_committed > 0.0:
        print(f"  Hinge has yielded — bilinear pushover successful.")
    else:
        print(f"  Hinge stayed elastic (load did not reach yield).")


if __name__ == "__main__":
    main()
