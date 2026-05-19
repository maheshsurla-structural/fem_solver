"""Single-story 3D moment frame with a rigid floor diaphragm.

Four columns rise H=3.0 m from a fixed base to a floor at the same XY
corners. A master node sits at the floor centre and is tied to the four
column tops by a RigidDiaphragm constraint (perp_dir=2 → XY plane). A
horizontal load applied at the master is distributed in parallel through
the four columns.

For a square plan with identical columns, each column is a fixed-free
cantilever in bending: lateral stiffness 3 EI / H^3 per column. Four in
parallel give 12 EI / H^3, so:

    u_x_master = F * H^3 / (12 E I)

Run::

    python examples/05_rigid_diaphragm_frame.py
"""
from femsolver import (
    BeamColumn3D,
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
)


def main() -> None:
    L = 6.0
    H = 3.5
    E = 2.0e11
    nu = 0.3
    A = 5.0e-3
    Iy = 4.17e-5
    Iz = 4.17e-5
    J = 8.33e-5
    F = 1.0e5  # lateral load at floor centre

    m = Model(ndm=3, ndf=6)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)

    # base nodes at the four corners (fixed) and column tops at z=H
    base_xy = [(0.0, 0.0), (L, 0.0), (L, L), (0.0, L)]
    for i, (x, y) in enumerate(base_xy):
        m.add_node(i + 1, x, y, 0.0)
        m.fix(i + 1, [1, 1, 1, 1, 1, 1])
    for i, (x, y) in enumerate(base_xy):
        m.add_node(i + 5, x, y, H)

    # diaphragm master at floor centre — pin its out-of-plane DOFs since no
    # element touches it (only in-plane translations + rotation about Z are
    # stiffened by the diaphragm).
    m.add_node(9, L / 2.0, L / 2.0, H)
    m.fix(9, [0, 0, 1, 1, 1, 0])

    for i in range(4):
        m.add_element(
            BeamColumn3D(
                tag=i + 1, nodes=(i + 1, i + 5), material=mat,
                area=A, Iy=Iy, Iz=Iz, J=J, vecxz=(1.0, 0.0, 0.0),
            )
        )

    m.rigid_diaphragm(master=9, slaves=[5, 6, 7, 8], perp_dir=2)
    m.add_nodal_load(9, [F, 0.0, 0.0, 0.0, 0.0, 0.0])

    info = LinearStaticAnalysis(m).run()
    expected = F * H ** 3 / (12.0 * E * Iz)
    u_master = m.node(9).disp[0]

    print(f"system size: {info['neq']} equations, {info['n_constraints']} MP constraints")
    print(f"master u_x: {u_master:.6e} m  (expected {expected:.6e} m, rel err {abs(u_master-expected)/expected:.2e})")
    for s in (5, 6, 7, 8):
        d = m.node(s).disp
        print(f"  slave {s}: u=({d[0]:.6e}, {d[1]:.6e}, {d[2]:.6e})  theta_z={d[5]:.3e}")
    for tag in (1, 2, 3, 4):
        Rx = m.node(tag).reaction[0]
        Ry = m.node(tag).reaction[1]
        Rz = m.node(tag).reaction[2]
        print(f"  base {tag} reaction: Rx={Rx:.3e}  Ry={Ry:.3e}  Rz={Rz:.3e}")


if __name__ == "__main__":
    main()
