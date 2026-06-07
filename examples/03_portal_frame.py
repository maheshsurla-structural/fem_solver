"""Two-dimensional portal frame with lateral load.

      ____F___
     |        |
     |        |
     |        |
    /_/      /_/

Two columns, one beam, fixed at base, lateral load at top-left.
"""
from femsolver import Model, ElasticIsotropic, BeamColumn2D, LinearStaticAnalysis
from femsolver.results import write_vtk_unstructured


def main() -> None:
    E = 200e9
    A_col = 1e-2
    Iz_col = 8.333e-5
    A_bm = 1.5e-2
    Iz_bm = 1.5e-4
    H, B = 3.0, 5.0
    F = 50e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)        # base left
    m.add_node(2, B, 0.0)          # base right
    m.add_node(3, 0.0, H)          # top left
    m.add_node(4, B, H)            # top right
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(BeamColumn2D(1, (1, 3), mat, A_col, Iz_col))  # left column
    m.add_element(BeamColumn2D(2, (2, 4), mat, A_col, Iz_col))  # right column
    m.add_element(BeamColumn2D(3, (3, 4), mat, A_bm, Iz_bm))    # roof beam
    m.fix(1, [1, 1, 1])
    m.fix(2, [1, 1, 1])
    m.add_nodal_load(3, [F, 0.0, 0.0])
    info = LinearStaticAnalysis(m).run()
    print(f"system size: {info['neq']} equations")
    print(f"top-left  drift: ({m.node(3).disp[0]:.4e}, {m.node(3).disp[1]:.4e}) m, "
          f"rot {m.node(3).disp[2]:.4e} rad")
    print(f"top-right drift: ({m.node(4).disp[0]:.4e}, {m.node(4).disp[1]:.4e}) m, "
          f"rot {m.node(4).disp[2]:.4e} rad")
    for tag, e in m.elements.items():
        N = e.end_forces_local[0]
        Vy = e.end_forces_local[1]
        Mz = e.end_forces_local[2]
        print(f"element {tag} end-1: N={N:.2f} Vy={Vy:.2f} Mz={Mz:.2f}")
    write_vtk_unstructured(m, "portal_frame.vtk", deformation_scale=200.0)
    print("wrote portal_frame.vtk")


if __name__ == "__main__":
    main()
