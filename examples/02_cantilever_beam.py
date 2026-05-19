"""Cantilever beam, point load at tip. Compare to PL^3/(3EI)."""
from femsolver import Model, ElasticIsotropic, BeamColumn2D, LinearStaticAnalysis


def main() -> None:
    E, A, Iz = 200e9, 1e-2, 8.333e-6
    L, P = 3.0, 1e3
    m = Model(ndm=2, ndf=3)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    m.add_nodal_load(2, [0.0, -P, 0.0])
    LinearStaticAnalysis(m).run()
    w_fem = m.node(2).disp[1]
    w_th = -P * L ** 3 / (3.0 * E * Iz)
    print(f"tip deflection (FEM):        {w_fem:.6e} m")
    print(f"tip deflection (analytical): {w_th:.6e} m")
    print(f"relative error:              {abs(w_fem - w_th) / abs(w_th):.2e}")


if __name__ == "__main__":
    main()
