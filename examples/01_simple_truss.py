"""Three-bar pin-jointed truss under tip load.

       3
      /|\
     / | \
    /  |  \
   1---+---2

Nodes 1 and 2 are supports; node 3 carries a downward load.
"""
import numpy as np

from femsolver import Model, ElasticIsotropic, Truss2D, LinearStaticAnalysis


def main() -> None:
    E, A, P = 200e9, 1e-4, 10e3
    m = Model(ndm=2, ndf=2)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 2.0, 0.0)
    m.add_node(3, 1.0, np.sqrt(3.0))
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)
    m.add_element(Truss2D(1, (1, 3), mat, A))
    m.add_element(Truss2D(2, (2, 3), mat, A))
    m.add_element(Truss2D(3, (1, 2), mat, A))
    m.fix(1, [1, 1])
    m.fix(2, [0, 1])
    m.add_nodal_load(3, [0.0, -P])
    info = LinearStaticAnalysis(m).run()
    print(f"system size: {info['neq']} equations")
    print(f"node 3 displacement: ({m.node(3).disp[0]:.6e}, {m.node(3).disp[1]:.6e}) m")
    for tag, e in m.elements.items():
        print(f"element {tag}: axial force = {e.axial_force:.3f} N")


if __name__ == "__main__":
    main()
