"""Plane-stress plate in tension, meshed with Q4 elements."""
import numpy as np

from femsolver import Model, ElasticIsotropic, Quad4, LinearStaticAnalysis
from femsolver.io import write_vtk_unstructured


def main() -> None:
    E, nu = 200e9, 0.3
    Lx, Ly, t = 4.0, 1.0, 0.05
    nx, ny = 8, 4
    sigma = 1.0e7  # 10 MPa applied at right edge

    m = Model(ndm=2, ndf=2)
    mat = ElasticIsotropic(1, E=E, nu=nu)
    m.add_material(mat)

    grid = {}
    tag = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            m.add_node(tag, Lx * i / nx, Ly * j / ny)
            grid[(i, j)] = tag
            tag += 1

    etag = 1
    for j in range(ny):
        for i in range(nx):
            n1 = grid[(i, j)]
            n2 = grid[(i + 1, j)]
            n3 = grid[(i + 1, j + 1)]
            n4 = grid[(i, j + 1)]
            m.add_element(Quad4(etag, (n1, n2, n3, n4), mat, thickness=t))
            etag += 1

    # left edge: u_x = 0 ; bottom-left: u_y = 0 too
    for j in range(ny + 1):
        m.fix(grid[(0, j)], [1, 0])
    m.fix(grid[(0, 0)], [1, 1])

    # right edge: distribute sigma * t * dy as nodal forces
    dy = Ly / ny
    for j in range(ny + 1):
        f = sigma * t * (dy / 2.0 if j == 0 or j == ny else dy)
        m.add_nodal_load(grid[(nx, j)], [f, 0.0])

    info = LinearStaticAnalysis(m).run()
    u_max = max(abs(n.disp[0]) for n in m.nodes.values())
    print(f"system size: {info['neq']} equations")
    print(f"max u_x: {u_max:.6e} m")
    # analytical: epsilon = sigma/E (plane stress, no transverse constraint), u_x_max = epsilon * Lx
    eps_th = sigma / E
    print(f"target eps = sigma/E = {eps_th:.6e};  u_x_target = {eps_th * Lx:.6e}")
    write_vtk_unstructured(m, "plate_tension.vtk", deformation_scale=500.0)
    print("wrote plate_tension.vtk")


if __name__ == "__main__":
    main()
