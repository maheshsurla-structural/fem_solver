"""Tests for ShellSection abstraction + LayeredShellSection (Phase 14.7).

The layered-shell machinery is validated through six properties:

1. **Section construction guard rails** — invalid thickness / k_shear.
2. **Single-layer ElasticShellSection matches the material+thickness
   constructor** — passing ``section=ElasticShellSection(...)`` to a
   shell element gives bit-identical results to the legacy path.
3. **Multi-layer same-material section equals single-layer** — three
   equal layers of the same material reproduce the homogeneous case.
4. **Sandwich plate stiffer than equivalent monolithic core**, softer
   than monolithic face — the textbook sandwich-beam check.
5. **Asymmetric stack produces nonzero coupling matrix** — required
   for membrane-bending coupling.
6. **A through-the-element analysis with a layered section converges
   to the homogeneous result when all layers share the same material**
   (sanity end-to-end check).
"""
import numpy as np
import pytest

from femsolver import (
    ElasticIsotropic,
    ElasticShellSection,
    LayeredShellSection,
    Model,
    ShellLayer,
    ShellMITC4,
    ShellTri3,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis


# ====================================================== guard rails

def test_elastic_shell_section_rejects_invalid_thickness():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    with pytest.raises(ValueError, match="thickness"):
        ElasticShellSection(mat, thickness=0.0)
    with pytest.raises(ValueError, match="thickness"):
        ElasticShellSection(mat, thickness=-0.1)


def test_elastic_shell_section_rejects_invalid_k_shear():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    with pytest.raises(ValueError, match="k_shear"):
        ElasticShellSection(mat, thickness=0.01, k_shear=0.0)


def test_shell_layer_rejects_zero_thickness():
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    with pytest.raises(ValueError, match="thickness"):
        ShellLayer(mat, thickness=0.0)


def test_layered_section_rejects_empty():
    with pytest.raises(ValueError, match="at least one layer"):
        LayeredShellSection(layers=[])


# ====================================================== D matrices

def test_elastic_shell_section_D_matches_legacy_formulas():
    """ElasticShellSection D matrices match the hard-coded formulas
    inside ShellMITC4."""
    E, nu, t = 2.0e11, 0.3, 0.01
    mat = ElasticIsotropic(1, E=E, nu=nu)
    sec = ElasticShellSection(mat, thickness=t)
    # Membrane: E*t/(1-nu^2)
    f = E * t / (1.0 - nu * nu)
    Dm_expected = f * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, 0.5 * (1.0 - nu)],
    ])
    assert np.allclose(sec.D_membrane(), Dm_expected, rtol=1e-12)
    # Bending: Dm * t^2 / 12
    assert np.allclose(sec.D_bending(), Dm_expected * (t ** 2 / 12.0), rtol=1e-12)
    # Coupling: zero
    assert np.allclose(sec.D_coupling(), np.zeros((3, 3)))
    # Shear: k*G*t * I
    G = E / (2.0 * (1.0 + nu))
    k = 5.0 / 6.0
    assert np.allclose(sec.D_shear(), k * G * t * np.eye(2), rtol=1e-12)


def test_layered_section_symmetric_stack_has_zero_coupling():
    """A symmetric stack (top half mirrors bottom half) has zero
    membrane-bending coupling."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    sec = LayeredShellSection.from_layers_centered([
        (mat, 0.001), (mat, 0.001), (mat, 0.001), (mat, 0.001),
    ])
    assert np.allclose(sec.D_coupling(), np.zeros((3, 3)), atol=1e-12)


def test_layered_section_asymmetric_stack_has_nonzero_coupling():
    """Asymmetric stack: stiff face on top only -> membrane-bending
    coupling is nonzero."""
    mat_stiff = ElasticIsotropic(1, E=2e11, nu=0.3)
    mat_soft = ElasticIsotropic(2, E=1e9, nu=0.3)
    sec = LayeredShellSection.from_layers_centered([
        (mat_soft, 0.005), (mat_soft, 0.005),
        (mat_stiff, 0.001), (mat_stiff, 0.001),
    ])
    assert np.max(np.abs(sec.D_coupling())) > 1e-3


def test_layered_section_same_material_equals_single_layer():
    """Splitting a homogeneous material into N layers should give
    identical D matrices to a single layer."""
    mat = ElasticIsotropic(1, E=2e11, nu=0.3)
    t = 0.01
    sec_single = ElasticShellSection(mat, thickness=t)
    sec_layered = LayeredShellSection.from_layers_centered([
        (mat, t / 5), (mat, t / 5), (mat, t / 5),
        (mat, t / 5), (mat, t / 5),
    ])
    assert np.allclose(sec_single.D_membrane(), sec_layered.D_membrane(),
                        rtol=1e-12)
    assert np.allclose(sec_single.D_bending(), sec_layered.D_bending(),
                        rtol=1e-12)
    assert np.allclose(sec_single.D_shear(), sec_layered.D_shear(),
                        rtol=1e-12)
    # D_coupling for symmetric stack is zero up to floating-point noise
    # from summing z_mid contributions that cancel symmetrically.
    assert np.max(np.abs(sec_layered.D_coupling())) < 1e-6 * np.max(
        np.abs(sec_layered.D_membrane())
    )


def test_sandwich_bending_stiffer_than_core_softer_than_face():
    """Sandwich plate (stiff-soft-stiff): bending stiffness lies
    *between* the all-stiff and all-soft monolithic equivalents."""
    mat_face = ElasticIsotropic(1, E=2e11, nu=0.3)
    mat_core = ElasticIsotropic(2, E=1e9, nu=0.3)
    t_face, t_core = 0.001, 0.008
    sec_sand = LayeredShellSection.from_layers_centered([
        (mat_face, t_face), (mat_core, t_core), (mat_face, t_face),
    ])
    t_total = 2 * t_face + t_core
    sec_face = ElasticShellSection(mat_face, thickness=t_total)
    sec_core = ElasticShellSection(mat_core, thickness=t_total)
    Db_sand = sec_sand.D_bending()[0, 0]
    Db_face = sec_face.D_bending()[0, 0]
    Db_core = sec_core.D_bending()[0, 0]
    # Sandwich is between core-only and face-only
    assert Db_core < Db_sand < Db_face
    # And dramatically closer to face than to core (the whole point of
    # a sandwich)
    assert Db_sand / Db_core > 50.0, \
        f"sandwich should be >>50x stiffer than all-core: {Db_sand/Db_core:.1f}"


# ====================================================== end-to-end

def test_mitc4_section_path_matches_material_path():
    """Using ``section=ElasticShellSection(mat, t)`` gives identical
    results to passing ``material, thickness`` directly."""
    E, nu, t = 2.0e11, 0.3, 0.01
    mat = ElasticIsotropic(1, E=E, nu=nu)
    L = 1.0; P = 1.0; N = 6

    def run(use_section: bool):
        m = Model(ndm=3, ndf=6); m.add_material(mat)
        nL = N + 1
        for j in range(nL):
            for i in range(nL):
                m.add_node(j * nL + i + 1, i * L / N, j * L / N, 0.0)
        etag = 1
        for j in range(N):
            for i in range(N):
                n1 = j * nL + i + 1; n2 = n1 + 1
                n3 = n2 + nL; n4 = n1 + nL
                if use_section:
                    sec = ElasticShellSection(mat, t)
                    m.add_element(ShellMITC4(etag, (n1, n2, n3, n4),
                                              mat, section=sec))
                else:
                    m.add_element(ShellMITC4(etag, (n1, n2, n3, n4), mat, t))
                etag += 1
        for j in range(nL):
            for i in range(nL):
                if i in (0, N) or j in (0, N):
                    m.fix(j * nL + i + 1, [0, 0, 1, 0, 0, 0])
        m.fix(1, [1, 1, 1, 0, 0, 0])
        m.fix(N + 1, [0, 1, 1, 0, 0, 0])
        ic = (N // 2) * nL + N // 2 + 1
        m.add_nodal_load(ic, [0, 0, -P, 0, 0, 0])
        LinearStaticAnalysis(m).run()
        return -m.node(ic).disp[2]

    w_a = run(False)
    w_b = run(True)
    assert w_a == pytest.approx(w_b, rel=1e-12)


def test_mitc4_layered_section_uniform_material_matches_legacy():
    """A 3-layer section of the SAME material is bit-identical to the
    single-layer / material+thickness paths."""
    E, nu, t = 2.0e11, 0.3, 0.01
    mat = ElasticIsotropic(1, E=E, nu=nu)
    L = 1.0; P = 1.0; N = 6

    def run(section_layers):
        m = Model(ndm=3, ndf=6); m.add_material(mat)
        nL = N + 1
        for j in range(nL):
            for i in range(nL):
                m.add_node(j * nL + i + 1, i * L / N, j * L / N, 0.0)
        etag = 1
        for j in range(N):
            for i in range(N):
                n1 = j * nL + i + 1; n2 = n1 + 1
                n3 = n2 + nL; n4 = n1 + nL
                if section_layers is None:
                    m.add_element(ShellMITC4(etag, (n1, n2, n3, n4), mat, t))
                else:
                    sec = LayeredShellSection.from_layers_centered(section_layers)
                    m.add_element(ShellMITC4(etag, (n1, n2, n3, n4),
                                              mat, section=sec))
                etag += 1
        for j in range(nL):
            for i in range(nL):
                if i in (0, N) or j in (0, N):
                    m.fix(j * nL + i + 1, [0, 0, 1, 0, 0, 0])
        m.fix(1, [1, 1, 1, 0, 0, 0])
        m.fix(N + 1, [0, 1, 1, 0, 0, 0])
        ic = (N // 2) * nL + N // 2 + 1
        m.add_nodal_load(ic, [0, 0, -P, 0, 0, 0])
        LinearStaticAnalysis(m).run()
        return -m.node(ic).disp[2]

    w_legacy = run(None)
    w_layered = run([(mat, t / 3), (mat, t / 3), (mat, t / 3)])
    assert w_legacy == pytest.approx(w_layered, rel=1e-12)


def test_tri3_accepts_section_argument():
    """ShellTri3 supports the same section= override as ShellMITC4."""
    mat = ElasticIsotropic(1, E=2.0e11, nu=0.3)
    sec = ElasticShellSection(mat, thickness=0.01)
    m = Model(ndm=3, ndf=6); m.add_material(mat)
    m.add_node(1, 0, 0, 0); m.add_node(2, 1, 0, 0); m.add_node(3, 0, 1, 0)
    e1 = ShellTri3(1, (1, 2, 3), mat, section=sec)
    m.add_element(e1)
    K1 = e1.K_global()
    # Build a parallel model for the material+thickness variant
    m2 = Model(ndm=3, ndf=6); m2.add_material(mat)
    m2.add_node(1, 0, 0, 0); m2.add_node(2, 1, 0, 0); m2.add_node(3, 0, 1, 0)
    e2 = ShellTri3(1, (1, 2, 3), mat, thickness=0.01)
    m2.add_element(e2)
    K2 = e2.K_global()
    assert np.allclose(K1, K2, rtol=1e-12)
