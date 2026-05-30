"""Phase 34 tests -- shear-wall fiber sections, shear flexibility,
coupling-beam helper.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from femsolver import (
    BeamColumn2D,
    ConcreteKentPark,
    ConcreteMander,
    CrackedSectionFactors,
    ElasticIsotropic,
    LinearStaticAnalysis,
    Model,
    UniaxialMenegottoPinto,
    aci318_cracked_factors,
    add_coupling_beam_2d,
    asce41_wall_factors,
    i_wall_section_3d,
    l_wall_section_3d,
    t_wall_section_3d,
    u_wall_section_3d,
    wall_base_shear_spring_stiffness,
    wall_lateral_stiffness,
    wall_section_2d,
    wall_shear_area,
)


# ============================================================ helpers

def _materials():
    unc = ConcreteKentPark(fpc=30.0e6, eps_c0=0.002,
                            fpcu=6.0e6, eps_cu=0.0035)
    conf = ConcreteMander(fpc=45.0e6, eps_c0=0.004)
    steel = UniaxialMenegottoPinto(E=2.0e11, b=0.01, sigma_y=420.0e6)
    return unc, conf, steel


# ============================================================ 2D wall section

def test_wall_section_2d_gross_area():
    """Total fiber area == L_w * t_w."""
    unc, conf, steel = _materials()
    L_w, t_w = 4.0, 0.30
    sec = wall_section_2d(
        L_w=L_w, t_w=t_w, L_be=0.5,
        web_concrete=unc, boundary_concrete=conf,
        rebar_material=steel,
    )
    assert sec.gross_area == pytest.approx(L_w * t_w, rel=1.0e-12)


def test_wall_section_2d_centroid_symmetric():
    """Symmetric wall has centroid at y=0."""
    unc, conf, steel = _materials()
    sec = wall_section_2d(
        L_w=4.0, t_w=0.30, L_be=0.5,
        web_concrete=unc, boundary_concrete=conf,
        rebar_material=steel,
    )
    assert sec.centroid_y == pytest.approx(0.0, abs=1.0e-10)


def test_wall_section_2d_gross_Iz_close_to_theory():
    """Gross I_z ~ t_w * L_w^3 / 12."""
    unc, conf, steel = _materials()
    L_w, t_w = 4.0, 0.30
    sec = wall_section_2d(
        L_w=L_w, t_w=t_w, L_be=0.5,
        web_concrete=unc, boundary_concrete=conf,
        rebar_material=steel,
        n_web_fibers=40, n_be_fibers=10,
    )
    I_theory = t_w * L_w ** 3 / 12.0
    # 0.2% error allowed for discretization
    assert sec.gross_Iz == pytest.approx(I_theory, rel=2.0e-3)


def test_wall_section_2d_rejects_invalid():
    unc, conf, steel = _materials()
    with pytest.raises(ValueError, match="L_w"):
        wall_section_2d(L_w=-1.0, t_w=0.30, L_be=0.5,
                        web_concrete=unc, boundary_concrete=conf,
                        rebar_material=steel)
    with pytest.raises(ValueError, match="L_be"):
        wall_section_2d(L_w=2.0, t_w=0.30, L_be=2.5,
                        web_concrete=unc, boundary_concrete=conf,
                        rebar_material=steel)
    with pytest.raises(ValueError, match="t_w"):
        wall_section_2d(L_w=4.0, t_w=-0.30, L_be=0.5,
                        web_concrete=unc, boundary_concrete=conf,
                        rebar_material=steel)


def test_wall_section_2d_fibers_within_bounds():
    """All fibers are within [-L_w/2, +L_w/2]."""
    unc, conf, steel = _materials()
    L_w = 4.0
    sec = wall_section_2d(
        L_w=L_w, t_w=0.30, L_be=0.5,
        web_concrete=unc, boundary_concrete=conf,
        rebar_material=steel,
    )
    y_min = min(f.y for f in sec.fibers)
    y_max = max(f.y for f in sec.fibers)
    assert y_min >= -L_w / 2.0 - 1.0e-12
    assert y_max <= +L_w / 2.0 + 1.0e-12


# ============================================================ 3D wall sections

def test_t_wall_section_3d_gross_area():
    """T-section area = web_area + flange_area."""
    unc, conf, steel = _materials()
    sec = t_wall_section_3d(
        web_length=4.0, web_thickness=0.30,
        flange_length=3.0, flange_thickness=0.25,
        L_be=0.5,
        web_concrete=unc, boundary_concrete=conf,
        rebar_material=steel,
    )
    expected = 4.0 * 0.30 + 3.0 * 0.25
    assert sec.gross_area == pytest.approx(expected, rel=1.0e-12)


def test_i_wall_section_3d_gross_area():
    """I-section: web + 2 flanges."""
    unc, conf, steel = _materials()
    sec = i_wall_section_3d(
        web_length=4.0, web_thickness=0.30,
        flange_length=2.0, flange_thickness=0.40,
        L_be=0.0,
        web_concrete=unc, boundary_concrete=conf,
        rebar_material=steel,
    )
    expected = 4.0 * 0.30 + 2.0 * (2.0 * 0.40)
    assert sec.gross_area == pytest.approx(expected, rel=1.0e-12)


def test_u_wall_section_3d_gross_area():
    """U-section: web + 2 flange returns."""
    unc, conf, steel = _materials()
    sec = u_wall_section_3d(
        web_length=4.0, web_thickness=0.30,
        flange_length=2.0, flange_thickness=0.30,
        L_be=0.5,
        web_concrete=unc, boundary_concrete=conf,
        rebar_material=steel,
    )
    expected = 4.0 * 0.30 + 2.0 * (2.0 * 0.30)
    assert sec.gross_area == pytest.approx(expected, rel=1.0e-12)


def test_l_wall_section_3d_gross_area_approx():
    """L-section: leg 1 + leg 2 (slight overlap at corner is OK)."""
    unc, conf, steel = _materials()
    sec = l_wall_section_3d(
        leg1_length=4.0, leg1_thickness=0.30,
        leg2_length=3.0, leg2_thickness=0.25,
        L_be=0.5,
        web_concrete=unc, boundary_concrete=conf,
        rebar_material=steel,
    )
    # Both legs separately = 4*0.30 + 3*0.25 (no corner overlap in our layout)
    expected = 4.0 * 0.30 + 3.0 * 0.25
    assert sec.gross_area == pytest.approx(expected, rel=0.01)


# ============================================================ shear utilities

def test_wall_shear_area_rectangular():
    """A_v = (5/6) * L_w * t_w by default."""
    A_v = wall_shear_area(L_w=4.0, t_w=0.30)
    assert A_v == pytest.approx((5.0 / 6.0) * 4.0 * 0.30, rel=1.0e-12)


def test_wall_lateral_stiffness_tall_wall_flexure_dominant():
    """For H/L_w >> 1, shear contribution is small."""
    E = 30.0e9
    G = E / (2.0 * 1.2)
    res = wall_lateral_stiffness(L_w=4.0, t_w=0.30, H=30.0, E=E, G=G)
    assert res["alpha_flex"] > 0.9
    assert res["alpha_shear"] < 0.1


def test_wall_lateral_stiffness_squat_wall_shear_dominant_fraction():
    """For H/L_w ~ 1, shear is appreciable (> 30%)."""
    E = 30.0e9
    G = E / (2.0 * 1.2)
    res = wall_lateral_stiffness(L_w=4.0, t_w=0.30, H=4.0, E=E, G=G)
    assert res["alpha_shear"] > 0.30


def test_wall_lateral_stiffness_cracked_reduces_k():
    """Cracked-section factor reduces lateral stiffness."""
    E = 30.0e9
    G = E / (2.0 * 1.2)
    uncracked = wall_lateral_stiffness(L_w=4.0, t_w=0.30, H=30.0,
                                        E=E, G=G)
    cracked = wall_lateral_stiffness(L_w=4.0, t_w=0.30, H=30.0,
                                      E=E, G=G,
                                      I_eff_factor=0.35,
                                      A_v_eff_factor=0.5)
    assert cracked["k_lat"] < uncracked["k_lat"]


def test_aci318_cracked_factors_wall():
    f = aci318_cracked_factors("wall_cracked")
    assert f.I_eff_over_I_g == 0.35
    assert "ACI 318" in f.code


def test_aci318_cracked_factors_unknown_raises():
    with pytest.raises(ValueError, match="unknown"):
        aci318_cracked_factors("unknown_member")


def test_asce41_wall_factors_flexure():
    f = asce41_wall_factors(flexure_or_shear="flexure")
    assert f.I_eff_over_I_g == 0.5
    assert f.A_v_eff_over_A_g == 0.4
    assert "ASCE 41" in f.code


def test_asce41_wall_factors_shear():
    f = asce41_wall_factors(flexure_or_shear="shear")
    assert f.I_eff_over_I_g == 0.8
    assert f.A_v_eff_over_A_g == 1.0


def test_wall_base_shear_spring_stiffness():
    """K_v = G * A_v / H."""
    G = 10.0e9
    L_w, t_w, H = 4.0, 0.30, 3.0
    K = wall_base_shear_spring_stiffness(L_w=L_w, t_w=t_w, H=H, G=G)
    A_v = (5.0 / 6.0) * L_w * t_w
    assert K == pytest.approx(G * A_v / H, rel=1.0e-12)


# ============================================================ coupling beam

def _two_wall_model():
    """Two-wall planar setup with optional coupling at the top."""
    mat = ElasticIsotropic(1, E=30.0e9, nu=0.2, rho=2400.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    A_wall = 4.0 * 0.30
    Iz_wall = 0.30 * 4.0 ** 3 / 12.0
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 0.0, 3.0)
    m.add_element(BeamColumn2D(1, (1, 2), mat, A_wall, Iz_wall))
    m.fix(1, [1, 1, 1])
    m.add_node(3, 8.0, 0.0)
    m.add_node(4, 8.0, 3.0)
    m.add_element(BeamColumn2D(2, (3, 4), mat, A_wall, Iz_wall))
    m.fix(3, [1, 1, 1])
    return m, mat


def test_coupling_beam_face_nodes_at_correct_positions():
    m, mat = _two_wall_model()
    r = add_coupling_beam_2d(
        m, centroid_node_1=2, centroid_node_2=4,
        L_w1=4.0, L_w2=4.0, material=mat,
        next_node_tag=5, next_element_tag=3,
        A=0.25 * 0.50, Iz=0.25 * 0.50 ** 3 / 12.0,
    )
    assert m.node(r.face_node_1).coords[0] == pytest.approx(2.0)
    assert m.node(r.face_node_2).coords[0] == pytest.approx(6.0)


def test_coupling_beam_increases_lateral_stiffness():
    """Lateral stiffness with coupling > without (un-coupled walls).

    The coupling beam transfers a portion of wall 1's lateral load
    to wall 2 through axial + flexure of the beam plus rigid offsets.
    """
    m, mat = _two_wall_model()
    add_coupling_beam_2d(
        m, centroid_node_1=2, centroid_node_2=4,
        L_w1=4.0, L_w2=4.0, material=mat,
        next_node_tag=5, next_element_tag=3,
        A=0.25 * 0.50, Iz=0.25 * 0.50 ** 3 / 12.0,
    )
    m.add_nodal_load(2, [1000.0e3, 0.0, 0.0])
    LinearStaticAnalysis(m).run()
    u_coupled = m.node(2).disp[0]
    # Without coupling, single wall: u = F * H^3 / (3 EI)
    A_wall = 4.0 * 0.30
    Iz_wall = 0.30 * 4.0 ** 3 / 12.0
    H = 3.0
    u_solo = 1000.0e3 * H ** 3 / (3.0 * 30.0e9 * Iz_wall)
    assert u_coupled < u_solo


def test_coupling_beam_couples_wall_responses():
    """With coupling, both walls deflect together (wall 2 gets pulled)."""
    m, mat = _two_wall_model()
    add_coupling_beam_2d(
        m, centroid_node_1=2, centroid_node_2=4,
        L_w1=4.0, L_w2=4.0, material=mat,
        next_node_tag=5, next_element_tag=3,
        A=0.25 * 0.50, Iz=0.25 * 0.50 ** 3 / 12.0,
    )
    m.add_nodal_load(2, [1000.0e3, 0.0, 0.0])
    LinearStaticAnalysis(m).run()
    u1 = m.node(2).disp[0]
    u2 = m.node(4).disp[0]
    assert abs(u2) > 0.05 * abs(u1)        # wall 2 follows


def test_coupling_beam_rejects_overlapping_walls():
    """If walls overlap (face 2 left of face 1), raise."""
    m, mat = _two_wall_model()
    # Walls are at x=0 and x=8; with L_w1=5, L_w2=5, faces would be 2.5 and 5.5
    # That's fine. With L_w1=10, L_w2=10, face_1=5, face_2=3 -> overlap.
    with pytest.raises(ValueError, match="overlap"):
        add_coupling_beam_2d(
            m, centroid_node_1=2, centroid_node_2=4,
            L_w1=10.0, L_w2=10.0, material=mat,
            next_node_tag=5, next_element_tag=3,
            A=0.25, Iz=0.001,
        )


def test_coupling_beam_rejects_non_aligned_walls():
    """Centroid nodes must be at the same elevation."""
    mat = ElasticIsotropic(1, E=30.0e9, nu=0.2, rho=2400.0)
    m = Model(ndm=2, ndf=3)
    m.add_material(mat)
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 0.0, 3.0)
    m.add_node(3, 8.0, 0.0)
    m.add_node(4, 8.0, 4.0)        # different elevation!
    with pytest.raises(ValueError, match="elevation"):
        add_coupling_beam_2d(
            m, centroid_node_1=2, centroid_node_2=4,
            L_w1=4.0, L_w2=4.0, material=mat,
            next_node_tag=5, next_element_tag=3,
            A=0.25, Iz=0.001,
        )
