"""Phase D.1.1 tests -- timber material reference databases."""
from __future__ import annotations

import pytest

from femsolver.materials.timber import (
    EC5_C_CLASS,
    EC5_GL_CLASS,
    EC5_T_CLASS,
    IS883_CLASSES,
    NDS_GLULAM,
    NDS_SAWN_LUMBER,
    TimberMaterial,
    get_ec5_class,
    get_is883_class,
    get_nds_timber,
    list_ec5_classes,
    list_nds_species,
)


# ============================================================ TimberMaterial dataclass

class TestTimberMaterial:
    def test_basic_construction(self):
        m = TimberMaterial(
            name="test", species="test", grade="test", code="NDS",
            E_0_mean=11e9, E_90_mean=370e6, G_mean=690e6,
            f_b_k=24e6, f_t_0_k=14e6, f_t_90_k=0.4e6,
            f_c_0_k=21e6, f_c_90_k=2.5e6, f_v_k=4e6,
            density_k=350, density_mean=420,
        )
        assert m.E_0_mean == 11e9
        assert m.E == 11e9   # alias
        assert m.G == 690e6
        assert m.density == 420

    def test_E_0_05_default(self):
        """If E_0_05 is None, it defaults to 0.67 * E_0_mean (EC5)."""
        m = TimberMaterial(
            name="test", species="test", grade="test", code="EC5",
            E_0_mean=11e9, E_90_mean=370e6, G_mean=690e6,
            f_b_k=24e6, f_t_0_k=14e6, f_t_90_k=0.4e6,
            f_c_0_k=21e6, f_c_90_k=2.5e6, f_v_k=4e6,
            density_k=350, density_mean=420,
        )
        assert m.E_0_05 == pytest.approx(0.67 * 11e9, rel=1e-12)

    def test_E_0_05_explicit(self):
        m = TimberMaterial(
            name="test", species="test", grade="test", code="NDS",
            E_0_mean=11e9, E_0_05=7.4e9,
            E_90_mean=370e6, G_mean=690e6,
            f_b_k=24e6, f_t_0_k=14e6, f_t_90_k=0.4e6,
            f_c_0_k=21e6, f_c_90_k=2.5e6, f_v_k=4e6,
            density_k=350, density_mean=420,
        )
        assert m.E_0_05 == 7.4e9

    def test_rejects_invalid_code(self):
        with pytest.raises(ValueError, match="code"):
            TimberMaterial(
                name="x", species="x", grade="x", code="BOGUS",
                E_0_mean=1, E_90_mean=1, G_mean=1,
                f_b_k=1, f_t_0_k=1, f_t_90_k=1,
                f_c_0_k=1, f_c_90_k=1, f_v_k=1,
                density_k=1, density_mean=1,
            )

    def test_rejects_negative_E(self):
        with pytest.raises(ValueError, match="positive"):
            TimberMaterial(
                name="x", species="x", grade="x", code="NDS",
                E_0_mean=-1, E_90_mean=1, G_mean=1,
                f_b_k=1, f_t_0_k=1, f_t_90_k=1,
                f_c_0_k=1, f_c_90_k=1, f_v_k=1,
                density_k=1, density_mean=1,
            )

    def test_compatible_with_section_designer(self):
        """The .E and .density attributes let TimberMaterial slot into
        the unified Section's MaterialZone like ConcreteMaterial does."""
        from femsolver.sections import (
            MaterialZone, Section, PolygonGeometry,
        )
        m = get_ec5_class("C24")
        sec = Section(
            geometry=PolygonGeometry.rectangle(0.2, 0.4),
            zones=[MaterialZone(material=m, name="C24 beam")],
            name="200x400 C24",
        )
        es = sec.elastic_section_3d()
        # EA = E * A = 11e9 * 0.08 = 8.8e8
        assert es.EA == pytest.approx(11e9 * 0.08, rel=1e-12)


# ============================================================ NDS database

class TestNDS:
    def test_dfl_ss_matches_published(self):
        """NDS-2024 Supplement Table 4A, Douglas Fir-Larch Select
        Structural: F_b = 1500 psi, F_t = 1000 psi, F_c = 1700 psi,
        F_c_perp = 625 psi, F_v = 180 psi, E = 1.9 msi, E_min = 0.690
        msi, rho = 32 pcf."""
        m = get_nds_timber("DFL-SS")
        _PSI = 6_894.757
        assert m.f_b_k == pytest.approx(1500 * _PSI, rel=1e-9)
        assert m.f_t_0_k == pytest.approx(1000 * _PSI, rel=1e-9)
        assert m.f_c_0_k == pytest.approx(1700 * _PSI, rel=1e-9)
        assert m.f_c_90_k == pytest.approx(625 * _PSI, rel=1e-9)
        assert m.f_v_k == pytest.approx(180 * _PSI, rel=1e-9)
        # E = 1.9 msi = 1.9 * 6.895 GPa = 13.10 GPa
        assert m.E_0_mean == pytest.approx(1.9e6 * _PSI, rel=1e-9)
        # E_min = 0.690 msi = 4.76 GPa
        assert m.E_0_05 == pytest.approx(0.690e6 * _PSI, rel=1e-9)
        # rho = 32 pcf = 32 * 16.018 = 512.6 kg/m^3
        assert m.density_mean == pytest.approx(32 * 16.0185, rel=1e-9)

    def test_southern_pine_higher_strength_than_dfl(self):
        """Southern Pine SS has higher F_b than DFL SS."""
        dfl = get_nds_timber("DFL-SS")
        sp = get_nds_timber("SP-SS")
        assert sp.f_b_k > dfl.f_b_k

    def test_spf_lower_strength_than_dfl(self):
        """Spruce-Pine-Fir is generally lower-strength than DFL."""
        dfl = get_nds_timber("DFL-SS")
        spf = get_nds_timber("SPF-SS")
        assert spf.f_b_k < dfl.f_b_k

    def test_glulam_higher_F_b_than_solid(self):
        """Glulam 24F-V4 has F_b = 2400 psi (much higher than DFL SS at
        1500 psi)."""
        gl = get_nds_timber("24F-V4")
        sol = get_nds_timber("DFL-SS")
        assert gl.f_b_k > sol.f_b_k

    def test_glulam_24f_v4_matches_published(self):
        """NDS Table 5A 24F-V4: F_b = 2400 psi, E = 1.8 msi."""
        m = get_nds_timber("24F-V4")
        _PSI = 6_894.757
        assert m.f_b_k == pytest.approx(2400 * _PSI, rel=1e-9)
        assert m.E_0_mean == pytest.approx(1.8e6 * _PSI, rel=1e-9)

    def test_grade_ordering_within_species(self):
        """Within Douglas Fir-Larch, Select Structural > No.1 > No.2"""
        ss = get_nds_timber("DFL-SS")
        n1 = get_nds_timber("DFL-1")
        n2 = get_nds_timber("DFL-2")
        assert ss.f_b_k > n1.f_b_k > n2.f_b_k
        assert ss.E_0_mean >= n1.E_0_mean >= n2.E_0_mean

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_nds_timber("NONEXISTENT-99")

    def test_list_species_returns_sorted(self):
        names = list_nds_species()
        assert names == sorted(names)
        assert "DFL-SS" in names
        assert len(names) >= 9   # 3 species x 3 grades minimum


# ============================================================ EC5 database

class TestEC5:
    def test_c24_matches_en_338(self):
        """EN 338:2016 Table 1, C24:
        f_m_k = 24 MPa, f_t_0_k = 14 MPa, f_t_90_k = 0.4 MPa
        f_c_0_k = 21 MPa, f_c_90_k = 2.5 MPa, f_v_k = 4.0 MPa
        E_0_mean = 11 GPa, E_0_05 = 7.4 GPa
        E_90_mean = 370 MPa, G_mean = 690 MPa
        rho_k = 350 kg/m^3, rho_mean = 420 kg/m^3"""
        m = get_ec5_class("C24")
        assert m.f_b_k == 24e6
        assert m.f_t_0_k == 14e6
        assert m.f_t_90_k == 0.4e6
        assert m.f_c_0_k == 21e6
        assert m.f_c_90_k == 2.5e6
        assert m.f_v_k == 4.0e6
        assert m.E_0_mean == 11e9
        assert m.E_0_05 == 7.4e9
        assert m.E_90_mean == 0.37e9
        assert m.G_mean == 0.69e9
        assert m.density_k == 350
        assert m.density_mean == 420

    def test_c30_higher_strength_than_c24(self):
        c24 = get_ec5_class("C24")
        c30 = get_ec5_class("C30")
        assert c30.f_b_k > c24.f_b_k
        assert c30.E_0_mean > c24.E_0_mean

    def test_gl28h_matches_en_14080(self):
        """EN 14080:2013 Table 5, GL28h homogeneous glulam:
        f_m_k = 28 MPa, E_0_mean = 12.6 GPa, rho_k = 425."""
        m = get_ec5_class("GL28h")
        assert m.f_b_k == 28e6
        assert m.E_0_mean == 12.6e9
        assert m.density_k == 425

    def test_glulam_higher_f_b_than_solid_same_E(self):
        """GL24h has same nominal f_m as C24 (24 MPa) but typically
        higher E because glulam laminations are quality-selected."""
        gl24 = get_ec5_class("GL24h")
        c24 = get_ec5_class("C24")
        assert gl24.f_b_k == c24.f_b_k   # both 24 MPa
        assert gl24.E_0_mean >= c24.E_0_mean

    def test_t_class_high_tension(self):
        """T-class (tension-graded) has high f_t_0 for its f_b."""
        t24 = get_ec5_class("T24")
        c24 = get_ec5_class("C24")
        # T24 has higher tension parallel than C24
        assert t24.f_t_0_k > c24.f_t_0_k

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_ec5_class("ZZZ")

    def test_list_classes_three_families(self):
        d = list_ec5_classes()
        assert "C" in d
        assert "GL" in d
        assert "T" in d
        assert "C24" in d["C"]
        assert "GL28h" in d["GL"]


# ============================================================ IS 883 database

class TestIS883:
    def test_class_ordering(self):
        """Class I > II > III in strength + stiffness."""
        c1 = get_is883_class("IS-Class-I")
        c2 = get_is883_class("IS-Class-II")
        c3 = get_is883_class("IS-Class-III")
        assert c1.f_b_k > c2.f_b_k > c3.f_b_k
        assert c1.E_0_mean > c2.E_0_mean > c3.E_0_mean

    def test_class_ii_matches_is_883(self):
        """IS 883:2016 Class II: f_b = 12 MPa, f_t = 8.5 MPa,
        f_c = 7.8 MPa, E = 9.8 GPa."""
        c2 = get_is883_class("IS-Class-II")
        assert c2.f_b_k == 12e6
        assert c2.f_t_0_k == 8.5e6
        assert c2.f_c_0_k == 7.8e6
        assert c2.E_0_mean == 9.8e9

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_is883_class("IS-Class-X")


# ============================================================ Cross-code sanity

class TestCrossCodeSanity:
    """Check that strength/stiffness magnitudes are reasonable across
    the three codes -- they should all be in the same order of
    magnitude for similar species."""

    def test_softwood_E_in_range(self):
        """Softwood E_0 typically 7-14 GPa across all codes."""
        for m in [
            get_nds_timber("DFL-SS"),
            get_nds_timber("SPF-SS"),
            get_ec5_class("C24"),
            get_ec5_class("C30"),
            get_is883_class("IS-Class-III"),
        ]:
            assert 5e9 < m.E_0_mean < 16e9, (
                f"{m.name} E_0 = {m.E_0_mean/1e9:.1f} GPa out of range"
            )

    def test_softwood_f_b_in_range(self):
        """Softwood f_b typically 8-30 MPa."""
        for m in [
            get_ec5_class("C24"),
            get_ec5_class("C30"),
            get_is883_class("IS-Class-II"),
            get_is883_class("IS-Class-III"),
        ]:
            assert 5e6 < m.f_b_k < 50e6, (
                f"{m.name} f_b = {m.f_b_k/1e6:.1f} MPa out of range"
            )

    def test_compression_lower_than_bending(self):
        """For solid timber, f_c_0 is typically less than 1.2 * f_b."""
        for m in [
            get_nds_timber("DFL-SS"),
            get_ec5_class("C24"),
            get_is883_class("IS-Class-II"),
        ]:
            assert m.f_c_0_to_f_b_ratio < 1.5, (
                f"{m.name} f_c/f_b = {m.f_c_0_to_f_b_ratio:.2f}"
            )

    def test_shear_much_smaller_than_bending(self):
        """f_v typically ~0.05-0.20 of f_b."""
        for m in [
            get_nds_timber("DFL-SS"),
            get_ec5_class("C24"),
            get_ec5_class("GL28h"),
        ]:
            ratio = m.f_v_k / m.f_b_k
            assert 0.05 <= ratio <= 0.25, (
                f"{m.name} f_v/f_b = {ratio:.3f}"
            )

    def test_E_90_much_smaller_than_E_0(self):
        """E_90 is typically 1/30 of E_0 for softwoods."""
        for m in [
            get_nds_timber("DFL-SS"),
            get_ec5_class("C24"),
            get_is883_class("IS-Class-I"),
        ]:
            ratio = m.E_90_mean / m.E_0_mean
            assert 0.01 < ratio < 0.10, (
                f"{m.name} E_90/E_0 = {ratio:.3f}"
            )
