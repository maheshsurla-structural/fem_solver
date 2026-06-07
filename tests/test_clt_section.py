"""Phase D.1.2 tests -- CLT panel section."""
from __future__ import annotations

import pytest

from femsolver.materials.timber import get_ec5_class, get_nds_timber
from femsolver.sections import CLTLayer, CLTSection


def _make_5ply_C24_100mm():
    """Standard 5-ply 100mm CLT panel from C24, 20mm per ply."""
    c24 = get_ec5_class("C24")
    return CLTSection(
        layers=[
            CLTLayer(thickness=0.020, material=c24, angle_deg=0),
            CLTLayer(thickness=0.020, material=c24, angle_deg=90),
            CLTLayer(thickness=0.020, material=c24, angle_deg=0),
            CLTLayer(thickness=0.020, material=c24, angle_deg=90),
            CLTLayer(thickness=0.020, material=c24, angle_deg=0),
        ],
        name="100mm-5ply-C24",
    )


# ============================================================ CLTLayer

class TestCLTLayer:
    def test_basic(self):
        c24 = get_ec5_class("C24")
        l = CLTLayer(thickness=0.02, material=c24, angle_deg=0)
        assert l.thickness == 0.02
        assert l.material is c24
        assert l.angle_deg == 0

    def test_negative_thickness_rejected(self):
        c24 = get_ec5_class("C24")
        with pytest.raises(ValueError, match="thickness"):
            CLTLayer(thickness=-0.01, material=c24)

    def test_oblique_angle_rejected(self):
        """CLT in practice only uses 0/90."""
        c24 = get_ec5_class("C24")
        with pytest.raises(ValueError, match="angle"):
            CLTLayer(thickness=0.02, material=c24, angle_deg=45)


# ============================================================ basic properties

class TestCLTSectionBasic:
    def test_total_thickness(self):
        panel = _make_5ply_C24_100mm()
        assert panel.total_thickness == pytest.approx(0.100, rel=1e-12)

    def test_n_layers(self):
        panel = _make_5ply_C24_100mm()
        assert panel.n_layers == 5

    def test_symmetric_layup_detected(self):
        panel = _make_5ply_C24_100mm()
        assert panel.is_symmetric is True

    def test_asymmetric_layup_detected(self):
        """A layup where TOP and BOTTOM layers differ in thickness
        (not mirror-symmetric) should return False."""
        c24 = get_ec5_class("C24")
        panel = CLTSection(
            layers=[
                CLTLayer(thickness=0.030, material=c24, angle_deg=0),    # thicker top
                CLTLayer(thickness=0.020, material=c24, angle_deg=90),
                CLTLayer(thickness=0.020, material=c24, angle_deg=0),    # thinner bottom
            ],
        )
        assert panel.is_symmetric is False

    def test_mass_per_area(self):
        """5 x 20mm x 420 kg/m^3 = 42 kg/m^2."""
        panel = _make_5ply_C24_100mm()
        assert panel.mass_per_area() == pytest.approx(0.100 * 420, rel=1e-9)

    def test_empty_layers_rejected(self):
        with pytest.raises(ValueError):
            CLTSection(layers=[])


# ============================================================ stiffness

class TestCLTStiffness:
    """Hand calc for 5-ply 100mm C24 panel:
    Layers 1, 3, 5: 0° (parallel to strong), E_0 = 11 GPa, t = 20mm
    Layers 2, 4: 90°, E_90 = 0.37 GPa, t = 20mm

    NA at mid-thickness (50mm) by symmetry.

    EI strong:
    - 0° layers: 3 layers contribute E_0 * (b*t^3/12 + b*t*d^2)
      * Layer 1 (d = -40mm): 11e9 * (0.020^3/12 + 0.020 * 0.0016) = 359,333
      * Layer 3 (d = 0): 7,333
      * Layer 5 (d = +40mm): 359,333
    - 90° layers: 2 layers at E_90 = 0.37 GPa
      * Layer 2 (d = -20mm): 0.37e9 * (0.020^3/12 + 0.020 * 0.0004) = 3,206
      * Layer 4 (d = +20mm): 3,206
    - Total EI = 732,411 N.m^2/m
    """

    def test_strong_axis_EI_hand_calc(self):
        panel = _make_5ply_C24_100mm()
        EI = panel.EI_eff_per_width(strong_axis=True)
        # Hand calc above: 732,411 N.m^2
        assert EI == pytest.approx(732_411, rel=0.01)

    def test_strong_axis_EA_hand_calc(self):
        """EA = sum of E*t over layers.
        3 layers at E=11GPa, t=0.020: 3 * 11e9 * 0.020 = 660e6
        2 layers at E=0.37GPa: 2 * 0.37e9 * 0.020 = 14.8e6
        Total = 674.8e6 N/m"""
        panel = _make_5ply_C24_100mm()
        EA = panel.EA_per_width(strong_axis=True)
        assert EA == pytest.approx(674.8e6, rel=1e-3)

    def test_neutral_axis_at_midheight_for_symmetric(self):
        panel = _make_5ply_C24_100mm()
        y_NA = panel.neutral_axis_from_top(strong_axis=True)
        assert y_NA == pytest.approx(0.050, abs=1e-9)

    def test_weak_axis_smaller_EI_than_strong(self):
        """For 5-ply with 3 strong and 2 cross layers, strong-axis EI
        should be much larger than weak-axis EI."""
        panel = _make_5ply_C24_100mm()
        EI_strong = panel.EI_eff_per_width(strong_axis=True)
        EI_weak = panel.EI_eff_per_width(strong_axis=False)
        assert EI_strong > EI_weak
        # Ratio typically 3-4x for 5-ply panels
        assert 2.5 < EI_strong / EI_weak < 5.0

    def test_E_0_E_90_ratio_drives_anisotropy(self):
        """If we set E_90 = E_0 (hypothetical), the panel becomes
        isotropic and strong/weak EI should be equal."""
        from femsolver.materials.timber import TimberMaterial
        iso = TimberMaterial(
            name="iso", species="iso", grade="iso", code="EC5",
            E_0_mean=11e9, E_90_mean=11e9, G_mean=0.69e9,
            f_b_k=24e6, f_t_0_k=14e6, f_t_90_k=0.4e6,
            f_c_0_k=21e6, f_c_90_k=2.5e6, f_v_k=4e6,
            density_k=350, density_mean=420,
        )
        panel = CLTSection(
            layers=[
                CLTLayer(thickness=0.020, material=iso, angle_deg=0),
                CLTLayer(thickness=0.020, material=iso, angle_deg=90),
                CLTLayer(thickness=0.020, material=iso, angle_deg=0),
                CLTLayer(thickness=0.020, material=iso, angle_deg=90),
                CLTLayer(thickness=0.020, material=iso, angle_deg=0),
            ],
        )
        EI_s = panel.EI_eff_per_width(strong_axis=True)
        EI_w = panel.EI_eff_per_width(strong_axis=False)
        assert EI_s == pytest.approx(EI_w, rel=1e-9)
        # Both should equal isotropic plate: E*b*h^3/12 = 11e9*0.1^3/12 = 916,667
        assert EI_s == pytest.approx(11e9 * 0.1 ** 3 / 12.0, rel=1e-9)


# ============================================================ gamma method

class TestGammaMethod:
    def test_gamma_reduces_EI(self):
        """Rolling-shear softening of cross layers gives γ < 1 for
        the outer load-carrying layers -> EI_eff < EI_full."""
        panel = _make_5ply_C24_100mm()
        gm = panel.gamma_method(span=5.0, strong_axis=True)
        assert gm["EI_eff"] <= gm["EI_full"]
        # For 5m span 100mm panel: ~2-5% reduction typical
        ratio = gm["EI_eff"] / gm["EI_full"]
        assert 0.95 < ratio < 1.0

    def test_gamma_smaller_at_shorter_spans(self):
        """Smaller span -> more rolling shear softening -> smaller γ."""
        panel = _make_5ply_C24_100mm()
        gm_short = panel.gamma_method(span=2.0)
        gm_long = panel.gamma_method(span=10.0)
        # Average γ at short span should be < at long span
        avg_short = sum(gm_short["gammas"]) / len(gm_short["gammas"])
        avg_long = sum(gm_long["gammas"]) / len(gm_long["gammas"])
        assert avg_short < avg_long

    def test_gamma_one_for_long_span(self):
        """At very long spans, γ → 1 (no softening)."""
        panel = _make_5ply_C24_100mm()
        gm = panel.gamma_method(span=100.0)
        ratio = gm["EI_eff"] / gm["EI_full"]
        assert ratio > 0.999

    def test_negative_span_rejected(self):
        panel = _make_5ply_C24_100mm()
        with pytest.raises(ValueError, match="span"):
            panel.gamma_method(span=-1.0)


# ============================================================ beam-strip adapter

class TestCLTBeamStrip:
    def test_returns_unified_section(self):
        from femsolver.sections import Section
        panel = _make_5ply_C24_100mm()
        strip = panel.beam_strip(width=1.0, strong_axis=True)
        assert isinstance(strip, Section)
        # Geometry should be 1.0 wide x 0.1 thick rectangle
        assert strip.geometry.width == pytest.approx(1.0)
        assert strip.geometry.depth == pytest.approx(0.1)

    def test_strip_EI_matches_panel(self):
        """The unified Section's ElasticSection3D EIz should match
        EI_eff_per_width * width."""
        panel = _make_5ply_C24_100mm()
        strip = panel.beam_strip(width=1.0, strong_axis=True)
        es = strip.elastic_section_3d()
        EI_expected = panel.EI_eff_per_width(strong_axis=True) * 1.0
        assert es.EIz == pytest.approx(EI_expected, rel=1e-6)

    def test_strip_width_scales_EI(self):
        """A 2m wide strip should have 2x the EI of a 1m strip."""
        panel = _make_5ply_C24_100mm()
        s1 = panel.beam_strip(width=1.0, strong_axis=True)
        s2 = panel.beam_strip(width=2.0, strong_axis=True)
        es1 = s1.elastic_section_3d()
        es2 = s2.elastic_section_3d()
        # Width doubles -> A doubles, I doubles -> EI doubles
        assert es2.EIz == pytest.approx(2 * es1.EIz, rel=1e-6)

    def test_weak_axis_strip(self):
        """Weak-axis strip should give weak-axis EI."""
        panel = _make_5ply_C24_100mm()
        s_strong = panel.beam_strip(strong_axis=True)
        s_weak = panel.beam_strip(strong_axis=False)
        es_s = s_strong.elastic_section_3d()
        es_w = s_weak.elastic_section_3d()
        assert es_s.EIz > es_w.EIz


# ============================================================ NDS variant

class TestNDSVariant:
    def test_5ply_DFL_panel(self):
        """5-ply CLT from Douglas Fir-Larch No. 1 instead of C24."""
        dfl = get_nds_timber("DFL-1")
        panel = CLTSection(
            layers=[
                CLTLayer(thickness=0.020, material=dfl, angle_deg=0),
                CLTLayer(thickness=0.020, material=dfl, angle_deg=90),
                CLTLayer(thickness=0.020, material=dfl, angle_deg=0),
                CLTLayer(thickness=0.020, material=dfl, angle_deg=90),
                CLTLayer(thickness=0.020, material=dfl, angle_deg=0),
            ],
            name="NDS-DFL1-100mm",
        )
        # DFL-1: E_0 = 1.7 msi = 11.72 GPa (slightly higher than C24 at 11)
        EI = panel.EI_eff_per_width(strong_axis=True)
        # Should be in similar ballpark to C24 result (732 kN.m^2/m)
        assert 700_000 < EI < 800_000


# ============================================================ asymmetric layup

class TestAsymmetricLayup:
    def test_NA_offset_for_asymmetric(self):
        """If we use different-thickness layers (asymmetric), the NA
        is no longer at mid-thickness."""
        c24 = get_ec5_class("C24")
        panel = CLTSection(
            layers=[
                CLTLayer(thickness=0.040, material=c24, angle_deg=0),    # thicker top
                CLTLayer(thickness=0.020, material=c24, angle_deg=90),
                CLTLayer(thickness=0.020, material=c24, angle_deg=0),
            ],
        )
        y_NA = panel.neutral_axis_from_top(strong_axis=True)
        # NA should be pulled toward the thicker (stiffer) top
        # Total thickness 80mm; if symmetric NA would be at 40mm.
        # Here we expect y_NA < 40mm (NA closer to top)
        assert y_NA < 0.040
        assert panel.is_symmetric is False
