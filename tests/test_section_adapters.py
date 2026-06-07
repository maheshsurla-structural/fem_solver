"""Phase II.6 tests -- Section adapter layer.

Verifies the lazy adapters from a unified :class:`Section`:

* ``fiber_section_2d / fiber_section_3d`` -- auto polygon discretization
* ``as_aisc_section / as_eurocode_section / as_indian_section`` -- legacy
  catalogue dataclass round-trips
* ``as_aci_concrete_section`` -- legacy ConcreteSection conversion
* ``rc_rectangular_section`` factory + ``ReinforcementLayout.from_rectangular_layers``
"""
from __future__ import annotations

import pytest

from femsolver.materials.uniaxial import (
    ConcreteKentPark,
    UniaxialBilinear,
    UniaxialElastic,
)
from femsolver.sections import (
    ReinforcementLayout,
    Section,
    aisc_section,
    eurocode_section,
    hollow_rect_section,
    indian_section,
    rc_rectangular_section,
    rectangular_section,
)


# ============================================================ ReinforcementLayout factories

class TestReinforcementLayoutFromLayers:
    def test_bottom_layer_distributed(self):
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6,
            bottom_bars=[(510e-6, "#8")] * 3,
            bottom_cover=0.04,
        )
        assert rl.n_bars == 3
        for bar in rl.bars:
            # Bottom layer at y = -h/2 + cover
            assert bar.y == pytest.approx(-0.3 + 0.04)
            assert bar.area == pytest.approx(510e-6)

    def test_bars_centered_when_single(self):
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6,
            bottom_bars=[(510e-6, "#8")],
            bottom_cover=0.04,
        )
        assert rl.bars[0].z == pytest.approx(0.0)

    def test_bars_distributed_uniformly(self):
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6,
            bottom_bars=[(510e-6, "#8")] * 3,
            bottom_cover=0.04,
        )
        z_positions = sorted(b.z for b in rl.bars)
        # 3 bars across inner width 0.3-2*0.04 = 0.22
        assert z_positions[0] == pytest.approx(-0.11)
        assert z_positions[-1] == pytest.approx(+0.11)

    def test_both_layers(self):
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6,
            bottom_bars=[(510e-6, "#8")] * 3,
            top_bars=[(285e-6, "#6")] * 2,
        )
        assert rl.n_bars == 5
        bottom = [b for b in rl.bars if b.y < 0]
        top = [b for b in rl.bars if b.y > 0]
        assert len(bottom) == 3
        assert len(top) == 2
        assert rl.total_area == pytest.approx(3 * 510e-6 + 2 * 285e-6)

    def test_side_bars_at_explicit_y(self):
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6,
            side_bars=[(200e-6, "#5", 0.1), (200e-6, "#5", -0.1)],
        )
        assert rl.n_bars == 2
        assert {b.y for b in rl.bars} == {0.1, -0.1}


# ============================================================ rc_rectangular_section

class TestRcRectangularSection:
    def test_returns_section(self):
        sec = rc_rectangular_section(b=0.3, h=0.6)
        assert isinstance(sec, Section)
        assert sec.family == "rect"
        assert sec.area == pytest.approx(0.18, rel=1e-12)

    def test_attaches_concrete_material(self):
        from femsolver.design.concrete.section import ConcreteMaterial
        cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
        sec = rc_rectangular_section(b=0.3, h=0.6, concrete=cm)
        assert sec.primary_material is cm
        assert sec.zones[0].name == "concrete"

    def test_attaches_reinforcement(self):
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6, bottom_bars=[(510e-6, "#8")] * 3,
        )
        sec = rc_rectangular_section(
            b=0.3, h=0.6, reinforcement=rl,
        )
        assert sec.reinforcement is rl
        assert sec.reinforcement.n_bars == 3

    def test_default_name(self):
        sec = rc_rectangular_section(b=0.3, h=0.6)
        assert sec.name == "RC 300x600"


# ============================================================ fiber_section_2d

class TestFiberSection2D:
    def test_total_area_matches_geometry(self):
        mat = UniaxialElastic(E=30e9)
        sec = rectangular_section(b=0.3, h=0.6)
        fs = sec.fiber_section_2d(material=mat, n_z=4, n_y=10)
        assert fs.gross_area == pytest.approx(0.18, rel=1e-10)

    def test_inertia_converges_with_discretization(self):
        """Finer fiber grid -> closer to closed form."""
        mat = UniaxialElastic(E=30e9)
        sec = rectangular_section(b=0.3, h=0.6)
        I_closed = 0.3 * 0.6**3 / 12
        fs_coarse = sec.fiber_section_2d(material=mat, n_z=2, n_y=4)
        fs_fine = sec.fiber_section_2d(material=mat, n_z=4, n_y=20)
        err_coarse = abs(fs_coarse.gross_Iz - I_closed) / I_closed
        err_fine = abs(fs_fine.gross_Iz - I_closed) / I_closed
        assert err_fine < err_coarse
        assert err_fine < 0.01   # 1% with 4x20

    def test_uses_primary_material_if_no_explicit(self):
        mat = UniaxialElastic(E=30e9)
        sec = rectangular_section(b=0.3, h=0.6, material=mat)
        fs = sec.fiber_section_2d(n_z=2, n_y=4)
        # 2x4 = 8 fibers, all using mat
        assert len(fs.fibers) == 8
        for f in fs.fibers:
            assert isinstance(f.material, UniaxialElastic)

    def test_requires_uniaxial_material(self):
        sec = rectangular_section(b=0.3, h=0.6)
        with pytest.raises(TypeError):
            sec.fiber_section_2d(material="not-a-material")

    def test_raises_without_any_material(self):
        sec = rectangular_section(b=0.3, h=0.6)
        with pytest.raises(ValueError, match="material"):
            sec.fiber_section_2d()

    def test_hollow_section_discretization_excludes_hole(self):
        """Fibers should populate only the wall of a hollow rectangle."""
        mat = UniaxialElastic(E=200e9)
        sec = hollow_rect_section(b=0.2, h=0.1, t=0.01)
        fs = sec.fiber_section_2d(material=mat, n_z=20, n_y=10)
        # Total area must match A = 0.2*0.1 - 0.18*0.08 = 0.0056
        assert fs.gross_area == pytest.approx(0.0056, rel=1e-2)

    def test_rc_section_includes_rebar_fibers(self):
        concrete = UniaxialElastic(E=30e9)
        steel = UniaxialBilinear(E=200e9, sigma_y=420e6, b=0.01)
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6,
            bottom_bars=[(510e-6, "#8")] * 3,
            top_bars=[(285e-6, "#6")] * 2,
            steel_material=steel,
        )
        sec = rc_rectangular_section(
            b=0.3, h=0.6, concrete=concrete, reinforcement=rl,
        )
        fs = sec.fiber_section_2d(material=concrete, n_z=4, n_y=10)
        # 4*10 = 40 concrete + 5 rebar
        assert len(fs.fibers) == 45

    def test_rebar_without_material_raises(self):
        concrete = UniaxialElastic(E=30e9)
        # Bars constructed without a UniaxialMaterial
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6, bottom_bars=[(510e-6, "#8")] * 3,
        )
        sec = rc_rectangular_section(
            b=0.3, h=0.6, concrete=concrete, reinforcement=rl,
        )
        with pytest.raises(ValueError, match="rebar"):
            sec.fiber_section_2d(material=concrete, n_z=4, n_y=10)


# ============================================================ fiber_section_3d

class TestFiberSection3D:
    def test_GJ_from_self_J(self):
        """GJ should derive from G_resolved * self.J."""
        # Use a high-level material with explicit E (for G resolution)
        class _Steel:
            E = 200e9
            nu = 0.3
            def get_response(self, eps):
                return 0.0, self.E
            def commit_state(self): pass
            def revert_state(self): pass
            def clone(self):
                return _Steel()
        from femsolver.materials.uniaxial.base import UniaxialMaterial
        UniaxialMaterial.register(_Steel)

        sec = rectangular_section(b=0.3, h=0.6, material=_Steel())
        fs = sec.fiber_section_3d(material=_Steel(), n_z=2, n_y=4)
        # G = 200e9 / (2 * 1.3); J of 0.3x0.6 rectangle ~ Roark formula
        expected_J = sec.J
        G_expected = 200e9 / (2 * 1.3)
        assert fs.GJ == pytest.approx(G_expected * expected_J, rel=1e-6)

    def test_explicit_GJ_override(self):
        mat = UniaxialElastic(E=200e9)
        sec = rectangular_section(b=0.3, h=0.6)
        fs = sec.fiber_section_3d(material=mat, n_z=2, n_y=4, GJ=1.5e8)
        assert fs.GJ == pytest.approx(1.5e8, rel=1e-12)


# ============================================================ legacy adapters

class TestLegacyAdapters:
    def test_as_aisc_section_round_trip(self):
        sec = aisc_section("W14x90")
        ss = sec.as_aisc_section()
        assert ss.designation == "W14x90"

    def test_as_aisc_requires_catalogue_ref(self):
        sec = rectangular_section(b=0.3, h=0.6)
        with pytest.raises(ValueError, match="catalogue_ref"):
            sec.as_aisc_section()

    def test_as_eurocode_section_round_trip(self):
        sec = eurocode_section("IPE 300")
        sp = sec.as_eurocode_section()
        assert sp.name == "IPE 300"
        assert sp.family == "IPE"

    def test_as_eurocode_for_HEA(self):
        sec = eurocode_section("HEA 200")
        sp = sec.as_eurocode_section()
        assert sp.family == "HEA"

    def test_as_indian_section_round_trip(self):
        sec = indian_section("ISMB 400")
        sp = sec.as_indian_section()
        assert sp.name == "ISMB 400"
        assert sp.family == "ISMB"

    def test_as_aci_concrete_section_round_trip(self):
        from femsolver.design.concrete.section import (
            ConcreteMaterial,
            ConcreteSection,
        )
        cm = ConcreteMaterial(fc_prime=30e6, fy=420e6)
        rl = ReinforcementLayout.from_rectangular_layers(
            b=0.3, h=0.6,
            bottom_bars=[(510e-6, "#8")] * 3,
            top_bars=[(285e-6, "#6")] * 2,
            bottom_cover=0.040, top_cover=0.040,
        )
        sec = rc_rectangular_section(
            b=0.3, h=0.6, concrete=cm, reinforcement=rl,
        )
        aci = sec.as_aci_concrete_section()
        assert isinstance(aci, ConcreteSection)
        assert aci.b == pytest.approx(0.3, rel=1e-12)
        assert aci.h == pytest.approx(0.6, rel=1e-12)
        assert aci.rebar.bottom_bars == ("#8", "#8", "#8")
        assert aci.rebar.top_bars == ("#6", "#6")
        assert aci.rebar.bottom_cover == pytest.approx(0.040, rel=1e-9)

    def test_as_aci_requires_rect_family(self):
        sec = aisc_section("W14x90")
        with pytest.raises(ValueError, match="rect"):
            sec.as_aci_concrete_section()

    def test_as_aci_requires_concrete_material(self):
        sec = rectangular_section(b=0.3, h=0.6)
        with pytest.raises(ValueError, match="ConcreteMaterial"):
            sec.as_aci_concrete_section()
