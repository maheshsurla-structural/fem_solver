"""Fiber-consistent section M-phi driver + recorders (plan P4 / G8).

Exercises :func:`fiber_section_moment_curvature` (drive a stateful fiber
section through curvature at constant axial force) and the three recorders in
:mod:`femsolver.results.recorders` used to log the run to CSV.
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from femsolver import (
    ConcreteMander,
    FiberSection2D,
    FiberSection3D,
    fiber_section_moment_curvature,
    rc_circular_column_section,
)
from femsolver.core.node import Node
from femsolver.materials.uniaxial import (
    ReinforcingSteelKinematic,
    UniaxialElastic,
    UniaxialReinforcingSteel,
)
from femsolver.results import FiberRecorder, NodeRecorder, SectionRecorder
from femsolver.sections.analysis import mander_confined_circular


# ---------------------------------------------------------------- driver

def test_elastic_mphi_matches_EI_kappa():
    """For a linear-elastic fiber section at zero axial force, M = E*Iz*kappa
    exactly (the fiber sum reproduces the closed-form elastic response)."""
    E = 30000.0
    sec = FiberSection2D.rectangular(300.0, 600.0, 40, UniaxialElastic(E))
    EI = E * sec.gross_Iz
    kaps = np.linspace(0.0, 1e-5, 12)
    res = fiber_section_moment_curvature(sec, kaps, N_target=0.0)
    assert res["converged"]
    for k, M in zip(res["kappa"], res["M"]):
        assert M == pytest.approx(EI * k, rel=1e-9, abs=1e-6)
    # zero axial force => zero axial strain at every step
    assert all(abs(e) < 1e-14 for e in res["eps_a"])


def _caltrans_section():
    D, cover, Ec, fc, eps_c0, Es, fy = (
        84.0, 2.0, 3605.0, 5.0, 0.002219, 29000.0, 68.0)
    conf = mander_confined_circular(
        fco=fc, eps_co=eps_c0, D_core=79.0, hoop_area=0.79, hoop_spacing=6.0,
        fyh=68.0, rho_long=0.0257, hoop_type="hoop", clear_spacing=5.0)
    return rc_circular_column_section(
        diameter=D, cover=cover, core_concrete=conf.to_material(Ec=Ec),
        cover_concrete=ConcreteMander(fpc=fc, eps_c0=eps_c0, Ec=Ec),
        n_bars=56, bar_area=2.25,
        steel=UniaxialReinforcingSteel(Es, fy, 95.0, 0.0075, 0.09),
        n_core_rings=8, n_cover_rings=2, n_wedges=24)


def test_axial_force_held_constant():
    """The driver holds the target axial force at every converged step while
    curvature grows (Caltrans benchmark section at P = 2400 kip compression)."""
    sec = _caltrans_section()
    kaps = np.linspace(0.0, 5e-4, 40)
    res = fiber_section_moment_curvature(sec, kaps, N_target=-2400.0)
    assert res["n_converged"] == len(kaps)
    assert all(abs(N - (-2400.0)) < 1e-6 for N in res["N"])
    # moment grows monotonically with curvature over this range
    assert all(res["M"][i + 1] >= res["M"][i] - 1.0
               for i in range(len(res["M"]) - 1))
    assert max(res["M"]) > 0.0


def test_axial_compression_raises_moment():
    """Modest axial compression raises the moment at a fixed curvature for a
    column section (P-M interaction, compression side of balanced)."""
    kap = np.array([3e-4])
    m0 = fiber_section_moment_curvature(
        _caltrans_section(), kap, N_target=0.0)["M"][0]
    mc = fiber_section_moment_curvature(
        _caltrans_section(), kap, N_target=-2400.0)["M"][0]
    assert mc > m0


def test_driver_rejects_3d_section():
    sec = FiberSection3D.rectangular(300.0, 600.0, 10, 10,
                                     UniaxialElastic(30000.0), GJ=1.0e9)
    with pytest.raises(ValueError):
        fiber_section_moment_curvature(sec, [0.0, 1e-5], N_target=0.0)


def test_cyclic_history_is_path_dependent():
    """With a kinematic steel, revisiting a curvature after a reversal gives a
    different moment (the section carries history), confirming the driver
    commits state between steps."""
    Es, fy = 29000.0, 68.0
    steel = ReinforcingSteelKinematic(Es, fy, 95.0, 0.0075, 0.09)
    sec = FiberSection2D.rectangular(20.0, 30.0, 30, steel)
    # +peak, back to zero, the return-to-zero moment is not the virgin zero
    kaps = list(np.linspace(0.0, 8e-4, 20)) + list(np.linspace(8e-4, 0.0, 20))
    res = fiber_section_moment_curvature(sec, kaps, N_target=0.0)
    assert res["converged"]
    assert abs(res["M"][-1]) > 1.0        # residual moment at kappa = 0


# ---------------------------------------------------------------- recorders

def test_section_recorder_matches_driver(tmp_path):
    sec = FiberSection2D.rectangular(300.0, 600.0, 40, UniaxialElastic(30000.0))
    rec = SectionRecorder(sec)
    kaps = np.linspace(0.0, 1e-5, 8)
    res = fiber_section_moment_curvature(sec, kaps, N_target=0.0,
                                         section_recorder=rec)
    a = rec.arrays()
    assert len(rec.rows) == len(kaps)
    assert np.allclose(a["Mz"], res["M"])
    assert np.allclose(a["kappa_z"], res["kappa"])
    # round-trips to CSV with the right header + row count
    p = tmp_path / "sec.csv"
    rec.to_csv(p)
    with open(p, newline="") as fh:
        reader = list(csv.DictReader(fh))
    assert len(reader) == len(kaps)
    assert set(["step", "eps_a", "kappa_z", "N", "Mz"]).issubset(reader[0])


def test_fiber_recorder_long_format(tmp_path):
    sec = FiberSection2D.rectangular(10.0, 20.0, 8, UniaxialElastic(1000.0))
    fr = FiberRecorder(sec)
    kaps = [0.0, 1e-4, 2e-4]
    fiber_section_moment_curvature(sec, kaps, N_target=0.0, fiber_recorder=fr)
    assert len(fr.rows) == len(kaps) * len(sec.fibers)      # tidy/long format
    # at the last step, a top fiber (y>0) is in compression (sigma < 0) under
    # positive curvature: eps_f = -y*kappa, sigma = E*eps_f
    last = [r for r in fr.rows if r["step"] == 2]
    top = max(last, key=lambda r: r["y"])
    assert top["sigma"] == pytest.approx(1000.0 * (-top["y"] * 2e-4), rel=1e-9)
    fr.to_csv(tmp_path / "fib.csv")               # writes without error


def test_node_recorder_reads_dofs():
    nd = Node(7, np.array([0.0, 0.0]), 3)
    rec = NodeRecorder(nd, dofs=[0, 1])
    nd.disp[:] = [0.1, -0.2, 0.0]
    nd.reaction[:] = [5.0, 6.0, 7.0]
    rec.record()
    nd.disp[:] = [0.3, -0.4, 0.0]
    rec.record()
    a = rec.arrays()
    assert list(a["u0"]) == [0.1, 0.3]
    assert list(a["u1"]) == [-0.2, -0.4]
    assert list(a["R0"]) == [5.0, 5.0]
    assert rec.columns == ["step", "u0", "u1", "R0", "R1"]
