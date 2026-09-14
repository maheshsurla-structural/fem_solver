"""Moving-load time-history / VBI (bridge plan T2.1).

Validates the moving-force dynamic analysis: consistent static distribution
(peak = PL³/48EI), the quasi-static low-speed limit (DAF → 1), and the growth
of the dynamic amplification factor with speed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from femsolver import (BeamColumn2D, EigenAnalysis,  # noqa: E402
                       ElasticIsotropic, Model)
from femsolver.analysis.damping import RayleighDamping  # noqa: E402
from femsolver.bridges import Lane  # noqa: E402
from femsolver.bridges.moving_force import (MovingForceAnalysis,  # noqa: E402
                                            VehicleAxles)

E, A, I, RHO, L, N = 3.0e10, 0.5, 0.05, 2500.0, 25.0, 20


def _beam(rho=RHO):
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=E, nu=0.2, rho=rho)
    m.add_material(mat)
    for i in range(N + 1):
        m.add_node(i + 1, i * L / N, 0.0)
    for i in range(N):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, I))
    m.fix(1, [1, 1, 0])
    m.fix(N + 1, [0, 1, 0])
    return m


def _lane(m):
    return Lane(node_tags=list(range(1, N + 2)),
               stations=[i * L / N for i in range(N + 1)],
               load_dof=1, gravity_sign=-1.0)


def _damping(m):
    w1 = 2 * np.pi * EigenAnalysis(m, num_modes=2).run()["frequencies_hz"][0]
    return RayleighDamping.from_modes(w1, 0.02, 3 * w1, 0.02)


# ---------------------------------------------------------- vehicle
def test_vehicle_axles_validation():
    with pytest.raises(ValueError):
        VehicleAxles(axle_loads=[1.0, 2.0], axle_offsets=[0.0])
    v = VehicleAxles(axle_loads=[35e3, 145e3, 145e3],
                     axle_offsets=[0.0, 4.3, 8.6])
    assert v.length == pytest.approx(8.6)


# ------------------------------------------------- static distribution
def test_static_pass_matches_closed_form():
    m = _beam()
    mid = N // 2 + 1
    P = 1.0e5
    res = MovingForceAnalysis(m, _lane(m), VehicleAxles([P], [0.0]), 20.0,
                              track=(mid, 1), damping=_damping(m)).run()
    # the quasi-static peak (force near midspan) = P L³ / 48 EI
    assert res["peak_static"] == pytest.approx(P * L ** 3 / (48 * E * I),
                                               rel=1e-3)


# ------------------------------------------------- dynamic amplification
def test_low_speed_is_quasi_static():
    m = _beam()
    mid = N // 2 + 1
    res = MovingForceAnalysis(m, _lane(m), VehicleAxles([1e5], [0.0]), 1.0,
                              track=(mid, 1), damping=_damping(m)).run()
    assert res["DAF"] == pytest.approx(1.0, abs=0.02)


def test_daf_grows_with_speed():
    mid = N // 2 + 1
    veh = VehicleAxles([1e5], [0.0])
    daf = []
    for v in (5.0, 40.0):
        m = _beam()
        res = MovingForceAnalysis(m, _lane(m), veh, v, track=(mid, 1),
                                  damping=_damping(m),
                                  free_vibration_time=0.3).run()
        daf.append(res["DAF"])
    assert daf[0] == pytest.approx(1.0, abs=0.05)      # slow → quasi-static
    assert daf[1] > 1.1                                 # fast → amplified
    assert daf[1] > daf[0]


def test_multi_axle_train_runs():
    m = _beam()
    mid = N // 2 + 1
    truck = VehicleAxles([35e3, 145e3, 145e3], [0.0, 4.3, 8.6])
    res = MovingForceAnalysis(m, _lane(m), truck, 25.0, track=(mid, 1),
                              damping=_damping(m)).run()
    assert len(res["times"]) == len(res["dynamic_disp"])
    assert res["peak_dynamic"] > 0.0 and res["DAF"] > 0.0


# --------------------------------------------------------------- guards
def test_rejects_3d_model():
    m = Model(ndm=3, ndf=6)
    with pytest.raises(ValueError, match="2-D"):
        MovingForceAnalysis(m, Lane([1, 2, 3]), VehicleAxles([1.0], [0.0]),
                            10.0, track=(1, 2))


def test_zero_mass_raises():
    m = _beam(rho=0.0)
    mid = N // 2 + 1
    with pytest.raises(ValueError, match="density"):
        MovingForceAnalysis(m, _lane(m), VehicleAxles([1e5], [0.0]), 10.0,
                            track=(mid, 1)).run()
