"""Sprung-mass vehicle–bridge interaction (bridge plan T2.1b).

Validates the coupled Newmark VBI: static reference, the quasi-static limit,
the contact-force balance (mean = weight), consistency with the moving-force
model, and DAF growth with speed.
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
from femsolver.bridges.vbi import SprungMassVehicle, VBIAnalysis  # noqa: E402

E, A, I, RHO, L, N, G = 3.0e10, 0.5, 0.05, 2500.0, 25.0, 20, 9.80665


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


def _lane():
    return Lane(node_tags=list(range(1, N + 2)),
               stations=[i * L / N for i in range(N + 1)],
               load_dof=1, gravity_sign=-1.0)


def _damping(m):
    w1 = 2 * np.pi * EigenAnalysis(m, num_modes=2).run()["frequencies_hz"][0]
    return RayleighDamping.from_modes(w1, 0.02, 3 * w1, 0.02)


def _vehicle(ms=20000.0, fv=2.0, zeta=0.1):
    ks = ms * (2 * np.pi * fv) ** 2
    cs = 2 * zeta * np.sqrt(ks * ms)
    return SprungMassVehicle(mass=ms, stiffness=ks, damping=cs)


def test_natural_frequency():
    v = _vehicle(ms=20000.0, fv=2.0)
    assert v.natural_frequency == pytest.approx(2.0, rel=1e-6)


def test_static_reference_matches_closed_form():
    veh = _vehicle()
    m = _beam()
    res = VBIAnalysis(m, _lane(), veh, 20.0, track=(N // 2 + 1, 1),
                      bridge_damping=_damping(m)).run()
    W = veh.mass * G
    assert res["peak_static"] == pytest.approx(W * L ** 3 / (48 * E * I),
                                               rel=1e-3)


def test_low_speed_quasi_static_and_contact_equals_weight():
    veh = _vehicle()
    m = _beam()
    res = VBIAnalysis(m, _lane(), veh, 1.0, track=(N // 2 + 1, 1),
                      bridge_damping=_damping(m)).run()
    W = veh.mass * G
    cf = np.array(res["contact_force"])[:, 0]
    assert res["DAF"] == pytest.approx(1.0, abs=0.02)
    assert cf.mean() == pytest.approx(W, rel=1e-3)
    # at crawl speed the contact force barely varies from the static weight
    assert (cf.max() - cf.min()) < 0.01 * W


def test_contact_force_mean_is_weight_at_speed():
    veh = _vehicle()
    m = _beam()
    res = VBIAnalysis(m, _lane(), veh, 30.0, track=(N // 2 + 1, 1),
                      bridge_damping=_damping(m),
                      free_vibration_time=0.3).run()
    cf = np.array(res["contact_force"])[:, 0]
    assert cf.mean() == pytest.approx(veh.mass * G, rel=0.05)


def test_vbi_matches_moving_force_when_vehicle_dynamics_secondary():
    veh = _vehicle()
    W = veh.mass * G
    mid = N // 2 + 1
    m1 = _beam()
    mf = MovingForceAnalysis(m1, _lane(), VehicleAxles([W], [0.0]), 20.0,
                             track=(mid, 1), damping=_damping(m1),
                             free_vibration_time=0.5).run()
    m2 = _beam()
    vb = VBIAnalysis(m2, _lane(), veh, 20.0, track=(mid, 1),
                     bridge_damping=_damping(m2),
                     free_vibration_time=0.5).run()
    assert vb["DAF"] == pytest.approx(mf["DAF"], abs=0.05)


def test_daf_grows_with_speed():
    veh = _vehicle()
    mid = N // 2 + 1
    daf = []
    for v in (2.0, 40.0):
        m = _beam()
        daf.append(VBIAnalysis(m, _lane(), veh, v, track=(mid, 1),
                               bridge_damping=_damping(m),
                               free_vibration_time=0.3).run()["DAF"])
    assert daf[0] == pytest.approx(1.0, abs=0.05)
    assert daf[1] > daf[0] and daf[1] > 1.1


def test_multiple_vehicles_run():
    m = _beam()
    vehs = [_vehicle(), _vehicle(ms=15000.0, fv=2.5)]
    vehs[1].offset = 10.0
    res = VBIAnalysis(m, _lane(), vehs, 25.0, track=(N // 2 + 1, 1),
                      bridge_damping=_damping(m)).run()
    assert np.array(res["contact_force"]).shape[1] == 2
    assert res["DAF"] > 0.0


def test_guards():
    with pytest.raises(ValueError, match="2-D"):
        VBIAnalysis(Model(ndm=3, ndf=6), Lane([1, 2, 3]), _vehicle(), 10.0,
                    track=(1, 2))
    m = _beam(rho=0.0)
    with pytest.raises(ValueError, match="mass"):
        VBIAnalysis(m, _lane(), _vehicle(), 10.0,
                    track=(N // 2 + 1, 1)).run()
