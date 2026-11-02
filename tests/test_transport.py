import math

import numpy as np
import pytest

from shinai_pi.config import load_cell
from shinai_pi.transport import G, sample_transport

CELL = load_cell()


def test_same_seed_gives_same_profile():
    a = sample_transport(CELL.transport, CELL.layout.layers, seed=7)
    b = sample_transport(CELL.transport, CELL.layout.layers, seed=7)
    assert a == b
    assert np.allclose(a.effective_gravity(1.0), b.effective_gravity(1.0))


def test_parameters_stay_within_configured_ranges():
    t = CELL.transport
    for seed in range(50):
        p = sample_transport(t, CELL.layout.layers, seed)
        assert t.brake_decel[0] <= p.brake_decel <= t.brake_decel[1]
        assert t.lateral_amp[0] <= p.lateral_amp <= t.lateral_amp[1]
        assert t.wrap_failure_layers[0] <= len(p.loose_layers) <= t.wrap_failure_layers[1]
        assert max(p.loose_layers) == CELL.layout.layers - 1     # film fails from the top


def test_gravity_is_normal_before_and_after_transport():
    p = sample_transport(CELL.transport, CELL.layout.layers, seed=1)
    assert np.allclose(p.effective_gravity(-0.1), [0, 0, -G])
    assert np.allclose(p.effective_gravity(p.duration + 0.1), [0, 0, -G])


def test_braking_pushes_load_forward_with_configured_peak():
    p = sample_transport(CELL.transport, CELL.layout.layers, seed=3)
    ts = np.linspace(0, p.duration, 3000)
    horiz = np.array([p.effective_gravity(t)[:2] for t in ts])
    along = horiz @ np.array([math.cos(p.heading), math.sin(p.heading)])
    # Braking acceleration of the truck is opposite to travel; the load is pushed forward (+heading).
    assert along.max() == pytest.approx(p.brake_decel, rel=0.05)
    assert along.min() > -0.5 * p.brake_decel


def test_vertical_component_stays_downward():
    for seed in range(20):
        p = sample_transport(CELL.transport, CELL.layout.layers, seed)
        for t in np.linspace(0, p.duration, 500):
            assert p.effective_gravity(t)[2] < -0.5 * G
