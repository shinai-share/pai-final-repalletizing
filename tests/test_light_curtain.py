"""Light-curtain simulation and the scripted operator reset, without starting Isaac Sim."""
import importlib

import pytest

from shinai_pi.config import load_cell


@pytest.fixture()
def backend_module():
    # isaac_backend imports the runtime module, which needs Kit only when instantiated.
    return importlib.import_module('shinai_pi.sim.isaac_backend')


class FakeRuntime:
    def __init__(self):
        self.cell = load_cell()
        self.sim_time = 0.0


def make(mod, keys):
    rt = FakeRuntime()
    return mod.IsaacBackend(rt, mod.PersonScript(keys)), rt


def test_person_script_interpolates_and_holds(backend_module):
    p = backend_module.PersonScript([(0, 0, 0), (10, 10, 0)])
    assert p.position(5) == (5, 0)
    assert p.position(20) == (10, 0)
    assert backend_module.PersonScript().position(3) is None


def test_curtain_breaks_inside_the_guarded_zone_and_reset_waits_two_seconds(backend_module):
    be, rt = make(backend_module, [(0, -3.4, -2.4), (10, -3.4, -2.4), (13, -1.0, -1.35), (17, -1.0, -1.35), (20, -3.4, -2.4)])
    rt.sim_time = 5.0
    assert be.curtain_clear()
    rt.sim_time = 15.0
    assert not be.curtain_clear()
    assert be.operator_reset_requested() is None
    rt.sim_time = 20.0
    assert be.curtain_clear()
    assert be.operator_reset_requested() is None          # clear for less than two seconds
    rt.sim_time = 22.5
    assert be.operator_reset_requested() == '作業者'
