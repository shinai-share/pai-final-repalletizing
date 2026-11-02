"""Each case's contents differ: mass and centre of mass are drawn per case, per load (seed)."""
import numpy as np

from shinai_pi.config import load_cell
from shinai_pi.layout import initial_cases
from shinai_pi.variation import vary_load

CELL = load_cell()
SKU = CELL.sku('BEV-500x24')


def test_variation_is_reproducible_and_stays_in_the_configured_ranges():
    spawns = initial_cases(CELL, SKU)
    a, b = vary_load(spawns, CELL.variation, seed=3), vary_load(spawns, CELL.variation, seed=3)
    assert a == b
    v = CELL.variation
    coms = np.array([s.com for s in a])
    masses = np.array([s.mass for s in a])
    assert np.all(np.abs(coms[:, :2]) <= v.com_xy + 1e-12) and np.all(np.abs(coms[:, 2]) <= v.com_z + 1e-12)
    assert np.all((masses >= v.mass[0]) & (masses <= v.mass[1]))
    assert np.std(coms[:, 0]) > v.com_xy / 4            # the contents really are uneven, not all centred


def test_different_loads_differ_and_positions_are_untouched():
    spawns = initial_cases(CELL, SKU)
    a, b = vary_load(spawns, CELL.variation, seed=3), vary_load(spawns, CELL.variation, seed=4)
    assert [s.com for s in a] != [s.com for s in b]
    assert [s.pos for s in a] == [s.pos for s in spawns]


def test_without_variation_cases_are_uniform_and_nominal():
    s = initial_cases(CELL, SKU)[0]
    assert s.com == (0.0, 0.0, 0.0) and s.mass is None
