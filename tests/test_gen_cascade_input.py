import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('gen', Path(__file__).resolve().parents[1] / 'scripts' / 'gen_cascade_data.py')
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

SIZE = (0.4, 0.3, 0.25)
TRUTH = {'a': (np.array([0.0, 0.0, 0.125]), np.array([1.0, 0, 0, 0])),
         'b': (np.array([0.0, 0.0, 0.375]), np.array([1.0, 0, 0, 0]))}


class Seen:
    def __init__(self, g):
        self.g = g

    def graph(self):
        return self.g


def test_camera_input_comes_from_the_look_and_never_from_the_truth():
    marker = object()
    called = []
    g = gen.belief_graph('camera', None, SIZE, 0.0, np.random.default_rng(0), lambda: called.append(1) or Seen(marker))
    assert g is marker and called == [1]


def test_truth_with_noise_stays_as_before():
    g = gen.belief_graph('truth_noise', TRUTH, SIZE, 0.0, np.random.default_rng(0), None)
    assert set(g.boxes) == {'a', 'b'}


def test_an_unknown_input_is_refused():
    with pytest.raises(ValueError):
        gen.belief_graph('lidar', TRUTH, SIZE, 0.0, np.random.default_rng(0), None)
