import math

from shinai_pi.config import load_cell
from shinai_pi.evaluation.truth import classify, quat_tilt_deg, summarize
from shinai_pi.layout import CaseSpawn

CELL = load_cell()
P = CELL.pallet
SP = [CaseSpawn('a', 'x', (-0.95, 0.0, 0.254), 0, 0), CaseSpawn('b', 'x', (-0.95, 0.3, 0.474), 0, 1)]


def q_about_x(deg):
    h = math.radians(deg) / 2
    return (math.cos(h), math.sin(h), 0, 0)


def test_tilt_from_quaternion():
    assert quat_tilt_deg((1, 0, 0, 0)) == 0
    assert abs(quat_tilt_deg(q_about_x(30)) - 30) < 1e-6


def test_classification_states():
    final = {'a': ((-0.95, 0.0, 0.254), (1, 0, 0, 0)), 'b': ((-0.95, 0.30, 0.474), (1, 0, 0, 0))}
    assert summarize(classify(SP, final, P.a_center, P.size)) == {'intact': 2, 'shifted': 0, 'tilted': 0, 'fallen': 0, 'floor': 0}
    final['a'] = ((-0.90, 0.0, 0.254), (1, 0, 0, 0))
    final['b'] = ((-0.30, 0.0, 0.14), q_about_x(80))
    c = classify(SP, final, P.a_center, P.size)
    assert c['a']['state'] == 'shifted'
    assert c['b']['state'] == 'floor'


def test_box_that_dropped_a_layer_but_stayed_on_pallet_is_fallen():
    final = {'a': ((-0.95, 0.0, 0.254), (1, 0, 0, 0)), 'b': ((-0.95, 0.45, 0.30), q_about_x(20))}
    assert classify(SP, final, P.a_center, P.size)['b']['state'] == 'fallen'
