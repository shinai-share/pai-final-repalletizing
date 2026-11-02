"""Naming the state a case is in after the collapse (requirements v05 R3, 11 categories). The name is
reported with every pick and counted per run, so the variety a run met can be read off it."""
import math

import numpy as np

from shinai_pi.config import load_cell
from shinai_pi.perception.obb import OBB, quat_to_matrix, yaw_matrix
from shinai_pi.pile import Box, PileBelief
from shinai_pi.states import CATEGORIES, classify

CELL = load_cell()
SKU = CELL.skus[0]
W, D, H = SKU.size
P = CELL.pallet
TOP = P.size[2]
A = P.a_center


def belief_of(boxes):
    pa = OBB.make((A[0], A[1], TOP / 2), P.size)
    b = PileBelief(floor=0.0, static=[pa])
    b.observe([Box(k, tuple(o.center), tuple(o.size), rotation=o.rotation) for k, o in boxes.items()], 'test')
    return b


def tipped(deg, about=(1, 0, 0)):
    h = math.radians(deg) / 2
    return quat_to_matrix((math.cos(h), about[0] * math.sin(h), about[1] * math.sin(h), about[2] * math.sin(h)))


def name(boxes, key):
    code, label = classify(key, belief_of(boxes), SKU, P)
    assert label == CATEGORIES[code]
    return code


def test_an_upright_case_with_nothing_on_it():
    assert name({'c': OBB.make((A[0], A[1], TOP + H / 2), SKU.size)}, 'c') == 1


def test_upright_but_tilted_a_little_and_more():
    assert name({'c': OBB.make((A[0], A[1], TOP + H / 2 + 0.02), SKU.size, tipped(10))}, 'c') == 2
    assert name({'c': OBB.make((A[0], A[1], TOP + H / 2 + 0.05), SKU.size, tipped(30))}, 'c') == 3


def test_lying_on_its_side_and_standing_on_its_end():
    side = OBB.make((A[0], A[1], TOP + D / 2), SKU.size, tipped(90))
    end = OBB.make((A[0], A[1], TOP + W / 2), SKU.size, tipped(90, about=(0, 1, 0)))
    assert name({'c': side}, 'c') == 4
    assert name({'c': end}, 'c') == 5


def test_a_case_leaning_on_its_neighbour_and_the_neighbour_it_leans_on():
    # Tipped 25 degrees towards an upright neighbour, one corner on the pallet and one on the
    # neighbour's face (the geometry of the support-graph test, at case size).
    upright = OBB.make((A[0], A[1], TOP + H / 2), SKU.size)
    lean = OBB.make((0.0, 0.0, 0.0), SKU.size, tipped(-25))
    c = lean.corners()
    lean = OBB.make((A[0], A[1] + D / 2 + 0.0005 - c[:, 1].min(), TOP - c[:, 2].min()), SKU.size, lean.rotation)
    boxes = {'upright': upright, 'lean': lean}
    assert name(boxes, 'lean') == 7
    assert name(boxes, 'upright') != 9                       # leant on, not stood on


def test_a_case_with_another_on_it():
    low = OBB.make((A[0], A[1], TOP + H / 2), SKU.size)
    high = OBB.make((A[0], A[1], TOP + 1.5 * H), SKU.size)
    assert name({'low': low, 'high': high}, 'low') == 9
    assert name({'low': low, 'high': high}, 'high') == 1


def test_a_case_on_the_floor_beside_the_pallet():
    assert name({'c': OBB.make((A[0], A[1] - P.size[1] / 2 - 0.4, H / 2), SKU.size)}, 'c') == 10


def test_a_box_twice_the_size_of_a_case_may_be_two_stuck_together():
    assert name({'c': OBB.make((A[0], A[1], TOP + H / 2), (2 * W, D, H))}, 'c') == 11
