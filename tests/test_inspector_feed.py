import json

import numpy as np

from shinai_pi.config import load_cell
from shinai_pi.inspector_feed import belief_snapshot, case_label
from shinai_pi.perception.voxels import VoxelMap
from shinai_pi.pile import Box, PileBelief

CELL = load_cell()


def test_labels_are_for_people():
    assert case_label('case_07') == 'ケース 7'


def test_snapshot_marks_next_pick_predicted_falls_and_unknown_floor():
    b = PileBelief(floor=0.0)
    b.observe([Box('case_00', (0, 0, 0.125), (0.4, 0.3, 0.25)), Box('case_01', (0, 0, 0.375), (0.4, 0.3, 0.25)),
               Box('case_02', (1.0, 0, 0.125), (0.4, 0.3, 0.25))], 'top')
    vm = VoxelMap((-1, -1, 0), (1.5, 1, 0.6), 0.05)
    vm.free_n[:] = 1
    vm.free_n[:5, :5, :] = 0                     # an unseen corner
    b.set_known_space(vm, ((-1, -1, 0), (1.5, 1, 0.6)))
    s = belief_snapshot(b, 12.0, next_pick='case_01', predicted=['case_02'], cell=CELL, progress={'placed': 3})
    by = {x['id']: x for x in s['boxes']}
    assert by['case_01']['state'] == 'next' and by['case_02']['state'] == 'falls'
    assert by['case_01']['label'] == 'ケース 1'
    assert len(by['case_00']['poly']) == 4
    assert s['unknown_cells'] and s['unknown_litres'] > 0
    assert s['progress'] == {'placed': 3}
    json.dumps(s, ensure_ascii=False)                      # serialisable as-is
    xs = [c[0] for c in s['unknown_cells']]
    assert max(xs) < -0.7                                  # only the unseen corner is reported


def test_snapshot_shows_where_the_mass_may_sit_what_holds_each_case_and_the_staging_tables():
    b = PileBelief(floor=0.0, com_margin=0.06)
    b.observe([Box('case_00', (0, 0, 0.125), (0.4, 0.3, 0.25)), Box('case_01', (0.17, 0, 0.375), (0.4, 0.3, 0.25))], 'top')
    s = belief_snapshot(b, 1.0, cell=CELL)
    by = {x['id']: x for x in s['boxes']}
    assert by['case_01']['com'] == [0.17, 0.0] and by['case_01']['com_margin'] == 0.06
    assert by['case_01']['stability'] == 'uncertain' and by['case_00']['stability'] == 'stable'
    assert by['case_01']['rests_on'] == ['case_00'] and by['case_00']['rests_on'] == []
    assert s['staging'] == [list(t) for t in CELL.staging.tables]
