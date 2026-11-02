"""Which cup stiffness holds a case through a quarter turn, and how far it droops.

The worst case for the turn: a 14 kg case lying on its side (gripped on its 0.36 x 0.22 face, so it
hangs 0.135 m from the cups) with its contents 60 mm further out. For each stiffness, grip with the
same back-off as the executive, test-lift, turn a quarter at the slow wrist speed and hold 2 s.

    .venv-isaac/Scripts/python scripts/probe_turn.py
Writes artifacts/probe_turn.json."""
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.layout import CaseSpawn  # noqa: E402
from shinai_pi.motion import limits_for, plan_move  # noqa: E402
from shinai_pi.perception.obb import quat_to_matrix  # noqa: E402
from shinai_pi.sim.runtime import IsaacRuntime  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STIFFNESS = [2000.0, 5000.0, 10000.0, 20000.0]
base = load_cell()
sku = base.sku('BEV-500x24')
R2 = math.sqrt(0.5)
rt = IsaacRuntime(base)
rows = []
for k in STIFFNESS:
    cell = dataclasses.replace(base, gripper=dataclasses.replace(base.gripper, cup_tilt_stiffness=k))
    rt.cell = cell
    g = cell.gantry
    a = cell.pallet.a_center
    pos = (a[0], a[1], cell.pallet.size[2] + sku.size[1] / 2)
    # Lying on its side; contents 60 mm towards the face that will hang lowest when turned.
    rt.new_episode()
    rt.build([CaseSpawn('case_00', sku.id, pos, 0.0, 0, com=(0.0, 0.0, -0.06), mass=14.0, quat=(R2, R2, 0.0, 0.0))])
    rt.start()

    def move(goal, slow=False, settle=0.4):
        goal = np.asarray(goal, float)
        if len(goal) == 4:
            goal = np.r_[goal, rt.gantry_target[4:]]
        prof = plan_move(rt.gantry_target.copy(), goal, *limits_for(g, slow))
        for i in range(int(math.ceil(prof.duration / cell.physics.dt)) + 1):
            rt.set_gantry_target(prof.position(i * cell.physics.dt))
            rt.step(1)
        rt.step(int(settle / cell.physics.dt))

    z_face = cell.pallet.size[2] + sku.size[1] - g.origin_height
    rt.step(int(0.5 / cell.physics.dt))
    move([a[0], a[1], z_face + 0.4, 0.0])
    move([a[0], a[1], z_face + 0.005, 0.0], slow=True)
    rt.grip(True)
    rt.step(int(0.4 / cell.physics.dt))
    move([a[0], a[1], z_face + 0.4, 0.0], slow=True)
    held_lift = rt.gripped()
    turned = None
    held_turn = []
    if held_lift:
        move([a[0], a[1], z_face + 0.4, 0.0, 0.0, -math.pi / 2], slow=True, settle=2.0)
        held_turn = rt.gripped()
        _, quat = rt.truth_case_poses()['case_00']
        # Droop: angle between the case's 0.22 m edge (its local z) and vertical; 0 when fully righted.
        R = quat_to_matrix(quat)
        turned = math.degrees(math.acos(min(1.0, abs(float(R[2, 2])))))
    rows.append({'cup_tilt_stiffness': k, 'held_after_lift': bool(held_lift), 'held_after_turn': bool(held_turn),
                 'droop_deg': None if turned is None else round(turned, 2)})
    print('ROW', json.dumps(rows[-1]), flush=True)
(ROOT / 'artifacts' / 'probe_turn.json').write_text(json.dumps({'mass_kg': 14.0, 'com_case_m': [0, 0, -0.06], 'rows': rows}, indent=1),
                                                     encoding='utf-8')
rt.close()
