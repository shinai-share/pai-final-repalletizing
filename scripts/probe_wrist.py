"""Calibrate and check the wrist force sensing in Isaac Sim, and the quarter turn.

For each of several cases of known mass and centre of mass: grip it, lift it 4 cm, read the
wrench the wrist passes to the vacuum head (empty reading subtracted), and compare it with the
true weight and moment r x F. Isaac reports these joint reactions scaled by a constant; the gain
per channel is fitted on the first loads and checked on the last one, like calibrating a load cell
with known weights. Then the case is turned a quarter about the plate's long side and the sag of
the cups under its moment is measured.

    .venv-isaac/Scripts/python scripts/probe_wrist.py
Writes artifacts/probe_wrist.json."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.layout import CaseSpawn, pallet_slots  # noqa: E402
from shinai_pi.motion import limits_for, plan_move  # noqa: E402
from shinai_pi.perception.obb import quat_to_matrix  # noqa: E402
from shinai_pi.sim.runtime import IsaacRuntime  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOADS = [(11.0, (0.05, -0.03, 0.0)), (8.0, (-0.04, 0.05, 0.02)), (13.5, (0.02, 0.04, -0.02))]   # last one checks the fit
cell = load_cell()
sku = cell.sku('BEV-500x24')
g = cell.gantry
src = pallet_slots(cell, sku, cell.pallet.a_center)[4]
rt = IsaacRuntime(cell)


def move(goal, slow=False, settle=0.4):
    goal = np.asarray(goal, float)
    if len(goal) == 4:
        goal = np.r_[goal, rt.gantry_target[4:]]
    prof = plan_move(rt.gantry_target.copy(), goal, *limits_for(g, slow))
    for k in range(int(math.ceil(prof.duration / cell.physics.dt)) + 1):
        rt.set_gantry_target(prof.position(k * cell.physics.dt))
        rt.step(1)
    rt.step(int(settle / cell.physics.dt))


def reading(n=60):
    vals = []
    for _ in range(n):
        rt.step(1)
        w = rt.head_wrench()
        vals.append([w[2], w[3], w[4]])          # head frame: force z, torque x, torque y (raw)
    return np.mean(vals, axis=0)


rows = []
for mass, com in LOADS:
    rt.new_episode()
    rt.build([CaseSpawn('case_00', sku.id, src.center, 0.0, 0, com=com, mass=mass)])
    rt.start()
    top = src.center[2] + sku.size[2] / 2
    z_face = top - g.origin_height
    safe = z_face + 0.5
    rt.step(int(0.5 / cell.physics.dt))
    move([src.center[0], src.center[1], safe, 0.0])
    move([src.center[0], src.center[1], z_face + 0.06, 0.0])
    tare = reading()
    move([src.center[0], src.center[1], z_face + 0.003, 0.0], slow=True)
    rt.grip(True)
    rt.step(int(0.4 / cell.physics.dt))
    move([src.center[0], src.center[1], z_face + 0.04, 0.0], slow=True)
    raw = reading() - tare
    pos, quat = rt.truth_case_poses()['case_00']
    q_tool = rt.gantry_state()[0]
    tool = np.array([q_tool[0], q_tool[1], g.origin_height + q_tool[2]])
    r = pos + quat_to_matrix(quat) @ np.asarray(com) - tool       # tool is level and unrotated here
    weight = mass * 9.81
    true = np.array([weight, -weight * r[1], weight * r[0]])         # (F, tau_x, tau_y) of a hanging load
    move([src.center[0], src.center[1], safe, 0.0])
    move([src.center[0], src.center[1], safe, 0.0, 0.0, math.pi / 2], slow=True, settle=1.0)
    pos2, quat2 = rt.truth_case_poses()['case_00']
    turned = 2 * math.degrees(math.acos(min(1.0, abs(float(quat2[0])))))
    rows.append({'mass_kg': mass, 'com_case_m': com, 'raw': raw.tolist(), 'true': true.tolist(),
                 'held_after_turn': rt.gripped(), 'case_turned_deg': round(turned, 2),
                 'sag_deg': round(90.0 - turned, 2), 'turn_moment_Nm': round(weight * (sku.size[2] / 2), 2)})

fit = rows[:-1]
gain = [float(np.mean([r['true'][i] / r['raw'][i] for r in fit])) for i in range(3)]
check = rows[-1]
predicted = [gain[i] * check['raw'][i] for i in range(3)]
out = {'loads': rows, 'gain': [round(v, 3) for v in gain],
       'check': {'true': [round(v, 2) for v in check['true']], 'predicted': [round(v, 2) for v in predicted],
                 'error': [round(predicted[i] - check['true'][i], 3) for i in range(3)]}}
(ROOT / 'artifacts' / 'probe_wrist.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print('PROBE', json.dumps(out))
rt.close()
