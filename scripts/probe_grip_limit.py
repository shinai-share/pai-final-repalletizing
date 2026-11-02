"""Where does the simulated cup really let go? (the Surface Gripper's force limit against the load)

Isaac reports joint reactions through the cups at about 1/8 of the load (probe_wrist, probe_zload).
If the gripper compares those same internal values with its limits, a set limit L lets go at about
8 L of real load - or, read the other way, at L / 8 if the scale runs the other way. Cases of known mass
and centre of mass hang on the cups, are held still, then turned half round, at several set limits; the
first moment the gripper lets go is recorded.

    .venv-isaac/Scripts/python scripts/probe_grip_limit.py
Writes artifacts/probe_grip_limit.json."""
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.layout import CaseSpawn, pallet_slots  # noqa: E402
from shinai_pi.motion import limits_for, plan_move  # noqa: E402
from shinai_pi.sim.runtime import IsaacRuntime  # noqa: E402

import os
MODE = os.environ.get('GRIP_PROBE', 'turn')
if MODE == 'scale':           # static hang: where does a set limit of 8 x 300 N really let go?
    LOADS = [(100.0, (0.0, 0.0, 0.0)), (140.0, (0.0, 0.0, 0.0))]
    LIMITS = [(2400.0, 1200.0)]
    TWISTS = [10.0]
else:
    LOADS = [(8.0, (0.0, 0.0, 0.0)), (14.0, (0.0, 0.0, 0.0)), (14.0, (0.06, 0.06, 0.0))]
    LIMITS = [(300.0, 150.0)]
    TWISTS = [10000.0, 100.0, 10.0]


def main():
    base = load_cell()
    rt = IsaacRuntime(base)
    rows = []
    for (coax, shear), twist in [(lim, tw) for lim in LIMITS for tw in TWISTS]:
        cell = dataclasses.replace(base, gripper=dataclasses.replace(base.gripper, coaxial_force_limit=coax, shear_force_limit=shear,
                                                                     cup_twist_stiffness=twist))
        rt.cell = cell
        g = cell.gantry
        sku = cell.sku('BEV-500x24')
        src = pallet_slots(cell, sku, cell.pallet.a_center)[4]
        dt = cell.physics.dt
        for mass, com in LOADS:
            rt.new_episode()
            rt.build([CaseSpawn('case_00', sku.id, src.center, 0.0, 0, mass=mass, com=com)])
            rt.start()
            rt.step(int(0.5 / dt))
            lost = {'t': None, 'phase': None}

            def run(goal, slow, phase, settle=0.5):
                goal = np.asarray(goal, float)
                goal = np.r_[goal, rt.gantry_target[4:]] if len(goal) == 4 else goal
                prof = plan_move(rt.gantry_target.copy(), goal, *limits_for(g, slow))
                for k in range(int(math.ceil(prof.duration / dt)) + int(settle / dt)):
                    rt.set_gantry_target(prof.position(min(k * dt, prof.duration)))
                    rt.step(1)
                    if k % 12 == 0:
                        rt.capture('overhead_b')                   # a rendered update, as in a mission
                    if phase != 'approach' and lost['t'] is None and rt.gripped() != ['case_00']:
                        lost.update(t=rt.sim_time, phase=phase)
            z_face = src.center[2] + sku.size[2] / 2 - g.origin_height
            run([src.center[0], src.center[1], z_face + 0.05, 0.0], False, 'approach')
            run([src.center[0], src.center[1], z_face + cell.fork.grip_gap, 0.0], True, 'approach')
            rt.grip(True)
            rt.step(int(0.4 / dt))
            run([src.center[0], src.center[1], z_face + 0.3, 0.0], True, 'lift')
            run([src.center[0], src.center[1], z_face + 0.3, 0.0], True, 'hold', settle=2.0)
            if MODE != 'scale':
                run([src.center[0], src.center[1], z_face + 0.3, math.pi], True, 'turn_slow')
                run([src.center[0] + 1.5, src.center[1], z_face + 0.3, math.pi], False, 'carry_fast')
            rows.append({'coaxial_limit_N': coax, 'shear_limit_N': shear, 'twist_stiffness': twist, 'mass_kg': mass, 'com_m': list(com),
                         'per_cup_static_N': round(mass * 9.81 / 4, 1), 'lost_in': lost['phase']})
            print('GRIP', json.dumps(rows[-1]), flush=True)
    (ROOT / 'artifacts' / 'probe_grip_limit.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
    rt.close()


if __name__ == '__main__':
    main()
