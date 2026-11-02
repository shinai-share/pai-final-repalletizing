"""Calibrate the up-down axis reading (z_load) like a load cell, so that touching a case with the
empty cups can be stated in newtons (requirements v05 R1, the damage limits of chapter 3).

The z drive's reaction is read raw by the executive to feel the empty tool touch a face. Isaac reports
joint reactions scaled (the wrist reads about 1/8), so the raw threshold says nothing in newtons until
the scale is measured. Known masses hang on the cups; the raw change from the empty reading is fitted
to their weight on the first loads and checked on the last. The touch threshold and the touch peaks
recorded by the stage gates are then converted.

    .venv-isaac/Scripts/python scripts/probe_zload.py
Writes artifacts/probe_zload.json. A virtual work cell: no real load cell is calibrated here."""
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

LOADS = [8.0, 14.0, 11.0]               # the last one checks the fit
G = 9.81


def main():
    cell = load_cell()
    sku = cell.sku('BEV-500x24')
    g = cell.gantry
    src = pallet_slots(cell, sku, cell.pallet.a_center)[4]
    rt = IsaacRuntime(cell)
    dt = cell.physics.dt

    def move(goal, slow=False, settle=0.4):
        goal = np.asarray(goal, float)
        if len(goal) == 4:
            goal = np.r_[goal, rt.gantry_target[4:]]
        prof = plan_move(rt.gantry_target.copy(), goal, *limits_for(g, slow))
        for k in range(int(math.ceil(prof.duration / dt)) + 1):
            rt.set_gantry_target(prof.position(k * dt))
            rt.step(1)
        rt.step(int(settle / dt))

    def reading(seconds=0.5):
        vals = []
        for _ in range(int(seconds / dt)):
            rt.step(1)
            vals.append(rt.z_load())
        return float(np.mean(vals))

    # Inertia: the empty tool accelerated at a known rate. The reading for a force on the links themselves
    # (inertia, and a contact on the plate) is compared with mass x acceleration.
    rt.new_episode()
    rt.build([CaseSpawn('case_00', sku.id, src.center, 0.0, 0, mass=11.0)])
    rt.start()
    rt.step(int(0.5 / dt))
    moving = sum(float(rt.stage.GetPrimAtPath(f'/World/cell/gantry/{n}').GetAttribute('physics:mass').Get())
                 for n in ('ram', 'tool', 'wrist_p', 'wrist_r', 'head', 'fork_bar_l', 'fork_bar_r', 'fork_tine_l', 'fork_tine_r'))
    a = float(g.a_max[2])
    z0 = float(rt.gantry_target[2])
    still = reading(0.3)
    inertial = []
    for k in range(int(0.2 / dt)):                         # constant acceleration downwards for 0.2 s
        tt = (k + 1) * dt
        rt.set_gantry_target(np.r_[rt.gantry_target[:2], z0 - 0.5 * a * tt * tt, rt.gantry_target[3:]])
        rt.step(1)
        if k > int(0.05 / dt):                             # past the start-up swing
            inertial.append(rt.z_load() - still)
    inertia_gain = moving * a / abs(float(np.mean(inertial)))
    rows = []
    for mass in LOADS:
        rt.new_episode()
        rt.build([CaseSpawn('case_00', sku.id, src.center, 0.0, 0, mass=mass)])
        rt.start()
        z_face = src.center[2] + sku.size[2] / 2 - g.origin_height
        rt.step(int(0.5 / dt))
        move([src.center[0], src.center[1], z_face + 0.25, 0.0])
        tare = reading()
        move([src.center[0], src.center[1], z_face + cell.fork.grip_gap, 0.0], slow=True)
        rt.grip(True)
        rt.step(int(0.4 / dt))
        move([src.center[0], src.center[1], z_face + 0.25, 0.0], slow=True)
        rows.append({'mass_kg': mass, 'raw': reading() - tare, 'weight_N': mass * G, 'held': rt.gripped() == ['case_00']})
    fit = rows[:-1]
    gain = float(np.mean([r['weight_N'] / r['raw'] for r in fit]))
    check = rows[-1]
    out = {'loads': rows, 'gain_N_per_raw': round(gain, 4),
           'inertia': {'moving_mass_kg': round(moving, 2), 'accel_mps2': a, 'expected_N': round(moving * a, 1),
                       'raw': round(abs(float(np.mean(inertial))), 2), 'gain_N_per_raw': round(inertia_gain, 3)},
           'check': {'weight_N': round(check['weight_N'], 2), 'predicted_N': round(gain * check['raw'], 2)},
           'touch_threshold': {'before_calibration_raw': 40.0,
                               'before_calibration_newtons_as_plate_force': round(inertia_gain * 40.0, 1),
                               'now_newtons': cell.contact.touch_detect}}
    # Touch peaks the stage gates recorded (raw), in newtons.
    peaks = []
    for f in (ROOT / 'artifacts').glob('stages_hand2/*/trial_*/events.jsonl'):
        for line in f.read_text(encoding='utf-8').splitlines():
            e = json.loads(line)
            if e['type'] == 'GRIP_OK' and e['data'].get('touch_peak_raw') is not None:
                peaks.append(abs(gain) * e['data']['touch_peak_raw'])
    if peaks:
        out['recorded_touch_peaks_N'] = {'n': len(peaks), 'max': round(max(peaks), 1), 'median': round(float(np.median(peaks)), 1)}
    (ROOT / 'artifacts' / 'probe_zload.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('ZLOAD', json.dumps(out, ensure_ascii=False), flush=True)
    rt.close()


if __name__ == '__main__':
    main()
