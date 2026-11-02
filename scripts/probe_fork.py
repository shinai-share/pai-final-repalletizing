"""F0: measure how the suction + under-support hand behaves in the physics model, before the
executive uses it (requirements v05, design drawings v01, environment spec v01).

For each case of known mass and centre of mass: grip it by the top, lift, weigh it on the wrist,
slide the tines under it, push them up to a share of the weight, carry it fast to pallet B, land
the tines, hand the weight back to the cups, pull the tines out and set the case down. Ground truth
is read only here, to score the model; the robot never sees it.

What F0 settles (Codex reviews, 2026-10-17):
  * the scale and sign of each tine's joint reading (calibrated against the drive's own force cap);
  * whether the cups carry the remaining force AND moment at once, cup by cup (rigid-plate statics
    on the measured wrench: a model value, labelled so);
  * how the tine readings move when the tines touch down (the landing test is fixed only after this);
  * drag when the tines are pulled out, slip in transit, penetration of tines into the case;
  * with --seam, a raised seam on the case bottom; with --hz, another physics rate.

    .venv-isaac/Scripts/python scripts/probe_fork.py [--hz 240] [--seam 0.002] [--out artifacts/f0]
Results in a virtual work cell only: no carton, leak or friction value here is a real measurement."""
import argparse
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
from shinai_pi.perception.obb import quat_to_matrix  # noqa: E402
from shinai_pi.sim.runtime import IsaacRuntime  # noqa: E402

G = 9.81
# (mass kg, centre of mass in the case frame): centred, the worst offsets of the load range, light.
LOADS = [(12.0, (0.0, 0.0, 0.0)), (14.0, (0.06, 0.0, 0.0)), (14.0, (0.06, 0.06, -0.03)), (8.0, (-0.06, -0.06, 0.03))]
CUP_COAXIAL = None      # filled from the config: per-cup limits of the Surface Gripper
CUP_SHEAR = None


def main():
    global CUP_COAXIAL, CUP_SHEAR
    ap = argparse.ArgumentParser()
    ap.add_argument('--hz', type=float, default=None, help='physics rate (default: the config)')
    ap.add_argument('--seam', type=float, default=0.0, help='height of a raised seam on the case bottom, m')
    ap.add_argument('--out', default=str(ROOT / 'artifacts' / 'f0'))
    ap.add_argument('--loads', type=int, nargs='*', default=None, help='indices into LOADS')
    args = ap.parse_args()
    cell = load_cell()
    if args.hz:
        cell = dataclasses.replace(cell, physics=dataclasses.replace(cell.physics, dt=1.0 / args.hz))
    CUP_COAXIAL, CUP_SHEAR = cell.gripper.planning_coaxial_limit, cell.gripper.planning_shear_limit   # what the robot plans on
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rt = IsaacRuntime(cell)
    rows = []
    for i, (mass, com) in enumerate(LOADS):
        if args.loads is not None and i not in args.loads:
            continue
        rows.append(Trial(rt, cell, mass, com, args.seam).run())
        print('TRIAL', json.dumps(rows[-1], ensure_ascii=False), flush=True)
    report = {'physics_hz': round(1.0 / cell.physics.dt, 1), 'seam_m': args.seam, 'trials': rows,
              'criteria': CRITERIA, 'passed': all(r['passed'] for r in rows)}
    name = f"probe_fork_{int(round(1.0 / cell.physics.dt))}hz" + (f"_seam{int(args.seam * 1000)}mm" if args.seam else '')
    (out / f'{name}.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print('F0', json.dumps({k: v for k, v in report.items() if k != 'trials'}, ensure_ascii=False), flush=True)
    rt.close()


# F0 pass conditions, fixed before running (design drawings v01, section 7).
CRITERIA = {'internal_force_N': 200.0, 'penetration_mm': 1.0, 'drag_mm': 2.0, 'slip_mm': 2.0,
            'cup_utilisation': 1.0, 'tine_load_N': 100.0}


class Trial:
    def __init__(self, rt, cell, mass, com, seam):
        self.rt, self.cell, self.mass, self.com, self.seam = rt, cell, mass, np.asarray(com, float), seam
        self.g = cell.gantry
        self.f = cell.fork
        self.sku = cell.sku('BEV-500x24')
        self.src = pallet_slots(cell, self.sku, cell.pallet.a_center)[4]
        self.dst = pallet_slots(cell, self.sku, cell.pallet.b_center)[4]
        self.dt = cell.physics.dt

    # ------------------------------------------------------------------ helpers
    def move(self, goal, slow=False, settle=0.3, record=None):
        goal = np.asarray(goal, float)
        if len(goal) == 4:
            goal = np.r_[goal, self.rt.gantry_target[4:]]
        prof = plan_move(self.rt.gantry_target.copy(), goal, *limits_for(self.g, slow))
        for k in range(int(math.ceil(prof.duration / self.dt)) + 1):
            self.rt.set_gantry_target(prof.position(k * self.dt))
            self.rt.step(1)
            if record is not None:
                record()
        self.rt.step(int(settle / self.dt))

    def fork(self, extension, insert, seconds=None, record=None, z_speed=0.2, y_speed=0.1):
        """Move both forks to (extension, insert), ramping the targets at the fork's speeds (a step
        target would saturate the force-capped drives and read as contact)."""
        e0 = float(np.mean(-self.rt.fork_target[:2]))
        y0 = float(np.mean(self.rt.fork_target[2:]))
        n = max(1, int(math.ceil(max(abs(extension - e0) / z_speed, abs(insert - y0) / y_speed) / self.dt)))
        for k in range(1, n + 1):
            a = k / n
            self.rt.set_fork_target([e0 + a * (extension - e0)] * 2, [y0 + a * (insert - y0)] * 2)
            self.rt.step(1)
            if record is not None:
                record()
        self.rt.step(int(0.3 / self.dt))

    def mean(self, fn, seconds=0.25):
        vals = []
        for _ in range(max(1, int(round(seconds / self.dt)))):
            self.rt.step(1)
            vals.append(fn())
        return np.mean(vals, axis=0)

    def tool(self):
        q = self.rt.gantry_state()[0]
        return np.array([q[0], q[1], self.g.origin_height + q[2]]), float(q[3])

    def case(self):
        pos, quat = self.rt.truth_case_poses()['case_00']
        return np.asarray(pos, float), quat_to_matrix(quat)

    def case_in_tool(self):
        (t, yaw), (p, R) = self.tool(), self.case()
        c, s = math.cos(yaw), math.sin(yaw)
        d = p - t
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]]), R

    def tine_penetration_mm(self):
        """How far a tine's top is above the case bottom (positive: inside the case), true geometry."""
        p, R = self.case()
        bottom = p[2] - (R @ np.array([0, 0, self.sku.size[2] / 2]))[2]
        t, _ = self.tool()
        e, _ = self.rt.fork_state()
        return float(max(0.0, (t[2] - float(e.min())) - bottom) * 1000)

    # ------------------------------------------------------------------ the run
    def run(self):
        rt, f, sku = self.rt, self.f, self.sku
        rt.new_episode()
        rt.build([CaseSpawn('case_00', sku.id, (self.src.center[0], self.src.center[1], self.src.center[2] + self.seam), 0.0, 0,
                            com=tuple(self.com), mass=self.mass)])
        if self.seam:
            add_seam(rt, sku, self.seam)
        rt.start()
        h = sku.size[2]
        top = self.src.center[2] + h / 2 + self.seam              # a seam under the case lifts it by its height
        z_face = top - self.g.origin_height
        safe = z_face + 0.5
        sx, sy = self.src.center[0], self.src.center[1]
        rt.step(int(0.8 / self.dt))
        r = {'mass_kg': self.mass, 'com_m': self.com.tolist(), 'seam_m': self.seam}

        # 1. Empty readings (tare), forks retracted.
        self.move([sx, sy, z_face + 0.06, 0.0])
        tare_w = self.mean(lambda: np.array(rt.wrist_load()))
        tare_f = self.mean(rt.fork_raw_forces)

        # 2. Grip, lift 4 cm on the cups alone, weigh.
        self.move([sx, sy, z_face + f.grip_gap, 0.0], slow=True)
        rt.grip(True)
        rt.step(int(0.4 / self.dt))
        self.move([sx, sy, z_face + 0.04, 0.0], slow=True)
        w = self.mean(lambda: np.array(rt.wrist_load())) - tare_w
        r['weighed_kg'] = round(float(w[0] / G), 3)
        # Centre of mass under the tool from the test lift (the cup-head reading is valid only while
        # the tines carry nothing: once they push, Isaac subtracts their force from it 1:1).
        com_tool = (float(w[2] / w[0]), float(-w[1] / w[0]))
        r['com_estimate_mm'] = [round(v * 1000, 1) for v in com_tool]
        r['held_after_lift'] = rt.gripped() == ['case_00']

        # 3. Clear the pallet by more than the tines' reach below the case, then slide them in.
        self.move([sx, sy, z_face + 0.25, 0.0], slow=True)
        held, _ = self.case_in_tool()
        bottom_depth = -(held[2] - h / 2)                                   # case bottom below the cup face
        e_in = bottom_depth + f.clearance
        insert_peak = [0.0]
        self.fork(e_in, 0.0)
        self.fork(e_in, f.insert_range[1],
                  record=lambda: insert_peak.__setitem__(0, max(insert_peak[0], float(np.abs(rt.fork_raw_forces()[2:] - tare_f[2:]).max()))))
        r['insert_peak_raw'] = round(insert_peak[0], 2)

        # 4. Push up with a capped force: the drives saturate at the cap, so the applied force is
        #    known exactly and calibrates the readings (scale and sign).
        weight = float(w[0])
        cap = f.support_fraction * weight / 2
        before, _ = self.case_in_tool()
        rt.set_fork_force_caps([cap, cap], [f.y_force_limit] * 2)
        rt.set_fork_target([bottom_depth - 0.03] * 2, [f.insert_range[1]] * 2)   # beyond reach: the cap governs
        rt.step(int(1.5 / self.dt))
        raw = self.mean(rt.fork_raw_forces) - tare_f
        gain = [cap / raw[0] if abs(raw[0]) > 1e-6 else None, cap / raw[1] if abs(raw[1]) > 1e-6 else None]
        r['tine_gain'] = [None if v is None else round(float(v), 4) for v in gain]
        after, _ = self.case_in_tool()
        r['pushed_up_mm'] = round(float(after[2] - before[2]) * 1000, 2)
        r['penetration_mm'] = round(self.tine_penetration_mm(), 2)

        # 5. Lock the tines where they stand, keeping the force: the spring holds cap at an offset of cap / k.
        e_now, _ = rt.fork_state()
        rt.set_fork_target(e_now - cap / f.drive_stiffness, [f.insert_range[1]] * 2)
        rt.set_fork_force_caps([f.z_force_limit] * 2, [f.y_force_limit] * 2)
        rt.step(int(0.5 / self.dt))
        tl = self.tine_loads(gain, tare_f)
        r['tine_loads_N'] = [round(v, 1) for v in tl]
        # The cups carry what the tines do not (balance with the weight measured before the tines).
        span = (max(-sku.size[1] / 2, f.bar_y - f.tine_size[1] + f.insert_range[1]), f.bar_y + f.insert_range[1])
        tine_y = (max(span[0], -sku.size[1] / 2) + min(span[1], sku.size[1] / 2)) / 2   # middle of the tine under the case
        suction = suction_wrench(weight, com_tool, tl, (-f.bar_x, f.bar_x), tine_y)
        r['suction_share'] = round(float(suction[0] / weight), 3)
        r['internal_force_N'] = round(float(max(0.0, -suction[0])), 1)      # tines push the case into the cups
        r['cups'] = cup_check(suction, self.g)

        # 6. Carry fast to pallet B and back over it; watch the loads and the case on the cups.
        start_rel, _ = self.case_in_tool()
        peak = {'tine': 0.0}

        def watch():
            t = self.tine_loads(gain, tare_f)
            peak['tine'] = max(peak['tine'], max(t))
        self.move([self.dst.center[0], self.dst.center[1], z_face + 0.25, 0.0], record=watch)
        end_rel, _ = self.case_in_tool()
        r['carry_tine_peak_N'] = round(peak['tine'], 1)
        r['carry_slip_mm'] = round(float(np.linalg.norm(end_rel[:2] - start_rel[:2])) * 1000, 2)
        r['held_after_carry'] = rt.gripped() == ['case_00']

        # 7. Land the tines on pallet B at touch speed; record how the tine readings move.
        dst_top = self.dst.center[2] - h / 2                                # pallet deck under the slot
        land_face = dst_top + bottom_depth + f.tine_size[2] - self.g.origin_height
        self.move([self.dst.center[0], self.dst.center[1], land_face + 0.03, 0.0], slow=True)
        pre = np.array(self.tine_loads(gain, tare_f))
        p_land0, _ = self.case()
        trace = []
        q = rt.gantry_state()[0].copy()
        landed_at = None
        for k in range(int(0.06 / (self.g.touch_speed * self.dt))):
            q[2] -= self.g.touch_speed * self.dt
            rt.set_gantry_target(q)
            rt.step(1)
            t = np.array(self.tine_loads(gain, tare_f))
            trace.append((t - pre).tolist())
            if landed_at is None and np.abs(t - pre).max() > 15.0 and len(trace) >= 3 and \
                    all(np.abs(np.array(x)).max() > 15.0 for x in trace[-3:]):
                landed_at = k
                break
        rt.set_gantry_target(rt.gantry_state()[0])
        rt.step(int(0.3 / self.dt))
        post = np.array(self.tine_loads(gain, tare_f))
        r['landing'] = {'detected': landed_at is not None,
                        'tine_change_N': [round(float(v), 1) for v in (post - pre)],
                        'sign': 'increase' if (post - pre).sum() > 0 else 'decrease',   # recorded, not relied on
                        'peak_change_N': round(float(max((np.abs(np.array(x)).max() for x in trace), default=0.0)), 1),
                        'trace_N_first': [[round(v, 1) for v in x] for x in trace[:3]],
                        'trace_N_last': [[round(v, 1) for v in x] for x in trace[-3:]]}

        # 8. Hand the weight back to the cups: raise the tool 3 mm while the tines stay on the deck.
        self.case_before_raise, _ = self.case()
        r['landing']['case_moved_mm'] = round(float(np.linalg.norm(self.case_before_raise[:2] - p_land0[:2])) * 1000, 2)
        e_now, _ = rt.fork_state()
        rt.set_fork_force_caps([20.0, 20.0], [f.y_force_limit] * 2)
        rt.set_fork_target(e_now + 0.006, [f.insert_range[1]] * 2)
        q = rt.gantry_state()[0].copy()
        q[2] += 0.003
        rt.set_gantry_target(q)
        rt.step(int(0.5 / self.dt))
        residual = self.tine_loads(gain, tare_f)
        r['residual_tine_N'] = [round(v, 1) for v in residual]

        # 9. Pull the tines out; the case must not move.
        p0, _ = self.case()
        r['raise_moved_case_mm'] = round(float(np.linalg.norm(p0[:2] - self.case_before_raise[:2])) * 1000, 2)
        drag = [0.0]
        pull_peak = [0.0]

        def pulling():
            drag[0] = max(drag[0], float(np.linalg.norm(self.case()[0][:2] - p0[:2])))
            pull_peak[0] = max(pull_peak[0], float(np.abs(rt.fork_raw_forces()[2:] - tare_f[2:]).max()))
        self.fork(float(rt.fork_state()[0].mean()), 0.0, record=pulling)
        r['pull_force_peak_N'] = round(pull_peak[0], 1)
        r['drag_mm'] = round(drag[0] * 1000, 2)
        rt.set_fork_force_caps([f.z_force_limit] * 2, [f.y_force_limit] * 2)
        self.fork(f.extension_range[0], 0.0, seconds=1.5)

        # 10. Set it down on the cups: lower slowly until the weight leaves them, release, look.
        q = rt.gantry_state()[0].copy()
        for _ in range(int(0.03 / (self.g.touch_speed * self.dt))):
            q[2] -= self.g.touch_speed * self.dt
            rt.set_gantry_target(q)
            rt.step(1)
            if rt.wrist_load()[0] - tare_w[0] < 0.5 * weight:
                break
        rt.grip(False)
        rt.step(int(0.3 / self.dt))
        self.move([self.dst.center[0], self.dst.center[1], z_face + 0.3, 0.0], slow=True)
        rt.step(int(3.0 / self.dt))
        p, R = self.case()
        r['placed_xy_err_mm'] = round(float(np.linalg.norm(p[:2] - np.asarray(self.dst.center[:2]))) * 1000, 2)
        r['placed_tilt_deg'] = round(math.degrees(math.acos(min(1.0, abs(float(R[2, 2]))))), 2)
        r['passed'] = judge(r)
        return r

    def tine_loads(self, gain, tare):
        raw = self.rt.fork_raw_forces() - tare
        return [float(raw[i] * gain[i]) if gain[i] is not None else 0.0 for i in range(2)]


def judge(r):
    c = CRITERIA
    return bool(r['held_after_lift'] and r['held_after_carry'] and r['internal_force_N'] < c['internal_force_N']
                and r['penetration_mm'] <= c['penetration_mm'] and r['drag_mm'] <= c['drag_mm']
                and r['carry_slip_mm'] <= c['slip_mm'] and r['cups']['utilisation'] <= c['cup_utilisation']
                and max(r['tine_loads_N']) <= c['tine_load_N'] and r['carry_tine_peak_N'] <= c['tine_load_N'] * 1.25
                and r['landing']['detected'])


def suction_wrench(weight, com, tines, tine_x, tine_y):
    """(force, torque about tool x, torque about tool y) left to the cups, in the convention of the
    wrist reading for a hanging load (weight, -W r_y, W r_x): the weight at its centre of mass minus
    the tines pushing up at (tine_x[i], tine_y)."""
    ft = sum(tines)
    tx = -weight * com[1] + sum(t * tine_y for t in tines)
    ty = weight * com[0] - sum(t * x for t, x in zip(tines, tine_x))
    return (weight - ft, tx, ty)


def cup_check(wrist, g):
    """What each cup carries, from the force and moments left to the cups.

    Model value (rigid plate, least-norm split over the cups), not a measured per-cup force: Isaac's
    Surface Gripper does not report forces per attachment. The force and the two moments are split
    together, so a cup is checked for its share of all three at once."""
    fs, tx_s, ty_s = (float(v) for v in wrist)
    cups = np.asarray(g.cup_offsets, float)
    n = len(cups)
    sx2 = float(np.sum(cups[:, 0] ** 2))
    sy2 = float(np.sum(cups[:, 1] ** 2))
    loads = [fs / n + ty_s * c[0] / sx2 - tx_s * c[1] / sy2 for c in cups]
    util = max(max(load, 0.0) / CUP_COAXIAL for load in loads)
    return {'suction_force_N': round(fs, 1), 'suction_moment_Nm': [round(tx_s, 2), round(ty_s, 2)],
            'per_cup_N': [round(v, 1) for v in loads], 'utilisation': round(util, 3),
            'note': 'model value: rigid-plate split of the measured wrench'}


def add_seam(rt, sku, height):
    """A raised seam along the middle of the case bottom (where the flaps meet), as a collider."""
    from shinai_pi.sim import cell as cellmod
    from shinai_pi.sim import usd_build as ub
    w, d, h = sku.size
    ub.box(rt.stage, f'{cellmod.CASES}/case_00/seam', (w * 0.95, 0.03, height), pos=(0, 0, -h / 2 - height / 2))


if __name__ == '__main__':
    main()
