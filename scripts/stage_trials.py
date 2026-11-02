"""Stage trials of requirements v04 (chapter 8) in Isaac Sim, from the easy end up.

    S0      one upright case
    S1-side one case lying on its side (0.27 m up)
    S1-end  one case standing on its end (0.36 m up)

Each trial draws the case's mass and centre of mass (physics only), a small offset and turn on
pallet A, then runs the same executive as the full mission. Ground truth scores the trial apart
from the executive: placement error after settling (position, yaw, tilt), the centre-of-mass
estimate from the test lift, whether the case was righted, and the end state.

    .venv-isaac/Scripts/python scripts/stage_trials.py --stage S1-side --trials 10 [--record 0]
Writes artifacts/stages/<stage>/trial_<n>/ and artifacts/stages/<stage>/summary.json."""
import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.evaluation.truth import quat_tilt_deg  # noqa: E402
from shinai_pi.evaluation.resume import gate_rows, pending  # noqa: E402
from shinai_pi.evaluation.contacts import ContactLog  # noqa: E402
from shinai_pi.events import EventLog  # noqa: E402
from shinai_pi.executive import Executive, MissionConfig  # noqa: E402
from shinai_pi.layout import CaseSpawn  # noqa: E402
from shinai_pi.perception.obb import quat_to_matrix  # noqa: E402
from shinai_pi.variation import vary_load  # noqa: E402
from shinai_pi.viewpoints import park_pose  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R2 = math.sqrt(0.5)
POSES = {                     # (orientation before the yaw, edge that ends up vertical)
    'S0': ((1.0, 0.0, 0.0, 0.0), 2),
    'S1-side': ((R2, R2, 0.0, 0.0), 1),
    'S1-end': ((R2, 0.0, R2, 0.0), 0),
    'S1-tilt': (None, None),                 # slid half off the pallet: 30 deg, on the floor and the pallet's edge
}
TARGET = {}                               # the case each stage is about, where a scene has more than one


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def scene(cell, sku, stage, trial, tilt_deg=None):
    rng = np.random.default_rng(1000 * (list(POSES).index(stage) + 1) + trial)
    base, up_axis = POSES[stage]
    if stage == 'S1-tilt':
        return leaning_scene(cell, sku, stage, trial, rng, 30.0 if tilt_deg is None else tilt_deg)
    if tilt_deg is not None:
        raise ValueError(f'a tilt angle sets the leaning scene only, not {stage}')
    yaw = rng.uniform(-0.3, 0.3)
    q = qmul((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), base)
    a = cell.pallet.a_center
    pos = (a[0] + rng.uniform(-0.08, 0.08), a[1] + rng.uniform(-0.08, 0.08), cell.pallet.size[2] + sku.size[up_axis] / 2)
    spawn = CaseSpawn('case_00', sku.id, pos, 0.0, 0, quat=tuple(float(v) for v in q))
    return vary_load([spawn], cell.variation, seed=500 + 37 * trial + list(POSES).index(stage))


def leaning_scene(cell, sku, stage, trial, rng, tilt_deg):
    """A case that slid half off pallet A in transport: its lower edge on the floor between the
    pallets, its bottom face resting on the pallet's edge, tilted `tilt_deg` about its short axis.
    The centre of gravity lies between the two contacts, so it stays as it is."""
    a = cell.pallet.a_center
    w, d, h = sku.size
    tilt = math.radians(tilt_deg)
    edge_x, edge_z = a[0] + cell.pallet.size[0] / 2, cell.pallet.size[2]
    # Bottom corners of the tilted box relative to its centre: the low one at +x, the high one at -x.
    low_x = (w / 2) * math.cos(tilt) - (h / 2) * math.sin(tilt)
    low_z = -(w / 2) * math.sin(tilt) - (h / 2) * math.cos(tilt)
    cz = -low_z + 0.001
    cx = edge_x + (edge_z - 0.001) / math.tan(tilt) - low_x
    q = (math.cos(tilt / 2), 0.0, math.sin(tilt / 2), 0.0)
    spawn = CaseSpawn('case_00', sku.id, (cx, a[1] + rng.uniform(-0.15, 0.15), cz), 0.0, 0, quat=q)
    return vary_load([spawn], cell.variation, seed=500 + 37 * trial + list(POSES).index(stage))


def run_trial(rt, cell, sku, stage, trial, out, record, overrides=None, on_event=None, tilt_deg=None):
    """One trial. `overrides` replaces fields of the target case's spawn (mass, com) for experiments
    that change one factor at a time; `on_event` also receives every event."""
    from shinai_pi.sim.isaac_backend import IsaacBackend, PersonScript
    spawns = scene(cell, sku, stage, trial, tilt_deg)
    if overrides:
        spawns = [dataclasses.replace(s, **overrides) if s.case_id == TARGET.get(stage, 'case_00') else s for s in spawns]
    spawn = next(s for s in spawns if s.case_id == TARGET.get(stage, 'case_00'))
    rt.new_episode()
    rt.build(spawns, wrapped_layers=[])
    rt.start()
    rt.set_gantry_target(park_pose(cell))
    rt.step(int(1.0 / cell.physics.dt))
    target = TARGET.get(stage, 'case_00')
    pos0, quat0 = rt.truth_case_poses()[target]
    out.mkdir(parents=True, exist_ok=True)
    ev = EventLog(f'{stage}-{trial}', out / 'events.jsonl')
    if on_event:
        ev.subscribe(on_event)
    be = IsaacBackend(rt, PersonScript(()))
    be.carried_weight = 9.81 * cell.variation.mass[0]
    ex = Executive(cell, be, ev, MissionConfig(sku.id, expected_cases=len(spawns)))
    feed = None
    if record:
        from shinai_pi.inspector_feed import RunRecorder
        feed = RunRecorder(out, ex, cell)
        ev.subscribe(feed.on_event)
        be.record_frames(out / 'frames', every_s=1.0 / 15, views=('presentation', 'side', 'closeup'))
    contacts = ContactLog(path=out / 'truth_contacts.jsonl', gap_s=0.1, raw_path=out / 'truth_contacts_raw.jsonl')      # evaluation only: the robot never reads it
    rt.watch_contacts(contacts)
    t0 = time.perf_counter()
    rep = ex.run()
    wall = time.perf_counter() - t0
    rt.step_hooks.clear()
    contacts.close(rt.sim_time, {k: p for k, (p, _) in rt.truth_case_poses().items()})
    ev.close()
    if feed:
        feed.close()
    pos, quat = rt.truth_case_poses()[target]
    res = {'stage': stage, 'trial': trial, 'status': rep.status, 'sim_seconds': rep.seconds, 'wall_seconds': round(wall, 1),
           'mass_kg': round(spawn.mass, 2), 'com_case_mm': [round(v * 1000, 1) for v in spawn.com],
           'reoriented': bool(ev.of('REORIENTED')), 'handed_over': rep.handed_over, 'placed': rep.placed,
           'failed_checks': rep.failed_checks, 'drops': len(ev.of('DROP_DETECTED'))}
    placed = [e.data for e in ev.of('PLACED') if e.data['case'] == target]
    if placed:
        slot = ex.slots[placed[-1]['slot']]
        R = quat_to_matrix(quat)
        horiz = [i for i in range(3) if abs(R[2, i]) < 0.7]
        i_long = max(horiz, key=lambda i: sku.size[i])
        yaw_deg = math.degrees(math.atan2(R[1, i_long], R[0, i_long]))
        d = (yaw_deg - slot.yaw_deg) % 180.0
        res['truth'] = {'position_error_vec_mm': [round(float(v) * 1000, 1) for v in np.asarray(pos[:2]) - np.asarray(slot.center[:2])],
                        'position_error_mm': round(float(np.linalg.norm(np.asarray(pos[:2]) - np.asarray(slot.center[:2]))) * 1000, 1),
                        'height_error_mm': round(float(pos[2] - slot.center[2]) * 1000, 1),
                        'yaw_error_deg': round(min(d, 180.0 - d), 2), 'tilt_deg': round(quat_tilt_deg(quat), 2),
                        'standing_edge_m': round(float(sku.size[int(np.argmax(np.abs(R[2, :])))]), 3)}
    loads = [e.data for e in ev.of('LOAD_MEASURED') if e.data['case'] == target]
    if loads:
        # Measured hanging level: for a case gripped tilted, after the wrist has levelled it.
        com_world = quat_to_matrix(quat0) @ np.asarray(spawn.com) if stage != 'S1-tilt' else None
        first = loads[0]
        if com_world is not None:
            res['com_estimate_error_mm'] = round(float(np.linalg.norm(np.asarray(first['com_offset_mm']) - com_world[:2] * 1000)), 1)
        res['mass_estimate_error_kg'] = round(first['mass_kg'] - spawn.mass, 2)
    checks = [e.data for e in ev.of('PLACEMENT_CHECKED') if e.data['case'] == target]
    if checks:
        res['seen'] = checks[-1]
    settled = [e.data for e in ev.of('PLACEMENT_SETTLED') if e.data['case'] == target]
    if settled:
        res['settled'] = settled[-1]
    (out / 'result.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    return res


def gate(stage, rows):
    """Requirements v04 chapter 8: S0 10 of 10; S1 9 of 10 or better, by physical contact only."""
    def ok(r):
        t = r.get('truth')
        return (r['status'] == 'verified' and t is not None and t['position_error_mm'] <= 10 and t['yaw_error_deg'] <= 2
                and t['tilt_deg'] <= 2 and abs(t['standing_edge_m'] - 0.22) < 0.01 and r['drops'] == 0)
    passed = sum(1 for r in rows if ok(r))
    need = len(rows) if stage == 'S0' else math.ceil(0.9 * len(rows))
    return {'passed': passed, 'trials': len(rows), 'required': need, 'gate_passed': passed >= need}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=list(POSES), required=True)
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--first', type=int, default=0, help='first trial number; trials that already have a result are skipped')
    ap.add_argument('--record', type=int, nargs='*', default=[], help='trial numbers to record frames for')
    ap.add_argument('--out', default=None, help='output folder (default artifacts/stages/<stage>)')
    args = ap.parse_args()
    cell = load_cell()
    sku = cell.sku('BEV-500x24')
    from shinai_pi.sim.runtime import IsaacRuntime
    rt = IsaacRuntime(cell)
    base = Path(args.out) if args.out else ROOT / 'artifacts' / 'stages' / args.stage
    for n in pending(range(args.first, args.first + args.trials), lambda n: base / f'trial_{n:02d}'):
        r = run_trial(rt, cell, sku, args.stage, n, base / f'trial_{n:02d}', n in args.record)
        print('TRIAL', json.dumps({k: r.get(k) for k in ('stage', 'trial', 'status', 'reoriented', 'truth', 'com_estimate_error_mm',
                                                         'mass_estimate_error_kg', 'handed_over', 'drops')}, ensure_ascii=False), flush=True)
    rows = gate_rows(base)             # every finished trial in the folder, from this process or an earlier one
    summary = {'stage': args.stage, 'gate': gate(args.stage, rows), 'trials': rows}
    (base / 'summary.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding='utf-8')
    print('GATE', json.dumps(summary['gate']))
    rt.close()
