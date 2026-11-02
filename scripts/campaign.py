"""Repeat a fixed task and record every episode for learning (experiment plan v01, chapters 3-5).

    .venv-isaac/Scripts/python scripts/campaign.py --campaign tilt30_A --stage S1-tilt --trials 2 1 6 9 --repeats 3
    .venv-isaac/Scripts/python scripts/campaign.py --campaign tilt30_B_mass --stage S1-tilt --trials 2 --mass 8 11 14

Layer A repeats the same scene (same trial index: pose, mass, centre of mass) to see whether a result
reproduces; layer B changes one factor (--mass, --com-x) of a fixed scene; layer C runs scenes kept
out of every adjustment. Each run is an episode under artifacts/experiments/<campaign>/: a physics-step
trace of what the robot observed and commanded, the truth apart, and separate verdicts. A rerun gets a
new episode id; nothing is overwritten."""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import stage_trials as st  # noqa: E402
from shinai_pi.config import load_cell  # noqa: E402
from shinai_pi.experiments import EpisodeStore  # noqa: E402
from shinai_pi.perception.obb import quat_to_matrix  # noqa: E402

GRIP_CODE = {'Open': 0, 'Closed': 1, 'Closing': 2}


def verdicts(res, events):
    """Quality, safety and validity judged apart (plan chapter 2)."""
    t = res.get('truth')
    placed_ok = (res['status'] == 'verified' and t is not None and t['position_error_mm'] <= 10 and t['yaw_error_deg'] <= 2
                 and t['tilt_deg'] <= 2 and abs(t['standing_edge_m'] - 0.22) < 0.01)
    kinds = [e['type'] for e in events]
    held_for_person = res['status'] == 'interrupted'
    return {'quality': 'success' if placed_ok else ('safe_hold' if held_for_person else 'failure'),
            'safety': {'drops': res.get('drops', 0), 'damage_risk': kinds.count('DAMAGE_RISK'),
                       'vacuum_lost': kinds.count('VACUUM_LOST'), 'safety_stops': kinds.count('SAFETY_STOP'),
                       'near_miss': kinds.count('NEAR_MISS')},
            'validity': 'valid'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--stage', default='S1-tilt', choices=list(st.POSES))
    ap.add_argument('--trials', type=int, nargs='+', required=True)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--mass', type=float, nargs='*', default=[None])
    ap.add_argument('--com-x', type=float, nargs='*', default=[None], help='centre of mass along the case long axis, m')
    ap.add_argument('--tilt-deg', type=float, nargs='*', default=[None], help='lean of the S1-tilt case, degrees')
    ap.add_argument('--layer', default='A', choices=['A', 'B', 'C'])
    ap.add_argument('--support-fraction', type=float, default=None, help='share of the weight the tines push up with (a factor of layer B)')
    args = ap.parse_args()
    from shinai_pi.sim.runtime import IsaacRuntime
    cell = load_cell()
    if args.support_fraction is not None:
        import dataclasses
        cell = dataclasses.replace(cell, fork=dataclasses.replace(cell.fork, support_fraction=args.support_fraction))
    sku = cell.sku('BEV-500x24')
    rt = IsaacRuntime(cell)
    store = EpisodeStore(ROOT / 'artifacts' / 'experiments', args.campaign)
    orig_step = rt.step
    live = {'ep': None, 'state': 'START', 'fork_be': None}

    def step(n=1, render=False):
        orig_step(n, render)
        ep = live['ep']
        if ep is None or not getattr(rt, 'gantry', None):
            return
        q, _ = rt.gantry_state()
        w = rt.head_wrench()
        fk = rt.fork_raw_forces()
        row = {'t': rt.sim_time, 'state': live['state'], 'obs_gripped': float(bool(rt.gripped())),
               'obs_grip_status': float(GRIP_CODE.get(rt.gripper_status(), -1)),
               'obs_tine_l': float(fk[0]), 'obs_tine_r': float(fk[1]), 'obs_slide_l': float(fk[2]), 'obs_slide_r': float(fk[3]),
               'obs_head_fz': float(w[2]), 'obs_head_tx': float(w[3]), 'obs_head_ty': float(w[4]),
               'obs_zload': float(rt.z_load())}
        for i in range(6):
            row[f'obs_q{i}'] = float(q[i])
            row[f'cmd_q{i}'] = float(rt.gantry_target[i])
        e, y = rt.fork_state()
        row.update(obs_fork_ext=float(np.mean(e)), obs_fork_ins=float(np.mean(y)), cmd_fork_ext=float(-np.mean(rt.fork_target[:2])))
        ep.add_trace(row)

    rt.step = step
    events = []

    def on_event(e):
        live['state'] = e.type
        events.append({'t': e.t, 'type': e.type, 'data': e.data})

    for trial in args.trials:
        for mass, com_x, tilt in itertools.product(args.mass, args.com_x, args.tilt_deg):
            overrides = {}
            tag = ''
            if tilt is not None:
                tag += f'-tilt{tilt:g}'
            if mass is not None:
                overrides['mass'] = mass
                tag += f'-m{mass:g}'
            if com_x is not None:
                base = st.scene(cell, sku, args.stage, trial)[0].com
                overrides['com'] = (com_x, base[1], base[2])
                tag += f'-cx{int(round(com_x * 1000))}'
            scenario = f'{args.stage}-t{trial:03d}{tag}'
            for rep in range(args.repeats):
                eid = f'{scenario}-r{rep}'
                ep = store.open(eid, scenario_id=scenario, family=f'{args.stage}-t{trial:03d}', repeat=rep,
                                seeds={'scene_trial': trial}, extra={'layer': args.layer, 'overrides': overrides,
                                                                     'tilt_deg': tilt,
                                                                     'support_fraction': cell.fork.support_fraction,
                                                                     'physics_dt': cell.physics.dt})
                events.clear()
                live.update(ep=ep, state='START')
                try:
                    res = st.run_trial(rt, cell, sku, args.stage, trial, ep.dir / 'run', False, overrides=overrides or None,
                                       on_event=on_event, tilt_deg=tilt)
                except Exception as exc:                    # keep the record of a run that broke
                    live['ep'] = None
                    ep.abort(f'{type(exc).__name__}: {exc}')
                    print('EPISODE', json.dumps({'id': eid, 'aborted': str(exc)}, ensure_ascii=False), flush=True)
                    continue
                live['ep'] = None
                spawn = st.scene(cell, sku, args.stage, trial, tilt)[0]
                pos, quat = rt.truth_case_poses()['case_00']
                ep.write_truth({'mass_kg': overrides.get('mass', spawn.mass), 'com_case_m': list(overrides.get('com', spawn.com)),
                                'final_pos': [float(v) for v in pos], 'final_R': quat_to_matrix(quat).tolist(), 'run': res})
                v = verdicts(res, events)
                ep.write_result({**v, 'status': res['status'], 'truth_errors': res.get('truth'),
                                 'events': [x for x in events if x['type'] in ('FORK_IN', 'VACUUM_LOST', 'SAFETY_STOP', 'HANDOVER',
                                                                              'NEAR_MISS', 'DAMAGE_RISK', 'CONTACT_LOAD', 'FORK_OUT')]})
                ep.complete()
                print('EPISODE', json.dumps({'id': eid, 'quality': v['quality'], 'status': res['status'], **v['safety']},
                                            ensure_ascii=False), flush=True)
    rt.close()


if __name__ == '__main__':
    main()
