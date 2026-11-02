"""Run one full re-stacking mission in Isaac Sim and score it against ground truth.

Example:  python scripts/run_mission.py --seed 8 --policy shinai
Writes artifacts/runs/<policy>_seed<seed>/{events.jsonl, result.json, manifest.json}"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shinai_pi.config import DEFAULT_CELL, load_cell  # noqa: E402
from shinai_pi.evaluation.contacts import ContactLog  # noqa: E402
from shinai_pi.events import EventLog  # noqa: E402
from shinai_pi.evaluation.harness import PalletExitClock, TruthRecorder  # noqa: E402
from shinai_pi.evaluation.resume import pending  # noqa: E402
from shinai_pi.evaluation.trace import StepTrace  # noqa: E402
from shinai_pi.executive import Executive, MissionConfig  # noqa: E402
from shinai_pi.layout import initial_cases  # noqa: E402
from shinai_pi.transport import sample_transport  # noqa: E402
from shinai_pi.variation import vary_load  # noqa: E402


def git_sha():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return 'unknown'


def _collapse_filmer(rt, out, fps, views):
    """Pictures of the transport collapse for people, into the same folders as the mission's."""
    from PIL import Image
    dirs = {v: out / ('frames' if v == 'presentation' else f'frames_{v}') for v in views}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    state = {'next': rt.sim_time}

    def on_step():
        if rt.sim_time < state['next']:
            return
        name = f'{int(round(rt.sim_time * 1000)):08d}.jpg'
        for v, d in dirs.items():
            Image.fromarray(rt.render_rgb(v)).save(d / name, quality=90)
        state['next'] += 1.0 / fps
    return on_step


def run_one(rt, cell, seed, policy, sku_id, person=None, out_root=ROOT / 'artifacts' / 'runs', on_event=None, record=False, max_picks=None,
            fps=4.0, views=('presentation',), trace=False, cascade_model=str(ROOT / 'models' / 'cascade_v2.json')):
    from shinai_pi.sim.isaac_backend import IsaacBackend, PersonScript
    from shinai_pi.sim.runtime import run_transport
    sku = cell.sku(sku_id)
    profile = sample_transport(cell.transport, cell.layout.layers, seed)
    spawns = vary_load(initial_cases(cell, sku), cell.variation, seed)
    rt.new_episode()
    rt.build(spawns, wrapped_layers=[l for l in range(cell.layout.layers) if l not in profile.loose_layers])
    rt.start()
    from shinai_pi.viewpoints import park_pose
    rt.set_gantry_target(park_pose(cell))
    rt.step(int(1.0 / cell.physics.dt))
    out = out_root / f'{policy}_seed{seed:03d}'
    out.mkdir(parents=True, exist_ok=True)
    film = _collapse_filmer(rt, out, fps, views) if record else None
    clock, steps = PalletExitClock(cell), [0]
    every = max(1, round(0.1 / cell.physics.dt))       # when each case left pallet A in the transport, to 0.1 s

    def on_step():
        if film:
            film()
        steps[0] += 1
        if steps[0] % every == 0:
            clock.tick(rt.sim_time, rt.truth_case_poses())
    transport = run_transport(rt, profile, on_step=on_step)
    ev = EventLog(f'{policy}-{seed}', out / 'events.jsonl')
    backend = IsaacBackend(rt, PersonScript(person or ()))
    backend.carried_weight = 9.81 * cell.variation.mass[0]      # the lightest case the load can hold
    # 'shinai_learned' is the same policy with the cascade predictor learned from physics outcomes.
    kw = {'policy': policy, 'expected_cases': len(spawns)}     # the shipping record of the load
    if policy == 'shinai_learned':
        kw.update(policy='shinai', cascade_model=cascade_model)
    if max_picks is not None:
        kw['max_cycles'] = max_picks
    cfg = MissionConfig(sku.id, **kw)
    ex = Executive(cell, backend, ev, cfg)
    rec = TruthRecorder(rt.truth_case_poses, cell, ex.slots, left_in_transport=clock.left)
    ev.subscribe(rec.on_event)
    feed = None
    if record:
        from shinai_pi.inspector_feed import RunRecorder
        feed = RunRecorder(out, ex, cell)
        ev.subscribe(feed.on_event)
        backend.record_frames(out / 'frames', every_s=1.0 / fps, views=views)
        from shinai_pi.sensor_log import SensorLog
        backend.sensor_log = SensorLog(out / 'sensors')     # what the robot's own cameras saw, for the Inspector
    if on_event:
        ev.subscribe(on_event)
    contacts = ContactLog(path=out / 'truth_contacts.jsonl', gap_s=0.1, raw_path=out / 'truth_contacts_raw.jsonl')      # evaluation only: the robot never reads it
    steps = (StepTrace(out / 'trace', *rt.contact_bodies(), meta={'mass_kg': rt.body_masses(), 'dt': cell.physics.dt, 'gravity': -9.81})
             if trace else None)       # evaluation only, like the contacts
    rt.watch_contacts(contacts, steps)
    t0 = time.perf_counter()
    report = ex.run()
    wall = time.perf_counter() - t0
    rt.step_hooks.clear()
    contacts.close(rt.sim_time, {k: p for k, (p, _) in rt.truth_case_poses().items()})
    if steps:
        steps.close()
    ev.close()
    if feed:
        feed.close()
    score = rec.summary(report, set(ex.belief.boxes) if ex.belief else set(), held=rt.gripped())
    result = {'seed': seed, 'policy': policy, 'sku': sku.id, 'status': report.status, 'max_picks': max_picks, 'sim_seconds': report.seconds,
              'wall_seconds': round(wall, 1), 'observations': report.observations, 'side_observations': report.side_observations,
              'safety_stops': report.safety_stops, 'grip_failures': report.grip_failures, 'fallback_picks': report.fallback_picks,
              'learned_decisions': report.learned_decisions, 'learned_changed': report.learned_changed,
              'remaining': report.remaining, 'unseen_outside_reach': report.unseen_outside_reach, 'not_found': report.not_found,
              'transport': transport,
              'brake_decel': round(profile.brake_decel, 2), 'loose_layers': list(profile.loose_layers), **score}
    (out / 'result.json').write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding='utf-8')
    manifest = {'episode': f'{policy}-{seed}', 'git_commit': git_sha(), 'seed': seed, 'policy': policy,
                'cell_config': DEFAULT_CELL.name, 'cell_config_sha256': hashlib.sha256(DEFAULT_CELL.read_bytes()).hexdigest(),
                'simulator': 'NVIDIA Isaac Sim 6.2.0.0', 'physics_dt': cell.physics.dt, 'python': platform.python_version(),
                'run_datetime_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'result_status': report.status,
                'layout': [cell.layout.cols, cell.layout.rows, cell.layout.layers, cell.layout.interlock]}
    if policy == 'shinai_learned':
        manifest.update(cascade_model=Path(cascade_model).name,
                        cascade_model_sha256=hashlib.sha256(Path(cascade_model).read_bytes()).hexdigest())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[8])
    ap.add_argument('--policies', nargs='+', default=['shinai'])
    ap.add_argument('--sku', default='BEV-500x24')
    ap.add_argument('--record', action='store_true', help='save presentation frames and belief snapshots for the Inspector')
    ap.add_argument('--max-picks', type=int, default=None, help='stop after this many pick attempts (comparison runs)')
    ap.add_argument('--out', default=None, help='output directory for runs (default artifacts/runs)')
    ap.add_argument('--fps', type=float, default=4.0, help='recorded frames per simulated second')
    ap.add_argument('--views', nargs='+', default=['presentation'], help='recorded views: presentation side closeup')
    ap.add_argument('--layers', type=int, default=None,
                    help='build the load with fewer layers (stage S3: two layers, the top one loose)')
    ap.add_argument('--trace', action='store_true', help='keep every physics step of motion, tool state and contacts (trace/)')
    ap.add_argument('--scenarios', action='store_true',
                    help='each seed is a collapse scenario (shinai_pi/scenarios.py): layers, product and transport vary')
    ap.add_argument('--cascade-model', default='models/cascade_v2.json', help="the learned predictor of 'shinai_learned'")
    ap.add_argument('--grid', type=int, nargs=3, default=None, metavar=('COLS', 'ROWS', 'LAYERS'),
                    help='a smaller block of cases, layers not turned (a short demonstration load)')
    ap.add_argument('--resume', action='store_true', help='skip the seeds whose run already has a result (to continue a stopped batch)')
    ap.add_argument('--person-at', type=float, default=None,
                    help='simulated second at which a worker walks into the cell (light-curtain scenario)')
    args = ap.parse_args()
    cell = load_cell()
    if args.layers is not None:
        import dataclasses
        cell = dataclasses.replace(cell, layout=dataclasses.replace(cell.layout, layers=args.layers))
    if args.grid is not None:
        from shinai_pi.layout import with_grid
        cell = with_grid(cell, *args.grid)
    from shinai_pi.sim.runtime import IsaacRuntime
    rt = IsaacRuntime(cell)
    out_root = Path(args.out) if args.out else ROOT / 'artifacts' / 'runs'
    for seed in args.seeds:
        for policy in args.policies:
            if args.resume and not pending([seed], lambda s: out_root / f'{policy}_seed{s:03d}'):
                continue
            person = None
            if args.person_at is not None:
                t0 = args.person_at          # walks in from the side (not across the camera), stands in the zone, leaves
                person = [(0.0, -3.4, -2.4), (t0, -3.4, -2.4), (t0 + 3.0, -1.0, -1.35), (t0 + 7.0, -1.0, -1.35), (t0 + 10.0, -3.4, -2.4)]
            run_cell, sku_id = cell, args.sku
            if args.scenarios:
                from shinai_pi import scenarios
                scen = scenarios.sample(cell, seed)
                run_cell, sku_id = scen.cell, scen.sku.id
                print('SCENARIO', json.dumps({**scen.meta, 'split': scenarios.split(scen.meta)}, ensure_ascii=False), flush=True)
            rt.cell = run_cell
            r = run_one(rt, run_cell, seed, policy, sku_id, person=person, record=args.record, max_picks=args.max_picks,
                        out_root=out_root, fps=args.fps, views=tuple(args.views), trace=args.trace,
                        cascade_model=str(ROOT / args.cascade_model))
            print('RESULT', json.dumps({k: r[k] for k in ('seed', 'policy', 'status', 'sim_seconds', 'wall_seconds', 'placed_on_b',
                                                          'handed_over', 'left_on_a', 'secondary_collapses', 'disturbed_cases',
                                                          'observations', 'side_observations', 'drops_in_transit', 'prediction')},
                                       ensure_ascii=False), flush=True)
    rt.close()
