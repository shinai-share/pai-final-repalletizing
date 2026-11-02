"""Draw a recorded run again, as it moved, for people: every case and the tool put where the run's physics-step
trace (run_mission.py --trace) has them, in a scene dressed for presentation (shinai_pi/sim/visual_detail.py).

Nothing is simulated here. The cases are kinematic, nothing collides, gravity is off, and the robot's cameras and
decisions are not run: the pictures show the recorded motion and nothing else. After placing each frame the scene is
read back and compared with the trace row it came from; the largest differences go into the manifest, and a frame
off by more than PLACE_TOL is an error.

    .venv-isaac/Scripts/python scripts/rerender_trace.py --run DIR (--layers 2 | --scenario) --out DIR [--from S] [--to S]
        [--speed 4] [--fps 30] [--views presentation closeup] [--uhd]
Writes <out>/frames_<view>/<ms>.jpg and <out>/manifest.json. Exit 1 on any error (Isaac exits 0 after one)."""
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shinai_pi.config import DEFAULT_CELL, load_cell  # noqa: E402
from shinai_pi.layout import initial_cases  # noqa: E402
from shinai_pi.variation import vary_load  # noqa: E402

PLACE_TOL_M = 1e-3             # a case drawn more than this from where the trace has it is an error
DOF_TOL = 1e-3                 # m or rad, each joint
ASSETS = ROOT / 'artifacts' / 'presentation_assets'


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_trace(folder):
    idx = json.loads((folder / 'trace_index.json').read_text(encoding='utf-8'))
    parts = {k: [] for k in ('t', 'pose', 'tool_gantry_q', 'tool_fork_state')}
    for k in range(idx['chunks']):
        with np.load(folder / f'trace_{k:05d}.npz') as z:
            for key in parts:
                parts[key].append(z[key])
    tr = {k: np.concatenate(v) for k, v in parts.items()}
    if len(tr['t']) != idx['steps']:
        raise RuntimeError(f"trace has {len(tr['t'])} steps, its index says {idx['steps']}")
    return idx, tr


def git(*cmd):
    return subprocess.run(['git', *cmd], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def inert(stage):
    """No body is pushed by another: every collider off (the run's contacts are in the trace). The cases stay
    dynamic bodies, with gravity off and no velocity: a kinematic body's pose set through the physics view is not
    written back to what the renderer reads (smoke1 drew every case where it was built)."""
    from pxr import UsdPhysics
    n = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            n += 1
    return n


TINES_OUT_M = 0.0              # extension past which a tine is out of its guide (retracted rest -0.08)


def tines_out(fork_state):
    """Both tines out from their guides (tool_fork_state = extension left, right, stroke left, right)."""
    return bool(min(fork_state[0], fork_state[1]) > TINES_OUT_M)


def scene(base, seed, scenario, layers, sku_id=None, grid=None):
    """The pile the run was built with: a collapse scenario's (run_mission.py --scenarios), or the base cell with
    the layers given."""
    if scenario:
        from shinai_pi import scenarios
        s = scenarios.sample(base, seed)
        return s.cell, s.sku, s.spawns
    if layers is None and grid is None:
        raise ValueError('a run made without --scenarios needs --layers')
    if grid is not None:
        from shinai_pi.layout import with_grid
        cell = with_grid(base, *grid)
    else:
        cell = dataclasses.replace(base, layout=dataclasses.replace(base.layout, layers=layers))
    sku = cell.sku(sku_id or 'BEV-500x24')
    return cell, sku, vary_load(initial_cases(cell, sku), cell.variation, seed)


def main(a):
    run = Path(a.run)
    manifest_run = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if manifest_run['cell_config_sha256'] != sha256(DEFAULT_CELL):
        raise RuntimeError('configs/cell.toml is not the one the run was made with')
    idx, tr = load_trace(run / 'trace')
    t_all = tr['t']
    t0 = a.t_from if a.t_from is not None else float(t_all[0])
    t1 = a.t_to if a.t_to is not None else float(t_all[-1])
    times = np.arange(t0, t1, a.speed / a.fps)

    cell, _, spawns = scene(load_cell(), int(manifest_run['seed']), a.scenario, a.layers, manifest_run.get('sku'), a.grid)
    if a.uhd:
        cell = dataclasses.replace(cell, rendering=dataclasses.replace(
            cell.rendering, people_resolution=(3840, 2160), inset_resolution=(1920, 1080)))
    from shinai_pi.sim.runtime import IsaacRuntime
    rt = IsaacRuntime(cell)                      # starts Kit: the USD schemas below import only after it
    from shinai_pi.sim import visual_detail as vd
    rt.build(spawns, wrapped_layers=())
    cases = [r for r in idx['rows'] if r.startswith('case_')]
    if sorted(cases) != sorted(rt.case_ids):
        raise RuntimeError(f'the scene has cases {rt.case_ids}, the trace {cases}')
    colliders = inert(rt.stage)
    vd.textures(ASSETS)
    look = vd.apply(rt.stage, ASSETS, cell)
    rt.start()
    rt.set_gravity((0.0, 0.0, 0.0))
    row_of = [idx['rows'].index(c) for c in rt.case_ids]
    from isaacsim.core.experimental.prims import XformPrim
    from shinai_pi.sim import cell as cellmod
    drawn = XformPrim([f'{cellmod.CASES}/{c}' for c in rt.case_ids])
    ndof = len(rt.gantry_dof_names)
    out = Path(a.out)
    dirs = {v: out / f'frames_{v}' for v in a.views}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    worst = {'case_m': 0.0, 'dof': 0.0}
    tines = {}
    start = time.perf_counter()
    for k, t in enumerate(times):
        i = int(min(np.searchsorted(t_all, t), len(t_all) - 1))
        pose = tr['pose'][i]
        rt.cases.set_world_poses(positions=pose[row_of, :3], orientations=pose[row_of, 3:])
        rt.cases.set_velocities(np.zeros((len(row_of), 3)), np.zeros((len(row_of), 3)))
        q = np.asarray(rt.gantry.get_dof_positions().numpy()).reshape(-1).copy()
        q[rt._dof_index] = tr['tool_gantry_q'][i]
        e_l, e_r, y_l, y_r = tr['tool_fork_state'][i]
        q[rt._fork_index] = (-e_l, -e_r, y_l, y_r)
        rt.gantry.set_dof_positions(q.reshape(1, -1))
        rt.gantry.set_dof_velocities(np.zeros((1, ndof)))
        rt.gantry.set_dof_position_targets(q.reshape(1, -1))
        rt._sm.step(steps=1)                       # writes the poses back to USD, which the renderer reads
        got = np.asarray(drawn.get_world_poses()[0].numpy())     # what the renderer reads, not the physics view
        dq = np.asarray(rt.gantry.get_dof_positions().numpy()).reshape(-1) - q
        worst['case_m'] = max(worst['case_m'], float(np.max(np.linalg.norm(got - pose[row_of, :3], axis=1))))
        worst['dof'] = max(worst['dof'], float(np.max(np.abs(dq[np.r_[rt._dof_index, rt._fork_index]]))))
        if worst['case_m'] > PLACE_TOL_M or worst['dof'] > DOF_TOL:
            raise RuntimeError(f'frame {k} at {t:.3f} s drawn off the trace: {worst}')
        name = f'{int(round(float(t_all[i]) * 1000)):08d}.jpg'
        tines[name] = tines_out(tr['tool_fork_state'][i])
        for v, d in dirs.items():
            Image.fromarray(rt.render_rgb(v)).save(d / name, quality=92)
        if k % 50 == 0:
            print(f'FRAME {k + 1}/{len(times)} t={t:.1f} worst={worst}', flush=True)
    manifest = {'what': 'the recorded run drawn again from its physics-step trace; nothing simulated',
                'run': str(run), 'run_commit': manifest_run['git_commit'], 'seed': manifest_run['seed'],
                'trace_steps': idx['steps'], 'from_s': t0, 'to_s': t1, 'speed': a.speed, 'fps': a.fps,
                'frames': len(times), 'views': list(a.views),
                'resolution': {'people': list(cell.rendering.people_resolution),
                               'inset': list(cell.rendering.inset_resolution)},
                'largest_difference_from_trace': worst, 'tolerance': {'case_m': PLACE_TOL_M, 'dof': DOF_TOL},
                'colliders_switched_off': colliders, 'appearance': look,
                'commit': git('rev-parse', 'HEAD'),
                'dirty_paths': git('status', '--porcelain', '--', 'shinai_pi', 'scripts', 'configs').splitlines(),
                'command': ' '.join(sys.argv), 'wall_s': round(time.perf_counter() - start, 1)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8',
                                       newline='\n')
    (out / 'tines.json').write_text(json.dumps(tines), encoding='utf-8', newline='\n')
    print('DONE', json.dumps({'frames': len(times), 'worst': worst}), flush=True)
    sys.stdout.flush()
    rt.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='a run folder with trace/ (run_mission.py --trace)')
    ap.add_argument('--layers', type=int, default=None, help='the layers the run was built with (not with --scenario)')
    ap.add_argument('--grid', type=int, nargs=3, default=None, help='the run was made with run_mission.py --grid')
    ap.add_argument('--scenario', action='store_true', help='the run was made with run_mission.py --scenarios')
    ap.add_argument('--out', required=True)
    ap.add_argument('--from', dest='t_from', type=float, default=None, help='simulated second to start at')
    ap.add_argument('--to', dest='t_to', type=float, default=None)
    ap.add_argument('--speed', type=float, default=4.0, help='simulated seconds per second of video')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--views', nargs='+', default=['presentation', 'closeup'])
    ap.add_argument('--uhd', action='store_true', help='3840 x 2160 overview and 1920 x 1080 close-up')
    try:
        main(ap.parse_args())
    except BaseException as e:                   # Isaac ends the process with 0 after an exception: it must not read as a pass
        print('FAILED', type(e).__name__, e, flush=True)
        os._exit(1)
