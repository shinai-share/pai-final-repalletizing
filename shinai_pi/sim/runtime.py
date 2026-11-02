"""Isaac Sim runtime: owns the Kit app, the stage, stepping, actuation and sensing.

The cognitive core never imports this module. It talks to the cell through
`shinai_pi.backend.CellBackend`, which this runtime implements for Isaac Sim.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from ..config import Cell

cellmod = None   # imported after Kit starts: pxr.PhysxSchema is only available then
PEOPLE_VIEWS = ('presentation', 'side', 'closeup')   # pictures only: the robot never reads them



# Isaac reports these joint reactions scaled by a constant (about 1/8) with the torques mirrored.
# Calibrated like a load cell with cases of known mass and centre of mass (scripts/probe_wrist.py,
# artifacts/probe_wrist.json: fitted on 11 kg and 8 kg, checked on 13.5 kg within 0.14 N and 0.08 N m).
WRIST_GAIN = (7.989, -7.993, -7.988)
GRIPPER_STATES = {'Open': 0, 'Closing': 1, 'Closed': 2}   # as the trace keeps the Surface Gripper's status
TRACE_MAX_POINTS = 8192   # contact points one step can report to the trace; the count per step is kept to show a shortfall

class IsaacRuntime:
    def __init__(self, cell: Cell, headless: bool = True):
        os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
        from isaacsim import SimulationApp
        t0 = time.perf_counter()
        self.app = SimulationApp({'headless': headless, 'width': 1280, 'height': 720,
                                  'anti_aliasing': cell.rendering.antialiasing})
        self.boot_seconds = time.perf_counter() - t0
        import omni.timeline
        import omni.usd
        global cellmod
        from . import cell as cellmod
        self.cell = cell
        self.ctx = omni.usd.get_context()
        self.timeline = omni.timeline.get_timeline_interface()
        self._cams = {}
        self.new_episode()

    def new_episode(self):
        """Fresh stage in the same Kit app (a process can start SimulationApp only once)."""
        if self.timeline.is_playing():
            self.timeline.stop()
            self.app.update()
        self._destroy_cameras()
        self.ctx.new_stage()
        self.app.update()
        self.stage = self.ctx.get_stage()
        self.case_ids = []
        self.wrap_paths = []
        self.sim_time = 0.0
        self.step_hooks = []                 # called after every physics step (evaluation only)
        self.gantry_target = np.zeros(len(cellmod.GANTRY_JOINTS))

    # ---------------------------------------------------------------- build / start
    def build(self, spawns, wrapped_layers=()):
        info = cellmod.build(self.stage, self.cell, spawns)
        self.case_ids = info['cases']
        # Contact reports let evaluation read the forces between bodies (watch_contacts); they do not change the physics.
        from pxr import PhysxSchema
        for q in [f'{cellmod.CASES}/{c}' for c in self.case_ids] + self._tool_parts():
            PhysxSchema.PhysxContactReportAPI.Apply(self.stage.GetPrimAtPath(q)).CreateThresholdAttr().Set(0.0)
        self.wrap_paths = cellmod.wrap_cases(self.stage, self.cell, spawns, set(wrapped_layers))
        return info

    def start(self):
        from isaacsim.core.experimental.prims import Articulation, RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.robot.surface_gripper import _surface_gripper
        self._sm = SimulationManager
        SimulationManager.set_physics_dt(self.cell.physics.dt)
        self.timeline.play()
        self.app.update()
        self.gantry = Articulation(cellmod.GANTRY + '/base')
        self.cases = RigidPrim([f'{cellmod.CASES}/{c}' for c in self.case_ids]) if self.case_ids else None
        self.gripper = _surface_gripper.acquire_surface_gripper_interface()
        self.gantry_dof_names = list(self.gantry.dof_names)
        # Commands and readings are in GANTRY_JOINTS order; the articulation may order its DOFs otherwise.
        self._dof_index = np.array([self.gantry_dof_names.index(j) for j in cellmod.GANTRY_JOINTS])
        self._fork_index = np.array([self.gantry_dof_names.index(j) for j in cellmod.FORK_JOINTS])
        self._head_link = list(self.gantry.link_names).index('head')
        f = self.cell.fork
        self._targets = np.zeros(len(self.gantry_dof_names))
        self._targets[self._fork_index] = (-f.extension_range[0], -f.extension_range[0], 0.0, 0.0)   # retracted
        self.fork_target = self._targets[self._fork_index].copy()

    # ---------------------------------------------------------------- stepping
    def step(self, n: int = 1, render: bool = False):
        for _ in range(n):
            if render:
                self.app.update()
            else:
                self._sm.step(steps=1)
            self.sim_time += self.cell.physics.dt
            for hook in self.step_hooks:
                hook()

    # ---------------------------------------------------------------- gantry
    def set_gantry_target(self, q):
        q = np.asarray(q, dtype=float)
        if len(q) == 4:                                  # a gantry move keeps the wrist where it is
            q = np.r_[q, self.gantry_target[4:]]
        self.gantry_target = q
        self._targets[self._dof_index] = q
        self.gantry.set_dof_position_targets(self._targets.reshape(1, -1))

    def gantry_state(self):
        q = np.asarray(self.gantry.get_dof_positions().numpy()).reshape(-1)[self._dof_index]
        v = np.asarray(self.gantry.get_dof_velocities().numpy()).reshape(-1)[self._dof_index]
        return q, v

    def _joint_forces(self):
        return np.asarray(self.gantry.get_dof_projected_joint_forces().numpy()).reshape(-1)[self._dof_index]

    def z_load(self) -> float:
        """Raw reaction of the z axis drive (uncalibrated, about 1/8 of a hanging load and including
        the ram's own inertia): used only to feel the empty tool push on a face."""
        return float(self._joint_forces()[2])

    def head_wrench(self):
        """Force and torque the wrist passes to the vacuum head (its incoming joint force): what a
        six-axis force sensor between wrist and head reads, in the head frame, about the centre of
        the cup face."""
        force, torque = self.gantry.get_link_incoming_joint_force()
        f = np.asarray(force.numpy()).reshape(-1, 3)[self._head_link]
        t = np.asarray(torque.numpy()).reshape(-1, 3)[self._head_link]
        return np.r_[f, t]

    def wrist_load(self):
        """(load along the tool axis, torque about tool x, torque about tool y), calibrated so that
        a hanging load reads (weight, r x F)."""
        w = self.head_wrench()
        return float(WRIST_GAIN[0] * w[2]), float(WRIST_GAIN[1] * w[3]), float(WRIST_GAIN[2] * w[4])

    # ---------------------------------------------------------------- fork (under-supports)
    def set_fork_target(self, extension, insert):
        """Per side (left, right): tine top this far below the cup face, tine slid this far along +y."""
        e = np.asarray(extension, float).reshape(2)
        y = np.asarray(insert, float).reshape(2)
        self.fork_target = np.r_[-e, y]
        self._targets[self._fork_index] = self.fork_target
        self.gantry.set_dof_position_targets(self._targets.reshape(1, -1))

    def set_fork_force_caps(self, z_caps, y_caps):
        """Force caps of the bar drives (pushing a tine up against the case bottom presses no harder
        than this) and of the tine drives."""
        caps = np.asarray(self.gantry.get_dof_max_efforts().numpy()).reshape(-1).copy()
        caps[self._fork_index] = np.r_[np.asarray(z_caps, float).reshape(2), np.asarray(y_caps, float).reshape(2)]
        self.gantry.set_dof_max_efforts(caps.reshape(1, -1))

    def fork_state(self):
        """(extension per side, insert per side), measured."""
        q = np.asarray(self.gantry.get_dof_positions().numpy()).reshape(-1)[self._fork_index]
        return -q[:2], q[2:]

    def fork_raw_forces(self):
        """Projected joint forces of (bar l, bar r, tine l, tine r): uncalibrated reactions along each
        joint axis. Scale and sign are found by scripts/probe_fork.py."""
        return np.asarray(self.gantry.get_dof_projected_joint_forces().numpy()).reshape(-1)[self._fork_index]

    def tool_height(self, q) -> float:
        return self.cell.gantry.origin_height + float(q[2])

    # ---------------------------------------------------------------- gripper
    def grip(self, close: bool):
        if close:
            self.gripper.close_gripper(cellmod.GRIPPER)
        else:
            self.gripper.open_gripper(cellmod.GRIPPER)

    def gripped(self) -> list:
        return [p.rsplit('/', 1)[-1] for p in self.gripper.get_gripped_objects(cellmod.GRIPPER)]

    def gripper_status(self) -> str:
        return str(self.gripper.get_gripper_status(cellmod.GRIPPER)).rsplit('.', 1)[-1]

    # ---------------------------------------------------------------- transport (deck frame)
    def set_gravity(self, g_vec):
        """Applied directly to the running PhysX scene (USD attribute edits are not seen
        between physics-only steps)."""
        self._sm.get_physics_sim_view().set_gravity([float(v) for v in g_vec])

    def gravity(self):
        return tuple(self._sm.get_physics_sim_view().get_gravity())

    def cut_wrap(self):
        cellmod.cut_wrap(self.stage, self.wrap_paths)

    # ---------------------------------------------------------------- person model (for the light curtain scenario)
    def add_person_model(self):
        """A visual-only stand-in for a worker (no collision): the light curtain reads its position."""
        from pxr import UsdGeom
        from . import usd_build as ub
        path = cellmod.CELL + '/person'
        xf = UsdGeom.Xform.Define(self.stage, path)
        mat = ub.material(self.stage, cellmod.CELL + '/mat/person', (0.95, 0.55, 0.10))
        body = UsdGeom.Capsule.Define(self.stage, path + '/body')
        body.CreateRadiusAttr(0.22)
        body.CreateHeightAttr(1.25)
        ub.set_pose(body.GetPrim(), (0, 0, 0.85))
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(mat)
        self._person = xf.GetPrim()
        return self._person

    def set_person_position(self, xy):
        from . import usd_build as ub
        ub.set_pose(self._person, (float(xy[0]), float(xy[1]), 0.0))

    # ---------------------------------------------------------------- ground truth (evaluation only)
    def truth_case_poses(self) -> dict:
        pos, quat = self.cases.get_world_poses()
        pos = np.asarray(pos.numpy())
        quat = np.asarray(quat.numpy())
        return {cid: (pos[i].copy(), quat[i].copy()) for i, cid in enumerate(self.case_ids)}

    def _tool_parts(self):
        """The bodies of the tool that can touch a case: the plate (on the wrist) and the fork bars and tines."""
        parts = [cellmod.GANTRY + '/wrist_r'] + [f'{cellmod.GANTRY}/fork_{k}_{s}' for k in ('bar', 'tine') for s in ('l', 'r')]
        return [q for q in parts if self.stage.GetPrimAtPath(q).IsValid()]

    def watch_contacts(self, log, trace=None):
        """Feed `log` (evaluation.contacts.ContactLog) the force between every case and tool part and every other
        one, the pallets, the tables and the floor, after every physics step. With `trace`
        (evaluation.trace.StepTrace made from `contact_bodies()`), also every step's motion, tool state and
        contact points. Evaluation only."""
        from isaacsim.core.experimental.prims import RigidPrim
        rows, cols = self._contact_paths()
        view = RigidPrim(rows, contact_filter_paths=cols, max_contact_count=TRACE_MAX_POINTS if trace else 0)
        row_names = [q.rsplit('/', 1)[1] for q in rows]
        names = [q.rsplit('/', 1)[1] for q in cols]
        dt = self.cell.physics.dt

        def hook():
            m = np.asarray(view.get_contact_force_matrix(dt=dt).numpy())
            pos = np.asarray(self.cases.get_world_poses()[0].numpy())
            log.step(self.sim_time, dt, row_names, names, m, {c: pos[i] for i, c in enumerate(self.case_ids)})
            if trace is not None:
                p, q = (np.asarray(a.numpy()) for a in view.get_world_poses())
                lin, ang = (np.asarray(a.numpy()) for a in view.get_velocities())
                fn, pts, nrm, _, counts, starts = (np.asarray(a.numpy()) for a in view.get_contact_force_data(dt=dt))
                ft = np.asarray(view.get_friction_data(dt=dt)[0].numpy())
                gq, gv = self.gantry_state()
                trace.step(self.sim_time, np.hstack([p, q]), np.hstack([lin, ang]), m,
                           detail=(fn, pts, nrm, counts, starts, ft),
                           tool={'gantry_q': gq, 'gantry_v': gv, 'z_load': self.z_load(), 'head_wrench': self.head_wrench(),
                                 'fork_raw': self.fork_raw_forces(), 'fork_state': np.hstack(self.fork_state()),
                                 'gripped': [row_names.index(k) if k in row_names else -1 for k in self.gripped()[:1]] or [-1],
                                 'points': int(counts.sum()), 'gripper': GRIPPER_STATES.get(self.gripper_status(), -1)})
        self.step_hooks.append(hook)
        return names

    def contact_bodies(self):
        """(rows, cols) by name, as watch_contacts reads them: a StepTrace is made with these."""
        rows, cols = self._contact_paths()
        return [q.rsplit('/', 1)[1] for q in rows], [q.rsplit('/', 1)[1] for q in cols]

    def body_masses(self):
        """Mass (kg) of every case and tool part as built: the trace's force balance needs it."""
        from pxr import UsdPhysics
        rows, _ = self._contact_paths()
        return {q.rsplit('/', 1)[1]: float(UsdPhysics.MassAPI(self.stage.GetPrimAtPath(q)).GetMassAttr().Get() or 0.0) for q in rows}

    def _contact_paths(self):
        rows = [f'{cellmod.CASES}/{c}' for c in self.case_ids] + self._tool_parts()
        fixed = [f'{cellmod.CELL}/{n}' for n in ('pallet_a', 'pallet_b', 'floor')]
        fixed += [str(q.GetPath()) for q in self.stage.GetPrimAtPath(cellmod.CELL).GetChildren() if q.GetName().startswith('staging_')]
        return rows, rows + [q for q in fixed if self.stage.GetPrimAtPath(q).IsValid()]

    # ---------------------------------------------------------------- cameras
    def _camera(self, key):
        if key in self._cams:
            return self._cams[key]
        import omni.replicator.core as rep
        r = self.cell.rendering
        res = {'tool': self.cell.tool_camera.resolution, 'presentation': r.people_resolution,
               'side': r.inset_resolution, 'closeup': r.inset_resolution,
               'under': self.cell.under_camera.resolution,
               'down': self.cell.down_camera.resolution,
               'overhead_b': self.cell.overhead_camera.resolution_b or self.cell.overhead_camera.resolution,
               }.get(key, self.cell.overhead_camera.resolution)
        rp = rep.create.render_product(cellmod.CAMERAS[key], tuple(res))
        annots = {}
        names = ('rgb',) if key in PEOPLE_VIEWS else ('rgb', 'distance_to_image_plane', 'instance_id_segmentation_fast', 'camera_params')
        for name in names:
            kwargs = {'init_params': {'colorize': False}} if name.startswith('instance') else {}
            a = rep.AnnotatorRegistry.get_annotator(name, **kwargs)
            a.attach(rp)
            annots[name] = a
        self._cams[key] = (rp, annots)
        return self._cams[key]

    def _activate(self, key):
        """Render only the camera being read: every enabled render product is drawn on each step."""
        for k, (rp, _) in self._cams.items():
            rp.hydra_texture.set_updates_enabled(k == key)

    def capture(self, key: str) -> dict:
        """Render the current state synchronously and return the frame. Physics does not advance.

        Annotator data lags the render loop by a frame or more; stepping the replicator
        orchestrator with wait_for_render makes the returned frame match the present state."""
        import omni.replicator.core as rep
        _, a = self._camera(key)
        self._activate(key)
        self._sm.step(steps=1)                            # physics writes poses back to USD, which the renderer reads
        self.sim_time += self.cell.physics.dt
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=1, pause_timeline=False, delta_time=0.0, wait_for_render=True)
            if np.asarray(a['distance_to_image_plane'].get_data()).size:
                break
        else:
            raise RuntimeError(f'camera {key} produced no data')
        seg = a['instance_id_segmentation_fast'].get_data()
        params = a['camera_params'].get_data()
        return {
            'rgb': np.asarray(a['rgb'].get_data())[..., :3].copy(),
            'depth': np.asarray(a['distance_to_image_plane'].get_data()).copy(),
            'instance': np.asarray(seg['data']).copy(),
            'instance_map': {int(k): str(v) for k, v in seg['info']['idToLabels'].items()},
            'view': np.asarray(params['cameraViewTransform']).reshape(4, 4).copy(),
            'projection': np.asarray(params['cameraProjection']).reshape(4, 4).copy(),
            'resolution': tuple(int(v) for v in params['renderProductResolution']),
            'stamp': self.sim_time,
        }

    def _destroy_cameras(self):
        for rp, annots in self._cams.values():
            for a in annots.values():
                a.detach()
            rp.destroy()
        self._cams = {}

    def render_rgb(self, key: str = 'presentation') -> np.ndarray:
        """A picture for people (no physics step): what the presentation camera sees now."""
        import omni.replicator.core as rep
        _, a = self._camera(key)
        self._activate(key)
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=self.cell.rendering.people_subframes, pause_timeline=False,
                                  delta_time=0.0, wait_for_render=True)
            img = np.asarray(a['rgb'].get_data())
            if img.size:
                return img[..., :3].copy()
        raise RuntimeError(f'camera {key} produced no image')

    def close(self):
        try:
            self.timeline.stop()
        finally:
            self.app.close()


def run_transport(rt: IsaacRuntime, profile, settle_speed: float = 0.01, max_settle_s: float = 8.0, on_step=None) -> dict:
    """Play the transport event, restore gravity, wait until the load is at rest, cut the film.
    `on_step` is called after every physics step (to film the collapse)."""
    dt = rt.cell.physics.dt
    n = int(math.ceil(profile.duration / dt))
    for k in range(n):
        rt.set_gravity(profile.effective_gravity(k * dt))
        rt.step(1)
        if on_step:
            on_step()
    rt.set_gravity((0.0, 0.0, -9.81))
    rt.cut_wrap()
    settled_after = None
    for k in range(int(max_settle_s / dt)):
        rt.step(1)
        if on_step:
            on_step()
        if k % 12 == 0 and rt.cases is not None:
            lin, ang = rt.cases.get_velocities()
            if float(np.max(np.linalg.norm(np.asarray(lin.numpy()), axis=1))) < settle_speed and \
               float(np.max(np.linalg.norm(np.asarray(ang.numpy()), axis=1))) < 10 * settle_speed:
                settled_after = k * dt
                break
    return {'transport_s': round(n * dt, 3), 'settled_after_s': None if settled_after is None else round(settled_after, 3)}
