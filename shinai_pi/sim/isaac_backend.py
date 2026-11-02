"""CellBackend on Isaac Sim: drives the gantry articulation and the suction gripper, renders the
cameras, and simulates the light curtain from a person model's position.

The cognitive core only sees what this interface returns. Ground truth stays in the runtime and
is read by the evaluation harness, never through this class."""
from __future__ import annotations

import math

import numpy as np

from ..backend import MoveResult
from ..hand import TouchdownDetector
from ..motion import limits_for, plan_move
from ..perception.geometry import CameraFrame, intrinsics_from_projection, normalise_instance_map
from ..safety import Action, TickState
from .runtime import IsaacRuntime

CONTACT_ERROR = 0.012          # m of z tracking error that means the empty tool met something
CONTACT_SURE = 4.0             # a change this many thresholds at once is contact, not noise
TOUCH_SETTLE_S = 0.15          # s at the start of a touch-down search where the drive's own start-up swing is not contact
                               # (probe 2026-10-18: -103 N in the first steps, 0.3 mm of travel; the search starts 6 cm above the face)
CONTACT_TICKS = 3              # control ticks a load change must last to count as contact
CONTACT_ERROR_CARRYING = 0.03  # m: the same while carrying (the load cell normally answers first)
UNLOAD_FRACTION = 0.5          # touchdown when this share of the carried weight leaves the load cell
RESET_CLEAR_S = 2.0            # the scripted operator resets after the area has been clear this long


WRIST_AVERAGE_S = 0.25
FORK_TICKS = 3                 # physics steps a tine's resistance must last to count
UNLOAD_CAP = 20.0              # N the tines keep on the surface while the cups take the case back
TRANSFER_S = 1.0               # s over which the weight passes between tines and cups
UNLOAD_RAISE = 0.003           # m the tool rises off the landed tines
RELEASE_DROP = 0.005           # m the tines drop off the case bottom before sliding out
PRESS_BEYOND = 0.03            # m past the case bottom the press aims: the force cap, not the target, ends it

class PersonScript:
    """A person model walking into the cell and out again: list of (t, x, y) keyframes."""

    def __init__(self, keys=()):
        self.keys = sorted(keys)

    def position(self, t):
        if not self.keys:
            return None
        if t <= self.keys[0][0]:
            return self.keys[0][1:]
        for (t0, x0, y0), (t1, x1, y1) in zip(self.keys, self.keys[1:]):
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
                return (x0 + a * (x1 - x0), y0 + a * (y1 - y0))
        return self.keys[-1][1:]


class IsaacBackend:
    def __init__(self, rt: IsaacRuntime, person: PersonScript | None = None, guard_zone=None, control_every: int = 2):
        self.rt = rt
        self.sensor_log = None                        # shinai_pi.sensor_log.SensorLog: keep what the robot's cameras saw
        self.cell = rt.cell
        self.person = person or PersonScript()
        g = self.cell.gantry
        self.zone = guard_zone or ((g.x_range[0] - 0.3, g.x_range[1] + 0.3), (g.y_range[0] - 0.3, g.y_range[1] + 0.3))
        self.control_every = control_every
        self._clear_since = 0.0
        self._was_clear = True
        self.carried_weight = 9.81 * max(sk.mass for sk in self.cell.skus)   # set per mission by the runner
        self.fork_engaged = False
        self._fork_tare = np.zeros(4)
        self.frame_dir = None
        self.frame_views = ('presentation',)
        self.frame_every = 0.25
        self._next_frame = 0.0

    def record_frames(self, directory, every_s: float = 0.25, views=('presentation',)):
        """Save pictures for people every `every_s` simulated seconds (for the Inspector and video):
        the presentation camera into `directory`, other views into `directory`_<view>."""
        self.frame_dir = directory
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.frame_views = tuple(views)
        for v in self.frame_views[1:]:
            (directory.parent / f'{directory.name}_{v}').mkdir(parents=True, exist_ok=True)
        self.frame_every = every_s
        self._next_frame = self.rt.sim_time

    def _frame(self):
        if self.frame_dir is None or self.rt.sim_time < self._next_frame:
            return
        from PIL import Image
        name = f'{int(round(self.rt.sim_time * 1000)):08d}.jpg'
        for v in self.frame_views:
            out = self.frame_dir if v == 'presentation' else self.frame_dir.parent / f'{self.frame_dir.name}_{v}'
            Image.fromarray(self.rt.render_rgb(v)).save(out / name, quality=90)
        self._next_frame += self.frame_every
        if self._next_frame < self.rt.sim_time:
            self._next_frame = self.rt.sim_time + self.frame_every
        self._person_prim = None

    # ------------------------------------------------------------------ helpers
    def _update_person(self):
        pos = self.person.position(self.rt.sim_time)
        if pos is None:
            return
        if self._person_prim is None:
            self._person_prim = self.rt.add_person_model()
        self.rt.set_person_position(pos)

    def curtain_clear(self):
        pos = self.person.position(self.rt.sim_time)
        if pos is None:
            clear = True
        else:
            (x0, x1), (y0, y1) = self.zone
            clear = not (x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1)
        if clear and not self._was_clear:
            self._clear_since = self.rt.sim_time
        self._was_clear = clear
        return clear

    def _tick(self, monitor, carrying):
        q, v = self.rt.gantry_state()
        return monitor(TickState(self.rt.sim_time, q, v, carrying, self.gripped(), self.curtain_clear(), supported=self.fork_engaged))

    def _hold(self):
        q, _ = self.rt.gantry_state()
        self.rt.set_gantry_target(q)
        for _ in range(int(0.25 / self.cell.physics.dt)):
            self.rt.step(1)
            self._frame()

    # ------------------------------------------------------------------ protocol
    def now(self):
        return self.rt.sim_time

    def gantry_q(self):
        return self.rt.gantry_state()[0]

    def operator_reset_requested(self):
        if self.curtain_clear() and self.rt.sim_time - self._clear_since >= RESET_CLEAR_S:
            return '作業者'
        return None

    def gripped(self):
        return self.rt.gripped()

    def wrist_load(self):
        """The force sensor's reading, averaged over a quarter second as a sensor filter would: a
        single physics step carries contact impulses many times the load."""
        vals = []
        for _ in range(int(round(WRIST_AVERAGE_S / self.cell.physics.dt))):
            self._update_person()
            self.rt.step(1)
            self._frame()
            vals.append(self.rt.wrist_load())
        return tuple(float(v) for v in np.mean(vals, axis=0))

    def grip(self, close):
        self.rt.grip(close)

    def idle(self, seconds, monitor, carrying=None):
        dt = self.cell.physics.dt
        for k in range(int(math.ceil(seconds / dt))):
            self._update_person()
            self.rt.step(1)
            self._frame()
            if k % self.control_every == 0 and self._tick(monitor, carrying).action is Action.STOP:
                return MoveResult(False, 'stop', k * dt)
        return MoveResult(True, seconds=seconds)

    def move(self, goal, monitor, carrying=None, speed=1.0, stop_on_contact=False, guard=None, fine=False):
        g = self.cell.gantry
        dt = self.cell.physics.dt
        start = np.asarray(self.rt.gantry_target, float)
        goal = np.asarray(goal, float)
        if len(goal) == 4:                              # a gantry move keeps the wrist where it is
            goal = np.r_[goal, start[4:]]
        prof = plan_move(start, goal, *limits_for(g, slow=speed < 1.0, touch=stop_on_contact, fine=fine))
        n = int(math.ceil(prof.duration / dt))
        t0 = self.rt.sim_time
        # Setting a case down: its weight leaves the cups (the calibrated head sensor). Pushing with
        # the empty tool: the z drive meets resistance. The z lag of a loaded slow descent reaches a
        # centimetre, so lag alone stops a carried case only well past that.
        load = (lambda: self.rt.wrist_load()[0]) if carrying else self.rt.z_load
        # Carrying: the cups hold the case's weight until it rests on something; compare with an
        # absolute floor rather than a reference read while the case may still bounce on the cups.
        baseline = 0.0 if carrying or not stop_on_contact else self._settled_load(load)
        gain = self.cell.contact.link_force_gain
        if stop_on_contact and not carrying and gain and abs(baseline) * gain > self.cell.contact.touch_static:
            # Already pressed against something before the search begins: whatever follows would be read
            # as a press on the face (S3 102: tines on pallet B's stack, -3,275 at the start).
            return MoveResult(False, 'collision', info={'kind': 'preload', 'load_N': round(abs(baseline) * gain, 1)})
        free = self.rt.z_load() if not carrying and not stop_on_contact else None
        struck = 0
        over, peak = 0, 0.0
        threshold = (UNLOAD_FRACTION * self.carried_weight if carrying else self._touch_raw()) if stop_on_contact else None
        # With the tines under the case they touch first: their loads change and stay changed.
        tines = TouchdownDetector(self.fork_loads(), self.carried_weight) if (carrying and self.fork_engaged and stop_on_contact) else None
        lag_limit = CONTACT_ERROR_CARRYING if carrying else CONTACT_ERROR
        every = max(1, int(round(self.cell.down_camera.every_s / dt)))
        for k in range(1, n + 1):
            if guard is not None and (k - 1) % every == 0:
                stop, report = guard()
                if stop:
                    self._hold()
                    return MoveResult(False, 'guard', self.rt.sim_time - t0, guard=report)
            target = prof.position(k * dt)
            self.rt.set_gantry_target(target)
            self._update_person()
            self.rt.step(1)
            self._frame()
            # A touch-down search reads its load every physics step: the stiff drive would push one
            # more step before the next control tick.
            if not stop_on_contact and k % self.control_every:
                continue
            if k % self.control_every == 0 and self._tick(monitor, carrying).action is Action.STOP:
                self._hold()
                return MoveResult(False, 'stop', self.rt.sim_time - t0)
            if free is not None and gain:
                dev = abs(self.rt.z_load() - free) * gain
                struck = struck + 1 if dev > self.cell.contact.collision else 0
                if struck >= CONTACT_TICKS:
                    self._hold()
                    return MoveResult(False, 'collision', self.rt.sim_time - t0, info={'kind': 'collision', 'load_N': round(dev, 1)})
            if tines is not None and target[2] < start[2]:
                if tines.update(self.fork_loads()):
                    self.last_contact = {'tines': True, 'peak_N': round(tines.peak, 1), 'carrying': carrying}
                    self._hold()
                    lasting = float(np.abs(np.asarray(self._fork_mean(0.1)) - tines.carried).max())
                    return MoveResult(True, 'contact', self.rt.sim_time - t0, contact=True,
                                      info={'kind': 'landing', 'static_N': lasting, 'peak_N': tines.peak})
                continue
            if stop_on_contact and target[2] < start[2] and not carrying and k * dt < TOUCH_SETTLE_S:
                continue
            if stop_on_contact and target[2] < start[2]:
                q, _ = self.rt.gantry_state()
                lag = float(q[2] - target[2])
                if carrying:
                    change = float(self.carried_weight - load())       # weight that has left the cups
                else:
                    change = float(abs(load() - baseline))
                # A single physics step can jump by tens of newtons (up to 1.5 times the threshold): a
                # change must last, unless it is far beyond what noise does.
                over = over + 1 if (threshold is not None and change > threshold) else 0
                touched = lag > lag_limit or over >= CONTACT_TICKS or (threshold is not None and change > CONTACT_SURE * threshold)
                peak = max(peak, change)
                if touched:
                    self.last_contact = {'lag_m': round(lag, 4), 'load_change_N': round(change, 1), 'peak_N': round(peak, 1),
                                         'carrying': carrying}
                    self._hold()
                    info = None
                    if not carrying:
                        # The empty cups on a face: the lasting press is what remains once the tool has stopped
                        # (averaged); the step at the moment of impact is the rigid model's spike, kept apart.
                        k = self.cell.contact.link_force_gain
                        lasting = abs(self._settled_load(load) - baseline)
                        info = ({'kind': 'touch', 'static_N': abs(k) * lasting, 'peak_N': abs(k) * peak} if k
                                else {'kind': 'touch', 'raw': (lasting, peak)})
                    return MoveResult(True, 'contact', self.rt.sim_time - t0, contact=True, info=info)
        settle = int(0.15 / dt)
        for k in range(settle):
            self.rt.step(1)
            self._frame()
            if k % self.control_every == 0 and self._tick(monitor, carrying).action is Action.STOP:
                self._hold()
                return MoveResult(False, 'stop', self.rt.sim_time - t0)
        return MoveResult(True, seconds=self.rt.sim_time - t0)

    def _touch_raw(self):
        """The empty cups' touch threshold in raw units of the up-down axis reading, from the force in
        newtons and the calibration. Without a calibration the cups would press blind: refuse."""
        c = self.cell.contact
        if c.link_force_gain <= 0:
            raise RuntimeError('上下軸の読みが校正されていない（scripts/probe_zload.py で校正する）')
        return c.touch_detect / c.link_force_gain

    def _settled_load(self, load, seconds=0.1):
        """Average a load reading while holding still, as the reference for a touch-down search."""
        vals = []
        for _ in range(max(1, int(round(seconds / self.cell.physics.dt)))):
            self.rt.step(1)
            self._frame()
            vals.append(load())
        return float(np.mean(vals))

    # ------------------------------------------------------------------ under-supports
    def _fork_ramp(self, extension, insert, monitor, resist=None):
        """Ramp both forks to (extension, insert) at the configured speeds. With `resist`, stop when
        a tine's resistance along its slide lasts above it (contact while sliding in or out)."""
        f = self.cell.fork
        dt = self.cell.physics.dt
        e0 = float(np.mean(-self.rt.fork_target[:2]))
        y0 = float(np.mean(self.rt.fork_target[2:]))
        n = max(1, int(math.ceil(max(abs(extension - e0) / f.z_speed, abs(insert - y0) / f.y_speed) / dt)))
        over, t0 = 0, self.rt.sim_time
        held = self.rt.gripped()
        carrying = held[0] if held else None
        for k in range(1, n + 1):
            a = k / n
            self.rt.set_fork_target([e0 + a * (extension - e0)] * 2, [y0 + a * (insert - y0)] * 2)
            self._update_person()
            self.rt.step(1)
            self._frame()
            if resist is not None:
                slide = np.abs(self.rt.fork_raw_forces() - self._fork_tare)[2:]
                over = over + 1 if slide.max() > resist else 0
                if over >= FORK_TICKS:
                    self._fork_hold()
                    return MoveResult(False, 'blocked', self.rt.sim_time - t0, contact=True)
            if k % self.control_every == 0 and self._tick(monitor, carrying).action is Action.STOP:
                self._fork_hold()
                return MoveResult(False, 'stop', self.rt.sim_time - t0)
        return MoveResult(True, seconds=self.rt.sim_time - t0)

    def _fork_hold(self):
        e, y = self.rt.fork_state()
        self.rt.set_fork_target(e, y)

    def _steps(self, seconds):
        for _ in range(int(round(seconds / self.cell.physics.dt))):
            self._update_person()
            self.rt.step(1)
            self._frame()

    def fork_insert(self, extension, insert, monitor):
        self._fork_tare = self.rt.fork_raw_forces().copy()
        r = self._fork_ramp(extension, 0.0, monitor)
        if not r.completed:
            return r
        return self._fork_ramp(extension, insert, monitor, resist=self.cell.fork.insert_force_limit)

    def fork_press(self, caps):
        """Push the tines up against the case bottom with capped drives (the cap is the force), then lock
        them where they stand, the drive's spring keeping the same force."""
        f = self.cell.fork
        e, y = self.rt.fork_state()
        self.rt.set_fork_force_caps([float(c) for c in caps], [f.y_force_limit] * 2)
        self.rt.set_fork_target(e - PRESS_BEYOND, y)
        self._steps(1.0)
        loads = self._fork_mean()
        e, y = self.rt.fork_state()
        self.rt.set_fork_target(e - np.asarray(loads) / f.drive_stiffness, y)
        self.rt.set_fork_force_caps([f.z_force_limit] * 2, [f.y_force_limit] * 2)
        self._steps(0.3)
        self.fork_engaged = True
        return tuple(float(v) for v in self._fork_mean())

    def _fork_mean(self, seconds=0.25):
        vals = []
        for _ in range(max(1, int(round(seconds / self.cell.physics.dt)))):
            self.rt.step(1)
            self._frame()
            vals.append(self.fork_loads())
        return np.mean(vals, axis=0)

    def fork_loads(self):
        """Load on each tine (N): the bar joints read it 1:1 (F0, artifacts/f0)."""
        raw = self.rt.fork_raw_forces() - self._fork_tare
        return (float(raw[0]), float(raw[1]))

    def fork_unload(self, monitor):
        """Landed: the tines stay on the surface with a light push while the tool rises a few
        millimetres, so the cups take the case back off them."""
        e, y = self.rt.fork_state()
        self.rt.set_fork_force_caps([UNLOAD_CAP] * 2, [self.cell.fork.y_force_limit] * 2)
        self.rt.set_fork_target(e + 2 * UNLOAD_RAISE, y)
        # Raise slowly: the cups take the weight back over a second, not in one step (a sudden load on the
        # cups makes the Surface Gripper let go).
        q = np.asarray(self.rt.gantry_target, float).copy()
        z0, n = float(q[2]), int(round(TRANSFER_S / self.cell.physics.dt))
        for k in range(1, n + 1):
            q[2] = z0 + UNLOAD_RAISE * k / n
            self.rt.set_gantry_target(q)
            self._steps(self.cell.physics.dt)
        self._steps(0.3)
        self.fork_engaged = False
        return tuple(float(v) for v in self._fork_mean())

    def fork_retract(self, monitor):
        f = self.cell.fork
        e, y = self.rt.fork_state()
        if self.fork_engaged:
            # Still under a hanging case: let the cups take its weight gradually, lowering the tines' push
            # to nothing over a second, before they leave it.
            loads = np.asarray(self.fork_loads(), float)
            n = int(round(TRANSFER_S / self.cell.physics.dt))
            for k in range(1, n + 1):
                self.rt.set_fork_force_caps(np.maximum(loads * (1 - k / n), 1.0), [f.y_force_limit] * 2)
                self._steps(self.cell.physics.dt)
            self.fork_engaged = False
        # Off the case bottom first (a tine still pushing up would rub it while sliding out). After a
        # landing the caps are already light: the tines rest on the surface and must not be driven into it.
        drop = float(min(np.mean(e) + RELEASE_DROP, f.extension_range[1]))
        r0 = self._fork_ramp(drop, float(np.mean(y)), monitor)
        if not r0.completed:
            return r0
        r = self._fork_ramp(drop, 0.0, monitor, resist=f.pull_force_limit)
        if not r.completed:
            return r
        self.rt.set_fork_force_caps([f.z_force_limit] * 2, [f.y_force_limit] * 2)
        r2 = self._fork_ramp(f.extension_range[0], 0.0, monitor)
        return MoveResult(r2.completed, r2.reason, r.seconds + r2.seconds)

    def capture(self, camera):
        f = self.rt.capture(camera)
        w, h = f['resolution']
        fx, fy = intrinsics_from_projection(f['projection'], w, h)
        frame = CameraFrame(depth=f['depth'], instance=f['instance'], instance_map=normalise_instance_map(f['instance_map']),
                            view=f['view'], fx=fx, fy=fy, width=w, height=h, stamp=f['stamp'], name=camera,
                            extra={'rgb': f['rgb']})
        if self.sensor_log is not None:
            try:
                self.sensor_log.write(frame)
            except OSError as e:                      # a full disk must not stop the robot; the gap is reported
                print(f'sensor log: {camera} at {f["stamp"]:.3f} not kept ({e})', flush=True)
        return frame
