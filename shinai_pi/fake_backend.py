"""Analytic stand-in for the Isaac cell, for fast tests of the decision logic.

Boxes are rigid OBBs. Removing a case makes every case that loses its quasi-static support
drop onto whatever is below it (a crude but deterministic collapse model). Rendering uses the
same analytic renderer as the perception tests. Not used for any reported result."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .backend import MoveResult
from .hand import fork_envelope
from .config import Cell
from .motion import limits_for, plan_move, tool_rotation
from .perception.geometry import look_at_view
from .perception.obb import OBB, yaw_matrix
from .perception.synthetic import render
from .pile import Box, SupportGraph
from .safety import Action, TickState
from .viewplan import tool_camera_world

DT = 0.05
OVERLAP = 0.01                 # m of footprint overlap before one box can rest on another
G = 9.81


wrist_matrix = tool_rotation


COLLISION_BELOW = 0.5           # m: tool face height under which injected collisions happen (tests)


class FakeBackend:
    def __init__(self, cell: Cell, boxes: dict, curtain_breaks=(), grip_failures=None, resolution=(320, 240), slips=None,
                 coms=None, masses=None, sag_deg=0.0, hang_tilt_deg=None, release_slips=None, release_pushes=None, fork_blocks=None,
                 vacuum_drops=None, unsupported_drops=None, obstacles=None, collisions=None):
        self.cell = cell
        self.t = 0.0
        self.q = np.array([(cell.pallet.a_center[0] + cell.pallet.b_center[0]) / 2, 0.0, 0.3, 0.0, 0.0, 0.0])
        self.boxes = dict(boxes)                      # key -> OBB (ground truth, private)
        p = cell.pallet
        self.static = {
            'pallet_a': OBB.make((p.a_center[0], p.a_center[1], p.size[2] / 2), p.size),
            'pallet_b': OBB.make((p.b_center[0], p.b_center[1], p.size[2] / 2), p.size),
        }
        st = cell.staging
        for i, (tx, ty) in enumerate(st.tables):
            self.static[f'staging_{i}'] = OBB.make((tx, ty, st.table_size[2] / 2), st.table_size)
        self.static.update(obstacles or {})          # fixed things in the cell (tests)
        self.fork_hits = []                           # (t, what) the empty tool's tines were driven into (truth, tests)
        # 'touch' | 'free' -> times the empty tool going down is already pressed / meets resistance (tests)
        self.collisions = Counter(collisions or ())
        self.coms = dict(coms or {})                  # case -> centre of mass in the case's own frame (m), truth
        self.masses = dict(masses or {})              # case -> kg, truth
        self.sag_deg = sag_deg                        # a case held on its side droops this much on the cups
        self.hang_tilt_deg = dict(hang_tilt_deg or {})  # case -> (about tool x, about tool y) while hanging level
        self.release_slips = dict(release_slips or {})  # case -> degrees it turns as the vacuum lets go
        # case -> metres it pushes its nearest neighbour away as it lands (S3 104: a case landed 32 mm off and
        # moved the case beside it, which had already been checked)
        self.release_pushes = dict(release_pushes or {})
        self._sag = 0.0
        self.attached = None
        self.rel = None
        self.curtain_breaks = list(curtain_breaks)    # [(t0, t1)]
        self.grip_failures = dict(grip_failures or {})
        self.slips = dict(slips or {})              # case -> degrees it turns on the cups after the floor camera
        self.resolution = resolution
        self._stopped_at = None
        self.collapses = []                           # truth log for tests
        f = cell.fork
        self.fork = {'ext': f.extension_range[0], 'ins': 0.0, 'loads': (0.0, 0.0), 'engaged': False, 'landed': False}
        # 'insert' | 'retract' -> times something is in the tines' way (tests); a set means once each
        self.fork_blocks = Counter(fork_blocks or ())
        self.vacuum_drops = dict(vacuum_drops or {})  # case -> times the cups let go while it rides on the tines (tests)
        self.sealed = True
        self.unsupported_drops = set(unsupported_drops or ())   # cases the tines do not carry when the cups let go (tests)

    # ------------------------------------------------------------------ protocol
    def now(self):
        return self.t

    def gantry_q(self):
        return self.q.copy()

    def curtain_clear(self):
        return not any(a <= self.t <= b for a, b in self.curtain_breaks)

    def operator_reset_requested(self):
        last_end = max([b for a, b in self.curtain_breaks if b <= self.t], default=None)
        if last_end is not None and self.t - last_end >= 2.0:
            return '作業者'
        return None

    def gripped(self):
        return [self.attached] if self.attached and self.sealed else []

    def _tick(self, monitor, carrying, v):
        if self.fork['engaged'] and self.vacuum_drops.get(self.attached, 0) > 0 and np.any(v):
            self.vacuum_drops[self.attached] -= 1
            self.sealed = False                        # the cups let go; the tines keep the case
            if self.attached in self.unsupported_drops:
                self.fork['loads'] = (0.0, 0.0)
        return monitor(TickState(self.t, self.q.copy(), v, carrying, self.gripped(), self.curtain_clear(),
                                 supported=self.fork['engaged']))

    def idle(self, seconds, monitor, carrying=None):
        end = self.t + seconds
        while self.t < end:
            self.t += DT
            if self._tick(monitor, carrying, np.zeros(len(self.q))).action is Action.STOP:
                return MoveResult(False, 'stop', seconds)
        return MoveResult(True, seconds=seconds)

    def move(self, goal, monitor, carrying=None, speed=1.0, stop_on_contact=False, guard=None, fine=False):
        g = self.cell.gantry
        goal = np.asarray(goal, float)
        if len(goal) == 4:                              # a gantry move keeps the wrist where it is
            goal = np.r_[goal, self.q[4:]]
        prof = plan_move(self.q, goal, *limits_for(g, slow=speed < 1.0, touch=stop_on_contact, fine=fine))
        kind = 'touch' if stop_on_contact else 'free'
        # injected only on the way down low, where something beside the case can be struck
        low = self.cell.gantry.origin_height + goal[2] < COLLISION_BELOW
        if not carrying and goal[2] < self.q[2] and low and self.collisions[kind] > 0:
            self.collisions[kind] -= 1
            return MoveResult(False, 'collision', info={'kind': 'preload' if stop_on_contact else 'collision',
                                                        'load_N': 2.0 * self.cell.contact.collision})
        t0, prev = self.t, self.q.copy()
        steps = max(1, int(math.ceil(prof.duration / DT)))
        every = max(1, int(round(self.cell.down_camera.every_s / DT)))
        for k in range(1, steps + 1):
            if guard is not None and (k - 1) % every == 0:
                stop, report = guard()
                if stop:
                    return MoveResult(False, 'guard', self.t - t0, guard=report)
            q = prof.position(k * DT)
            if stop_on_contact and q[2] < prev[2] and self._contact(q):
                lo, hi = q[2], prev[2]                 # bisect to the height of first contact
                for _ in range(12):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if self._contact(np.r_[q[:2], mid, q[3:]]) else (lo, mid)
                self.q = np.r_[q[:2], hi, q[3:]]
                self._carry_update(self.q)
                return MoveResult(True, 'contact', self.t - t0, contact=True)
            v = (q - prev) / DT
            self.q, prev = q, q
            self.t += DT
            self._carry_update(q)
            self._check_fork(q)
            if self._tick(monitor, carrying, v).action is Action.STOP:
                return MoveResult(False, 'stop', self.t - t0)
        return MoveResult(True, seconds=self.t - t0)

    def capture(self, camera):
        p = self.cell.pallet
        w, h = self.resolution
        if camera == 'tool':
            pos, R = tool_camera_world(self.cell, self.q)
            view = np.eye(4)
            view[:3, :3], view[3, :3] = R.T, pos
            view = np.linalg.inv(view)
            tc = self.cell.tool_camera
            f = tc.focal_mm / tc.aperture_mm * w
        elif camera == 'down':
            dc = self.cell.down_camera
            R = wrist_matrix(self.q)
            eye = np.array([self.q[0], self.q[1], self.cell.gantry.origin_height + self.q[2]]) + R @ np.asarray(dc.offset, float)
            view = look_at_view(eye, eye - R[:, 2], up=tuple(R[:, 1]))
            w, h = dc.resolution
            f = dc.focal_mm / 20.955 * w
        elif camera == 'under':
            uc = self.cell.under_camera
            eye = np.asarray(uc.position, float)
            view = look_at_view(eye, eye + np.array([0.0, 1e-4, 1.0]), up=(0, -1, 0))
            f = uc.focal_mm / 20.955 * w
        else:
            c = p.a_center if camera == 'overhead_a' else p.b_center
            up = (-1, 0, 0)                                                 # both overhead cameras are turned 90 deg
            view = look_at_view((c[0], c[1], self.cell.overhead_camera.height), (c[0] - 1e-4 * up[0], c[1] + 1e-4 * up[1], 0.0), up=up)
            f = self.cell.overhead_camera.focal_mm / 20.955 * w
        objects = {**{k: v for k, v in self.boxes.items()}, **self.static}
        frame = render(view, objects, width=w, height=h, fx=f, fy=f, stamp=self.t, name=camera)
        self.t += 0.3
        if camera == 'under' and self.attached in self.slips:
            turn = yaw_matrix(np.radians(self.slips.pop(self.attached)))
            self.rel = (turn @ self.rel[0], turn @ self.rel[1])
            self._carry_update(self.q)
        return frame

    def grip(self, close):
        if close and self.attached and not self.sealed:
            self.sealed = True                         # sealing again on the case still against the cups
            return
        if not close:
            if self.attached:
                self._release()
            return
        tool = np.array([self.q[0], self.q[1], self.cell.gantry.origin_height + self.q[2]])
        axis = wrist_matrix(self.q)[:, 2]
        R = wrist_matrix(self.q)
        cups = [tool + R @ np.array([cx, cy, 0.0]) for cx, cy in self.cell.gantry.cup_offsets]
        for k, o in self.boxes.items():
            c, n, half, (u, v) = o.top_face()
            along = float((tool - c) @ n)
            square = math.degrees(math.acos(min(1.0, abs(float(axis @ n))))) <= self.cell.gripper.max_face_tilt_deg
            # The cups seal on a face they are square to (within the bellows' give); the case is held
            # when at least two cups sit on the face, wherever on it (as the Isaac surface gripper).
            on_face = sum(1 for cp in cups if abs((cp - c) @ u) <= half[0] and abs((cp - c) @ v) <= half[1])
            if on_face >= 2 and -0.015 <= along <= 0.012 and square:
                if self.grip_failures.get(k, 0) > 0:
                    self.grip_failures[k] -= 1
                    return
                self.attached = k
                T = self._tool_pose(self.q)
                self.rel = (T[0].T @ (o.center - T[1]), T[0].T @ o.rotation)
                self._hang = self.hang_tilt_deg.get(k)
                self._hang_on = False
                self._grip_z = float(self.q[2])
                return

    # ------------------------------------------------------------------ under-supports
    def _held_bottom_depth(self):
        """Depth of the held case's lowest point below the cup face, and its extents in the tool frame."""
        R, p = self._tool_pose(self.q)
        o = self.boxes[self.attached]
        corners = (R.T @ (o.corners() - p).T).T
        return float(-corners[:, 2].min()), corners

    def fork_insert(self, extension, insert, monitor):
        f = self.cell.fork
        if not self.attached:
            return MoveResult(False, 'nothing held')
        depth, corners = self._held_bottom_depth()
        lo, hi = corners.min(0), corners.max(0)
        blocked = (self.fork_blocks['insert'] > 0 or extension < depth          # a tine would strike the case's side
                   or lo[1] < f.bar_y + f.bar_size[1] / 2                     # the case reaches the bars
                   or f.bar_x + f.tine_size[0] / 2 > min(-lo[0], hi[0]))       # a tine would miss the bottom
        seconds = abs(extension - self.fork['ext']) / 0.2 + insert / 0.1
        self.t += seconds
        if blocked:
            self.fork_blocks['insert'] -= 1
            self.fork.update(ext=extension, ins=0.0)
            return MoveResult(False, 'blocked', seconds, contact=True)
        self.fork.update(ext=extension, ins=insert)
        return MoveResult(True, seconds=seconds)

    def fork_press(self, caps):
        """The drives push up until they reach their caps: the tines then carry those forces."""
        if not self.attached or self.fork['ins'] <= 0:
            return (0.0, 0.0)
        depth, _ = self._held_bottom_depth()
        self.fork.update(ext=depth, loads=tuple(float(c) for c in caps), engaged=True)
        self.t += 1.0
        return self.fork['loads']

    def fork_loads(self):
        """Pushing up, the tines carry what their drives push with; with the cups let go, all of it."""
        if not self.fork['engaged']:
            return (0.0, 0.0)
        if self.sealed or self.attached in self.unsupported_drops:
            return self.fork['loads']
        w = G * self.masses.get(self.attached, self.cell.skus[0].mass)
        return (w / 2, w / 2)

    def fork_unload(self, monitor):
        """Landed: raise the tool a few millimetres while the tines stay on the surface; the cups take
        the weight back."""
        self.q = self.q + np.r_[0.0, 0.0, 0.003, np.zeros(len(self.q) - 3)]
        self._carry_update(self.q)
        self.fork.update(engaged=False, loads=(0.0, 0.0), ext=self.fork['ext'] + 0.003)
        self.t += 0.5
        return (0.0, 0.0)

    def fork_retract(self, monitor):
        f = self.cell.fork
        seconds = self.fork['ins'] / 0.1 + 0.5
        self.t += seconds
        if self.fork_blocks['retract'] > 0:
            self.fork_blocks['retract'] -= 1
            return MoveResult(False, 'blocked', seconds, contact=True)
        self.fork.update(ext=f.extension_range[0], ins=0.0, engaged=False, loads=(0.0, 0.0))
        return MoveResult(True, seconds=seconds)

    # ------------------------------------------------------------------ physics stand-in
    def _tool_pose(self, q):
        return wrist_matrix(q), np.array([q[0], q[1], self.cell.gantry.origin_height + q[2]])

    def wrist_load(self):
        """What the z axis and the wrist joints feel: (load N, torque about tool x, torque about tool y),
        torques about the centre of the cup face. Zero when nothing is held."""
        if not self.attached:
            return 0.0, 0.0, 0.0
        o = self.boxes[self.attached]
        m = self.masses.get(self.attached, self.cell.skus[0].mass)
        com = o.center + o.rotation @ np.asarray(self.coms.get(self.attached, (0.0, 0.0, 0.0)), float)
        R, p = self._tool_pose(self.q)
        r = R.T @ (com - p)
        f = R.T @ np.array([0.0, 0.0, -m * G])
        tau = np.cross(r, f)
        return m * G, float(tau[0]), float(tau[1])

    def _carry_update(self, q):
        if not self.attached:
            return
        # The cups bend under the moment of a case turned onto its side, towards the ground; a case
        # resting on something (tipped over on a table) does not hang on them and does not bend them.
        held = self.boxes[self.attached]
        resting = held.aabb()[0][2] <= self._support_top(held, {self.attached}) + 0.005
        want = self._sag if resting else -math.copysign(math.radians(self.sag_deg), q[5]) * min(1.0, abs(q[5]) / (math.pi / 2))
        if abs(want - self._sag) > 1e-9:
            d = want - self._sag
            c, s = math.cos(d), math.sin(d)
            Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            self.rel = (Rx @ self.rel[0], Rx @ self.rel[1])
            self._sag = want
        if getattr(self, '_hang', None) and not self._hang_on and q[2] > self._grip_z + 0.02:
            self._tilt_held(+1)
        R, p = self._tool_pose(q)
        o = self.boxes[self.attached]
        was_on_pile = o.center[2]
        self.boxes[self.attached] = OBB(p + R @ self.rel[0], R @ self.rel[1], o.size)
        if self.boxes[self.attached].center[2] > was_on_pile + 0.02 and not getattr(self, '_lifted', None) == self.attached:
            self._lifted = self.attached
            self._settle_without(self.attached)

    def _tilt_held(self, sign):
        """Hanging free, off-centre contents tip the case on the compliant cups (about the cup face);
        set down, it levels again (sign -1 undoes the tilt)."""
        if sign > 0:
            ax, ay = (math.radians(v) for v in self._hang)
            cx, sx, cy, sy = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay)
            self._hang_R = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rt = self._hang_R
        else:
            Rt = self._hang_R.T
        self.rel = (Rt @ self.rel[0], Rt @ self.rel[1])
        self._hang_on = sign > 0

    def _support_top(self, o: OBB, exclude):
        lo, hi = o.aabb()
        rest = 0.0
        for k, other in list(self.boxes.items()) + list(self.static.items()):
            if k in exclude:
                continue
            klo, khi = other.aabb()
            # Footprints must overlap by more than a few millimetres: a case set down grazing its
            # neighbour's edge slides off it rather than resting on it (box corners grow when turned).
            if np.all(klo[:2] + OVERLAP < hi[:2]) and np.all(lo[:2] + OVERLAP < khi[:2]) and khi[2] <= lo[2] + 0.02:
                rest = max(rest, khi[2])
        return rest

    def _check_fork(self, q):
        """Record the empty tool's tines inside anything (a held case rides on them, and is not counted)."""
        if self.attached:
            return
        f = self.cell.fork
        x0, x1, y0, y1, bottom = fork_envelope(f, extension=self.fork['ext'])
        y0 += self.fork['ins']
        band = f.tine_size[2]
        R, p = self._tool_pose(q)
        tines = OBB(p + R @ np.array([(x0 + x1) / 2, (y0 + y1) / 2, bottom + band / 2]), R, (x1 - x0, y1 - y0, band))
        for k, other in list(self.boxes.items()) + list(self.static.items()):
            if _overlap(tines, other) and not any(w == k for _, w in self.fork_hits[-1:]):
                self.fork_hits.append((round(self.t, 3), k))

    def _contact(self, q):
        R, p = self._tool_pose(q)
        if self.attached:
            o = OBB(p + R @ self.rel[0], R @ self.rel[1], self.boxes[self.attached].size)
            lo, _ = o.aabb()
            under = self.cell.fork.tine_size[2] if self.fork['ins'] > 0 else 0.0     # tines are the lowest point
            return lo[2] - under <= self._support_top(o, {self.attached}) + 0.002
        # The cups touch first: they stand proud of the plate, which may overhang a small face.
        cups = np.array([p + R @ np.array([cx, cy, -0.005]) for cx, cy in self.cell.gantry.cup_offsets])
        return any(np.any(b.contains(cups, margin=0.0)) for b in self.boxes.values())

    def _release(self):
        k = self.attached
        if getattr(self, '_hang_on', False):
            self._tilt_held(-1)
            self._hang = None
            R, p = self._tool_pose(self.q)
            self.boxes[k] = OBB(p + R @ self.rel[0], R @ self.rel[1], self.boxes[k].size)
        if k in self.release_slips:
            turn = yaw_matrix(np.radians(self.release_slips.pop(k)))
            o = self.boxes[k]
            self.boxes[k] = OBB(o.center, turn @ o.rotation, o.size)
        o = self.boxes[k]
        if o.tilt_deg() < 5.0:                         # released onto a flat surface, a case sits flat
            o = OBB(o.center, _levelled(o.rotation), o.size)
        lo, _ = o.aabb()
        rest = self._support_top(o, {k})
        self.boxes[k] = OBB(o.center - np.array([0, 0, lo[2] - rest]), o.rotation, o.size)
        if k in self.release_pushes:
            self._push_neighbour(k, self.release_pushes.pop(k))
        self.attached, self.rel, self._lifted, self._sag = None, None, None, 0.0

    def _push_neighbour(self, k, d):
        c = self.boxes[k].center
        level = [n for n, o in self.boxes.items() if n != k and abs(o.center[2] - c[2]) < 0.02]
        if not level:
            return
        n = min(level, key=lambda m: np.linalg.norm(self.boxes[m].center[:2] - c[:2]))
        o = self.boxes[n]
        away = np.r_[o.center[:2] - c[:2], 0.0]
        self.boxes[n] = OBB(o.center + d * away / np.linalg.norm(away), o.rotation, o.size)

    def _settle_without(self, removed):
        others = {k: v for k, v in self.boxes.items() if k != removed}
        g = SupportGraph([Box(k, tuple(v.center), tuple(v.size), rotation=v.rotation) for k, v in others.items()],
                         floor=0.0, static=list(self.static.values()))
        fallen = set()
        present = set(others)
        changed = True
        while changed:
            changed = False
            for k in sorted(present):
                if not g.stable(k, present):
                    fallen.add(k); present.discard(k); changed = True
        for k in sorted(fallen, key=lambda k: self.boxes[k].center[2]):
            o = self.boxes[k]
            lo, _ = o.aabb()
            rest = self._support_top(o, {k, removed})
            self.boxes[k] = OBB(o.center - np.array([0, 0, lo[2] - rest]), o.rotation, o.size)
        if fallen:
            self.collapses.append((self.t, removed, sorted(fallen)))

    # ------------------------------------------------------------------ truth (tests only)
    def truth(self):
        return dict(self.boxes)


def _levelled(R):
    """The nearest rotation whose box axis closest to vertical is exactly vertical."""
    R = np.asarray(R, float)
    i = int(np.argmax(np.abs(R[2, :])))
    up = R[:, i] * np.sign(R[2, i])
    axis = np.cross(up, [0.0, 0.0, 1.0])
    s, c = np.linalg.norm(axis), float(up[2])
    if s < 1e-12:
        return R
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + s * K + (1 - c) * K @ K) @ R


def _overlap(a: OBB, b: OBB) -> bool:
    """Separating-axis test of two boxes (strictly inside each other by more than a millimetre)."""
    Ra, Rb = np.asarray(a.rotation, float), np.asarray(b.rotation, float)
    ha, hb = np.asarray(a.size, float) / 2, np.asarray(b.size, float) / 2
    d = np.asarray(b.center, float) - np.asarray(a.center, float)
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    axes += [np.cross(Ra[:, i], Rb[:, j]) for i in range(3) for j in range(3)]
    for ax in axes:
        n = np.linalg.norm(ax)
        if n < 1e-9:
            continue
        ax = ax / n
        ra = float(np.sum(ha * np.abs(Ra.T @ ax)))
        rb = float(np.sum(hb * np.abs(Rb.T @ ax)))
        if abs(float(d @ ax)) >= ra + rb - 0.001:
            return False
    return True
