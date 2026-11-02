"""Mission executive: from "re-stack this pallet" to a sequence of observations and picks.

The human gives the goal. The order of looking, deciding and acting is chosen here from the
belief: pick when a safe pick exists; otherwise look where looking can change the decision;
otherwise fall back conservatively or hand the case to a person. Every step is an event."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import hand, structure
from .backend import CellBackend
from .belief_space import pile_region
from .config import Cell
from .events import EventLog
from .layout import open_long_side, pallet_slots
from .motion import tool_rotation
from .perception.clearance import path_clearance
from .perception.topface import measure_top_face
from .perception.boxfit import fit_all
from .perception.obb import OBB
from .perception.voxels import OCCUPIED, UNKNOWN, VoxelMap
from .pile import Box, PileBelief
from .safety import Action, Decision, SafetySupervisor
from .states import classify
from .viewplan import rank_viewpoints
from .viewpoints import park_pose, side_viewpoints

POLICIES = ('shinai', 'observe_all', 'top_first')
GRIP_OFFSET = 0.004            # m above the surface where a carried case is released
GRIP_GAP = 0.003               # m the cups back off a touched face before the vacuum is switched on (5 mm left the
                               # far cups of a tilted face near their 12 mm reach: some did not attach)
FINE_START = 0.015             # m above the expected face where the touch search slows to the fine speed
TOUCH_DEPTH = 0.03             # m below the estimated face the touch-down search may go
BACK_OFF = 0.05               # m the tool backs straight up off something it struck
NEIGHBOUR_REACH = 0.45         # m between slot centres: the cases a landing case can push (its side neighbours)
APPROACH = 0.06                # m above the face where the slow final approach starts
LIFT_SLOW = 0.04               # m lifted slowly before the fast lift
SLOW = 0.3                     # speed factor for approach, first lift and placing
VOXEL = 0.04
DECAY_EPOCH = 0.85             # confidence kept per world change for cases not re-observed
DECAY_NEAR = 0.6               # extra factor for cases near the one that was removed
NEAR_RADIUS = 0.6
MOVE_CHECK_CONFIDENCE = 0.8
PLACE_MARGIN = 0.02             # m around the footprint that must also be free (floor spots)
PLACE_INSET_B = 0.005           # m (plus one voxel) inside the footprint on B, where neighbours touch by design
PLACE_UNKNOWN_SHARE = 0.05     # edges of the placement volume may fall in sensor shadow
HANDOVER_STACK = 2             # cases per handover spot: a second one only on a case lying the same way
SUPPORT_SHARE = 0.6            # share of the footprint just under a stacked case that must be seen occupied
YAW_TOL_DEG = 2.0              # placement targets (requirements v04): position, yaw and tilt of a placed case
POS_TOL = 0.010                # m
TILT_TOL_DEG = 2.0
SETTLE_S = 3.0                 # s a released case is watched before it counts as placed
DRIFT_TOL = 0.005              # m it may move while settling
G = 9.81
WRIST_MISMATCH_DEG = 5.0       # largest angle left between cups and face after the wrist turns
TILT_DIRECT_DEG = 3.0          # faces tilted less than this are met with a level wrist
AIM_TRIES = 3                  # looks at a hanging case over its slot before letting it down
AIM_POS_TOL = 0.0015           # m: closer than this, no further correction
AIM_YAW_TOL_DEG = 0.3
AIM_YAW_DEADBAND_DEG = 3.0     # the rim around the plate reads the edge up to 2.5 deg off even at 1280 px: smaller turns are noise
AIM_CLAMPED_DEADBAND_DEG = 10.0  # pressed onto the tines the case cannot twist on the cups: a reading below this is the
                               # rim's noise (seen at 3.3 deg), not a turn; above it the tines come out to correct it
HANG_TILT_MAX_DEG = 8.0        # a case hanging on the cups tips a few degrees; a steeper fit is a bad plane
SHADOW_MARGIN = 0.3            # m around a case left for a person: its unseen shadow is theirs to check
REGRIP_OFFSET = 0.04           # m: cups this far from a case's centre hold it on part of the cups: take it again
RETREAT = 0.08                 # m the cups back away from a righted case before rising
UPRIGHT_TOL = 0.03             # m: measured height within this of the case height means it stands upright
CATCH_GAP = 0.02               # m below the pile's estimate of the hanging bottom: the cups give a few mm (8 in the
                               # stand-in) and the estimate is not measured yet; pressing up closes the gap
CLAMPED_TURN_MAX_DEG = 0.5     # a case pressed onto the tines is not turned: clamped, it does not follow the tool (the
                               # stand-in left 3 deg; on the cups they let go as a turn began, stage gates). A heading the
                               # measurement from below changes gets the tines out, the turn, and the tines in again


@dataclass
class MissionConfig:
    sku_id: str
    policy: str = 'shinai'
    max_cascade_risk: float = 0.25
    fallback_risk: float = 0.6
    max_observations_per_decision: int = 3
    min_view_info: float = 0.5
    grip_retries: int = 2
    max_cycles: int = 150
    max_seconds: float = 2400.0
    instruction: str = 'パレット A の荷を、パレット B へ積み替えてください'
    cascade_model: str | None = None      # path of a learned cascade predictor (models/cascade_v1.json)
    expected_cases: int | None = None     # cases on the load per its shipping record, if known


def long_edge_yaw_error_deg(obb, slot_yaw_deg):
    """Angle between a box's long horizontal edge and a slot's long-edge direction, folded to
    0..90 degrees (a box turned half a turn is in the same place)."""
    R, size = np.asarray(obb.rotation), np.asarray(obb.size)
    horizontal = [i for i in range(3) if abs(R[2, i]) < 0.7]
    i = max(horizontal, key=lambda j: size[j])
    yaw = math.degrees(math.atan2(R[1, i], R[0, i]))
    d = (yaw - slot_yaw_deg) % 180.0
    return min(d, 180.0 - d)


@dataclass
class MissionReport:
    status: str
    placed: list = field(default_factory=list)
    handed_over: list = field(default_factory=list)
    remaining: dict = field(default_factory=dict)
    observations: int = 0
    side_observations: int = 0
    safety_stops: int = 0
    grip_failures: int = 0
    fallback_picks: int = 0
    learned_decisions: int = 0             # picks chosen with a learned cascade model in use
    learned_changed: int = 0               # of those, picks where the geometric rule alone would have taken another case
    seconds: float = 0.0
    unseen_outside_reach: list = field(default_factory=list)
    not_found: int = 0                     # cases the load count says exist but the cell never showed
    failed_checks: list = field(default_factory=list)


class Executive:
    def __init__(self, cell: Cell, backend: CellBackend, events: EventLog, config: MissionConfig):
        if config.policy not in POLICIES:
            raise ValueError(f'policy must be one of {POLICIES}')
        self.cell, self.be, self.ev, self.cfg = cell, backend, events, config
        self.sku = cell.sku(config.sku_id)
        self.sup = SafetySupervisor(cell)
        self.slots = pallet_slots(cell, self.sku, cell.pallet.b_center)
        self.next_slot = 0
        self.slot_of: dict = {}
        self.handover_next = 0
        self.handover_stacks = {}  # spot index -> [(height, horizontal edges)] of cases put there, bottom first
        p = cell.pallet
        self.pallet_a = OBB.make((p.a_center[0], p.a_center[1], p.size[2] / 2), p.size)
        self.region = pile_region(cell)
        self.epoch_frames = []
        self.epoch_sides = set()
        self.prior: dict = {}
        self.removed = set()
        self.unpickable: dict = {}
        self.belief: PileBelief | None = None
        self.belief_stamp = -math.inf
        self.vm: VoxelMap | None = None
        self.vm_b: VoxelMap | None = None
        self.carrying: str | None = None
        self.report = MissionReport(status='running')
        self.obs_this_decision = 0
        self.holding_for_person = None
        self.failed_checks = set()     # placements that did not pass the check on B
        self.handover_reason = {}      # case -> why it was set aside for a person
        self.load = {}                 # case -> (mass kg, centre of mass from the cups in the tool frame, m)
        self.gripped_by_top = set()    # cases now held at the centre of their measured top face
        self.tines_refused = set()     # cases the tines met something under, over A: not tried again
        self.fork_engaged = False      # the tines are under the held case and carry most of its weight
        self.support = None            # hand.Support of the current transfer
        self.hang_up = {}              # case -> direction into the case from its bottom face (floor camera)
        self.blocked = {}              # case -> approaches stopped by something in the cups' path
        self.blocked_by = {}           # case -> what closed each yaw at its last stopped approach
        self.waiting = {}              # case -> the cases in its path that are taken first; then it is tried again
        self.collided = None           # the last unexpected contact of the empty tool, until a pick takes it up
        self.reaching = None           # the case the tool is going for
        self.gripped_crosswise = set()  # cases held with the cups across their length (a steep face)
        self.grip_yaw = {}             # case -> tool yaw it was gripped at
        self.fork_yaw = 0.0            # tool heading when the tines took the current case
        self.vacuum_losses_this_carry = 0
        self.damage_risk = set()       # cases a contact pressed over its limit (requirements v05 R1)
        self.states_met = {}           # state name (requirements v05 R3) -> cases picked in it
        self.cascade_model = None
        if config.cascade_model:
            from .learning.cascade import CascadeModel
            self.cascade_model = CascadeModel.load(config.cascade_model)

    # ================================================================== helpers
    def _emit(self, type_, **data):
        return self.ev.emit(self.be.now(), type_, **data)

    def _release(self, key, r):
        """Let go of a carried case, saying whether the landing was felt and by what reading, so the simulator's
        trace can be cut at this moment."""
        how = getattr(self.be, 'last_contact', None) if r.contact else None
        self._emit('RELEASED', case=key, felt_landing=bool(r.contact), how=dict(how) if how else None)
        self.be.grip(False)

    def _monitor(self, state):
        return self.sup.check(state)

    def _dims_for(self, key):
        return self.sku.size if key.startswith('case_') else None

    def _move(self, q, carrying=None, speed=1.0, stop_on_contact=False, guard=None, fine=False):
        ok, why = self.sup.validate_target(q)
        if not ok:
            self._emit('MOVE_REJECTED', target=list(map(float, q)), reason=why)
            return None
        r = self.be.move(np.asarray(q, float), self._monitor, carrying=carrying, speed=speed, stop_on_contact=stop_on_contact,
                         guard=guard, fine=fine)
        if r.reason == 'guard':
            return r
        if r.reason == 'collision':
            self._on_collision(r.info or {})
            return None
        if not r.completed:
            self._handle_stop()
            return None
        return r

    def _carry_move(self, q, speed=1.0, stop_on_contact=False):
        """Moves while holding a case resume after the stop is cleared: hanging in the air is the
        worse state. If the case was lost, the move is abandoned. A case on the cups alone, or on
        tines accepted only below the full acceleration, travels at the slow acceleration."""
        fast_ok = self.fork_engaged and self.support is not None and self.support.max_accel >= self.cell.gantry.a_max[0]
        if self.carrying and not fast_ok:
            speed = min(speed, SLOW)
        for _ in range(3):
            r = self._move(q, carrying=self.carrying, speed=speed if fast_ok else min(speed, SLOW), stop_on_contact=stop_on_contact)
            if r is not None:
                return r
            if self.carrying is None or self.holding_for_person is not None:
                return None
            fast_ok = self.fork_engaged and self.support is not None and self.support.max_accel >= self.cell.gantry.a_max[0]
        return None

    def _handle_stop(self):
        if self.sup.latched is None and self.carrying is not None and self.fork_engaged:
            return self._handle_vacuum_loss()
        reason = self.sup.latched or '停止'
        self.report.safety_stops += 1
        dropped = self.carrying is not None and self.carrying not in self.be.gripped()
        if dropped:
            self._emit('DROP_DETECTED', case=self.carrying)
            self.carrying = None
        self._emit('SAFETY_STOP', reason=reason, q=list(map(float, self.be.gantry_q())))
        waited = 0.0
        while True:
            self.be.idle(0.5, _hold_position)          # the gantry is stopped; nothing to supervise but time
            waited += 0.5
            op = self.be.operator_reset_requested()
            if op and self.sup.reset(op, self.be.curtain_clear()):
                self._emit('SAFETY_CLEARED', operator=op, waited_s=round(waited, 1))
                break
            if waited > 600:
                raise RuntimeError('safety stop was not cleared within 10 minutes')
        self._world_changed(near=None)

    def _on_collision(self, info):
        """The empty tool met something it should not have: record it, back straight up off it and let the
        pick give the case to a person. Not charged to the case being reached for: the force was elsewhere."""
        self.collided = {'kind': info.get('kind', 'collision'), 'load_N': info.get('load_N')}
        self._emit('COLLISION', case=self.reaching, **self.collided,
                   reason='工具が予期しない物に当たった。離れて止め、人が確かめる')
        q = np.asarray(self.be.gantry_q(), float).copy()
        q[2] += BACK_OFF
        self.be.move(q, self._monitor, speed=SLOW)

    def _record_contact(self, key, r):
        """Every contact's forces against the damage limits (requirements v05 R1): CONTACT_LOAD, and
        DAMAGE_RISK for a lasting force over its limit."""
        info = getattr(r, 'info', None)
        if not info:
            return
        v = hand.contact_verdict(info['kind'], info.get('static_N'), info.get('peak_N'), self.cell.contact, raw=info.get('raw'))
        self._emit('CONTACT_LOAD', case=key, **v)
        if v['damage_risk']:
            self.damage_risk.add(key)
            self._emit('DAMAGE_RISK', case=key, kind=v['kind'], static_N=v['static_N'], limit_N=v['limit_static_N'],
                       reason='接触の力が上限を超えた。商品の損傷のおそれ（人が確認）')

    def _handle_vacuum_loss(self):
        """The cups let go while the tines carry the case (design 9). Hold still, seal again, and
        carry the rest of the way slowly: the tines' pads are slippery on their own."""
        key = self.carrying
        tines = self.be.fork_loads()
        self.vacuum_losses_this_carry += 1
        mass, com = self.load.get(key, (self.sku.mass, np.zeros(2)))
        ok, why = hand.can_recover(tines, mass * G, com, self.cell.fork)
        if self.vacuum_losses_this_carry > 1:
            ok, why = False, f'吸着が 1 回の運搬で {self.vacuum_losses_this_carry} 回目に外れた'
        sealed = False
        if ok:
            self.be.grip(True)
            self.be.idle(0.3, _hold_position)
            sealed = key in self.be.gripped()
            self.support = hand.Support(self.support.accepted, self.support.mode, self.cell.gantry.slow_accel,
                                        self.support.share, '吸着が一度外れたため遅く運ぶ') if self.support else None
        self._emit('VACUUM_LOST', case=key, tines_N=[round(v, 1) for v in tines], sealed_again=sealed, recoverable=ok,
                   reason=('吸着が外れたが、爪がケースを支えている。止めて吸い直し、残りはゆっくり運ぶ' if ok
                           else f'吸着が外れた。{why}ため、続けない'))
        if not sealed:
            # Hold it where it is: without the cups it cannot be handed back off the tines at its place,
            # and moving a case whose support is in doubt would be guessing.
            self._emit('HANDOVER', case=key, reason=(f'{why}。その場で止めて待機する（人が受け取る）' if not ok
                                                     else '吸着を吸い直せないため、爪で支えたまま待機する（人が受け取る）'))
            self.holding_for_person = key

    def _world_changed(self, near):
        self.sup.note_world_change(self.be.now())
        self.epoch_frames, self.epoch_sides = [], set()
        self.obs_this_decision = 0
        if self.belief is not None:
            for k, b in self.belief.boxes.items():
                if k in self.removed:
                    continue
                f = DECAY_EPOCH
                if near is not None and np.linalg.norm(np.asarray(b.center) - near) < NEAR_RADIUS:
                    f *= DECAY_NEAR
                self.prior[k] = Box(k, b.center, b.size, b.confidence * f, b.rotation, b.top_visible)

    def _park(self):
        q = park_pose(self.cell)
        q[3] = self.be.gantry_q()[3]
        return self._move(q)

    def _rule_choice(self, bel):
        """The case the geometric rule alone would take from the same belief, by the same selection as _decide
        (None: the rule would have looked again or stopped instead of taking one)."""
        pickable = [c for c in bel.candidates() if c.pickable]
        risk = {c.box_id: bel.rule_risk(c.box_id) for c in pickable}
        safe = [c for c in pickable if risk[c.box_id] <= self.cfg.max_cascade_risk]
        if safe:
            return max(safe, key=lambda c: (round(c.height, 3), -risk[c.box_id])).box_id
        if pickable:
            low = min(pickable, key=lambda c: (round(risk[c.box_id], 6), -c.height, c.box_id))
            if risk[low.box_id] <= self.cfg.fallback_risk:
                return low.box_id
        return None

    # ================================================================== perception
    def _observe(self, camera, label):
        frame = self.be.capture(camera)
        frame.name = label
        self.epoch_frames.append(frame)
        self.report.observations += 1
        if label.startswith('side_'):
            self.report.side_observations += 1
            self.epoch_sides.add(label)
        return frame

    def look(self, cameras):
        """Belief of pallet A from these cameras alone, as at the start of a mission (also used to make training data
        whose input is what the robot sees)."""
        if not cameras:
            raise ValueError('a look needs at least one camera')
        for c in cameras:
            self._observe(c, c)
        self._perceive()
        return self.belief

    def _perceive(self):
        lo, hi = self.region
        vm = VoxelMap(lo, hi, VOXEL)
        for f in self.epoch_frames:
            vm.integrate(f, step=2)
        est = fit_all(self.epoch_frames, self._dims_for, vm)
        boxes = {}
        for k, e in est.items():
            c = e.obb.center
            if k in self.removed or not np.all((c >= np.array(lo)) & (c <= np.array(hi))):
                continue
            boxes[k] = Box(k, tuple(c), tuple(e.obb.size), e.confidence, e.obb.rotation, e.top_visible)
        for k, b in self.prior.items():
            if k not in boxes and k not in self.removed:
                boxes[k] = b
        g = self.cell.gantry
        reach = ((g.x_range[0] + 0.05, g.x_range[1] - 0.05), (g.y_range[0] + 0.05, g.y_range[1] - 0.05))
        bel = PileBelief(floor=0.0, max_cascade_risk=self.cfg.max_cascade_risk, static=[self.pallet_a],
                         max_face_tilt_deg=self.cell.gripper.max_face_tilt_deg + math.degrees(g.pitch_range[1]),
                         tool_footprint=g.tool_size[:2],
                         box_volume=float(np.prod(self.sku.size)), reach=reach, cascade_model=self.cascade_model,
                         com_margin=self.sku.com_uncertainty,
                         case_dims=self.sku.size)
        bel.observe(list(boxes.values()), 'epoch', seen_fraction=1.0)
        bel.removed = set(self.removed)
        bel.set_known_space(vm, self.region)
        self.belief, self.vm, self.belief_stamp = bel, vm, self.be.now()
        self.prior = dict(boxes)
        cands = bel.candidates()
        unp = sum(1 for c in cands if not c.pickable)
        self._emit('BELIEF_UPDATED', cases=len(bel.boxes), unpickable=unp,
                   candidates=[{'case': c.box_id, 'risk': round(c.expected_cascade, 3), 'pickable': c.pickable,
                                'reason': c.reason, 'height': round(c.height, 3),
                                'confidence': round(bel.boxes[c.box_id].confidence, 3)} for c in cands],
                   hidden_probability=round(bel.hidden_box_probability(), 3),
                   unknown_fraction=round(vm.unknown_fraction(*self.region), 3),
                   views=[f.name for f in self.epoch_frames])
        return bel

    def _survey(self):
        if self._park() is None:
            return None
        self._observe('overhead_a', 'overhead_a')
        bel = self._perceive()
        self._emit('OBSERVED', view='overhead_a', cases=len(bel.boxes))
        return bel

    def _observe_side(self, vp, reason, info=None):
        self._emit('OBSERVE_REQUESTED', view=vp.name, reason=reason, info=None if info is None else round(info, 2))
        q = np.array(vp.q, float)
        if self._move(q) is None:
            return False
        self._observe('tool', f'side_{vp.name}')
        bel = self._perceive()
        self._emit('OBSERVED', view=f'side_{vp.name}', cases=len(bel.boxes))
        return True

    # ================================================================== deciding
    def _choose(self):
        """Returns ('pick', candidate, fallback) | ('observe', viewpoint, info, reason) | ('done', reason)."""
        bel = self.belief
        live = [k for k in bel.boxes if k not in self.removed]
        if not live:
            p, _ = self._hidden_in_reach()
            if p > 0.2 and self.cfg.policy != 'top_first' and self.obs_this_decision < self.cfg.max_observations_per_decision:
                best = self._best_view(require_info=False)
                if best is not None:
                    return ('observe', best.viewpoint, best.info, '見えていない場所にケースが残っている可能性')
            return ('done', 'A に見えるケースがない')
        cands = [c for c in bel.candidates() if c.box_id not in self.unpickable]
        pickable = [c for c in cands if c.pickable]
        if self.cfg.policy == 'top_first':
            if pickable:
                return ('pick', max(pickable, key=lambda c: c.height), False)
            return ('done', '吸着できるケースがない')
        if self.cfg.policy == 'observe_all':
            missing = [vp for vp in side_viewpoints(self.cell, self.cell.pallet.a_center) if f'side_{vp.name}' not in self.epoch_sides]
            if missing:
                return ('observe', missing[0], None, '毎回すべての方向から確認する方式')
        safe = [c for c in pickable if c.expected_cascade <= self.cfg.max_cascade_risk]
        if safe:
            return ('pick', max(safe, key=lambda c: (round(c.height, 3), -c.expected_cascade)), False)
        if self.cfg.policy == 'shinai' and self.obs_this_decision < self.cfg.max_observations_per_decision and bel.observation_gain() > 0:
            best = self._best_view(require_info=True)
            if best is not None:
                return ('observe', best.viewpoint, best.info, '取る順番を決めるのに足りない情報がある')
        if pickable and pickable[0].expected_cascade <= self.cfg.fallback_risk:
            return ('pick', pickable[0], True)
        # Nothing more to take here: before ending, look where a case could still hide (as when A
        # is empty), so the robot ends knowing what it leaves.
        p, _ = self._hidden_in_reach(for_person=[c.box_id for c in cands if c.reason == 'out of reach'])
        if p > 0.2 and self.obs_this_decision < self.cfg.max_observations_per_decision:
            best = self._best_view(require_info=False)
            if best is not None:
                return ('observe', best.viewpoint, best.info, '見えていない場所にケースが残っている可能性')
        if cands and all(c.reason == 'out of reach' for c in cands):
            return ('done', '残ったケースは腕が届かない場所にある（人が取り出す）')
        return ('done', '安全に取れるケースがない（人の確認が必要）')

    def _best_view(self, require_info):
        vps = [vp for vp in side_viewpoints(self.cell, self.cell.pallet.a_center) if f'side_{vp.name}' not in self.epoch_sides]
        if not vps:
            return None
        ranked = rank_viewpoints(self.cell, self.belief, vps, self.be.gantry_q())
        best = ranked[0]
        if require_info and best.info < self.cfg.min_view_info:
            return None
        return best

    # ================================================================== acting
    def _tool_yaw_for(self, axis_xy, reference):
        base = math.atan2(axis_xy[1], axis_xy[0])
        lo, hi = self.cell.gantry.yaw_range
        options = [base + k * math.pi for k in range(-3, 4) if lo <= base + k * math.pi <= hi]
        return min(options, key=lambda a: abs(a - reference))

    def _obstacle_top(self):
        top = self.cell.pallet.size[2]
        if self.vm is not None:
            s = self.vm.state()
            mask = (s == OCCUPIED) | (s == UNKNOWN)
            c = self.vm.centers()[mask]
            above_floor = c[c[:, 2] > VOXEL]
            if len(above_floor):
                top = max(top, float(above_floor[:, 2].max()) + VOXEL / 2)
        for k, b in self.belief.boxes.items():
            if k not in self.removed:
                top = max(top, float(b.obb.aabb()[1][2]))
        placed_layers = 1 + max((self.slots[i].layer for i in self.slot_of.values()), default=-1)
        top = max(top, self.cell.pallet.size[2] + placed_layers * self.sku.size[2])
        return top

    def _safe_z(self, carried_height):
        z = self._obstacle_top() + carried_height + self.cell.gantry.travel_clearance + 0.02
        return min(z, self.cell.gantry.origin_height + self.cell.gantry.z_range[1]) - self.cell.gantry.origin_height

    def _execute_pick(self, cand, fallback):
        bel = self.belief
        key = cand.box_id
        b = bel.boxes[key]
        o = b.obb
        ok, why = self.sup.validate_plan(self.belief_stamp)
        if not ok:
            self._emit('DECISION_DEFERRED', case=key, reason=why)
            return 'aborted'
        ok, why = self.sup.validate_payload(self.sku.mass)
        if not ok:
            self.unpickable[key] = why
            self._emit('HANDOVER', case=key, reason=why)
            return 'unpickable'
        g = bel.graph()
        predicted = g.fall_severity(key)
        alts = [{'case': c.box_id, 'risk': round(c.expected_cascade, 3), 'pickable': c.pickable} for c in bel.candidates()[:5]]
        code, state = classify(key, bel, self.sku, self.cell.pallet)
        self.states_met[state] = self.states_met.get(state, 0) + 1
        compared = {}
        if self.cascade_model is not None:
            rule = self._rule_choice(bel)
            compared = {'rule_choice': rule, 'rule_risk': round(bel.rule_risk(key), 3), 'model_changed_choice': rule != key}
            self.report.learned_decisions += 1
            self.report.learned_changed += int(rule != key)
        self._emit('PICK_PLANNED', case=key, risk=round(cand.expected_cascade, 3), fallback=fallback,
                   predicted_fall=sorted(predicted), alternatives=alts, confidence=round(b.confidence, 3),
                   state=state, state_code=code, **compared)
        if fallback:
            self.report.fallback_picks += 1
        c, n, half, (u, v) = o.top_face()
        long_axis = u if half[0] >= half[1] else v
        q_now = self.be.gantry_q()
        yaw = self._tool_yaw_for(long_axis[:2], q_now[3])
        z_face = float(c[2]) - self.cell.gantry.origin_height
        up_dim = float(o.size[o.up_axis])
        safe = self._safe_z(up_dim)
        self.reaching, self.collided = key, None
        held = self._grip_from_top(key, c, z_face, yaw, safe, normal=n)
        self.reaching = None
        if held == 'aborted' and self.collided is not None:
            self.unpickable[key] = f'取りに行く途中で工具がほかの物に当たった（{self.collided["load_N"]} N）。取らずに人が確かめる'
            self._emit('HANDOVER', case=key, reason=self.unpickable[key])
            self._world_changed(near=np.asarray(o.center))
            return 'aborted'
        if held == 'aborted':
            return 'aborted'
        if held == 'blocked':
            self.blocked[key] = self.blocked.get(key, 0) + 1
            if self.blocked[key] >= 2:
                # A case in the way is one the robot takes anyway: once it is gone, the way may be clear
                # (S3 103: case_13 blocked by case_12, which was taken next, and case_13 was never tried again).
                ahead = sorted(w for w in self.blocked_by.get(key, ()) if w in bel.boxes and w not in self.removed)
                if ahead:
                    self.waiting[key] = set(ahead)
                    self.unpickable[key] = 'ほかのケースが爪の通り道にある。先にそちらを取ってから取り直す'
                    self._emit('DECISION_DEFERRED', case=key, waiting_for=ahead, reason=self.unpickable[key])
                else:
                    self.unpickable[key] = '吸着板か爪の進路にほかの物があり、取れない（人が確認）'
                    self._emit('HANDOVER', case=key, reason=self.unpickable[key])
            self._world_changed(near=np.asarray(o.center))
            return 'aborted'
        if held == 'failed':
            self.unpickable[key] = '吸着できなかった'
            why = ('押し付けの力が上限を超えたため、吸着を繰り返さない（人が確認して取り出す）' if key in self.damage_risk
                   else '吸着できなかった（人が取り出す）')
            self._emit('HANDOVER', case=key, reason=why)
            self._move([c[0], c[1], max(safe, z_face + APPROACH), yaw])
            return 'unpickable'
        yaw = self.grip_yaw.get(key, yaw)
        placed_at, up_dim, horiz = self._measure_and_place(key, yaw, safe, up_dim, long_axis)
        if placed_at == 'aborted':
            return 'aborted'
        if placed_at is None:
            return self._put_back(key, o, c, yaw, up_dim, horiz, safe)
        dest_kind, target_xy, place_yaw, slot_index = placed_at
        if dest_kind == 'B':
            self._emit('PLACED', case=key, slot=slot_index, layer=self.slots[slot_index].layer)
            self.report.placed.append(key)
            self.slot_of[key] = slot_index
            self.next_slot += 1
        else:
            self._emit('HANDOVER', case=key, reason=self.handover_reason.get(key, '横倒しのため別置き（人が起こす）'),
                       area=dest_kind)
            self.report.handed_over.append(key)
            self.handover_next += 1
        self.removed.add(key)
        for k in [k for k, ahead in self.waiting.items() if key in ahead]:
            del self.waiting[k]
            self.unpickable.pop(k, None)
            self.blocked.pop(k, None)
        self._move([target_xy[0], target_xy[1], safe, place_yaw])
        self._world_changed(near=np.asarray(o.center))
        return self._after_pick(key, predicted, dest_kind)

    def _wrist_for(self, normal, yaw):
        """Wrist (pitch, roll) that turns the cups square to a face with this upward normal, the tool
        yaw kept (cups along the face's long edge). None if the wrist and the cups' own give together
        cannot reach it."""
        g = self.cell.gantry
        c_, s_ = math.cos(yaw), math.sin(yaw)
        n = np.array([c_ * normal[0] + s_ * normal[1], -s_ * normal[0] + c_ * normal[1], normal[2]])
        roll = -math.asin(max(-1.0, min(1.0, float(n[1]))))
        pitch = math.atan2(float(n[0]), float(n[2]))
        held_pitch = min(max(pitch, g.pitch_range[0]), g.pitch_range[1])
        # Cups 0.1 m off the plate centre seal across a few degrees of mismatch, not ten.
        if abs(pitch - held_pitch) > math.radians(WRIST_MISMATCH_DEG) or not g.roll_range[0] <= roll <= g.roll_range[1]:
            return None
        return held_pitch, roll

    def _grip_pose_for(self, normal, yaw):
        """(tool yaw, wrist, crosswise) meeting a tilted face square, or None. Along the face's long
        edge first; past the wrist's pitch, a quarter turn of the tool lets the roll meet it instead
        (the cups then lie across the case, which is taken again along it on a table)."""
        lo, hi = self.cell.gantry.yaw_range
        # A half turn of the tool meets the same face with the opposite pitch and roll: of the two, the one that keeps
        # the drawn-up fork bars off the ram (structure.wrist_allowed, review 2026-10-31).
        for options, crosswise in (((yaw, yaw + math.pi, yaw - math.pi), False),
                                   ((yaw + math.pi / 2, yaw - math.pi / 2, yaw + 3 * math.pi / 2, yaw - 3 * math.pi / 2), True)):
            for turned in sorted(options, key=lambda a: abs(a - yaw)):
                if lo <= turned <= hi:
                    wrist = self._wrist_for(normal, turned)
                    if wrist is not None and structure.wrist_allowed(*wrist):
                        return turned, wrist, crosswise
        return None

    def _grip_from_top(self, key, c, z_face, yaw, safe, normal=None):
        """Approach a top face along its normal, touch it, grip, lift clear and hang the case level.
        A face tilted beyond what the cups' bellows take is met with the wrist turned square to it.
        'held' | 'failed' | 'aborted'; the tool yaw it holds the case at is self.grip_yaw[key]."""
        self.gripped_by_top.discard(key)                        # a new grip: where it holds is not yet known
        self.gripped_crosswise.discard(key)
        n = np.array([0.0, 0.0, 1.0]) if normal is None else np.asarray(normal, float) / np.linalg.norm(normal)
        tilted = math.degrees(math.acos(min(1.0, float(n[2])))) > TILT_DIRECT_DEG
        if not tilted:
            n = np.array([0.0, 0.0, 1.0])

        def pose_at(y):
            return self._grip_pose_for(n, y) if tilted else (y, (0.0, 0.0), False)
        pose = pose_at(yaw)
        if pose is None:
            return 'failed'
        yaw, wrist, crosswise = pose
        oh = self.cell.gantry.origin_height
        face = np.array([c[0], c[1], z_face + oh])

        def at(point):                 # the yaw and wrist chosen below, read when called
            return [float(point[0]), float(point[1]), float(point[2]) - oh, yaw, wrist[0], wrist[1]]
        above = face + n * APPROACH
        if self._move([above[0], above[1], max(safe, above[2] - oh), yaw, 0.0, 0.0]) is None:
            return 'aborted'
        # From up here the downward camera sees what the plate and the tines will pass on the way down.
        # The case is gripped as well half turned round: of the two, the tool goes down the way that is clear.
        turned = self._half_turned(yaw)
        pose = self._clear_descent(key, face, n, [pose, None if turned is None else pose_at(turned)])
        if pose is None:
            return 'blocked'
        yaw, wrist, crosswise = pose
        if crosswise:
            self.gripped_crosswise.add(key)
        if wrist != (0.0, 0.0) and self._move([above[0], above[1], max(safe, above[2] - oh), yaw, *wrist], speed=SLOW) is None:
            return 'aborted'
        if self._move(at(above)) is None:
            return 'aborted'
        tare = np.asarray(self.be.wrist_load(), float)         # nothing held, nothing touched
        held = False
        for attempt in range(self.cfg.grip_retries + 1):
            # Touch then grip: move slowly along the normal until the load cell feels the face
            # (estimates of the face carry centimetre errors; the cups seal only within millimetres).
            # Search at the touch speed down to just above the expected face, then slower for the rest.
            guard = self._path_guard(key, face, n)
            r = self._move(at(face + n * FINE_START), speed=SLOW, stop_on_contact=True, guard=guard)
            if r is not None and r.completed and not r.contact and r.reason != 'guard':
                r = self._move(at(face - n * TOUCH_DEPTH), speed=SLOW, stop_on_contact=True, guard=guard, fine=True)
            if r is None:
                return 'aborted'
            if r.reason == 'guard':
                c, margin = r.guard
                if c.valid:
                    self._emit('NEAR_MISS', case=key, clearance_mm=round(c.blocked_at * 1000, 1), what=c.what,
                               stop_margin_mm=round(margin * 1000, 1), stopped_in_time=c.blocked_at > margin,
                               reason='吸着板の進路にほかの物がある。触れる前に止めた')
                else:
                    self._emit('NEAR_MISS', case=key, clearance_mm=None, what=None, stop_margin_mm=round(margin * 1000, 1),
                               stopped_in_time=True, reason='下向きのカメラが何も返さないため、進路を確かめられず止めた')
                q = self.be.gantry_q()
                if self._move(at(np.array([q[0], q[1], q[2] + oh]) + n * APPROACH)) is None:
                    return 'aborted'
                return 'blocked'
            if r.contact:
                self._record_contact(key, r)
                # Back off a few millimetres before the vacuum: gripped while pressed flat against the
                # plate, the cups' hold and the plate's contact fight each other (over 1 kN in Isaac).
                q = self.be.gantry_q()
                if self._move(at(np.array([q[0], q[1], q[2] + oh]) + n * GRIP_GAP), speed=SLOW) is None:
                    return 'aborted'
            self.be.grip(True)
            self.be.idle(0.3, self._monitor)
            if key in self.be.gripped():
                held = True
                break
            self.be.grip(False)
            self.report.grip_failures += 1
            self._emit('GRIP_FAILED', case=key, attempt=attempt + 1)
            if self._move(at(face + n * 0.05), speed=SLOW) is None:
                return 'aborted'
            if key in self.damage_risk:
                break                  # each retry of a face that resisted pressed harder (envelope_B2: 70 N, then 87 N)
        if not held:
            return 'failed'
        self._emit('GRIP_OK', case=key, wrist_deg=[round(math.degrees(v), 1) for v in wrist], crosswise=key in self.gripped_crosswise)
        self.grip_yaw[key] = yaw
        self.carrying = key
        q = self.be.gantry_q()
        if self._carry_move(at(np.array([q[0], q[1], q[2] + oh]) + n * LIFT_SLOW), speed=SLOW) is None:
            return 'aborted'
        if wrist != (0.0, 0.0):
            # Clear of the pile, turn the wrist back: the case hangs level from its face.
            q = self.be.gantry_q()
            if self._carry_move([q[0], q[1], safe, yaw, *wrist]) is None:
                return 'aborted'
            if self._carry_move([q[0], q[1], safe, yaw, 0.0, 0.0], speed=SLOW) is None:
                return 'aborted'
        self._test_lift(key, tare)
        q = self.be.gantry_q()
        if self._carry_move([q[0], q[1], safe, yaw]) is None:
            return 'aborted'
        return 'held'

    def _half_turned(self, yaw):
        lo, hi = self.cell.gantry.yaw_range
        y = yaw + math.pi if yaw + math.pi <= hi else yaw - math.pi
        return y if lo <= y <= hi else None

    def _clear_descent(self, key, face, normal, poses):
        """The first (yaw, wrist, crosswise) of `poses` whose plate and tines meet nothing but the case on the way
        down to its face, judged from the downward camera above it; None when every one would strike something."""
        g = self.cell.gantry
        frame = self.be.capture('down')
        q = self.be.gantry_q()
        tool = np.array([q[0], q[1], g.origin_height + q[2]])
        # Straight down to just above the face, then along its normal to it and the search beyond.
        reach = float(tool[2] - (face + normal * APPROACH)[2]) + APPROACH + TOUCH_DEPTH
        regions = [hand.fork_envelope(self.cell.fork)]
        nearest, tried = None, []
        for pose in poses:
            if pose is None:
                continue
            y, w, _ = pose
            c = path_clearance(frame, tool, tool_rotation([0.0, 0.0, 0.0, y, *w]), (g.tool_size[0] / 2, g.tool_size[1] / 2),
                               ignore={key}, reach=reach, regions=regions, direction=(0.0, 0.0, -1.0))
            if not c.valid:
                break
            if c.blocked_at is None:
                return pose
            tried.append({'yaw_deg': round(math.degrees(y), 1), 'what': c.what, 'clearance_mm': round(c.blocked_at * 1000, 1)})
            nearest = c if nearest is None or c.blocked_at < nearest.blocked_at else nearest
        self.blocked_by[key] = {t['what'] for t in tried}
        if nearest is None:
            self._emit('NEAR_MISS', case=key, clearance_mm=None, what=None, stop_margin_mm=None, stopped_in_time=True,
                       reason='下向きのカメラが何も返さないため、下ろす道を確かめられず止めた')
        else:
            self._emit('NEAR_MISS', case=key, clearance_mm=round(nearest.blocked_at * 1000, 1), what=nearest.what,
                       stop_margin_mm=None, stopped_in_time=True, candidates=tried,
                       reason='吸着板か爪の通り道にほかの物がある（工具を半回転しても同じ）。下ろす前に止めた')
        return None

    def _path_guard(self, key, face, normal):
        """During the last approach to a face: the downward camera's view of the plate's footprint. Stop
        if anything but the target case lies in the path, nearer than the face or than the stopping
        margin (requirements v05 R4, 5.1.1 and 5.2)."""
        g = self.cell.gantry
        dc = self.cell.down_camera
        margin = hand.stop_margin(g.touch_speed, dc.latency_s, g.slow_accel)

        def guard():
            q = self.be.gantry_q()
            tool = np.array([q[0], q[1], g.origin_height + q[2]])
            reach = float((tool - face) @ normal) + TOUCH_DEPTH          # to the face being reached, and the search beyond it
            c = path_clearance(self.be.capture('down'), tool, tool_rotation(q), (g.tool_size[0] / 2, g.tool_size[1] / 2),
                               ignore={key}, reach=reach, regions=[hand.fork_envelope(self.cell.fork)])
            # A camera that returns nothing is no clear path: stop as for an object.
            return (c.blocked_at is not None or not c.valid), (c, margin)
        return guard

    def _test_lift(self, key, tare):
        """Held just clear of the pile, the case hangs still: the z axis carries its weight and the
        wrist joints its moment. Weight and moment give the mass and where the mass sits."""
        if self.be.idle(0.3, self._monitor, carrying=key) is None:
            return
        f, tx, ty = np.asarray(self.be.wrist_load(), float) - tare
        if f <= 1.0:
            return
        self.load[key] = (f / G, np.array([ty / f, -tx / f]))     # r_x = tau_y / F, r_y = -tau_x / F

    def _measure_and_place(self, key, yaw, safe, up_dim, long_axis, only=None):
        """Measure the carried case from below and lower it onto the first verified free
        destination (or only onto `only`). Returns (placed_at | None | 'aborted', up_dim, edges).
        With `only` the case was lifted off B itself, so B is seen again first: the map still
        holds the case where it lay."""
        if self._tines_before_travel(key, up_dim, long_axis, only) == 'aborted':
            return 'aborted', up_dim, None
        yaw = float(self.be.gantry_q()[3])
        measured = self._measure_carried(key, yaw, safe)
        if measured == 'aborted':
            return 'aborted', up_dim, None
        if only is not None:
            self._observe_b()
        offset_tool = np.zeros(2)
        if measured is not None:
            up_dim, long_axis, offset_tool = measured
        upright = abs(up_dim - self.sku.size[2]) < UPRIGHT_TOL
        self._report_load(key, offset_tool)
        righting = not upright and only is None and bool(self.cell.staging.tables)
        self._emit('LIFTED', case=key, upright=upright, up_dim=round(up_dim, 3),
                   destination='B' if upright else ('reorient' if righting else 'handover'),
                   measured_from_below=measured is not None)
        unmeasured = measured is None and key not in self.gripped_by_top
        crosswise = key in self.gripped_crosswise
        if (upright and only is None and self.cell.staging.tables
                and (unmeasured or crosswise or (measured is not None and float(np.linalg.norm(offset_tool)) > REGRIP_OFFSET))):
            # Held far off centre, or held without knowing where (the floor camera could not see):
            # set it down where its top face can be measured, and take it again at its centre.
            why = ('床のカメラで下から測れなかったため、台に置いて上面を測り持ち直す' if unmeasured
                   else '急な斜面を横向きに吸着したため、台に置いて長手の向きで持ち直してから爪で運ぶ' if crosswise
                   else '吸着の位置がケースの中心から大きくずれていたため、台に置いて持ち直す')
            if self.fork_engaged:
                self._fork_out_in_air(key, why)
            r = self._set_down_and_regrip(key, yaw, safe, up_dim, offset_tool, reason=why)
            if r == 'aborted':
                return 'aborted', up_dim, None
            if r[0] == 'regripped':
                _, yaw2, safe2, axis2 = r
                return self._measure_and_place(key, yaw2, safe2, float(self.sku.size[2]), axis2)
            if r[0] == 'on_table':
                self.handover_reason[key] = r[2]
                return ('staging', r[1], 0.0, None), up_dim, None
        if righting and self.fork_engaged:
            self._fork_out_in_air(key, '下から測ると立っていなかったため、起こす前に爪を抜く')
        if righting:
            r = self._reorient(key, yaw, safe, up_dim, long_axis, offset_tool)
            if r == 'aborted':
                return 'aborted', up_dim, None
            if r[0] == 'regripped':
                _, yaw2, safe2, axis2 = r
                return self._measure_and_place(key, yaw2, safe2, float(self.sku.size[2]), axis2)
            if r[0] == 'on_table':
                self.handover_reason[key] = r[2]
                return ('staging', r[1], 0.0, None), up_dim, None
            self.handover_reason[key] = r[1]                    # still held: it goes to a handover spot
        rel = math.atan2(long_axis[1], long_axis[0]) - yaw          # case long axis relative to tool
        dims = sorted(self.sku.size, reverse=True)
        horiz = [d for d in dims]
        horiz.remove(min(dims, key=lambda d: abs(d - up_dim)))     # the two edges that lie horizontal
        spots = only if only is not None else self._destinations(upright, horiz, up_dim)
        placed_at = None
        for dest_kind, target_xy, slot_yaw, base_z, slot_index in spots:
            place_yaw = self._tool_yaw_for((math.cos(slot_yaw - rel), math.sin(slot_yaw - rel)), yaw)
            footprint = horiz if abs(math.cos(slot_yaw)) > 0.5 else horiz[::-1]
            clear, why = self._placement_clear(dest_kind, target_xy, footprint, base_z, up_dim,
                                               top_z=safe + self.cell.gantry.origin_height - up_dim)
            fork_slot = None
            if clear and dest_kind == 'B' and upright and only is None:
                yaw_bar = self._bar_yaw(slot_index, place_yaw, float(self.be.gantry_q()[3]))
                if (yaw_bar is not None and self.fork_engaged
                        and abs(yaw_bar - float(self.be.gantry_q()[3])) > math.radians(CLAMPED_TURN_MAX_DEG)):
                    self._fork_out_in_air(key, 'この区画には別の向きで置くため、回す前に爪を抜く')
                if yaw_bar is not None and self._bar_side_seen_clear(target_xy, yaw_bar, footprint, base_z, up_dim):
                    if not self.fork_engaged and key not in self.tines_refused:
                        # Turn to the placing heading on the cups alone (slowly), then hand the weight to the
                        # tines: a clamped case is never turned (stage gates: the cups let go as a turn began).
                        q = self.be.gantry_q()
                        if self._carry_move([q[0], q[1], q[2], yaw_bar]) is None:
                            return 'aborted', up_dim, horiz
                        self._fork_in(key, up_dim, offset_tool, horiz)
                    if self.fork_engaged:
                        place_yaw, fork_slot = yaw_bar, slot_index
            if clear and dest_kind == 'handover' and base_z > 0:
                clear, why = self._support_seen(target_xy, footprint, base_z)
            if not clear:
                self._emit('MOVE_REJECTED', target=[float(target_xy[0]), float(target_xy[1])], reason=why, case=key)
                continue
            # The case hangs off-centre on the cups by what the floor camera measured; aim the case, not the tool.
            c_, s_ = math.cos(place_yaw), math.sin(place_yaw)
            tool_xy = (target_xy[0] - (c_ * offset_tool[0] - s_ * offset_tool[1]),
                       target_xy[1] - (s_ * offset_tool[0] + c_ * offset_tool[1]))
            sight = (key, target_xy, slot_yaw) if dest_kind == 'B' else None
            outcome = self._lower_and_release(tool_xy, place_yaw, base_z, up_dim, safe, sight=sight, fork_slot=fork_slot)
            if outcome == 'aborted':
                return 'aborted', up_dim, horiz
            if outcome == 'placed':
                placed_at = (dest_kind, target_xy, place_yaw, slot_index)
                if dest_kind == 'handover':
                    self.handover_stacks.setdefault(slot_index, []).append((up_dim, tuple(horiz)))
                break
        return placed_at, up_dim, horiz

    def _report_load(self, key, offset_tool):
        if key not in self.load:
            return
        mass, r = self.load[key]
        com_tool = r - np.asarray(offset_tool, float)          # from the case centre, tool frame
        yaw = self.grip_yaw.get(key, float(self.be.gantry_q()[3]))   # as the case lay when it was taken
        c_, s_ = math.cos(yaw), math.sin(yaw)
        com_world = (c_ * com_tool[0] - s_ * com_tool[1], s_ * com_tool[0] + c_ * com_tool[1])
        self._emit('LOAD_MEASURED', case=key, mass_kg=round(float(mass), 2),
                   com_offset_mm=[round(float(v) * 1000, 1) for v in com_world],
                   com_from_cups_mm=[round(float(v) * 1000, 1) for v in r])

    def _reorient(self, key, yaw, safe, up_dim, long_axis, offset_tool):
        """Right a case that lies on a side or stands on an end by tipping it over its bottom edge on a
        righting table (requirements v05, stage 2). The cups hold the face now on top; the case is set
        down on the table on the cups alone, then the cup centre moves on an arc about the bottom edge
        while the wrist rolls a quarter, so the table carries the weight and the cups only the moment
        about the edge. The cups let go, back away, and the case is gripped again from the top.
        Returns 'aborted', ('regripped', yaw, safe, long_axis), ('on_table', xy, reason) or ('held', reason)."""
        g, st = self.cell.gantry, self.cell.staging
        dims = sorted(self.sku.size, reverse=True)
        horiz = list(dims)
        horiz.remove(min(dims, key=lambda d: abs(d - up_dim)))
        axis_len, w0 = horiz[0], horiz[1]                     # the long edge stays horizontal; tipped, w0 is the height
        if abs(w0 - self.sku.size[2]) > UPRIGHT_TOL:
            return ('held', '倒しても底面が下にならない向きのため別置き（人が起こす）')
        if g.roll_range[0] > -math.pi / 2:
            return ('held', '手首が倒す向きに回りきらないため別置き（人が起こす）')
        mass = self.load.get(key, (self.sku.mass, None))[0]
        need = hand.pivot_cup_force(mass, w0, up_dim)
        cups = len(g.cup_offsets)
        if need > cups * self.cell.gripper.planning_shear_limit / hand.CUP_SAFETY or need * math.hypot(w0 / 2, up_dim) > g.wrist_torque_limit:
            return ('held', '倒して起こすと吸着か手首の余裕が足りないため別置き（人が起こす）')
        half = st.table_size[1] / 2                             # cases tip along y (below)
        try:
            p = hand.pivot_offset(w0, up_dim, half, com=st.com_allowance)
        except ValueError:
            return ('held', '姿勢を直す台が小さく、倒した後に台に載らないため別置き（人が起こす）')
        self._observe_b()
        table = None
        for tx, ty in st.tables:
            clear, _ = self._placement_clear('handover', (tx, ty), (axis_len, w0), st.table_size[2], up_dim)
            if clear:
                table = (tx, ty)
                break
        if table is None:
            return ('held', '姿勢を直す台が空いていないため別置き（人が起こす）')
        tx, ty = table
        t = np.array([0.0, 1.0 if ty >= 0 else -1.0])            # tip away from the other table
        # Tool yaw whose +y is the tip direction: the bars then end on top, clear of the table.
        lo, hi = g.yaw_range
        base = 0.0 if t[1] > 0 else math.pi
        options = [base + 2 * k * math.pi for k in range(-2, 3) if lo <= base + 2 * k * math.pi <= hi]
        if not options:
            return ('held', '倒す向きに工具を向けられないため別置き（人が起こす）')
        psi = min(options, key=lambda a: abs(a - yaw))
        cp, sp = math.cos(psi), math.sin(psi)
        off_w = np.array([cp * offset_tool[0] - sp * offset_tool[1], sp * offset_tool[0] + cp * offset_tool[1]])
        centre = np.array([tx, ty]) + (p - w0 / 2) * t          # the lying case's centre on the table
        tool_xy = centre - off_w
        z_top = st.table_size[2] + up_dim - g.origin_height
        z_high = max(safe, z_top + APPROACH)
        if self._carry_move([tool_xy[0], tool_xy[1], z_high, psi]) is None:
            return 'aborted'
        if self._carry_move([tool_xy[0], tool_xy[1], z_top + APPROACH, psi]) is None:
            return 'aborted'
        r = self._carry_move([tool_xy[0], tool_xy[1], z_top - TOUCH_DEPTH, psi], speed=SLOW, stop_on_contact=True)
        if r is None:
            return 'aborted'
        if not r.contact:
            self._emit('HANDOVER', case=key, reason='姿勢を直す台に着いたことを感じないため、持ったまま待機する（人が確認）')
            self.holding_for_person = key
            return 'aborted'
        q = self.be.gantry_q()
        cup = np.array([q[0], q[1], g.origin_height + q[2]])
        pivot = np.r_[cup[:2] + off_w + (w0 / 2) * t, st.table_size[2]]
        for c, roll in hand.pivot_path(pivot, cup, t)[1:]:
            if self._carry_move([c[0], c[1], c[2] - g.origin_height, psi, 0.0, roll], speed=SLOW) is None:
                return 'aborted'
        self._release(key, r)
        self.carrying = None
        self.be.idle(0.3, self._monitor)
        q = self.be.gantry_q()
        away = q[:2] + RETREAT * t                               # the cups face the tip direction now
        if self._move([away[0], away[1], q[2], psi, 0.0, q[5]], speed=SLOW) is None:
            return 'aborted'
        if self._move([away[0], away[1], z_high, psi, 0.0, q[5]]) is None:
            return 'aborted'
        if self._move([away[0], away[1], z_high, psi, 0.0, 0.0], speed=SLOW) is None:
            return 'aborted'
        self._last_standing = None
        r = self._regrip_on_table(key, table)
        if self._last_standing is not None:
            self._emit('REORIENTED', case=key, table=[float(tx), float(ty)], method='tip_on_table', from_height_m=round(up_dim, 3),
                       roll_deg=-90.0, cup_force_needed_N=round(need, 1), standing_m=round(self._last_standing, 3))
        return r

    def _grip_by_top(self, key, top, safe):
        """Grip a case at the centre of its measured top face, cups along its long edge."""
        a = math.radians(top.long_edge_yaw_deg)
        axis = np.array([math.cos(a), math.sin(a), 0.0])
        yaw = self._tool_yaw_for(axis[:2], float(self.be.gantry_q()[3]))
        c = np.array([top.centre[0], top.centre[1], top.top_z])
        held = self._grip_from_top(key, c, float(top.top_z) - self.cell.gantry.origin_height, yaw, safe)
        return held, yaw, axis

    def _regrip_on_table(self, key, table):
        """A case standing on a staging table: look at it, check it stands upright, grip it at the
        centre of its top face. ('regripped', yaw, safe, axis) | ('on_table', table, reason) | 'aborted'."""
        if self._park() is None:
            return 'aborted'
        top = measure_top_face(self._observe_b(), key)
        face = sorted(self.sku.size, reverse=True)[:2]
        if not _whole_face(top, face):
            return ('on_table', table, '台の上で上面の全体が見えない（人が確認）')
        standing = float(top.top_z) - self.cell.staging.table_size[2]
        if abs(standing - self.sku.size[2]) > UPRIGHT_TOL or top.tilt_deg > TILT_TOL_DEG + 3.0:
            return ('on_table', table, '台の上で正立を確かめられない（人が直す）')
        self._last_standing = standing
        safe = self._safe_z(float(self.sku.size[2]))
        held, yaw, axis = self._grip_by_top(key, top, safe)
        if held == 'aborted':
            return 'aborted'
        if held == 'failed':
            return ('on_table', table, '台の上で吸着できない（人が取り出す）')
        return ('regripped', yaw, safe, axis)

    def _free_table(self, footprint, height):
        self._observe_b()
        st = self.cell.staging
        for tx, ty in st.tables:
            if self._placement_clear('handover', (tx, ty), footprint, st.table_size[2], height)[0]:
                return (tx, ty)
        return None

    def _set_down_and_regrip(self, key, yaw, safe, up_dim, offset_tool,
                             reason='吸着の位置がケースの中心から大きくずれていたため、台に置いて持ち直す'):
        """The cups took the case far from its centre (the pile estimate was off): it hangs on part
        of the cups. Set it down on a staging table and grip it again at its centre, as a person
        would put down a box held at one end and take it again."""
        footprint = (max(self.sku.size), max(self.sku.size))        # any heading on the table
        table = self._free_table(footprint, up_dim)
        if table is None:
            return ('held', '吸着がずれたが持ち直す台が空いていない')
        c_, s_ = math.cos(yaw), math.sin(yaw)
        tool_xy = (table[0] - (c_ * offset_tool[0] - s_ * offset_tool[1]), table[1] - (s_ * offset_tool[0] + c_ * offset_tool[1]))
        outcome = self._lower_and_release(tool_xy, yaw, self.cell.staging.table_size[2], up_dim, safe)
        if outcome == 'aborted':
            return 'aborted'
        if self._move([tool_xy[0], tool_xy[1], safe, yaw]) is None:
            return 'aborted'
        self._emit('REGRIPPED', case=key, offset_mm=round(float(np.linalg.norm(offset_tool)) * 1000, 1),
                    table=[float(table[0]), float(table[1])], reason=reason)
        r = self._regrip_on_table(key, table)
        if r != 'aborted' and r[0] == 'regripped':
            self.gripped_by_top.add(key)                        # gripped at the centre of a measured top face
        return r

    def _measure_carried(self, key, yaw, safe):
        """Carry the case over the floor camera and measure it from below: its height (from the
        bottom face to the cups) and the direction of its long edge. Estimates from the collapsed
        pile can swap the edges; a case placed 90 degrees off lands on its neighbour."""
        uc = self.cell.under_camera
        q = [uc.position[0], uc.position[1], safe, yaw]
        if self._carry_move(q) is None:
            return 'aborted'
        frame = self.be.capture('under')
        frame.name = 'under'
        ids = [i for i, k in frame.instance_map.items() if k == key]
        if not ids:
            self._emit('OBSERVED', view='under', cases=0, measured=False)
            return None
        pts, inst, _ = frame.unproject(step=1)
        mine = pts[np.isin(inst, ids)]
        if len(mine) < 50:
            self._emit('OBSERVED', view='under', cases=0, measured=False)
            return None
        tool_z = self.cell.gantry.origin_height + float(self.be.gantry_q()[2])
        # The bottom face as a plane: off-centre contents tip a hanging case a few degrees on the
        # compliant cups, so a height band would cut the face.
        low = mine[mine[:, 2] < mine[:, 2].min() + 0.03]
        _, _, vt = np.linalg.svd(low - low.mean(0), full_matrices=False)
        up = -vt[2] if vt[2][2] < 0 else vt[2]                      # into the case, from its bottom face
        if up[2] < math.cos(math.radians(HANG_TILT_MAX_DEG)):
            up = np.array([0.0, 0.0, 1.0])                            # a thin strip of points: no trustworthy plane
        bottom = mine[np.abs((mine - low.mean(0)) @ up) < 0.005]
        height = tool_z - float(np.median(bottom[:, 2]))
        centred = bottom[:, :2] - bottom[:, :2].mean(0)
        w, v = np.linalg.eigh(centred.T @ centred)
        long_axis = np.array([v[0, 1], v[1, 1], 0.0])
        # Centre of the bottom face from its extents along its own edges (robust to uneven point density).
        a, b = centred @ v[:, 1], centred @ v[:, 0]
        # The face must be seen whole: a case fallen over the floor camera can leave a strip, whose
        # centre and direction are not the face's. A strip is no measurement.
        seen = sorted((float(a.max() - a.min()), float(b.max() - b.min())), reverse=True)
        faces = [sorted(pair, reverse=True) for pair in ((self.sku.size[0], self.sku.size[1]), (self.sku.size[0], self.sku.size[2]),
                                                          (self.sku.size[1], self.sku.size[2]))]
        if not any(abs(seen[0] - f[0]) <= 0.02 and abs(seen[1] - f[1]) <= 0.02 for f in faces):
            self._emit('OBSERVED', view='under', cases=1, measured=False, reason='下面の一部しか見えない（床のカメラの上に物がある）',
                       seen_mm=[round(v * 1000) for v in seen])
            return None
        centre = bottom[:, :2].mean(0) + v[:, 1] * (a.max() + a.min()) / 2 + v[:, 0] * (b.max() + b.min()) / 2
        # Set down, the case levels again about the cups: aim the centre of its top face.
        centre = centre + height * up[:2] / max(up[2], 0.5)
        q = self.be.gantry_q()
        offset_world = centre - q[:2]
        c_, s_ = math.cos(-q[3]), math.sin(-q[3])
        offset_tool = np.array([c_ * offset_world[0] - s_ * offset_world[1], s_ * offset_world[0] + c_ * offset_world[1]])
        self.hang_up[key] = up
        self._emit('OBSERVED', view='under', cases=1, measured=True, height_m=round(height, 3),
                   long_axis_deg=round(math.degrees(math.atan2(long_axis[1], long_axis[0])), 1),
                   grip_offset_mm=[round(float(x) * 1000, 1) for x in offset_tool])
        return height, long_axis, offset_tool

    def _put_back(self, key, o, c, yaw, up_dim, horiz, safe):
        """No verified place: look again, and put the case back only if its old spot is seen empty
        (neighbours may have slid into it). Otherwise hold it and call a person."""
        if self._carry_move(list(park_pose(self.cell)[:3]) + [yaw]) is None:
            return 'aborted'
        self._observe('overhead_a', 'overhead_a')
        self._perceive()
        base = float(o.aabb()[0][2])
        clear, why = self._placement_clear('A', (c[0], c[1]), horiz, base, up_dim)
        if clear:
            self._emit('HANDOVER', case=key, reason='置き場が空いていないため、空いていると確かめた元の位置へ戻した（人が確認）')
            outcome = self._lower_and_release((c[0], c[1]), yaw, base, up_dim, safe)
            self.unpickable[key] = '置き場がない'
            self._world_changed(near=np.asarray(o.center))
            return 'unpickable' if outcome == 'placed' else 'aborted'
        self._emit('HANDOVER', case=key, reason=f'置き場がなく元の位置も使えないため、持ったまま待機する（人が受け取る）：{why}')
        self.holding_for_person = key
        return 'hold'

    @staticmethod
    def _stackable(stack, up_dim, longest):
        """A handover spot takes a case if it is empty, or holds fewer than the limit and its top
        case lies the same way (same height, so the same footprint: the pair is a stable column).
        A case standing on its end is never stacked: the column would be far taller than its base."""
        if not stack:
            return True
        on_end = abs(up_dim - longest) < UPRIGHT_TOL
        return not on_end and len(stack) < HANDOVER_STACK and abs(stack[-1][0] - up_dim) < UPRIGHT_TOL

    def _support_seen(self, xy, footprint, base_z):
        """A stacked case goes down only onto a case the camera sees there now, not one remembered."""
        vm = self.vm_b
        gx = np.linspace(-footprint[0] / 2 + 0.03, footprint[0] / 2 - 0.03, 5)
        gy = np.linspace(-footprint[1] / 2 + 0.03, footprint[1] / 2 - 0.03, 5)
        pts = np.array([(xy[0] + a, xy[1] + b, base_z - vm.res / 2) for a in gx for b in gy])
        if np.mean(vm.state_at(pts) == OCCUPIED) < SUPPORT_SHARE:
            return False, '下のケースが見えない'
        return True, ''

    def _destinations(self, upright, horiz, up_dim):
        """Where this case may go: the next pattern slot on B if it stands upright, otherwise
        the handover area (a person stands it up). Each item: (kind, xy, yaw, base z)."""
        out = []
        used = set(self.slot_of.values())
        free = [d for d in self.slots if d.index not in used]
        if upright and free:
            layer = min(d.layer for d in free)        # a layer is finished before the next one starts
            for d in free:
                if d.layer == layer:
                    out.append(('B', tuple(d.center[:2]), math.radians(d.yaw_deg), d.center[2] - self.sku.size[2] / 2, d.index))
            return out
        h = self.cell.handover
        # Empty spots first; a second layer only when every spot is taken.
        for second in (False, True):
            for i, (hx, hy) in enumerate(h.spots):
                stack = self.handover_stacks.get(i, [])
                if bool(stack) == second and self._stackable(stack, up_dim, max(self.sku.size)):
                    out.append(('handover', (hx, hy), math.radians(h.spot_yaw_deg), sum(c[0] for c in stack), i))
        return out

    def _placement_clear(self, kind, xy, footprint, base_z, height, top_z=None):
        """A case is only put where cameras have seen empty space (unknown is not empty). The
        footprint is checked with a margin, and the column it is lowered through must hold nothing."""
        vm = self.vm_b if kind in ('B', 'handover') else self.vm
        if vm is None:
            return False, '置き場をまだ見ていない'
        # On B the pattern packs cases flush against their neighbours; elsewhere keep a clear margin.
        m = -(vm.res + PLACE_INSET_B) if kind == 'B' else PLACE_MARGIN   # one voxel: a neighbour's surface voxel
        gx = np.linspace(-footprint[0] / 2 - m, footprint[0] / 2 + m, 7)
        gy = np.linspace(-footprint[1] / 2 - m, footprint[1] / 2 + m, 7)
        z_top = base_z + height + 0.05
        gz = np.linspace(base_z + 2 * vm.res, z_top, 5)         # rays stop one voxel short of a surface
        pts = np.array([(xy[0] + a, xy[1] + b, z) for a in gx for b in gy for z in gz])
        state = vm.state_at(pts)
        if np.any(state == OCCUPIED):
            return False, '置き場に物がある'
        if np.mean(state == UNKNOWN) > PLACE_UNKNOWN_SHARE:
            return False, '置き場を見て確かめられていない'
        if top_z is not None and top_z > z_top:
            col = np.array([(xy[0] + a, xy[1] + b, z) for a in gx for b in gy for z in np.linspace(z_top, top_z, 6)])
            if np.any(vm.state_at(col) == OCCUPIED):
                return False, '置き場の上の経路に物がある'
        return True, ''

    def _surface_top_near(self, xy, base_z, reach=0.45):
        """Highest occupied surface within reach of a spot, from the map that covers it."""
        tops = [base_z]
        for vm in (self.vm, self.vm_b):
            if vm is None:
                continue
            c = vm.centers()
            occ = vm.state() == OCCUPIED
            pts = c[occ]
            near = pts[(np.abs(pts[:, 0] - xy[0]) < reach) & (np.abs(pts[:, 1] - xy[1]) < reach)]
            if len(near):
                tops.append(float(near[:, 2].max()) + vm.res / 2)
        return max(tops)

    def _lower_and_release(self, xy, place_yaw, base_z, up_dim, safe, sight=None, fork_slot=None):
        """Traverse, lower with touchdown detection, release. 'placed' | 'blocked' | 'aborted'.
        With `sight` (case, target centre, slot yaw), B's camera squares the hanging case up over
        its slot before the last slow descent. With `fork_slot`, the tines under the case land first,
        hand the case back to the cups and come out on the slot's open side before it is set down;
        anywhere else they come out in the air first."""
        if self.fork_engaged and fork_slot is None:
            self._fork_out_in_air(self.carrying, 'この置き場では爪を抜く側の空きを確かめられないため、空中で抜いて吸着だけで置く')
        if self._carry_move([xy[0], xy[1], safe, place_yaw]) is None:
            return 'aborted'
        if self.fork_engaged:
            return self._land_on_tines(xy, place_yaw, base_z, up_dim, sight, fork_slot)
        z_top = base_z + up_dim - self.cell.gantry.origin_height
        # The fast descent stops with the case above everything around the spot (neighbours on
        # B, cases on the floor); the slow last part listens to the load cell.
        around = self._surface_top_near(xy, base_z)
        z_fast = max(z_top + APPROACH, around + up_dim + APPROACH - self.cell.gantry.origin_height)
        if self._carry_move([xy[0], xy[1], z_fast, place_yaw]) is None:
            return 'aborted'
        if sight is not None:
            xy, place_yaw = self._aim_by_sight(sight, np.asarray(xy, float), place_yaw, z_fast)
            if xy is None:
                return 'aborted'
        return self._set_down_on_cups(xy, place_yaw, z_top)

    def _set_down_on_cups(self, xy, place_yaw, z_top):
        r = self._carry_move([xy[0], xy[1], z_top - TOUCH_DEPTH, place_yaw], speed=SLOW, stop_on_contact=True)
        if r is None:
            return 'aborted'
        if not r.contact:
            self._emit('MOVE_REJECTED', target=[float(xy[0]), float(xy[1])], reason='着地を感じないまま下限に達した')
        self._release(self.carrying, r)
        self.carrying = None
        self.be.idle(0.2, self._monitor)
        return 'placed'

    def _land_on_tines(self, xy, place_yaw, base_z, up_dim, sight, slot_index):
        key = self.carrying
        f = self.cell.fork
        z_land = base_z + up_dim + f.tine_size[2] - self.cell.gantry.origin_height     # tines' underside on the surface
        around = self._surface_top_near(xy, base_z)
        z_fast = max(z_land + APPROACH, around + up_dim + f.tine_size[2] + APPROACH - self.cell.gantry.origin_height)
        if self._carry_move([xy[0], xy[1], z_fast, place_yaw]) is None:
            return 'aborted'
        if sight is not None:
            xy, place_yaw = self._aim_by_sight(sight, np.asarray(xy, float), place_yaw, z_fast)
            if xy is None:
                return 'aborted'
            if not self.fork_engaged:                          # the tines came out to turn the case
                return self._set_down_on_cups(xy, place_yaw, base_z + up_dim - self.cell.gantry.origin_height)
        r = self._carry_move([xy[0], xy[1], z_land - TOUCH_DEPTH, place_yaw], speed=SLOW, stop_on_contact=True)
        if r is None:
            return 'aborted'
        if not r.contact:
            self._emit('HANDOVER', case=key, reason='爪の着地を感じないまま下限に達したため、持ったまま待機する（人が確認）')
            self.holding_for_person = key
            return 'aborted'
        self._emit('FORK_LANDED', case=key, slot=slot_index, tines_N=[round(v, 1) for v in self.be.fork_loads()])
        self._record_contact(key, r)
        self._emit('FORK_UNLOADING', case=key, slot=slot_index)
        residual = self.be.fork_unload(self._monitor)
        self._emit('FORK_RETRACTING', case=key, slot=slot_index)
        out = self.be.fork_retract(self._monitor)
        if not out.completed and out.reason == 'blocked':
            # Held back now and then on a case below (S3 seeds 102, 105): the cups lift the case once more
            # by the unload's few millimetres and the tines are pulled again, once.
            self._emit('FORK_RETRIED', case=key, slot=slot_index, residual_N=[round(v, 1) for v in residual],
                       reason='爪を抜く途中で抵抗があったため、ケースを少し上げて一度だけ抜き直す')
            self._emit('FORK_UNLOADING', case=key, slot=slot_index)
            residual = self.be.fork_unload(self._monitor)
            self._emit('FORK_RETRACTING', case=key, slot=slot_index)
            out = self.be.fork_retract(self._monitor)
        if not out.completed:
            self._emit('HANDOVER', case=key, reason='爪を抜く途中で抵抗があり、ケースを引きずるおそれがあるため、持ったまま待機する（人が確認）')
            self.holding_for_person = key
            return 'aborted'
        self.fork_engaged, self.support = False, None
        self._emit('FORK_OUT', case=key, slot=slot_index, residual_N=[round(v, 1) for v in residual],
                   bar_direction=[round(float(v), 3) for v in hand.bar_direction(place_yaw)],
                   turned_rad=round(float(place_yaw - self.fork_yaw), 3))
        z_top = base_z + up_dim - self.cell.gantry.origin_height
        r = self._carry_move([xy[0], xy[1], z_top - TOUCH_DEPTH, place_yaw], speed=SLOW, stop_on_contact=True)
        if r is None:
            return 'aborted'
        if not r.contact:
            self._emit('MOVE_REJECTED', target=[float(xy[0]), float(xy[1])], reason='着地を感じないまま下限に達した')
        self._release(key, r)
        self.carrying = None
        self.be.idle(0.2, self._monitor)
        return 'placed'

    # ------------------------------------------------------------------ under-supports
    def _fork_in(self, key, up_dim, offset_tool, horiz):
        """Slide the tines under the held case and hand most of its weight to them (requirements v05
        P7). The transfer counts only when the measured loads pass hand.judge_support; otherwise the
        tines come out and the case travels slowly on the cups."""
        return self._tines_under(key, up_dim, offset_tool, horiz) and self._tines_take(key, horiz)

    def _tines_under(self, key, up_dim, offset_tool, horiz, gap=0.0):
        """Slide the tines in under the held case, just clear of its bottom: a catch, not yet a support."""
        f = self.cell.fork
        if key not in self.load:
            self._emit('FORK_SKIPPED', case=key, reason='重さと重心を量れていないため、爪に移さず吸着だけでゆっくり運ぶ')
            return False
        mass, com = self.load[key]
        weight = mass * G
        length, depth = max(horiz), min(horiz)
        ok, why = hand.fork_fits(offset_tool, depth, length, f)
        if not ok:
            self._emit('FORK_SKIPPED', case=key, reason=why + '。吸着だけでゆっくり運ぶ')
            return False
        up = self.hang_up.get(key, np.array([0.0, 0.0, 1.0]))
        tilt = math.acos(min(1.0, abs(float(up[2]))))
        try:
            extension = min(hand.tine_extension(up_dim, tilt, depth, f) + gap, f.extension_range[1])
        except ValueError:
            self._emit('FORK_SKIPPED', case=key, reason='爪がケースの底まで届かない。吸着だけでゆっくり運ぶ')
            return False
        r = self.be.fork_insert(extension, f.insert_range[1], self._monitor)
        if not r.completed:
            self.be.fork_retract(self._monitor)
            self._emit('FORK_SKIPPED', case=key, reason=('爪を差し込む途中で何かに当たった' if r.contact else '爪の動作が止まった')
                       + '。吸着だけでゆっくり運ぶ')
            return False
        return True

    def _tines_take(self, key, horiz):
        """Press the inserted tines up and hand them most of the case's weight, if the loads pass."""
        f = self.cell.fork
        mass, com = self.load[key]
        weight = mass * G
        depth = min(horiz)
        caps = hand.tine_caps(weight, float(com[0]), f)
        loads = self.be.fork_press(caps)
        support = hand.judge_support(loads, weight, com, depth, f, self.cell.gripper, self.cell.gantry)
        self._emit('FORK_IN', case=key, accepted=support.accepted, mode=support.mode, share=round(support.share, 3),
                   tines_N=[round(v, 1) for v in loads], pushes_N=[round(v, 1) for v in caps],
                   max_accel=round(support.max_accel, 2), reason=support.reason)
        if not support.accepted:
            self.be.fork_retract(self._monitor)
            self._emit('FORK_SKIPPED', case=key, reason=support.reason + '。吸着だけでゆっくり運ぶ')
            return False
        self.fork_engaged, self.support = True, support
        self.vacuum_losses_this_carry = 0
        self.fork_yaw = float(self.be.gantry_q()[3])           # the tool's heading when the tines took the case
        return True

    def _tines_before_travel(self, key, up_dim, long_axis, only=None):
        """Requirements v05 P7 before C1 (review 2026-10-31): the tines are there to keep a case from falling when
        the travel shakes it, so they go under an upright case right after the lift above A. The heading the
        first free slot needs is turned to here, with the tines under the case as a catch and not yet pressing,
        and only then do they take the weight. Not upright by the pile's estimate, or no slot with an open side:
        the measurement from below decides, as before."""
        if abs(up_dim - self.sku.size[2]) >= UPRIGHT_TOL or key in self.gripped_crosswise or key not in self.load:
            return None
        horiz = sorted(self.sku.size, reverse=True)
        horiz.remove(min(horiz, key=lambda d: abs(d - up_dim)))
        q = self.be.gantry_q()
        yaw = float(q[3])
        rel = math.atan2(long_axis[1], long_axis[0]) - yaw
        for _, _, slot_yaw, _, slot_index in (only if only is not None else self._destinations(True, horiz, up_dim)):
            place_yaw = self._tool_yaw_for((math.cos(slot_yaw - rel), math.sin(slot_yaw - rel)), yaw)
            yaw_bar = self._bar_yaw(slot_index, place_yaw, yaw)
            if yaw_bar is not None:
                break
        else:
            return None
        if not self._tines_under(key, up_dim, np.zeros(2), horiz, gap=CATCH_GAP):
            self.tines_refused.add(key)                       # something is in the way: carried on the cups, slowly
            return None
        if self._carry_move([q[0], q[1], q[2], yaw_bar]) is None:
            return 'aborted'
        if not self._tines_take(key, horiz):
            return None
        return 'taken'

    def _fork_out_in_air(self, key, reason):
        self.be.fork_retract(self._monitor)
        self.fork_engaged, self.support = False, None
        self._emit('FORK_OUT', case=key, slot=None, in_air=True, reason=reason)

    def _bar_yaw(self, slot_index, place_yaw, current_yaw):
        """The tool yaw, of those that align the case with its slot, that turns the bars to the slot's
        open long side with the least turn of the carried case. None if the slot has none."""
        n = open_long_side(self.slots, slot_index)
        if n is None:
            return None
        lo, hi = self.cell.gantry.yaw_range
        options = [place_yaw + k * math.pi for k in range(-4, 5)
                   if lo <= place_yaw + k * math.pi <= hi and float(hand.bar_direction(place_yaw + k * math.pi) @ n) > 0.9]
        return min(options, key=lambda y: abs(y - current_yaw)) if options else None

    def _bar_side_seen_clear(self, target_xy, yaw, footprint, base_z, up_dim):
        """Nothing seen in the space the bars come down in and the tines are pulled back through: from
        the case's face to the tines' back end, from just above the surface up to the case's top."""
        if self.vm_b is None:
            return False
        f = self.cell.fork
        d = hand.bar_direction(yaw)
        t = np.array([-d[1], d[0]])
        fx, fy = footprint
        depth = abs(fx * d[0]) + abs(fy * d[1])
        length = abs(fx * t[0]) + abs(fy * t[1])
        near, far = depth / 2 + 0.005, abs(f.bar_y) + f.tine_size[1]
        c = self.vm_b.centers()
        occ = c[self.vm_b.state() == OCCUPIED]
        rel = occ[:, :2] - np.asarray(target_xy, float)[:2]
        along, across = rel @ d, rel @ t
        inside = ((along > near + self.vm_b.res / 2) & (along < far) & (np.abs(across) < length / 2)
                  & (occ[:, 2] > base_z + 0.01 + self.vm_b.res / 2) & (occ[:, 2] < base_z + up_dim))
        return not bool(inside.any())

    def _aim_by_sight(self, sight, xy, place_yaw, z):
        """The top face of the hanging case shows around the plate. Measure its centre and long
        edge from B's camera and move the tool until they match the slot: the floor camera's
        measurement is older, and the case can turn or swing on the cups since."""
        key, target, slot_yaw = sight
        face = sorted(self.sku.size, reverse=True)[:2]
        shift_total, turn_total = np.zeros(2), 0.0
        for _ in range(AIM_TRIES):
            top = measure_top_face(self.be.capture('overhead_b'), key)
            if not _whole_face(top, face):
                break                                          # partly hidden: a wrong size means a wrong centre
            d = (top.long_edge_yaw_deg - math.degrees(slot_yaw)) % 180.0
            turn = math.radians(d if d <= 90.0 else d - 180.0)
            lo, hi = self.cell.gantry.yaw_range
            if abs(turn) < math.radians(AIM_YAW_DEADBAND_DEG):
                turn = 0.0                                     # noise
            if turn and self.fork_engaged:
                if abs(turn) < math.radians(AIM_CLAMPED_DEADBAND_DEG):
                    turn = 0.0
                else:
                    # Pressed onto the tines the case does not follow a turn: they come out here, over the slot, and
                    # the cups turn it and set it down (review 2026-10-31)
                    self._fork_out_in_air(key, '置く直前に向きのずれを見つけたため、区画の上で爪を抜き、吸着だけで向きを合わせて置く')
            if turn and not lo <= place_yaw - turn <= hi:
                alt = turn - math.copysign(math.pi, turn)     # the case is the same turned half round
                turn = alt if not self.fork_engaged and lo <= place_yaw - alt <= hi else 0.0   # else the check on B decides
            # Turning the tool turns the case about the tool point: move the tool so the case
            # centre ends on the target.
            rel = top.centre - xy
            c, s = math.cos(-turn), math.sin(-turn)
            rel_after = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
            shift = np.asarray(target, float) - (xy + rel_after)
            if np.linalg.norm(shift) < AIM_POS_TOL and abs(turn) < math.radians(AIM_YAW_TOL_DEG):
                break
            xy = xy + shift
            place_yaw -= turn
            shift_total += shift
            turn_total -= turn
            if self._carry_move([xy[0], xy[1], z, place_yaw], speed=SLOW) is None:
                return None, place_yaw
        if np.any(shift_total) or turn_total:
            self._emit('AIMED', case=key, shift_mm=[round(float(v) * 1000, 1) for v in shift_total],
                       yaw_correction_deg=round(math.degrees(turn_total), 1))
        return xy, place_yaw

    def _after_pick(self, key, predicted, dest_kind):
        before = {k: b for k, b in (self.prior or {}).items() if k not in self.removed}
        bel = self._survey()
        if bel is None:
            return 'done'
        moved = []
        for k, b in bel.boxes.items():
            p = before.get(k)
            # Only well-seen cases are compared: a poorly seen case's estimate wanders between views.
            if p is None or min(p.confidence / DECAY_EPOCH, b.confidence) < MOVE_CHECK_CONFIDENCE:
                continue
            if np.linalg.norm(np.asarray(b.center) - np.asarray(p.center)) > 0.05:
                moved.append(k)
        self._emit('PILE_CHANGED', removed=key, moved=sorted(moved), predicted=sorted(predicted),
                   unexpected=sorted(set(moved) - set(predicted)), missed=sorted(set(predicted) - set(moved)))
        if dest_kind == 'B':
            out = self._check_placement(key, repair=True)
            return self._check_neighbours(key) if out == 'done' else out
        self._observe_b()          # the handover strips are seen by B's camera: never reuse a stale map
        return 'done'

    def _observe_b(self):
        frame = self.be.capture('overhead_b')
        p = self.cell.pallet
        top = p.size[2] + self.cell.layout.layers * self.sku.size[2] + 0.3
        spots = np.array(self.cell.handover.spots, float)
        lo = np.minimum([p.b_center[0] - p.size[0] / 2 - 0.1, p.b_center[1] - p.size[1] / 2 - 0.1], spots.min(0) - 0.35)
        hi = np.maximum([p.b_center[0] + p.size[0] / 2 + 0.1, p.b_center[1] + p.size[1] / 2 + 0.1], spots.max(0) + 0.35)
        vm = VoxelMap((lo[0], lo[1], 0.0), (hi[0], hi[1], top), VOXEL)
        vm.integrate(frame, step=2)
        self.vm_b = vm
        return frame

    def _check_placement(self, key, repair):
        """Look at the placed case from B's camera, by its own top face (neighbours packed flush
        against it would pull a whole-box fit by a centimetre): position, height, the direction of
        its long edge and its tilt. A case can turn on the cups after it was measured; askew, it
        blocks the next slot. One repair: grip it again, measure it again, put it back square."""
        frame = self._observe_b()
        top = measure_top_face(frame, key)
        slot = self.slots[self.slot_of[key]]
        footprint = sorted(self.sku.size, reverse=True)[:2]
        if top is None or abs(top.extents[0] - footprint[0]) > 0.02 or abs(top.extents[1] - footprint[1]) > 0.02:
            self.failed_checks.add(key)
            self._emit('PLACEMENT_CHECKED', case=key, ok=False, reason='上面の全体が見えず確かめられない（未検証）')
            return 'done'
        err = float(np.linalg.norm(top.centre - np.asarray(slot.center[:2])))
        dz = float(top.top_z - (slot.center[2] + self.sku.size[2] / 2))
        d = (top.long_edge_yaw_deg - slot.yaw_deg) % 180.0
        yaw_err = min(d, 180.0 - d)
        ok = err <= POS_TOL and abs(dz) <= POS_TOL and yaw_err <= YAW_TOL_DEG and top.tilt_deg <= TILT_TOL_DEG
        self._emit('PLACEMENT_CHECKED', case=key, ok=ok, position_error_m=round(err, 4), height_error_m=round(dz, 4),
                   yaw_error_deg=round(yaw_err, 1), tilt_deg=round(top.tilt_deg, 1))
        if ok:
            return self._check_settled(key, top)
        if not repair:
            self.failed_checks.add(key)
            return 'done'
        return self._repair_placement(key, top, slot)

    def _check_neighbours(self, key):
        """A case landing beside others can push them after they were checked (S3 104: a case landed 32 mm off
        and moved the checked case beside it 10.9 mm). Look again at the placed cases next to it on its layer
        whose tops are in view, and put right any that has left its slot."""
        slot = self.slots[self.slot_of[key]]
        near = [k for k, i in self.slot_of.items() if k != key and k not in self.failed_checks
                and abs(self.slots[i].center[2] - slot.center[2]) < 0.01
                and np.linalg.norm(np.asarray(self.slots[i].center[:2]) - np.asarray(slot.center[:2])) <= NEIGHBOUR_REACH]
        if not near:
            return 'done'
        frame = self._observe_b()
        footprint = sorted(self.sku.size, reverse=True)[:2]
        for k in near:
            top = measure_top_face(frame, k)
            if top is None or abs(top.extents[0] - footprint[0]) > 0.02 or abs(top.extents[1] - footprint[1]) > 0.02:
                continue                         # not wholly in view: a part of a face is no measure of where it is
            s = self.slots[self.slot_of[k]]
            err = float(np.linalg.norm(top.centre - np.asarray(s.center[:2])))
            d = (top.long_edge_yaw_deg - s.yaw_deg) % 180.0
            yaw_err = min(d, 180.0 - d)
            if err <= POS_TOL and yaw_err <= YAW_TOL_DEG:
                continue
            self._emit('PLACEMENT_CHECKED', case=k, ok=False, position_error_m=round(err, 4), yaw_error_deg=round(yaw_err, 1),
                       tilt_deg=round(top.tilt_deg, 1), moved_by=key)
            out = self._repair_placement(k, top, s)
            if out != 'done':
                return out
            frame = self._observe_b()
        return 'done'

    def _check_settled(self, key, first):
        """A case that matched its slot just after release may still creep or lean. Watch it."""
        if self.be.idle(SETTLE_S, self._monitor) is None:
            return 'done'
        top = measure_top_face(self._observe_b(), key)
        if top is None:
            self.failed_checks.add(key)
            self._emit('PLACEMENT_SETTLED', case=key, ok=False, reason='B の画像に見えない', waited_s=SETTLE_S)
            return 'done'
        drift = float(np.linalg.norm(np.r_[top.centre - first.centre, top.top_z - first.top_z]))
        lean = abs(top.tilt_deg - first.tilt_deg)
        ok = drift <= DRIFT_TOL and lean <= TILT_TOL_DEG
        if not ok:
            self.failed_checks.add(key)
        self._emit('PLACEMENT_SETTLED', case=key, ok=ok, drift_mm=round(drift * 1000, 1), lean_deg=round(lean, 1),
                   waited_s=SETTLE_S)
        return 'done'

    def _repair_placement(self, key, top, slot):
        """Grip the misplaced case at the centre of its measured top face (a whole-box fit is pulled
        by the neighbours it touches) and put it back square."""
        up_dim = float(self.sku.size[2])
        safe = self._safe_z(up_dim)
        self._emit('PLACEMENT_REPAIRED', case=key, slot=slot.index, reason='向きか位置がずれているため、つかみ直して置き直す')
        held, yaw, long_axis = self._grip_by_top(key, top, safe)
        if held == 'aborted':
            return 'aborted'
        if held == 'failed':
            self._emit('HANDOVER', case=key, reason='B の上でずれたケースをつかみ直せない（人が直す）')
            return 'done'
        only = [('B', tuple(slot.center[:2]), math.radians(slot.yaw_deg), slot.center[2] - self.sku.size[2] / 2, slot.index)]
        placed_at, _, _ = self._measure_and_place(key, yaw, safe, up_dim, np.array([long_axis[0], long_axis[1], 0.0]), only=only)
        if placed_at == 'aborted':
            return 'aborted'
        if placed_at is None:
            self._emit('HANDOVER', case=key, reason='B の区画に置き直せないため、持ったまま待機する（人が受け取る）')
            self.holding_for_person = key
            return 'hold'
        self._move([slot.center[0], slot.center[1], safe, placed_at[2]])
        if self._park() is None:                                  # the tool above the case hides it from the camera
            return 'aborted'
        return self._check_placement(key, repair=False)

    # ================================================================== mission
    def run(self) -> MissionReport:
        self._emit('MISSION_ACCEPTED', instruction=self.cfg.instruction, policy=self.cfg.policy, sku=self.sku.id,
                   slots=len(self.slots))
        t0 = self.be.now()
        cycles = 0
        if self._survey() is None:
            self._survey()
        self._observe_b()
        while cycles < self.cfg.max_cycles and self.be.now() - t0 < self.cfg.max_seconds:
            if self.belief is None or self.belief_stamp < self.sup.last_world_change:
                self._survey()
                continue
            choice = self._choose()
            if choice[0] == 'done':
                return self._finish(choice[1], t0)
            if choice[0] == 'observe':
                self.obs_this_decision += 1
                _, vp, info, reason = choice
                self._observe_side(vp, reason, info)
                continue
            _, cand, fallback = choice
            outcome = self._execute_pick(cand, fallback)
            if outcome == 'aborted' and self.holding_for_person is not None:
                outcome = 'hold'
            if outcome == 'aborted' and self.be.gripped():
                # A move that could not be made with a case on the cups: never pick with it still held.
                held = self.be.gripped()[0]
                self._emit('HANDOVER', case=held, reason='動作を続けられないため、持ったまま待機する（人が受け取る）')
                self.holding_for_person = held
                outcome = 'hold'
            if outcome == 'hold':
                return self._finish('ケースを持ったまま人の対応を待つ', t0)
            cycles += 1
        return self._finish('時間または回数の上限', t0)

    def _hidden_in_reach(self, for_person=()):
        """Chance that unseen space within reach hides a case, and the unseen points left to people:
        outside the reach, or in the shadow of a case left for a person (who sees it on removing it)."""
        pts = self.belief.unexplained_points()
        (x0, x1), (y0, y1) = self.belief.reach
        inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        for k in for_person:
            o = self.belief.boxes[k].obb
            lo, hi = o.aabb()
            near = np.all((pts[:, :2] >= lo[:2] - SHADOW_MARGIN) & (pts[:, :2] <= hi[:2] + SHADOW_MARGIN), axis=1)
            inside &= ~near
        vol = float(np.sum(inside)) * VOXEL ** 3
        return 1.0 - math.exp(-vol / (0.5 * float(np.prod(self.sku.size)))), pts[~inside]

    def _finish(self, reason, t0):
        # A case beyond the gantry's reach is beyond the robot: name it for the person.
        far = {c.box_id for c in self.belief.candidates() if c.reason == 'out of reach'}
        for k in far:
            self.unpickable.setdefault(k, '腕が届かない場所にある（人が取り出す）')
        for k in self.waiting:
            # what it waited for was never taken: the way stays closed
            self.unpickable[k] = '吸着板か爪の進路にほかの物があり、取れない（人が確認）'
            self._emit('HANDOVER', case=k, reason=self.unpickable[k])
        self.waiting = {}
        remaining = {k: self.unpickable.get(k, '未処理') for k in self.belief.boxes if k not in self.removed}
        if remaining and far and set(remaining) <= far:
            reason = '残ったケースは腕が届かない場所にある（人が取り出す）'
        self.report.remaining = remaining
        self.report.seconds = round(self.be.now() - t0, 2)
        p_hidden, outside = self._hidden_in_reach(for_person=[k for k, why in remaining.items() if why != '未処理'])
        self.report.unseen_outside_reach = _floor_patches(outside)
        # Every case the load held is either moved, still known in the cell, or unaccounted for.
        # Unaccounted cases were never seen: thrown beyond the cell in transport, or hidden.
        if self.cfg.expected_cases is not None:
            seen = len(self.report.placed) + len(self.report.handed_over) + len(remaining)
            self.report.not_found = max(0, self.cfg.expected_cases - seen)
        if self.report.not_found:
            reason = f'{reason}。積荷の個数より {self.report.not_found} 個少ない：作業場の外に落ちた可能性がある（人が確認）'
        unresolved = sum(1 for why in remaining.values() if why == '未処理') + (1 if p_hidden > 0.2 else 0)
        self.report.status = end_status(remaining, unresolved, self.report.not_found, self.holding_for_person is not None,
                                        len(self.failed_checks), len(self.report.handed_over), damage_risk=len(self.damage_risk))
        self.report.failed_checks = sorted(self.failed_checks)
        done = self.report.status.startswith('verified')
        self._emit('MISSION_COMPLETE' if done else 'MISSION_INCOMPLETE', reason=reason, status=self.report.status,
                   failed_checks=self.report.failed_checks, placed=len(self.report.placed),
                   handed_over=len(self.report.handed_over), remaining=remaining, seconds=self.report.seconds,
                   observations=self.report.observations, safety_stops=self.report.safety_stops,
                   hidden_probability_in_reach=round(p_hidden, 3), check_by_person=self.report.unseen_outside_reach,
                   not_found=self.report.not_found, states_met=self.states_met, damage_risk=len(self.damage_risk))
        return self.report


def _whole_face(top, face, tol=0.015):
    """A top face measured whole: its seen edges match the case's (long, short) face within tol."""
    return top is not None and abs(top.extents[0] - face[0]) <= tol and abs(top.extents[1] - face[1]) <= tol


def end_status(remaining, unresolved_remaining, not_found, holding, failed_checks, handed, damage_risk=0):
    """How a mission ends (requirements v04 chapter 6; v05 chapter 3 adds 'moved, quality not passed'
    when a contact pressed a case over its limit: the cases are moved, but not all verified unharmed)."""
    if holding:
        return 'interrupted'
    if unresolved_remaining or not_found or failed_checks:
        return 'unresolved'
    if damage_risk:
        return 'moved_quality_unverified'
    if handed or remaining:
        return 'verified_with_handover'
    return 'verified'


def _floor_patches(points, cell=0.5):
    """Coarse floor areas (outside the robot's reach) that were never seen, for a person to check."""
    if len(points) == 0:
        return []
    keys = {(round(float(np.floor(x / cell) * cell + cell / 2), 2), round(float(np.floor(y / cell) * cell + cell / 2), 2)) for x, y in points[:, :2]}
    return [{'x': k[0], 'y': k[1], 'size_m': cell} for k in sorted(keys)]


def _hold_position(state):
    return Decision(Action.CONTINUE)
