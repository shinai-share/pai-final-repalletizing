"""The suction + under-support hand (requirements v05, design drawings v01).

Tool frame: x along the plate's long side, y across it, z up from the cup face. The two bars stand on
the -y side at x = +-bar_x; their tines slide along +y under the held case and push up. Forces in N.

Accounting (F0, artifacts/f0): the tine joints read their own load 1:1. The cup-head reading is not
the cups' load once the tines push (Isaac subtracts the tines' force from it), so what the cups carry
is the weight measured on the cups before the tines went in, minus the tines."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81
EVEN_SHARE = 0.70              # the tines take at least this share of the weight (design: suction <= 30 %)
EVEN_SPREAD = 0.30             # left and right within this share of their sum
CUP_SAFETY = 2.0               # every cup within half its limit, force and moment taken together
FACE_MARGIN = 0.01             # m between a bar and the held case's face
EDGE_MARGIN = 0.005            # m between a tine's outer edge and the case's edge


@dataclass(frozen=True)
class Support:
    accepted: bool
    mode: str                  # 'even' | 'uneven' | 'refused'
    max_accel: float           # m/s^2 the case may be carried at
    share: float               # of the weight on the tines
    reason: str = ''


def bar_direction(tool_yaw: float) -> np.ndarray:
    """World direction (x, y) from the held case towards the bars (tool -y)."""
    return np.array([math.sin(tool_yaw), -math.cos(tool_yaw)])


def tine_caps(weight: float, com_x: float, fork):
    """Push of (left, right) tine: together the support share of the weight, split so that the tines,
    at x = -+bar_x, carry the moment of contents off-centre along the tool x axis. Each within its
    limit; what a limit cuts off is left to the cups (judge_support checks they can take it)."""
    total = fork.support_fraction * weight
    diff = weight * com_x / fork.bar_x
    left = min(max((total - diff) / 2, 0.0), fork.tine_load_limit)
    right = min(max((total + diff) / 2, 0.0), fork.tine_load_limit)
    return left, right


class TouchdownDetector:
    """The tines meet the surface below: their loads change from what they carried in transit and
    stay changed. The direction is not relied on (F0: up at 120 Hz, down at 240 Hz)."""

    TICKS = 3                  # control ticks a change must last
    SURE = 4.0                 # a change this many thresholds at once is contact, not noise

    def __init__(self, carried, weight: float):
        self.carried = np.asarray(carried, float)
        self.threshold = max(10.0, 0.1 * weight)
        self.count = 0
        self.peak = 0.0

    def update(self, tines) -> bool:
        change = float(np.abs(np.asarray(tines, float) - self.carried).max())
        self.last = change
        self.peak = max(self.peak, change)
        self.count = self.count + 1 if change > self.threshold else 0
        return self.count >= self.TICKS or change > self.SURE * self.threshold


def tine_y(depth: float, fork) -> float:
    """Middle of the part of a fully inserted tine that lies under a case `depth` deep (tool y)."""
    lo = max(fork.bar_y + fork.insert_range[1] - fork.tine_size[1], -depth / 2)
    hi = min(fork.bar_y + fork.insert_range[1], depth / 2)
    return (lo + hi) / 2


def tine_extension(bottom_depth: float, tilt: float, depth: float, fork) -> float:
    """Tine top below the cup face for sliding in: under the case bottom by the clearance, and under
    the far edge of a bottom that a hanging tilt lowers."""
    e = bottom_depth + fork.clearance + depth / 2 * math.tan(abs(tilt))
    if e > fork.extension_range[1]:
        raise ValueError(f'tines cannot reach {e:.3f} m below the cups')
    return e


def fork_fits(offset_tool, depth: float, length: float, fork):
    """(ok, reason): the held case (centre `offset_tool` from the cup centre, `length` along tool x,
    `depth` along tool y) leaves room for the bars and has both tines under its bottom."""
    ox, oy = float(offset_tool[0]), float(offset_tool[1])
    if oy - depth / 2 < fork.bar_y + fork.bar_size[1] / 2 + FACE_MARGIN:
        return False, '吸着のずれで、ケースの側面が爪の棒に当たる位置にある'
    if fork.bar_x + fork.tine_size[0] / 2 > length / 2 - abs(ox) - EDGE_MARGIN:
        return False, '吸着のずれで、片方の爪がケースの底から外れる'
    if fork.bar_y + fork.insert_range[1] < oy:
        return False, '爪の先がケースの重心の下まで届かない'
    return True, ''


def suction_wrench(weight: float, com, tines, tine_x, tine_y_: float):
    """(force, torque about tool x, torque about tool y) left to the cups, in the wrist reading's
    convention for a hanging load (W, -W r_y, W r_x): the weight at its centre of mass minus the
    tines pushing up at (tine_x[i], tine_y_)."""
    tx = -weight * com[1] + sum(t * tine_y_ for t in tines)
    ty = weight * com[0] - sum(t * x for t, x in zip(tines, tine_x))
    return (weight - sum(tines), tx, ty)


def cup_loads(wrench, cup_offsets):
    """Axial load per cup (positive: the cup pulls) for a rigid plate: force and both moments split
    together (least-norm over the cups)."""
    f, tx, ty = wrench
    cups = np.asarray(cup_offsets, float)
    sx2, sy2 = float(np.sum(cups[:, 0] ** 2)), float(np.sum(cups[:, 1] ** 2))
    return [f / len(cups) + ty * c[0] / sx2 - tx * c[1] / sy2 for c in cups]


def cup_utilisation(wrench, cup_offsets, coaxial_limit: float) -> float:
    return max(max(v, 0.0) for v in cup_loads(wrench, cup_offsets)) / coaxial_limit


def judge_support(tines, weight: float, com_tool, depth: float, fork, gripper, gantry) -> Support:
    """Is the weight handed to the tines, and how fast may the case then be carried?

    The tines must take most of the weight (EVEN_SHARE). Evenly shared, the case travels at the
    gantry's full acceleration. Unevenly (contents off-centre), it travels only at an acceleration
    where every tine stays within its limit and every cup within half its limit with the remaining
    force and moment together; vertical acceleration scales every load by (1 + a/g), the worst case.
    Below the slow acceleration it is not accepted."""
    tl = [float(t) for t in tines]
    total = sum(tl)
    share = total / weight if weight > 0 else 0.0
    if max(tl) > fork.tine_load_limit:
        return Support(False, 'refused', 0.0, share, '片爪の荷重が上限を超えた')
    if min(tl) < 0 or share < EVEN_SHARE:
        return Support(False, 'refused', 0.0, share, '爪が重さを受け取っていない')
    even = abs(tl[0] - tl[1]) <= EVEN_SPREAD * total
    ty_ = tine_y(depth, fork)

    def ok(a):
        k = 1.0 + a / G
        if max(tl) * k > fork.tine_load_limit:
            return False
        w = suction_wrench(weight * k, com_tool, [t * k for t in tl], (-fork.bar_x, fork.bar_x), ty_)
        return cup_utilisation(w, gantry.cup_offsets, gripper.planning_coaxial_limit) <= 1.0 / CUP_SAFETY
    a_full = gantry.a_max[0]
    if ok(a_full):
        return Support(True, 'even' if even else 'uneven', a_full, share)
    lo, hi = 0.0, a_full
    if not ok(lo):
        return Support(False, 'refused', 0.0, share, '吸着が残りの力とモーメントを同時に受けられない')
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    if lo < gantry.slow_accel:
        return Support(False, 'refused', lo, share, '偏りが大きく、ゆっくりでも吸着と爪の余裕が足りない')
    return Support(True, 'uneven', lo, share)


# ---------------------------------------------------------------- righting by tipping on the table
TABLE_MARGIN = 0.02            # m between a case's centre of mass and the table edge, before and after tipping


def pivot_offset(w0: float, h0: float, table_half: float, com: float = 0.0) -> float:
    """Where the bottom edge the case tips over goes, along the tip direction from the table centre.

    A case `w0` wide (across the tip) and `h0` tall tips over that edge: its box centre starts w0/2
    behind it and ends h0/2 beyond it, and its centre of mass may lie up to `com` off the box centre
    in either direction - unseen, when it lies along the axis the case hung by. The edge is placed so
    both lie on the table with equal room; ValueError if the table cannot hold both."""
    p = (w0 - h0) / 4
    if p - w0 / 2 - com < -table_half + TABLE_MARGIN - 1e-9 or p + h0 / 2 + com > table_half - TABLE_MARGIN + 1e-9:
        raise ValueError('the righting table is too small to hold the case before and after it tips')
    return p


def pivot_cup_force(mass: float, w0: float, h0: float) -> float:
    """Largest force the cups (on the centre of the top face) must give while the case tips over its
    bottom edge: the table carries the weight, the cups supply the moment about the edge - pushing it
    over at the start (centre of mass w0/2 behind the edge), holding it back near the end (h0/2 beyond)."""
    moment = mass * G * max(w0, h0) / 2
    return moment / math.hypot(w0 / 2, h0)


def pivot_path(pivot, cup_centre, tip_dir, steps: int = 9):
    """[(cup-face centre, wrist roll)] for tipping a case a quarter turn over its bottom edge at
    `pivot`, towards the horizontal `tip_dir`. The tool yaw is the one whose +y is `tip_dir` (the bars
    then end on top); the wrist rolls the same angle negative, so the cups stay on the face."""
    pivot = np.asarray(pivot, float)
    t = np.asarray(tip_dir, float) / np.linalg.norm(tip_dir)
    rel = np.asarray(cup_centre, float) - pivot
    u, v = float(rel[:2] @ t), float(rel[2])
    side = rel[:2] - u * t                                     # along the edge: unchanged by the tip
    out = []
    for k in range(steps + 1):
        phi = (math.pi / 2) * k / steps
        u2 = u * math.cos(phi) + v * math.sin(phi)
        v2 = -u * math.sin(phi) + v * math.cos(phi)
        out.append((pivot + np.r_[side + u2 * t, v2], -phi))
    return out


def fork_envelope(fork, extension=None):
    """The under-supports as a box in the tool frame (x0, x1, y0, y1, bottom above the cup face). Retracted
    they stand 0.22 m out past the plate on its -y side, above the cup face, and go down with it: lowered
    beside pallet B, they struck B's second layer while the plate's footprint was clear (S3 102)."""
    e = fork.extension_range[0] if extension is None else extension
    tx, ty, tz = fork.tine_size
    x = fork.bar_x + tx / 2
    return (-x, x, fork.bar_y - ty, fork.bar_y + fork.bar_size[1] / 2, -e - tz)


def stop_margin(speed: float, latency: float, decel: float, error: float = 0.01) -> float:
    """Clearance the cups need to stop short of what they approach: measurement error, the distance
    covered while the sighting travels through camera, judgement and drive, and the braking distance
    (design 5.1.1: d > e + v T + v^2 / 2a)."""
    return error + speed * latency + speed * speed / (2 * decel)


RECOVER_SHARE = 0.20           # the tines' sum within this share of the measured weight
RECOVER_MARGIN = 0.02          # m the centre of mass keeps inside the tines' support


def can_recover(tines, weight: float, com_tool, fork):
    """(ok, reason): with the cups let go, may the case be sealed again and carried on slowly? Only
    when the tines really carry it (their sum near its weight, each within its limit) and its centre
    of mass lies between them with room to spare (requirements v05 12.4 (b))."""
    tl = [float(t) for t in tines]
    if max(tl) > fork.tine_load_limit:
        return False, '片爪の荷重が上限を超えている'
    if min(tl) < 0 or abs(sum(tl) - weight) > RECOVER_SHARE * weight:
        return False, '爪が重さを受けていない'
    reach = fork.bar_x - fork.tine_size[0] / 2 - RECOVER_MARGIN
    if abs(float(com_tool[0])) > reach:
        return False, '重心が爪の支えの範囲の端にある'
    return True, ''


def contact_verdict(kind: str, static, peak, limits, raw=None) -> dict:
    """Judge one contact against the damage limits (requirements v05 chapter 3). A lasting force over its
    limit is a damage risk. A one-step peak over its limit is counted apart: the rigid model's contact
    spikes are not judged until a smaller physics step shows they converge. A reading not calibrated
    in newtons is recorded raw and not judged."""
    lim_static = {'touch': limits.touch_static, 'landing': limits.landing_static}[kind]
    lim_peak = {'touch': limits.touch_peak, 'landing': limits.landing_peak}[kind]
    if static is None:
        return {'kind': kind, 'calibrated': False, 'raw': [round(float(v), 1) for v in (raw or ())],
                'damage_risk': False, 'spike_over': False}
    return {'kind': kind, 'calibrated': True, 'static_N': round(float(static), 1), 'peak_N': round(float(peak), 1),
            'limit_static_N': lim_static, 'limit_peak_N': lim_peak,
            'damage_risk': bool(static > lim_static), 'spike_over': bool(peak > lim_peak)}
