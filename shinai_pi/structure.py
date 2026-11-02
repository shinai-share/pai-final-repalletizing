"""The gantry's structure where parts slide in guides, as boxes (review 2026-10-29), and how the tool is fastened and
where it may move without meeting the rest of the gantry (review 2026-10-31).

The simulator joins two bodies by a joint whatever their shapes, so a part can hang free of what is meant to carry
it and still move: in the submission video the ram hung up to 0.6 m below its trolley, the fork bars had no mount,
and a fully inserted tine had left its bar. Here each sliding part (runner) is named with the guide it runs in, and
engagement() says how much of the guide it fills at a joint value. The scene (sim/cell.py) builds these boxes; the
tests and validate() check every slide over its whole travel.

Two more things a joint does not show (review 2026-10-31): a part fixed to the wrist that meets the plate along an
edge carries nothing (the fork's arm did), and the articulation's own parts do not collide in the simulator, so the
drawn-up bars could pass through the ram when the wrist rolled and pitched together. unfastened() follows the joins
from the plate; interference() samples the poses the gantry may take.

Every body's frame sits at the gantry origin (0, 0, origin_height) at joint zero, as the scene builds them; boxes
are in their body's frame. A prismatic joint at q moves its child body by q along the axis. The wrist turns about
the cup face's centre: yaw (ram -> tool) about z, then pitch about y, then roll about x."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

WRIST_LINK = 0.24                  # m from the cup face to the bottom of the ram column
FORK_SIDES = (('l', -1.0), ('r', 1.0))
BEAM_UNDERSIDE = 0.28              # m below the top of the frame: the bridge beam's underside, where the trolley hangs
SLEEVE_CLEAR = 0.01                # m between a tine's sleeve and the face of the deepest case the cups hold
MIN_FASTENED_AREA = 0.002          # m^2 of face two fixed parts share to count as bolted (a 40 x 50 mm pad)
TOUCH_TOL = 1e-6
# The plate and the yoke bolted level with it lie at the roll axis: a quarter roll brings their -y edge to where the
# schematic wrist housing stands (0.08 m square on the plate's centre). A real wrist carries the roll on side brackets
# clear of them; the housing box is a stand-in, so it is not checked against them (the bars and guides are).
AT_THE_PIVOT = ('plate', 'yoke')


@dataclass(frozen=True)
class Box:
    body: str
    name: str
    size: tuple
    pos: tuple
    collide: bool = True


@dataclass(frozen=True)
class Slide:
    joint: str
    guide: Box                     # on the joint's parent
    runner: Box                    # on the child
    axis: int                      # 0 x, 1 y, 2 z
    travel: tuple                  # joint values
    min_engagement: float          # m of runner that must stay in the guide


def extent(box: Box, axis: int):
    return box.pos[axis] - box.size[axis] / 2, box.pos[axis] + box.size[axis] / 2


def engagement(slide: Slide, q: float) -> float:
    """Length along the axis over which the runner, moved by q, lies in its guide (negative: a gap)."""
    g0, g1 = extent(slide.guide, slide.axis)
    r0, r1 = extent(slide.runner, slide.axis)
    return min(g1, r1 + q) - max(g0, r0 + q)


def gantry_boxes(cell):
    g, s = cell.gantry, cell.structure
    top = g.beam_height - g.origin_height
    th = s.trolley_size[2]
    return [Box('trolley', 'body', tuple(s.trolley_size), (0.0, 0.0, top - BEAM_UNDERSIDE - th / 2), collide=False),
            Box('ram', 'column', (s.ram_section, s.ram_section, s.ram_length),
                (0.0, 0.0, WRIST_LINK + s.ram_length / 2))]


def wrist_link(cell):
    """The housing of the pitch and roll axes, on the tool body (it turns with yaw only)."""
    t = cell.gantry.tool_size[2]
    return Box('tool', 'wrist_link', (0.08, 0.08, WRIST_LINK - t), (0.0, 0.0, (WRIST_LINK + t) / 2), collide=False)


def plate(cell):
    g = cell.gantry
    return Box('wrist_r', 'plate', tuple(g.tool_size), (0.0, 0.0, g.tool_size[2] / 2))


def _guide_section(cell):
    f, s = cell.fork, cell.structure
    return f.bar_size[0] + 2 * s.fork_guide_wall, f.bar_size[1] + 2 * s.fork_guide_wall


def fork_mounts(cell):
    """The frame the fork hangs from, all fixed to the wrist: a yoke bolted face to face over the plate's -y side,
    and one guide per bar on the yoke's face rising from the cup face. No tie across the guides' tops: rolled a
    quarter towards -y, it would cross the ram (the guides stand clear of it at x = +-0.13)."""
    f, s, g = cell.fork, cell.structure, cell.gantry
    gx, gy = _guide_section(cell)
    back = f.bar_y + gy / 2                           # the guides' face towards the plate
    edge = -g.tool_size[1] / 2                        # the plate's -y face
    yoke = Box('wrist_r', 'yoke', (g.tool_size[0], edge - back, s.yoke_height),
               (0.0, (back + edge) / 2, s.yoke_height / 2))
    guides = [Box('wrist_r', f'fork_guide_{side}', (gx, gy, s.fork_guide_height),
                  (sx * f.bar_x, f.bar_y, s.fork_guide_height / 2)) for side, sx in FORK_SIDES]
    return [yoke, *guides]


def fork_boxes(cell, side):
    """The bar, its tine and their guides on one side: the bar runs through its guide on the yoke; the tine runs
    through a sleeve at the bar's foot and reaches back behind it."""
    f, s = cell.fork, cell.structure
    sx = dict(FORK_SIDES)[side]
    x = sx * f.bar_x
    bx, by, bz = f.bar_size
    tx, ty, tz = f.tine_size
    guide = next(b for b in fork_mounts(cell) if b.name == f'fork_guide_{side}')
    s0, s1 = s.tine_sleeve
    sleeve = Box(f'fork_bar_{side}', 'sleeve', (tx, s1 - s0, tz + s.tine_sleeve_lid),
                 (x, f.bar_y + (s0 + s1) / 2, (-tz + s.tine_sleeve_lid) / 2))
    return [guide,
            Box(f'fork_bar_{side}', 'geom', tuple(f.bar_size), (x, f.bar_y, -tz + bz / 2)),
            sleeve,
            Box(f'fork_tine_{side}', 'tine', tuple(f.tine_size), (x, f.bar_y - ty / 2, -tz / 2))]


def tool_frame(cell):
    """Every part fixed to the wrist."""
    return [plate(cell), *fork_mounts(cell)]


def slides(cell):
    g, f, s = cell.gantry, cell.fork, cell.structure
    trolley, column = gantry_boxes(cell)
    out = [Slide('z_joint', trolley, column, 2, tuple(g.z_range), s.ram_min_engagement)]
    e0, e1 = f.extension_range
    for side, _ in FORK_SIDES:
        guide, bar, sleeve, tine = fork_boxes(cell, side)
        out.append(Slide(f'fork_z_{side}', guide, bar, 2, (-e1, -e0), s.fork_min_engagement))
        out.append(Slide(f'fork_y_{side}', sleeve, tine, 1, tuple(f.insert_range), s.tine_min_engagement))
    return out


def ram_mass(cell) -> float:
    s = cell.structure
    inner = s.ram_section - 2 * s.ram_wall
    return (s.ram_section ** 2 - inner ** 2) * s.ram_length * s.steel_density


def _shared(a: Box, b: Box):
    """Overlap of two boxes of one body along each axis (negative: a gap)."""
    return [min(extent(a, k)[1], extent(b, k)[1]) - max(extent(a, k)[0], extent(b, k)[0]) for k in range(3)]


def joined(a: Box, b: Box, min_area: float = MIN_FASTENED_AREA) -> bool:
    """Welded into each other, or face to face over at least min_area. An edge or a corner is no join."""
    d = _shared(a, b)
    if min(d) < -TOUCH_TOL:
        return False
    touching = [k for k in range(3) if abs(d[k]) <= TOUCH_TOL]
    if not touching:
        return True
    if len(touching) > 1:
        return False
    i, j = (k for k in range(3) if k != touching[0])
    return d[i] * d[j] >= min_area


def unfastened(boxes, min_area: float = MIN_FASTENED_AREA):
    """The boxes that do not reach the first (the plate) through joins."""
    reached, frontier = {0}, [0]
    while frontier:
        i = frontier.pop()
        for j, b in enumerate(boxes):
            if j not in reached and joined(boxes[i], b, min_area):
                reached.add(j)
                frontier.append(j)
    return [b for j, b in enumerate(boxes) if j not in reached]


def tool_boxes_at(cell, extension: float, insert: float):
    """Every box that turns with the wrist, as (box, rotation, offset) in the wrist's frame, at a fork position."""
    out = [(b, np.eye(3), np.zeros(3)) for b in tool_frame(cell)]
    for side, _ in FORK_SIDES:
        _, bar, sleeve, tine = fork_boxes(cell, side)
        down = np.array([0.0, 0.0, -extension])
        out += [(bar, np.eye(3), down), (sleeve, np.eye(3), down), (tine, np.eye(3), down + np.array([0.0, insert, 0.0]))]
    return out


def wrist_allowed(pitch: float, roll: float) -> bool:
    """Rolled towards -y while pitched, the drawn-up bars lean into the ram. A half turn of the tool gives any face the
    opposite roll, so the planner takes that one (executive)."""
    return abs(pitch) < 1e-9 or roll >= 0.0


def _rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _overlap(ca, Ra, ha, cb, Rb, hb, margin=0.0) -> bool:
    """Oriented boxes (centre, rotation, half sizes) meet: no separating axis among the 15."""
    t = cb - ca
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    axes += [np.cross(Ra[:, i], Rb[:, j]) for i in range(3) for j in range(3)]
    for n in axes:
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        ra = sum(ha[i] * abs(Ra[:, i] @ n) for i in range(3))
        rb = sum(hb[i] * abs(Rb[:, i] @ n) for i in range(3))
        if abs(t @ n) > ra + rb + margin:
            return False
    return True


def interference(cell, wrist_rule: bool = True, steps=(7, 5, 9, 6)):
    """The first pose, in samples over yaw, pitch, roll and height and both ends of the fork's stroke, at which a box
    turning with the wrist meets the ram, the trolley or the wrist's housing; None if there is none."""
    g, f = cell.gantry, cell.fork
    n_yaw, n_pitch, n_roll, n_z = steps
    trolley, column = gantry_boxes(cell)
    link = wrist_link(cell)
    for yaw, pitch, roll, z in itertools.product(np.linspace(0.0, math.pi, n_yaw), np.linspace(*g.pitch_range, n_pitch),
                                                 np.linspace(*g.roll_range, n_roll), np.linspace(*g.z_range, n_z)):
        if wrist_rule and not wrist_allowed(pitch, roll):
            continue
        Rt = _rot(2, yaw)
        R = Rt @ _rot(1, pitch) @ _rot(0, roll)
        lift = np.array([0.0, 0.0, z])
        fixed = [(trolley, np.eye(3), np.zeros(3)), (column, np.eye(3), lift), (link, Rt, lift)]
        for extension in f.extension_range:
            for b, Rb, p in tool_boxes_at(cell, extension, 0.0):
                cb = R @ (np.asarray(b.pos) + p) + lift
                for o, Ro, po in fixed:
                    if o is link and b.name in AT_THE_PIVOT:
                        continue                       # the housing stands for the wrist's axes, which these turn about
                    co = Ro @ np.asarray(o.pos) + po
                    if _overlap(cb, R @ Rb, np.asarray(b.size) / 2, co, Ro, np.asarray(o.size) / 2):
                        return {'yaw': round(float(yaw), 3), 'pitch': round(float(pitch), 3), 'roll': round(float(roll), 3),
                                'z': round(float(z), 3), 'extension': extension, 'part': b.name, 'meets': o.name}
    return None


def check(cell, samples: int = 81):
    """The first slide that leaves its guide anywhere in its travel, a sleeve against a held case, or a part not
    fastened to the plate, as a message; None when there is none."""
    held_depth = max(sorted(sk.size, reverse=True)[1] for sk in cell.skus)
    for side, _ in FORK_SIDES:
        sleeve = next(b for b in fork_boxes(cell, side) if b.name == 'sleeve')
        if -held_depth / 2 - extent(sleeve, 1)[1] < SLEEVE_CLEAR - 1e-9:
            return f'fork_y_{side}: the tine sleeve reaches within {SLEEVE_CLEAR} m of a held case'
    for sl in slides(cell):
        lo, hi = sl.travel
        for k in range(samples):
            q = lo + (hi - lo) * k / (samples - 1)
            if engagement(sl, q) < sl.min_engagement - 1e-9:
                return f'{sl.joint}: at {q:.3f} only {engagement(sl, q):.3f} m of {sl.runner.name} in its guide ' \
                       f'(at least {sl.min_engagement:.3f})'
    loose = unfastened(tool_frame(cell))
    if loose:
        return f'{loose[0].name}: not fastened to the plate over a face'
    return None
