"""Belief over a collapsed pile: who rests on whom, what falls if a case is removed, how sure we
are, what we have not seen, and whether looking again could change the decision.

Only perceived boxes and the known-space map enter here. Simulator ground truth never does.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .perception.obb import OBB

CONTACT_SPACING = 0.04        # m between surface samples used to find contacts
HULL_MARGIN = 0.01            # m of tolerance when testing the centre of mass against the support polygon
HARMFUL_DROP = 0.25           # m; a case falling this far counts as one full secondary collapse
VERTICAL_COS = 0.7            # a supporting surface faces up (normal z above this)
LATERAL_SIN = 0.5             # a side surface (normal z magnitude below this)
UNEXPLAINED_WEIGHT = 0.5      # expected falls per touching case whose support we cannot explain
APPROACH_MARGIN = 0.015       # m the lift column is shrunk by, to tolerate neighbour pose error
ROOM_SHARE = 0.7              # a hidden region must span this share of each case dimension
MAYBE_WEIGHT = 0.5            # a case that falls only for some placements of its contents counts half
OVERLAP_TOL = 0.03            # m two estimates may interpenetrate before one of them is taken to be wrong


class Verdict(Enum):
    COMPLETE = 'complete'
    PICK = 'pick'
    NEED_OBSERVATION = 'need_observation'


@dataclass(frozen=True)
class Box:
    box_id: str
    center: tuple
    size: tuple
    confidence: float = 1.0
    rotation: np.ndarray | None = field(default=None, compare=False)
    top_visible: float = 1.0

    def __post_init__(self):
        if len(self.center) != 3 or len(self.size) != 3:
            raise ValueError('center and size must have three components')
        if not all(math.isfinite(v) for v in (*self.center, *self.size, self.confidence)):
            raise ValueError('box values must be finite')
        if any(s <= 0 for s in self.size):
            raise ValueError('box size must be positive')
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError('confidence must lie in [0, 1]')

    @property
    def obb(self) -> OBB:
        return OBB.make(self.center, self.size, self.rotation)


def _samples(obb: OBB) -> np.ndarray:
    per = max(5, int(math.ceil(float(np.max(obb.size)) / CONTACT_SPACING)) + 1)
    return obb.surface_samples(per)


def _face_normals(obb: OBB, pts) -> np.ndarray:
    """Outward normal of the face of obb nearest to each point."""
    local = obb.to_local(pts) / obb.half
    axis = np.argmax(np.abs(local), axis=1)
    sign = np.sign(local[np.arange(len(local)), axis])
    return obb.rotation[:, axis].T * sign[:, None]


def _support_margin(com_xy, pts_xy) -> float:
    """Signed distance (m) from the centre of mass to the support polygon: negative inside,
    positive outside; +1 when there is no support at all."""
    if len(pts_xy) == 0:
        return 1.0
    pts = np.unique(np.round(pts_xy, 4), axis=0)
    centred = pts - pts.mean(0)
    if len(pts) < 3 or np.linalg.matrix_rank(centred, tol=1e-4) < 2:
        return float(np.min(np.linalg.norm(pts - com_xy, axis=1)))
    from scipy.spatial import ConvexHull
    eq = ConvexHull(pts).equations
    return float(np.max(eq[:, :2] @ com_xy + eq[:, 2]))


def _inside_support(com_xy, pts_xy) -> bool:
    if len(pts_xy) == 0:
        return False
    pts = np.unique(np.round(pts_xy, 4), axis=0)
    if len(pts) == 1:
        return bool(np.linalg.norm(pts[0] - com_xy) <= HULL_MARGIN)
    centred = pts - pts.mean(0)
    if np.linalg.matrix_rank(centred, tol=1e-4) < 2:
        d = centred[np.argmax(np.linalg.norm(centred, axis=1))]
        d = d / np.linalg.norm(d)
        t = (pts - pts[0]) @ d
        q = com_xy - pts[0]
        along = q @ d
        off = np.linalg.norm(q - along * d)
        return bool(off <= HULL_MARGIN and t.min() - HULL_MARGIN <= along <= t.max() + HULL_MARGIN)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    return bool(np.all(hull.equations[:, :2] @ com_xy + hull.equations[:, 2] <= HULL_MARGIN))


class SupportGraph:
    """Quasi-static support relations from perceived poses (no physics engine).

    A box is stable when its centre of mass projects inside the polygon spanned by its contact
    points below the centre of mass (with the floor, static objects such as the pallet, or other
    boxes still present)."""

    def __init__(self, boxes, floor=0.0, tol=0.01, static=(), com_margin=0.0):
        self.boxes = {b.box_id: b for b in boxes}
        self.com_margin = com_margin                     # contents may sit this far from the box centre (m)
        self.obbs = {k: b.obb for k, b in self.boxes.items()}
        self.floor, self.tol = floor, tol
        self.static = list(static)
        self._vertical = {k: {} for k in self.boxes}     # k -> {supporter: points resting on its upper surface}
        self._lateral = {k: {} for k in self.boxes}      # k -> {neighbour: (points, neighbour face normals)}
        self._fixed = {k: np.empty((0, 3)) for k in self.boxes}   # resting on floor / static objects
        aabb = {k: o.aabb() for k, o in self.obbs.items()}
        for k, o in self.obbs.items():
            s = _samples(o)
            below = s[s[:, 2] < o.center[2] - 1e-6]
            fixed = [below[below[:, 2] <= floor + tol]]
            for st in self.static:
                m = st.signed_distance(below) <= tol
                if np.any(m):
                    fixed.append(below[m][_face_normals(st, below[m])[:, 2] > VERTICAL_COS])
            self._fixed[k] = np.vstack(fixed)
            lo_k, hi_k = aabb[k]
            for j, oj in self.obbs.items():
                if j == k:
                    continue
                lo_j, hi_j = aabb[j]
                if np.any(lo_k > hi_j + tol) or np.any(lo_j > hi_k + tol):
                    continue
                m = oj.signed_distance(below) <= tol
                if np.any(m):
                    up = _face_normals(oj, below[m])[:, 2] > VERTICAL_COS
                    if np.any(up):
                        self._vertical[k][j] = below[m][up]
                # A leaning case touches its neighbour anywhere on its surface, often above its
                # centre of mass, so side contacts are searched over the whole surface.
                m = oj.signed_distance(s) <= tol
                if np.any(m):
                    n = _face_normals(oj, s[m])
                    side = np.abs(n[:, 2]) < LATERAL_SIN
                    if np.any(side):
                        self._lateral[k][j] = (s[m][side], n[side])

    def _leaners(self, bid, present):
        """Neighbours whose side faces keep bid from tipping over its vertical support."""
        com = self.obbs[bid].center
        v = np.vstack([self._fixed[bid]] + [c for s, c in self._vertical[bid].items() if s in present])
        if len(v) == 0:
            return set()
        d = com[:2] - v[:, :2].mean(0)
        if np.linalg.norm(d) < 1e-9:
            return set()
        d = d / np.linalg.norm(d)
        out = set()
        for j, (pts, normals) in self._lateral[bid].items():
            if j not in present:
                continue
            ahead = (pts[:, :2] - com[:2]) @ d > -0.02
            opposing = normals[:, :2] @ d < -0.3
            if np.any(ahead & opposing):
                out.add(j)
        return out

    def _supporter_map(self):
        if not hasattr(self, '_sup_cache'):
            present = set(self.boxes)
            self._sup_cache = {}
            for k in self.boxes:
                out = set(self._vertical[k])
                if not self._stable_vertical(k, present):
                    out |= self._leaners(k, present)
                self._sup_cache[k] = out
        return self._sup_cache

    def supporters(self, bid):
        return set(self._supporter_map()[bid])

    def vertical_supporters(self, bid):
        """Boxes this one stands on."""
        return set(self._vertical[bid])

    def leaning_on(self, bid):
        """Neighbours this box needs at its side to stay up (it leans on them)."""
        return self.supporters(bid) - set(self._vertical[bid])

    def leans_on(self, bid):
        """Neighbours whose sides this box touches (it may lean on them)."""
        return set(self._lateral[bid])

    def supported_by(self, bid):
        return {o for o, s in self._supporter_map().items() if o != bid and bid in s}

    def on_floor(self, bid):
        return bool(np.any(self.obbs[bid].corners()[:, 2] <= self.floor + self.tol))

    def on_static(self, bid):
        return len(self._fixed[bid]) > 0 and not self.on_floor(bid)

    def floating(self, bid):
        return not self._vertical[bid] and not self._lateral[bid] and len(self._fixed[bid]) == 0

    def free(self):
        return {bid for bid in self.boxes if not self.supported_by(bid)}

    def _stable_vertical(self, bid, present) -> bool:
        pts = np.vstack([self._fixed[bid]] + [c for s, c in self._vertical[bid].items() if s in present])
        return _inside_support(self.obbs[bid].center[:2], pts[:, :2])

    def support_margin(self, bid, present) -> float:
        """How far the centre of mass sits outside its vertical support (negative = inside)."""
        pts = np.vstack([self._fixed[bid]] + [c for s, c in self._vertical[bid].items() if s in present])
        return _support_margin(self.obbs[bid].center[:2], pts[:, :2])

    def vertical_share(self, bid, supporter) -> float:
        """Share of bid's vertical contact points that rest on `supporter`."""
        total = len(self._fixed[bid]) + sum(len(c) for c in self._vertical[bid].values())
        return len(self._vertical[bid].get(supporter, ())) / total if total else 0.0

    def stable(self, bid, present) -> bool:
        """Centre of mass over the vertical support, or tipping blocked by a neighbour's side."""
        return self._stable_vertical(bid, present) or bool(self._leaners(bid, present))

    def stability(self, bid, present) -> str:
        """'stable' when every place the contents could put the centre of mass is over the support
        (or a neighbour's side blocks tipping); 'uncertain' when only some are; 'unstable' when the
        box centre itself is not."""
        if self._leaners(bid, present):
            return 'stable'
        pts = np.vstack([self._fixed[bid]] + [c for s, c in self._vertical[bid].items() if s in present])
        centre = self.obbs[bid].center[:2]
        if not _inside_support(centre, pts[:, :2]):
            return 'unstable'
        m = self.com_margin
        if m > 0 and not all(_inside_support(centre + np.array([dx, dy]), pts[:, :2]) for dx in (-m, m) for dy in (-m, m)):
            return 'uncertain'
        return 'stable'

    def would_maybe_fall(self, bid) -> set:
        """Cases that stay up if their contents are centred but may fall if they are not, once bid
        is removed (and those predicted to fall have gone)."""
        if self.com_margin <= 0:
            return set()
        cache = self.__dict__.setdefault('_maybe_cache', {})
        if bid not in cache:
            fallen = self.would_fall_if_removed(bid)
            everyone = set(self.boxes)
            present = everyone - {bid} - fallen
            touched = {bid} | fallen
            cache[bid] = {o for o in present
                          if (set(self._vertical[o]) | set(self._lateral[o])) & touched
                          and self.stability(o, present) == 'uncertain' and self.stability(o, everyone) == 'stable'}
        return set(cache[bid])

    def unexplained(self) -> set:
        """Cases the model cannot hold up although the observed pile is at rest. Something we do
        not see supports them (a hidden case, or a pose error); they are not predicted to fall,
        but they are not trusted either."""
        if not hasattr(self, '_unexplained'):
            everyone = set(self.boxes)
            self._unexplained = {k for k in self.boxes if not self.stable(k, everyone)}
        return self._unexplained

    def would_fall_if_removed(self, bid):
        cache = self.__dict__.setdefault('_fall_cache', {})
        if bid not in cache:
            cache[bid] = self._would_fall(bid)
        return set(cache[bid])

    def _would_fall(self, bid):
        present = set(self.boxes) - {bid}
        doubtful = self.unexplained()
        fallen = set()
        changed = True
        while changed:
            changed = False
            for o in sorted(present):
                if o in doubtful:
                    continue          # its real support is unknown: handled by possibly_affected
                touching = set(self._vertical[o]) | set(self._lateral[o])
                if bid not in touching and not (touching & fallen):
                    continue          # nothing it rests or leans on has changed
                if not self.stable(o, present):
                    fallen.add(o)
                    present.discard(o)
                    changed = True
        return fallen

    def possibly_affected(self, bid) -> set:
        """Cases with unexplained support that touch bid or a predicted faller."""
        touched = {bid} | self.would_fall_if_removed(bid)
        return {o for o in self.unexplained() if o != bid and (set(self._vertical[o]) | set(self._lateral[o])) & touched}

    def fall_severity(self, bid) -> dict:
        """Each case that would fall if bid were removed, weighted by how far it can drop
        (a case that only tips a few centimetres is not a collapse)."""
        fallen = self.would_fall_if_removed(bid)
        maybe = self.would_maybe_fall(bid)
        remaining = [k for k in self.boxes if k != bid and k not in fallen]
        out = {}
        for f in fallen | maybe:
            o = self.obbs[f]
            lo, hi = o.aabb()
            rest = self.floor
            for st in self.static:
                slo, shi = st.aabb()
                if np.all(slo[:2] <= hi[:2]) and np.all(lo[:2] <= shi[:2]):
                    rest = max(rest, shi[2])
            for k in remaining:
                klo, khi = self.obbs[k].aabb()
                if np.all(klo[:2] <= hi[:2]) and np.all(lo[:2] <= khi[:2]) and khi[2] <= o.center[2]:
                    rest = max(rest, khi[2])
            drop = float(lo[2] - rest)
            out[f] = float(np.clip(drop / HARMFUL_DROP, 0.1, 1.0)) * (MAYBE_WEIGHT if f in maybe else 1.0)
        return out


def support_graph(boxes, floor=0.0, tol=0.01, static=(), com_margin=0.0):
    return SupportGraph(boxes, floor=floor, tol=tol, static=static, com_margin=com_margin)


@dataclass(frozen=True)
class PickCandidate:
    box_id: str
    expected_cascade: float
    pickable: bool
    reason: str
    height: float = 0.0


class PileBelief:
    """Fuses perceived boxes and known space; decides pick / observe / complete."""

    def __init__(self, floor=0.0, max_cascade_risk=0.2, hidden_prior=0.5, static=(),
                 max_face_tilt_deg=15.0, min_top_visible=0.5, tool_footprint=(0.30, 0.22), box_volume=None,
                 reach=None, cascade_model=None, case_dims=None, com_margin=0.0):
        self.floor = floor
        self.max_cascade_risk = max_cascade_risk
        self.hidden_prior = hidden_prior
        self.static = list(static)
        self.max_face_tilt_deg = max_face_tilt_deg
        self.min_top_visible = min_top_visible
        self.tool_footprint = tool_footprint
        self.box_volume = box_volume
        self.reach = reach
        self.cascade_model = cascade_model     # learned from physics; None = geometric rule only
        self.com_margin = com_margin           # contents may sit this far from a box's centre
        self.case_dims = case_dims             # known case size: unknown slivers smaller than a case are ignored
        self.boxes = {}
        self.seen_fraction = 0.0
        self.viewpoints = []
        self.removed = set()
        self._space = None
        self._invalidate()

    # ------------------------------------------------------------------ inputs
    def observe(self, boxes, viewpoint, seen_fraction=1.0):
        if not 0.0 <= seen_fraction <= 1.0:
            raise ValueError('seen_fraction must lie in [0, 1]')
        for b in boxes:
            prev = self.boxes.get(b.box_id)
            # Keep the more confident estimate; a second viewpoint never lowers confidence.
            if prev is None or b.confidence >= prev.confidence:
                self.boxes[b.box_id] = b
        self.viewpoints.append(viewpoint)
        self.seen_fraction = max(self.seen_fraction, seen_fraction)
        self._invalidate()

    def replace(self, boxes, viewpoint):
        """A fresh perception pass after the pile changed: forget superseded estimates."""
        self.boxes = {b.box_id: b for b in boxes if b.box_id not in self.removed}
        self.viewpoints.append(viewpoint)
        self._invalidate()

    def set_known_space(self, voxel_map, region):
        """Unknown voxels inside `region` that no estimated box explains may hide cases."""
        lo, hi = np.asarray(region[0], float), np.asarray(region[1], float)
        centers = voxel_map.centers()
        state = voxel_map.state()
        inside = np.all((centers >= lo) & (centers <= hi), axis=-1) & (centers[..., 2] > self.floor + voxel_map.res)
        # Cases do not float: unknown space with seen-empty space beneath it in the same column
        # cannot hold a case resting on anything.
        free_below = np.zeros_like(state, dtype=bool)
        free_below[:, :, 1:] = np.cumsum(state[:, :, :-1] == 1, axis=2) > 0
        cand = inside & (state == 0) & ~free_below
        idx = np.argwhere(cand)
        pts = centers[cand]
        if len(pts):
            keep = np.ones(len(pts), bool)
            for k, b in self.boxes.items():
                if k not in self.removed:
                    keep &= ~b.obb.contains(pts, margin=voxel_map.res)
            for st in self.static:
                keep &= ~st.contains(pts, margin=voxel_map.res)
            pts, idx = pts[keep], idx[keep]
            if self.case_dims is not None and len(pts):
                pts = pts[self._room_for_a_case(idx, voxel_map.shape, voxel_map.res)]
        self._space = {'points': pts, 'voxel_volume': voxel_map.res ** 3}
        self._invalidate()

    def _room_for_a_case(self, idx, shape, res):
        """A case hides whole only in a connected unknown region it fits in. Slivers (for example at
        the edge of a camera's view) are dropped. A case partly in view is found by segmentation,
        so the region needs ROOM_SHARE of each case dimension, not all of it."""
        from scipy import ndimage
        mask = np.zeros(shape, bool)
        mask[tuple(idx.T)] = True
        labels, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
        need = np.sort(np.asarray(self.case_dims)) * ROOM_SHARE
        ok = np.zeros(n + 1, bool)
        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            extent = np.sort([(s.stop - s.start) * res for s in sl])
            ok[i] = bool(np.all(extent >= need))
        return ok[labels[tuple(idx.T)]]

    # ------------------------------------------------------------------ model
    def _invalidate(self):
        self._graph = None
        self._cands = None
        self._samples = {}

    def graph(self):
        if self._graph is None:
            self._graph = SupportGraph([b for i, b in self.boxes.items() if i not in self.removed],
                                       floor=self.floor, static=self.static, com_margin=self.com_margin)
        return self._graph

    def _box_volume(self):
        if self.box_volume:
            return self.box_volume
        vols = [float(np.prod(b.size)) for b in self.boxes.values()]
        return min(vols) if vols else 0.02

    def unexplained_litres(self):
        return None if self._space is None else len(self._space['points']) * self._space['voxel_volume'] * 1000

    def unexplained_points(self):
        return np.empty((0, 3)) if self._space is None else self._space['points']

    def hidden_box_probability(self, near: OBB | None = None, reach: float = 0.0, above: float | None = None) -> float:
        """Chance that unexplained unknown space holds a case. With `near`, only space within
        `reach` of that box; with `above`, only space at or above that height (a hidden case
        below a box cannot be resting on it)."""
        if self._space is None:
            return self.hidden_prior * (1.0 - self.seen_fraction)
        pts = self._space['points']
        if near is not None and len(pts):
            pts = pts[near.signed_distance(pts) <= reach]
        if above is not None and len(pts):
            pts = pts[pts[:, 2] >= above]
        vol = len(pts) * self._space['voxel_volume']
        return float(1.0 - math.exp(-vol / (0.5 * self._box_volume())))

    def rule_risk(self, bid):
        """cascade_risk by the geometric rule alone, whatever model is given: what the learned model is compared with."""
        return self.cascade_risk(bid, learned=False)

    def cascade_risk(self, bid, learned=True):
        """Expected number of cases that fall if bid is removed: geometry, plus pose uncertainty of
        everything resting on it, plus the chance that unseen cases rest on it."""
        g = self.graph()
        if learned and self.cascade_model is not None:
            from .learning.cascade import expected_falls
            risk = expected_falls(self.cascade_model, g, bid)
        else:
            risk = float(sum(g.fall_severity(bid).values()))
            risk += UNEXPLAINED_WEIGHT * len(g.possibly_affected(bid))
        for o in g.supported_by(bid):
            risk += 1.0 - self.boxes[o].confidence
        risk += 1.0 - self.boxes[bid].confidence
        if self._space is None:
            risk += self.hidden_box_probability()
        else:
            o = self.boxes[bid].obb
            risk += self.hidden_box_probability(near=o, reach=0.5 * float(np.max(o.size)), above=float(o.aabb()[0][2]))
        return risk

    def pickability(self, bid):
        b = self.boxes[bid]
        o = b.obb
        if self.reach is not None:
            (x0, x1), (y0, y1) = self.reach
            if not (x0 <= o.center[0] <= x1 and y0 <= o.center[1] <= y1):
                return False, 'out of reach'
        tilt = o.tilt_deg()
        if tilt > self.max_face_tilt_deg:
            return False, f'top face tilted {tilt:.0f} deg'
        if b.top_visible < self.min_top_visible:
            return False, f'top face only {b.top_visible:.0%} seen'
        # Two cases cannot share space. Where estimates overlap, the less confident one is wrong
        # somewhere, and cups aimed at it would land partly on its neighbour.
        lo, hi = o.aabb()
        for k, other in self.boxes.items():
            if k == bid or k in self.removed or other.confidence < b.confidence:
                continue
            olo, ohi = other.obb.aabb()
            if np.any(olo > hi) or np.any(lo > ohi):
                continue
            if k not in self._samples:
                self._samples[k] = _samples(other.obb)
            if np.any(o.contains(self._samples[k], margin=-OVERLAP_TOL)):
                return False, f'estimate overlaps {k}'
        c, n, half, (u, v) = o.top_face()
        tool_long, tool_short = max(self.tool_footprint) / 2, min(self.tool_footprint) / 2
        a_half, b_half = (max(half[0], tool_long), max(half[1], tool_short)) if half[0] >= half[1] else (max(half[0], tool_short), max(half[1], tool_long))
        w = np.cross(u, v)
        R = np.column_stack([u, v, w if w[2] >= 0 else -w])
        column = OBB.make(c + R[:, 2] * 1.505, (2 * a_half, 2 * b_half, 3.0), R)
        lo, hi = column.aabb()
        for k, other in self.boxes.items():
            if k == bid or k in self.removed:
                continue
            olo, ohi = other.obb.aabb()
            if np.any(olo > hi) or np.any(lo > ohi):
                continue
            if k not in self._samples:
                self._samples[k] = _samples(other.obb)
            if np.any(column.contains(self._samples[k], margin=-APPROACH_MARGIN)):
                return False, f'approach blocked by {k}'
        return True, 'ok'

    def candidates(self):
        if self._cands is not None:
            return list(self._cands)
        g = self.graph()
        out = []
        for bid in g.boxes:
            ok, why = self.pickability(bid)
            risk = self.cascade_risk(bid)
            if ok:
                why = 'free' if bid in g.free() else f'supports {sorted(g.supported_by(bid))}'
            out.append(PickCandidate(bid, risk, ok, why, float(self.boxes[bid].obb.top_face()[0][2])))
        self._cands = sorted(out, key=lambda c: (not c.pickable, round(c.expected_cascade, 6), -c.height, c.box_id))
        return list(self._cands)

    def next_pick(self):
        """Among safe candidates take the highest: unloading from the top down removes what
        rests on others before it can be disturbed."""
        safe = [c for c in self.candidates() if c.pickable and c.expected_cascade <= self.max_cascade_risk]
        return max(safe, key=lambda c: (round(c.height, 3), -c.expected_cascade)) if safe else None

    def observation_gain(self):
        """Share of the best candidate's risk that comes from uncertainty rather than geometry:
        zero when a safe pick exists, positive when looking again could change the choice."""
        cands = self.candidates()
        if not cands:
            return 0.0
        best = cands[0]
        if best.pickable and best.expected_cascade <= self.max_cascade_risk:
            return 0.0
        geometric = float(sum(self.graph().fall_severity(best.box_id).values()))
        return max(0.0, min(1.0, best.expected_cascade - geometric))

    def verdict(self):
        if not self.graph().boxes:
            return Verdict.COMPLETE
        return Verdict.PICK if self.next_pick() else Verdict.NEED_OBSERVATION

    # ------------------------------------------------------------------ closing the loop
    def commit_pick(self, bid):
        if bid not in self.boxes or bid in self.removed:
            raise ValueError('unknown or already removed box')
        self.removed.add(bid)
        self._invalidate()

    def verify_after_pick(self, boxes, tol=0.03):
        """Return ids whose pose moved more than tol since the last belief; the caller replans."""
        moved = []
        for b in boxes:
            prev = self.boxes.get(b.box_id)
            if prev is None or prev.box_id in self.removed:
                moved.append(b.box_id)
                continue
            if math.dist(prev.center, b.center) > tol:
                moved.append(b.box_id)
                self.boxes[b.box_id] = b
        if moved:
            self._invalidate()
        return moved
