"""The state a case is in after the collapse, named in the 11 categories of requirements v05 R3
(chapter 6). Only what the robot's own estimates show is used. Upside down (category 6) is not
named: the cameras give depth and object labels, not the print on the top, so a case is never
called upside down for want of seeing its label."""
from __future__ import annotations

import numpy as np

CATEGORIES = {
    0: '分類できない',
    1: '正立・上が開いている',
    2: '正立・傾き 5〜15 度',
    3: '傾き 15〜50 度',
    4: '横倒し',
    5: '立ち（端面が下）',
    7: 'もたれ（隣のケースに寄りかかっている）',
    8: '挟まれ・推定が隣と重なる',
    9: '上に別のケースが載っている',
    10: 'パレットの外・床',
    11: '1 個分でない大きさ（二つが密着しているおそれ）',
}
SIZE_TOL = 0.03                # m: an estimate's edges within this of the case's are one case
LEAN_TILT_DEG = 15.0           # tipped more than the cups' give and held up by a neighbour's side


def classify(bid, belief, sku, pallet):
    """(category, name) of case `bid` in the current belief."""
    b = belief.boxes[bid]
    o = b.obb
    if np.any(np.abs(np.sort(o.size) - np.sort(sku.size)) > SIZE_TOL):
        return _named(11)
    g = belief.graph()
    if any(bid in g.vertical_supporters(k) for k in belief.boxes if k != bid and k not in belief.removed):
        return _named(9)
    hx, hy = pallet.size[0] / 2, pallet.size[1] / 2
    if abs(o.center[0] - pallet.a_center[0]) > hx or abs(o.center[1] - pallet.a_center[1]) > hy:
        return _named(10)
    tilt = o.tilt_deg()
    if tilt > LEAN_TILT_DEG and g.leaning_on(bid):
        return _named(7)
    ok, why = belief.pickability(bid)
    if not ok and ('overlaps' in why or 'only' in why):
        return _named(8)
    up = float(o.size[o.up_axis])
    w, d, h = sku.size
    if abs(up - h) <= SIZE_TOL:
        return _named(1 if tilt <= 5.0 else 2 if tilt <= 15.0 else 3)
    if tilt > 15.0:
        return _named(3)
    if abs(up - d) <= SIZE_TOL:
        return _named(4)
    if abs(up - w) <= SIZE_TOL:
        return _named(5)
    return _named(0)


def _named(code):
    return code, CATEGORIES[code]
