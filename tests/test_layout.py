import numpy as np
import pytest

from shinai_pi.config import load_cell
from shinai_pi.layout import pallet_slots

CELL = load_cell()
SKU = CELL.skus[0]


def test_full_pallet_has_48_slots_in_layer_order():
    slots = pallet_slots(CELL, SKU, CELL.pallet.b_center)
    assert len(slots) == 48
    assert [s.layer for s in slots] == sorted(s.layer for s in slots)
    assert [s.index for s in slots] == list(range(48))


def test_no_overhang_and_no_overlap_within_layer():
    p = CELL.pallet
    for layer in range(CELL.layout.layers):
        slots = [s for s in pallet_slots(CELL, SKU, p.b_center) if s.layer == layer]
        rects = []
        for s in slots:
            fx, fy = s.footprint
            x0, x1 = s.center[0] - fx / 2, s.center[0] + fx / 2
            y0, y1 = s.center[1] - fy / 2, s.center[1] + fy / 2
            assert x0 >= p.b_center[0] - p.size[0] / 2 - 1e-9 and x1 <= p.b_center[0] + p.size[0] / 2 + 1e-9
            assert y0 >= p.b_center[1] - p.size[1] / 2 - 1e-9 and y1 <= p.b_center[1] + p.size[1] / 2 + 1e-9
            rects.append((x0, x1, y0, y1))
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                ox = min(a[1], b[1]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[2], b[2])
                assert ox <= 1e-9 or oy <= 1e-9


def test_layers_rest_on_each_other_and_interlock():
    slots = pallet_slots(CELL, SKU, CELL.pallet.b_center)
    h = SKU.size[2]
    for s in slots:
        assert s.center[2] == pytest.approx(CELL.pallet.size[2] + h * (s.layer + 0.5))
    yaws = {s.layer: s.yaw_deg for s in slots}
    assert yaws[0] != yaws[1] and yaws[0] == yaws[2]


def test_too_large_sku_raises():
    import dataclasses
    big = dataclasses.replace(SKU, size=(0.6, 0.6, 0.2))
    with pytest.raises(ValueError):
        pallet_slots(CELL, big, CELL.pallet.b_center)


def test_every_slot_has_a_long_side_still_open_when_its_case_goes_down():
    # The tines come out on one long side of a placed case (requirements v05 5.3): the fill order
    # must leave one of them free of earlier cases on that layer, or the tines cannot be pulled.
    import dataclasses
    from shinai_pi.layout import open_long_side
    cell = load_cell()
    cell = dataclasses.replace(cell, layout=dataclasses.replace(cell.layout, layers=2))
    slots = pallet_slots(cell, cell.skus[0], cell.pallet.b_center)
    for s in slots:
        n = open_long_side(slots, s.index)
        assert n is not None, s
        long_axis = np.array([1.0, 0.0]) if s.yaw_deg == 0.0 else np.array([0.0, 1.0])
        assert abs(float(n @ long_axis)) < 1e-9                     # a long side, not an end


def test_a_slot_boxed_in_on_both_long_sides_has_no_open_side():
    from shinai_pi.layout import Slot, open_long_side
    slots = [Slot(0, 0, (0.0, -0.3, 0.2), 0.0, (0.36, 0.27)), Slot(1, 0, (0.0, 0.3, 0.2), 0.0, (0.36, 0.27)),
             Slot(2, 0, (0.0, 0.0, 0.2), 0.0, (0.36, 0.27))]
    assert open_long_side(slots, 2) is None


def test_a_small_grid_load_has_its_cases_in_unturned_layers():
    from shinai_pi.config import load_cell
    from shinai_pi.layout import pallet_slots, with_grid
    cell = with_grid(load_cell(), 3, 1, 2)
    sku = cell.skus[0]
    slots = pallet_slots(cell, sku, cell.pallet.a_center)
    assert len(slots) == 6 and not cell.layout.interlock
    assert sorted({round(s.center[2], 3) for s in slots}).__len__() == 2
    lower = sorted((s.center[0], s.center[1]) for s in slots if s.layer == 0)
    upper = sorted((s.center[0], s.center[1]) for s in slots if s.layer == 1)
    assert lower == upper                                   # each top case stands on one below


def test_a_grid_needs_at_least_one_case_a_side():
    import pytest
    from shinai_pi.config import load_cell
    from shinai_pi.layout import with_grid
    with pytest.raises(ValueError):
        with_grid(load_cell(), 0, 1, 2)
