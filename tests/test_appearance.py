"""How the cell looks to people. The robot never judges by colour (it uses depth and object labels),
so these values are for the person watching: cases, pallets and floor must not be confused."""
import dataclasses

import numpy as np
import pytest

from shinai_pi.appearance import delta_e, lab_from_linear, lab_from_srgb8, saturation
from shinai_pi.config import load_cell, validate

CELL = load_cell()
LOOK = CELL.appearance


def test_white_and_black_are_the_ends_of_lightness():
    assert lab_from_linear((1.0, 1.0, 1.0))[0] == pytest.approx(100.0, abs=0.1)
    assert lab_from_linear((0.0, 0.0, 0.0))[0] == pytest.approx(0.0, abs=0.1)


def test_an_8_bit_picture_and_its_linear_colour_agree():
    assert np.allclose(lab_from_srgb8((188, 188, 188)), lab_from_linear((0.5, 0.5, 0.5)), atol=0.5)


def test_the_pallet_is_grey():
    assert saturation(LOOK.pallet_color) < 0.08


def test_every_case_colour_is_far_from_the_pallet_and_the_floor():
    # 25 in CIELAB is a difference anyone sees at a glance (2 is about the smallest visible).
    for c in LOOK.case_colors:
        assert delta_e(c, LOOK.pallet_color) >= 25
        assert delta_e(c, LOOK.floor_color) >= 25
    assert delta_e(LOOK.pallet_color, LOOK.floor_color) >= 15


def test_a_colour_outside_zero_to_one_is_rejected():
    bad = dataclasses.replace(CELL, appearance=dataclasses.replace(LOOK, pallet_color=(1.2, 0.5, 0.5)))
    with pytest.raises(ValueError, match='colour'):
        validate(bad)


def test_a_pallet_colour_close_to_a_case_colour_is_rejected():
    bad = dataclasses.replace(CELL, appearance=dataclasses.replace(LOOK, pallet_color=LOOK.case_colors[0]))
    with pytest.raises(ValueError, match='pallet'):
        validate(bad)


def test_pictures_for_people_are_full_hd_and_the_insets_keep_the_same_shape():
    r = CELL.rendering
    assert tuple(r.people_resolution) == (1920, 1080)
    w, h = r.inset_resolution
    assert w * 9 == h * 16 and w >= 960
    assert r.antialiasing in (0, 1, 2, 3, 4) and r.people_subframes >= 1
