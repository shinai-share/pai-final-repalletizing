"""The cell's dressing (stack light, cabinet, motors, cable ducts) is for the picture only:
it stays outside the gantry's legs, there is no fence or light-curtain post, and moving parts carry their own motors."""
from shinai_pi.config import load_cell
from shinai_pi.sim.visual_detail import dressing

CELL = load_cell()
ITEMS = dressing(CELL)
G = CELL.gantry
LEG_X = (G.x_range[1] - G.x_range[0] + 0.6) / 2
LEG_Y = (G.y_range[1] - G.y_range[0] + 0.6) / 2


def test_there_is_no_fence():
    assert not [i for i in ITEMS if i['group'] == 'fence']


def test_the_cabinet_stands_outside_the_gantry():
    for i in ITEMS:
        if i['group'] == 'cabinet':
            x, y, _ = i['pos']
            assert abs(x) > LEG_X or y > LEG_Y


def test_there_are_no_light_curtain_posts():
    assert not [i for i in ITEMS if i['group'] == 'light_curtain']


def test_motors_ride_on_the_axes_they_drive():
    parents = {i['parent'] for i in ITEMS if i['group'] == 'motor'}
    assert parents == {'bridge', 'trolley', 'ram'}


def test_every_item_has_a_size_and_a_colour():
    assert all(len(i['size']) == 3 and min(i['size']) > 0 and len(i['color']) == 3 for i in ITEMS)
