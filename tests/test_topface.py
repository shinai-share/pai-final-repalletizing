"""Measuring a placed case from its top face alone: neighbours packed flush against it must not
pull the estimate, and a case that leans must be reported as leaning."""
import math

import numpy as np

from shinai_pi.perception.geometry import look_at_view
from shinai_pi.perception.obb import OBB, quat_to_matrix, yaw_matrix
from shinai_pi.perception.synthetic import render
from shinai_pi.perception.topface import measure_top_face

SIZE = (0.36, 0.27, 0.22)


def overhead(boxes, res=(640, 480)):
    view = look_at_view((0.0, 0.0, 3.4), (1e-4, 0.0, 0.0), up=(-1, 0, 0))
    f = 24.0 / 20.955 * res[0]
    return render(view, boxes, width=res[0], height=res[1], fx=f, fy=f)


def test_a_case_packed_between_neighbours_is_measured_within_three_millimetres():
    centre = np.array([0.012, -0.007, 0.11])
    boxes = {'target': OBB.make(centre, SIZE, yaw_matrix(math.radians(1.0)))}
    for dx in (-0.365, 0.365):                                      # flush neighbours, 5 mm gaps
        boxes[f'n{dx}'] = OBB.make((centre[0] + dx, centre[1], 0.11), SIZE)
    m = measure_top_face(overhead(boxes), 'target')
    assert np.linalg.norm(m.centre - centre[:2]) < 0.003
    assert abs(m.top_z - 0.22) < 0.003
    assert abs(((m.long_edge_yaw_deg - 1.0 + 90) % 180) - 90) < 0.7
    assert m.tilt_deg < 1.0


def test_a_leaning_case_reports_its_tilt():
    q = (math.cos(math.radians(3) / 2), math.sin(math.radians(3) / 2), 0, 0)  # 3 degrees about x
    boxes = {'target': OBB.make((0.0, 0.0, 0.12), SIZE, quat_to_matrix(q))}
    m = measure_top_face(overhead(boxes), 'target')
    assert 2.0 < m.tilt_deg < 4.0
    assert m.normal[2] > 0.99 and m.normal[1] < -0.03          # tipped about +x: the face looks towards -y


def test_a_case_not_in_view_gives_nothing():
    boxes = {'other': OBB.make((0.0, 0.0, 0.11), SIZE)}
    assert measure_top_face(overhead(boxes), 'target') is None


def test_a_case_held_under_the_plate_is_measured_from_the_rim_that_shows_around_it():
    # Hanging on the cups, only a 2-3 cm rim of the top face shows around the plate (and less where
    # the column hides it). The rim alone must give the edge direction and centre.
    centre = np.array([0.02, -0.01, 0.40])
    yaw = math.radians(7.0)
    boxes = {'target': OBB.make(centre, SIZE, yaw_matrix(yaw)),
             'plate': OBB.make(centre + np.array([0.0, 0.0, 0.11 + 0.03]), (0.30, 0.22, 0.06), yaw_matrix(yaw)),
             'column': OBB.make(centre + np.array([0.0, 0.0, 0.11 + 0.5]), (0.10, 0.10, 0.8))}
    m = measure_top_face(overhead(boxes), 'target')
    assert np.linalg.norm(m.centre - centre[:2]) < 0.003
    assert abs(((m.long_edge_yaw_deg - 7.0 + 90) % 180) - 90) < 0.5
    assert abs(m.extents[0] - 0.36) < 0.01 and abs(m.extents[1] - 0.27) < 0.01


def test_at_the_resolution_of_b_camera_a_case_anywhere_on_the_pallet_reads_within_a_degree():
    # The edge direction comes from pixels: at 640 px (4.3 mm a pixel) a long edge reads up to two
    # degrees off, as much as the placement tolerance. B's camera has 1280 px for its checks.
    worst_yaw, worst_centre = 0.0, 0.0
    for cx in (-0.4, 0.0, 0.27):
        for cy in (-0.5, 0.3):
            for yd in (0.4, 1.0, 91.3):
                for z in (0.11, 0.33):
                    c = np.array([cx, cy, z])
                    frame = overhead({'t': OBB.make(c, SIZE, yaw_matrix(math.radians(yd)))}, res=(1280, 960))
                    m = measure_top_face(frame, 't')
                    worst_yaw = max(worst_yaw, abs(((m.long_edge_yaw_deg - yd + 90) % 180) - 90))
                    worst_centre = max(worst_centre, float(np.linalg.norm(m.centre - c[:2])))
    assert worst_yaw < 1.0
    assert worst_centre < 0.002
