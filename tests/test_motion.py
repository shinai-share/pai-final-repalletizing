import numpy as np
import pytest

from shinai_pi.motion import Profile, plan_move

V = np.array([1.2, 1.2, 0.7, 1.6])
A = np.array([1.5, 1.5, 1.5, 3.0])


def sample(p: Profile, n=400):
    ts = np.linspace(0, p.duration, n)
    return ts, np.array([p.position(t) for t in ts])


def test_reaches_goal_exactly_and_holds():
    p = plan_move([0, 0, 0, 0], [1.0, -0.5, 0.3, 1.0], V, A)
    assert np.allclose(p.position(p.duration), [1.0, -0.5, 0.3, 1.0])
    assert np.allclose(p.position(p.duration + 5), [1.0, -0.5, 0.3, 1.0])
    assert np.allclose(p.position(-1), [0, 0, 0, 0])


def test_respects_velocity_and_acceleration_limits():
    p = plan_move([0, 0, 0, 0], [3.0, 0.2, -1.2, 2.5], V, A)
    ts, xs = sample(p, 2000)
    dt = ts[1] - ts[0]
    vel = np.diff(xs, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    assert np.all(np.abs(vel) <= V * 1.01)
    assert np.all(np.abs(acc[2:-2]) <= A * 1.05)


def test_slow_accel_axis_with_short_distance_is_still_within_limits():
    a = np.array([5.0, 0.2, 5.0, 5.0])
    p = plan_move([0, 0, 0, 0], [3.0, 0.3, 0, 0], V, a)
    ts, xs = sample(p, 4000)
    dt = ts[1] - ts[0]
    acc = np.diff(np.diff(xs, axis=0) / dt, axis=0) / dt
    assert np.all(np.abs(acc[2:-2]) <= a * 1.05)


def test_all_axes_finish_together_and_move_monotonically():
    p = plan_move([0, 0, 0, 0], [2.0, 0.3, 0.0, -0.4], V, A)
    _, xs = sample(p)
    for k, goal in enumerate([2.0, 0.3, 0.0, -0.4]):
        d = np.diff(xs[:, k]) * np.sign(goal) if goal else np.diff(xs[:, k])
        assert np.all(d >= -1e-9)
    assert np.allclose(xs[-1], [2.0, 0.3, 0.0, -0.4])


def test_short_move_uses_triangular_profile():
    p = plan_move([0, 0, 0, 0], [0.1, 0, 0, 0], V, A)
    assert p.duration == pytest.approx(2 * np.sqrt(0.1 / 1.5), rel=1e-3)


def test_zero_move_has_zero_duration():
    p = plan_move([0.5, 0, 0, 0], [0.5, 0, 0, 0], V, A)
    assert p.duration == 0.0
    assert np.allclose(p.position(1.0), [0.5, 0, 0, 0])


@pytest.mark.parametrize('v,a', [([0, 1, 1, 1], A), (V, [1, 1, -1, 1]), ([np.nan, 1, 1, 1], A)])
def test_invalid_limits_rejected(v, a):
    with pytest.raises(ValueError):
        plan_move([0, 0, 0, 0], [1, 1, 1, 1], v, a)


def test_non_finite_target_rejected():
    with pytest.raises(ValueError):
        plan_move([0, 0, 0, 0], [np.inf, 0, 0, 0], V, A)


def test_six_axis_moves_keep_the_wrist_within_its_own_limits():
    from shinai_pi.config import load_cell
    from shinai_pi.motion import limits_for
    g = load_cell().gantry
    v, a = limits_for(g, slow=False)
    assert len(v) == len(a) == 6
    p = plan_move([0, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 1.57, 0], v, a)
    for k in range(1, 200):
        t = p.duration * k / 200
        dt = 1e-4
        speed = np.abs(p.position(t + dt) - p.position(t)) / dt
        assert np.all(speed <= v + 1e-3)


def test_slow_moves_turn_the_wrist_slowly_too():
    from shinai_pi.config import load_cell
    from shinai_pi.motion import limits_for
    g = load_cell().gantry
    v, _ = limits_for(g, slow=True)
    assert v[4] <= 0.25 * g.v_max[4] + 1e-9 and v[5] <= 0.25 * g.v_max[5] + 1e-9


def test_a_move_must_name_the_same_axes_as_its_limits():
    with pytest.raises(ValueError):
        plan_move([0, 0, 0, 0], [1, 0, 0, 0], [1] * 6, [1] * 6)


def test_a_touch_down_search_moves_slowly_enough_not_to_crush_a_case():
    from shinai_pi.config import load_cell
    from shinai_pi.motion import limits_for
    g = load_cell().gantry
    v, a = limits_for(g, slow=True, touch=True)
    assert np.all(v[:3] <= g.touch_speed + 1e-12) and g.touch_speed <= 0.06



def test_the_last_centimetres_of_a_touch_are_searched_slower():
    # The z drive is stiff: at the search speed a face's reaction grows about 50 N a physics step and
    # overshoots the touch threshold (stage gates 2026-10-18: 170-300 N). The last stretch goes slower.
    from shinai_pi.config import load_cell
    from shinai_pi.motion import limits_for
    g = load_cell().gantry
    v_search, _ = limits_for(g, slow=True, touch=True)
    v_fine, _ = limits_for(g, slow=True, touch=True, fine=True)
    assert v_fine[2] == g.fine_touch_speed < v_search[2] == g.touch_speed
