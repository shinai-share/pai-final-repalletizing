import numpy as np
import pytest

from shinai_pi.config import load_cell
from shinai_pi.safety import Action, SafetySupervisor, TickState

CELL = load_cell()


def sup():
    return SafetySupervisor(CELL)


def tick(**kw):
    base = dict(t=1.0, q=np.zeros(4), v=np.zeros(4), carrying=None, gripped=[], curtain_clear=True)
    base.update(kw)
    return TickState(**base)


def test_target_outside_joint_limits_is_rejected():
    ok, why = sup().validate_target(np.array([5.0, 0, 0, 0]))
    assert not ok and 'x' in why


def test_target_inside_limits_is_accepted():
    ok, _ = sup().validate_target(np.array([0.5, 0.2, -0.5, 1.0]))
    assert ok


def test_non_finite_target_is_rejected():
    ok, _ = sup().validate_target(np.array([np.nan, 0, 0, 0]))
    assert not ok


def test_light_curtain_breach_stops_and_latches_until_reset():
    s = sup()
    assert s.check(tick(curtain_clear=False)).action is Action.STOP
    assert s.check(tick(curtain_clear=True)).action is Action.STOP          # latched
    assert not s.reset(operator='unknown', curtain_clear=False)              # cannot reset while blocked
    assert s.reset(operator='作業者', curtain_clear=True)
    assert s.check(tick(curtain_clear=True)).action is Action.CONTINUE


def test_losing_the_carried_case_stops_motion():
    s = sup()
    assert s.check(tick(carrying='case_03', gripped=['case_03'])).action is Action.CONTINUE
    r = s.check(tick(carrying='case_03', gripped=[]))
    assert r.action is Action.STOP and 'case_03' in r.reason


def run_ticks(s, qs, dt=1 / 30, v=None):
    out = []
    for k, q in enumerate(qs):
        vk = (np.asarray(q) - np.asarray(qs[k - 1])) / dt if v is None and k else (np.zeros(6) if v is None else v)
        out.append(s.check(TickState(t=k * dt, q=np.asarray(q, float), v=np.asarray(vk, float), carrying=None, gripped=[],
                                     curtain_clear=True)).action)
    return out


def test_an_axis_moving_on_over_its_speed_limit_stops_within_a_tenth_of_a_second():
    s = sup()
    dt, v = 1 / 30, np.array(CELL.gantry.v_max) * 1.3
    acts = run_ticks(s, [v * k * dt for k in range(12)], dt)
    first = acts.index(Action.STOP)
    assert first * dt <= 0.1 + 2 * dt + 1e-9


def test_the_wrist_springing_back_when_the_cups_let_go_is_not_an_over_speed():
    # S3 seed 103: tipping on the table, the case's weight bent the roll 0.08 rad behind its target; when the
    # cups let go it sprang back within two physics steps and the joint read 2.56 rad/s (limit 1.5 x 1.25).
    s = sup()
    dt = 1 / 30
    qs = [np.r_[np.zeros(5), -1.652]] * 4 + [np.r_[np.zeros(5), -1.571]] * 8
    spike = np.r_[np.zeros(5), 2.56]
    acts = [s.check(TickState(t=k * dt, q=q, v=spike if 4 <= k < 7 else np.zeros(6), carrying=None, gripped=[],
                              curtain_clear=True)).action for k, q in enumerate(qs)]
    assert Action.STOP not in acts


def test_a_single_spike_of_the_speed_reading_does_not_stop():
    s = sup()
    v = np.array(CELL.gantry.v_max) * 1.3
    assert s.check(tick(v=v)).action is Action.CONTINUE           # contact impulse
    assert s.check(tick(v=np.zeros(4))).action is Action.CONTINUE


def test_plan_older_than_last_change_is_rejected():
    s = sup()
    s.note_world_change(t=10.0)
    ok, why = s.validate_plan(plan_stamp=9.0)
    assert not ok and 'stale' in why
    ok, _ = s.validate_plan(plan_stamp=10.5)
    assert ok


def test_payload_over_rating_is_rejected():
    ok, why = sup().validate_payload(CELL.gripper.rated_payload + 1)
    assert not ok


@pytest.mark.parametrize('clear,expected', [(0.30, True), (0.05, False)])
def test_traverse_clearance(clear, expected):
    s = sup()
    ok, _ = s.validate_clearance(load_bottom_z=1.0 + clear, obstacle_top_z=1.0)
    assert ok is expected


def test_a_case_that_leaves_the_cups_while_resting_on_the_tines_is_not_a_latched_stop():
    from shinai_pi.config import load_cell
    from shinai_pi.safety import Action, SafetySupervisor, TickState
    import numpy as np
    sup = SafetySupervisor(load_cell())
    d = sup.check(TickState(1.0, np.zeros(6), np.zeros(6), 'case_00', [], True, supported=True))
    assert d.action is Action.STOP and sup.latched is None and sup.vacuum_losses
    d = sup.check(TickState(2.0, np.zeros(6), np.zeros(6), 'case_00', [], True, supported=False))
    assert d.action is Action.STOP and sup.latched                   # on the cups alone it is a dropped case
