import numpy as np
import pytest

from shinai_pi.learning.cascade import FEATURES, CascadeModel, auc, calibration_bins, expected_falls, fit, removal_features
from shinai_pi.pile import Box, support_graph


def box(bid, cx, cy, cz, sx=0.4, sy=0.3, sz=0.25):
    return Box(bid, (cx, cy, cz), (sx, sy, sz))


def test_features_mark_the_case_resting_on_the_removed_one():
    g = support_graph([box('a', 0, 0, 0.125), box('b', 0, 0, 0.375), box('far', 2.0, 0, 0.125)], floor=0.0)
    keys, X = removal_features(g, 'a')
    assert keys == ['b']                                   # 'far' is not a neighbour
    f = dict(zip(FEATURES, X[0]))
    assert f['geo_falls'] == 1.0 and f['vertical_share_on_removed'] == pytest.approx(1.0)
    assert f['margin_after'] > 0 and f['margin_before'] < 0


def test_model_round_trips_as_json(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURES)))
    y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
    m = fit(X, y, {'source': 'test'})
    path = tmp_path / 'm.json'
    path.write_text(m.to_json(), encoding='utf-8')
    m2 = CascadeModel.load(path)
    assert np.allclose(m.predict(X), m2.predict(X))
    assert auc(m.predict(X), y) > 0.9


def test_model_with_other_features_is_refused(tmp_path):
    path = tmp_path / 'm.json'
    path.write_text('{"features": ["x"], "mean": [0], "scale": [1], "weights": [1], "bias": 0, "meta": {}}', encoding='utf-8')
    with pytest.raises(ValueError):
        CascadeModel.load(path)


def test_expected_falls_uses_the_model():
    g = support_graph([box('a', 0, 0, 0.125), box('b', 0, 0, 0.375)], floor=0.0)
    always = CascadeModel(np.zeros(len(FEATURES)), np.ones(len(FEATURES)), np.zeros(len(FEATURES)), 40.0, {})
    never = CascadeModel(np.zeros(len(FEATURES)), np.ones(len(FEATURES)), np.zeros(len(FEATURES)), -40.0, {})
    assert expected_falls(always, g, 'a') == pytest.approx(1.0)
    assert expected_falls(never, g, 'a') == pytest.approx(0.0)


def test_calibration_and_auc_helpers():
    assert auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    bins = calibration_bins([0.1, 0.15, 0.9, 0.95], [0, 0, 1, 1], n=2)
    assert bins[0]['observed'] == 0.0 and bins[1]['observed'] == 1.0


def test_belief_uses_a_learned_model_when_given():
    from shinai_pi.pile import PileBelief
    always = CascadeModel(np.zeros(len(FEATURES)), np.ones(len(FEATURES)), np.zeros(len(FEATURES)), 40.0, {})
    boxes = [box('a', 0, 0, 0.125), box('b', 0, 0, 0.375)]
    geo = PileBelief(floor=0.0); geo.observe(boxes, 'x', seen_fraction=1.0)
    learned = PileBelief(floor=0.0, cascade_model=always); learned.observe(boxes, 'x', seen_fraction=1.0)
    assert learned.cascade_risk('b') > geo.cascade_risk('b')       # the model says 'a' moves when 'b' goes


def test_the_rule_risk_stays_the_geometric_one_when_a_model_is_given():
    from shinai_pi.pile import PileBelief
    always = CascadeModel(np.zeros(len(FEATURES)), np.ones(len(FEATURES)), np.zeros(len(FEATURES)), 40.0, {})
    boxes = [box('a', 0, 0, 0.125), box('b', 0, 0, 0.375)]
    geo = PileBelief(floor=0.0); geo.observe(boxes, 'x', seen_fraction=1.0)
    learned = PileBelief(floor=0.0, cascade_model=always); learned.observe(boxes, 'x', seen_fraction=1.0)
    for k in ('a', 'b'):
        assert learned.rule_risk(k) == pytest.approx(geo.cascade_risk(k))
