"""The learned predictor is judged by the criteria fixed before any data (docs/generalisation_plan.md)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('judge_cascade', ROOT / 'scripts' / 'judge_cascade.py')
jc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jc)


def part(learned_auc, rule_auc, learned_brier, rule_brier):
    return {'learned': {'auc': learned_auc, 'brier': learned_brier}, 'geometric_rule': {'auc': rule_auc, 'brier': rule_brier}}


def test_both_held_out_sets_must_pass():
    good = part(0.90, 0.70, 0.01, 0.02)
    assert jc.judge({'seen': good, 'unseen': good})['passed'] is True
    v = jc.judge({'seen': good, 'unseen': part(0.79, 0.70, 0.01, 0.02)})
    assert v['passed'] is False and v['sets']['unseen']['passed'] is False and v['sets']['seen']['passed'] is True


def test_the_margin_and_the_brier_are_both_required():
    assert jc.judge({'seen': part(0.80, 0.70, 0.02, 0.02), 'unseen': part(0.80, 0.70, 0.02, 0.02)})['passed'] is True
    assert jc.judge({'seen': part(0.90, 0.70, 0.03, 0.02), 'unseen': part(0.90, 0.70, 0.01, 0.02)})['passed'] is False


def test_a_missing_held_out_set_is_not_a_pass():
    with pytest.raises(ValueError):
        jc.judge({'seen': part(0.9, 0.7, 0.01, 0.02)})


def test_the_criteria_are_the_ones_fixed_in_the_plan():
    assert jc.AUC_MARGIN == 0.10 and jc.SETS == ('seen', 'unseen')
