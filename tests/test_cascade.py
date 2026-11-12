"""Cascade joint budget: fused benign-validation FPR = B within 0.1% and never above B."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import BENIGN, KNOWN, NOVEL, budget_threshold, cascade_decision, cascade_thresholds  # noqa: E402


def test_budget_threshold_continuous():
    s = np.random.default_rng(0).random(10_000)
    t = budget_threshold(s, 200)
    assert (s >= t).sum() == 200


def test_budget_threshold_never_exceeds_with_ties():
    s = np.r_[np.zeros(9_000), np.full(500, 0.3), np.random.default_rng(1).random(500) + 1]
    for k in (100, 499, 500, 501, 800, 1000, 1001, 9_999):
        assert (s >= budget_threshold(s, k)).sum() <= k
    assert (s >= budget_threshold(s, 1000)).sum() == 1000      # the tied block fits exactly
    assert (s >= budget_threshold(s, 800)).sum() == 500        # it does not fit: step above the tie


def test_budget_threshold_edges():
    assert budget_threshold([1, 2, 3], 0) == float('inf')
    assert budget_threshold(np.ones(10), 5) == float('inf')     # all tied: nothing can be flagged within budget
    assert budget_threshold([1, 2, 3], 10) == 1


@pytest.mark.parametrize('share', [0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize('budget', [0.01, 0.02])
def test_cascade_hits_budget(share, budget):
    rng = np.random.default_rng(42)
    p, s = rng.beta(0.5, 20, 20_000), rng.gamma(2, 1, 20_000)
    r = cascade_thresholds(p, s, budget, share)
    assert abs(r['val_fpr'] - budget) <= 0.001
    assert r['val_fpr'] <= budget + 1e-12
    assert abs(r['val_fpr_stage1'] - (1 - share) * budget) <= 0.001
    if share == 0:
        assert r['t2'] == float('inf')


def test_cascade_ties_in_p_handed_to_stage2():
    # 97% of benign flows have p exactly 0 (typical for XGBoost) and a tie block straddles t1
    rng = np.random.default_rng(3)
    p = np.r_[np.zeros(9_700), np.full(150, 0.05), rng.random(150) * 0.5 + 0.5]
    s = rng.random(10_000)
    r = cascade_thresholds(p, s, budget=0.02, share=0.5)
    assert r['val_fpr'] <= 0.02 and abs(r['val_fpr'] - 0.02) <= 0.001   # budget t1 could not use goes to stage 2
    assert r['val_fpr_stage1'] <= 0.01


def test_cascade_decision_types():
    p = np.array([0.9, 0.9, 0.1, 0.1])
    s = np.array([5.0, 0.0, 5.0, 0.0])
    t = cascade_decision(p, s, t1=0.5, t2=1.0)
    assert list(t) == [KNOWN, KNOWN, NOVEL, BENIGN]


def test_cascade_on_test_uses_val_thresholds_only():
    # same thresholds applied to a shifted "test" set: FPR drifts, which is why test FPR is always reported
    rng = np.random.default_rng(7)
    p_v, s_v = rng.beta(0.5, 20, 10_000), rng.gamma(2, 1, 10_000)
    r = cascade_thresholds(p_v, s_v, 0.02, 0.5)
    p_t, s_t = rng.beta(0.6, 18, 10_000), rng.gamma(2.2, 1, 10_000)
    t = cascade_decision(p_t, s_t, r['t1'], r['t2'])
    assert (t != BENIGN).mean() > 0.02
