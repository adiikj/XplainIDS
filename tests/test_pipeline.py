"""End-to-end cascade on a small synthetic split: budget respected, evidence produced, save/load identical."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for m in ('xgboost', 'shap', 'torch'):
    pytest.importorskip(m)
from data import Split  # noqa: E402
from metrics import BENIGN, NOVEL  # noqa: E402
from pipeline import XplainIDSPipeline  # noqa: E402


def make_synthetic(seed=0):
    rng = np.random.default_rng(seed)

    def block(n_benign, n_known, n_new):
        Xb = rng.normal(0, 1, (n_benign, 6))
        # known attack: f0 AND f1 high, so XGBoost has to use both features.
        # zero-day: f0 high but f1 low. XGBoost tends to pass it (f1 is not high), but its SHAP vector is unusual
        # (f0 pushes towards attack, unlike benign flows). A zero-day that differed only in features XGBoost never
        # uses would have near-zero SHAP there and be invisible to a SHAP-based stage 2: a real limitation.
        Xk = rng.normal(0, 1, (n_known, 6)); Xk[:, 0] += 3; Xk[:, 1] += 3
        Xn = rng.normal(0, 1, (n_new, 6)); Xn[:, 0] += 3; Xn[:, 1] -= 3
        X = np.vstack([Xb, Xk, Xn]); X[:, 5] = rng.integers(0, 3, len(X))  # one categorical column
        y = np.r_[np.zeros(n_benign), np.ones(n_known + n_new)].astype(int)
        fam = np.array(['normal'] * n_benign + ['Known'] * n_known + ['New'] * n_new)
        return X, y, fam

    Xtr, ytr, ftr = block(4000, 1000, 0)
    Xva, yva, fva = block(3000, 300, 0)
    Xte, yte, fte = block(2000, 300, 300)
    return Split('synthetic/lofo/New', ['f0', 'f1', 'f2', 'f3', 'f4', 'proto'], Xtr, ytr, ftr, Xva, yva, fva,
                 Xte, yte, fte, fte == 'New', categories={'proto': ['tcp', 'udp', 'icmp']})


@pytest.fixture(scope='module')
def fitted():
    split = make_synthetic()
    return split, XplainIDSPipeline(budget=0.02, share=0.5, seed=0, max_fit=5000, max_val=3300).fit(split)


def test_budget_on_validation(fitted):
    _, pipe = fitted
    assert pipe.thresholds['val_fpr'] <= 0.02 + 1e-12
    assert abs(pipe.thresholds['val_fpr'] - 0.02) <= 0.001


def test_records_and_save_load(fitted, tmp_path):
    split, pipe = fitted
    out = pipe.score(split.X_test)
    t = out['alert_type']
    benign = split.y_test == 0
    assert (t[split.zd_test] != BENIGN).mean() > 3 * (t[benign] != BENIGN).mean()   # zero-days flagged far more often
    novel = np.where(t == NOVEL)[0]
    assert len(novel) > 0
    if len(novel):
        rec = pipe.record(int(novel[0]), out, split.X_test)
        assert rec['alert_type'] == 'NOVEL' and 1 <= len(rec['evidence']) <= 5 and 'phi_hat' in rec['evidence'][0]
    path = tmp_path / 'pipe.pkl'
    pipe.save(path)
    again = XplainIDSPipeline.load(path).score(split.X_test[:500])
    np.testing.assert_allclose(again['s'], out['s'][:500], rtol=1e-5, atol=1e-6)
    assert (again['alert_type'] == t[:500]).all()
