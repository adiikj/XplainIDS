"""Parallel TreeSHAP gives the same values in-process, with fork (Linux) and with spawn (Windows)."""

import multiprocessing as mp
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
pytest.importorskip('shap')
pytest.importorskip('xgboost')
import explain  # noqa: E402


@pytest.fixture(scope='module')
def fitted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 8))
    y = (X[:, 0] + X[:, 1] ** 2 > 1).astype(int)
    model = explain.fit_xgb(X, y, seed=0)
    return model, X


@pytest.mark.parametrize('method', [m for m in ('fork', 'spawn') if m in mp.get_all_start_methods()])
def test_parallel_matches_single_process(fitted, method, monkeypatch):
    model, X = fitted
    ex = explain.make_explainer(model, X, seed=0, background_n=100)
    single = explain.shap_values(ex, X[:2500], n_jobs=1)
    monkeypatch.setenv('XZ_MP_START', method)
    multi = explain.shap_values(ex, X[:2500], n_jobs=2)
    assert multi.shape == single.shape
    np.testing.assert_allclose(multi, single, atol=1e-6)
    # additivity, as asserted in first_stage
    p = model.predict_proba(X[:2500])[:, 1]
    assert np.abs(ex.expected_value + multi.sum(1) - p).max() < 1e-3
