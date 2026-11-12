"""
First stage: XGBoost classifier + TreeSHAP explanations (as in Barnard et al. 2022), with on-disk caching.
"""

import multiprocessing as mp
import os
import time

import numpy as np
import shap
import xgboost

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
SHAP_BACKGROUND = 500


def fit_xgb(X, y, seed):
    """XGBoost with the defaults Barnard et al. used (xgboost 1.x defaults written out explicitly)."""
    model = xgboost.XGBClassifier(objective="binary:logistic", n_estimators=100, max_depth=6, learning_rate=0.3,
                                  random_state=seed, n_jobs=os.cpu_count())
    model.fit(X, y)
    return model


def _build_explainer(model, background):
    return shap.TreeExplainer(model, background, feature_perturbation="interventional", model_output="probability")


def make_explainer(model, X_train, seed, background_n=SHAP_BACKGROUND):
    """Interventional TreeSHAP decomposing P(attack|x), as in the original code, but with a fixed-size background sample."""
    rng = np.random.default_rng(seed)
    background = X_train[rng.choice(len(X_train), size=min(background_n, len(X_train)), replace=False)]
    explainer = _build_explainer(model, background)
    explainer._xz_rebuild = (model, background)   # lets 'spawn' workers (Windows) rebuild an identical explainer
    return explainer


_EXPLAINER = None  # fork: set before forking so workers inherit it; spawn: rebuilt in each worker by _init_worker


def _init_worker(model, background):
    global _EXPLAINER
    _EXPLAINER = _build_explainer(model, background)


def _shap_chunk(X_chunk):
    return np.asarray(_EXPLAINER.shap_values(X_chunk), dtype=np.float32)


def _start_method():
    """'fork' where available (Linux/macOS, identical to the pilot runs); 'spawn' on Windows. Override with XZ_MP_START."""
    forced = os.environ.get('XZ_MP_START')
    if forced:
        return forced
    return 'fork' if 'fork' in mp.get_all_start_methods() else 'spawn'


def shap_values(explainer, X, n_jobs=None):
    """
    Interventional TreeSHAP is single-threaded in the shap package, so split rows across worker processes.
    n_jobs defaults to XZ_SHAP_JOBS or min(CPU count, 8); n_jobs=1 runs in-process. Interventional TreeSHAP is
    deterministic, so the result does not depend on the start method or the number of workers.
    """
    global _EXPLAINER
    # default: at most 8 workers (each holds its own copy of the model and libraries); XZ_SHAP_JOBS overrides
    n_jobs = n_jobs or int(os.environ.get('XZ_SHAP_JOBS', 0)) or min(os.cpu_count(), 8)
    if n_jobs == 1 or len(X) < 2000:
        return np.asarray(explainer.shap_values(X), dtype=np.float32)
    chunks = np.array_split(X, n_jobs * 4)
    method = _start_method()
    if method == 'fork':
        _EXPLAINER = explainer
        try:
            with mp.get_context('fork').Pool(n_jobs) as pool:
                out = pool.map(_shap_chunk, chunks)
        finally:
            _EXPLAINER = None
    else:
        # spawn: workers start fresh, so each one rebuilds the explainer from (model, background).
        # Scripts calling this must keep their entry point under `if __name__ == '__main__':`.
        model, background = explainer._xz_rebuild
        with mp.get_context(method).Pool(min(n_jobs, 61), initializer=_init_worker, initargs=(model, background)) as pool:
            out = pool.map(_shap_chunk, chunks)
    return np.concatenate(out, axis=0)


def first_stage(split, seed, cache=True):
    """
    Fit XGBoost on split.X_train, then return a dict with predicted attack probabilities and SHAP vectors
    for train/val/test, plus timing. Results are cached per (split, seed).
    """
    key = split.name.replace('/', '__') + f'__seed{seed}__n{len(split.X_train)}'
    path = os.path.join(CACHE_DIR, key + '.npz')
    if cache and os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}

    t0 = time.time()
    model = fit_xgb(split.X_train, split.y_train, seed)
    t_fit = time.time() - t0

    res = {'t_xgb_fit': np.array(t_fit)}
    for part in ['train', 'val', 'test']:
        res[f'p_{part}'] = model.predict_proba(getattr(split, f'X_{part}'))[:, 1].astype(np.float32)

    explainer = make_explainer(model, split.X_train, seed)
    t0 = time.time()
    for part in ['train', 'val', 'test']:
        res[f'shap_{part}'] = shap_values(explainer, getattr(split, f'X_{part}'))
    n_explained = len(split.X_train) + len(split.X_val) + len(split.X_test)
    res['t_shap_per_flow'] = np.array((time.time() - t0) / n_explained)

    # local accuracy check: base value + sum of SHAP values should reproduce P(attack|x)
    err = np.abs(explainer.expected_value + res['shap_test'].sum(1) - res['p_test']).max()
    assert err < 1e-3, f"SHAP additivity violated (max err {err})"

    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, **res)
    return res
