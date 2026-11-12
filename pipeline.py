"""
XplainIDS pipeline: XGBoost -> TreeSHAP -> stage-2 detector on SHAP vectors -> cascade with an exact
joint false-alarm budget -> alert records with type-aware evidence.

    pipe = XplainIDSPipeline(budget=0.02, share=0.5, seed=0).fit(split)
    out = pipe.score(split.X_test)                     # p, phi, s, alert_type
    rec = pipe.record(i, out, split.X_test)            # AlertRecord dict for flagged row i
    pipe.save(path); pipe = XplainIDSPipeline.load(path)

Data use: XGBoost trains on the full training split. SHAP and the stage-2 detector use a random subsample of
`max_fit` training rows and `max_val` validation rows (as in the extended pilot); the detector is fit on the BENIGN
rows of that subsample, and both thresholds come from its BENIGN validation rows. Test data is never touched in fit().
"""

import os
import pickle

import numpy as np

from detectors import InputTransform, make_detector
from evidence import build_record
from explain import _build_explainer, fit_xgb, make_explainer, shap_values
from metrics import BENIGN, cascade_decision, cascade_thresholds


def fit_first_stage(split, seed, max_fit=50000, max_val=20000, model=None):
    """
    XGBoost on the full training split, then TreeSHAP on a random subsample of max_fit training and max_val validation
    rows (the rows stage 2 and the thresholds use). Deterministic given the seed. Never touches test data.
    """
    rng = np.random.default_rng(seed)
    model = model if model is not None else fit_xgb(split.X_train, split.y_train, seed)
    tr = rng.choice(len(split.y_train), min(max_fit, len(split.y_train)), replace=False)
    va = rng.choice(len(split.y_val), min(max_val, len(split.y_val)), replace=False)
    explainer = make_explainer(model, split.X_train[tr], seed)
    return dict(model=model, explainer=explainer, tr=tr, va=va,
                phi_tr=shap_values(explainer, split.X_train[tr]), phi_va=shap_values(explainer, split.X_val[va]),
                p_va=model.predict_proba(split.X_val[va])[:, 1])


class XplainIDSPipeline:
    def __init__(self, budget=0.02, share=0.5, detector='ae_small', inp='shap', seed=0, max_fit=50000, max_val=20000):
        assert inp in ('shap', 'concat', 'raw')
        self.budget, self.share, self.detector, self.inp, self.seed = budget, share, detector, inp, seed
        self.max_fit, self.max_val = max_fit, max_val

    # ------------------------------------------------------------------ fit
    def fit(self, split, model=None, first=None):
        """
        model: optionally an already fitted XGBoost (same seed, same training data) to skip refitting.
        first: optionally the output of fit_first_stage() for this split and seed (run_grid shares it across detectors).
        """
        self.feature_names = list(split.feature_names)
        self.categories = split.categories or {}
        first = first or fit_first_stage(split, self.seed, self.max_fit, self.max_val, model)
        self.model, self._explainer = first['model'], first['explainer']
        self.background = self._explainer._xz_rebuild[1]
        tr, va, phi_tr, phi_va = first['tr'], first['va'], first['phi_tr'], first['phi_va']
        X_tr, y_tr, X_va, y_va = split.X_train[tr], split.y_train[tr], split.X_val[va], split.y_val[va]

        b_tr, b_va = y_tr == 0, y_va == 0
        self.T = InputTransform(self.inp).fit(X_tr[b_tr], phi_tr[b_tr])
        Z_tr, Z_va = self.T.transform(X_tr, phi_tr), self.T.transform(X_va, phi_va)
        self.det = make_detector(self.detector, self.seed).fit(Z_tr[b_tr], Z_va[b_va])

        p_va, s_va = first['p_va'], self.det.score(Z_va)
        self.thresholds = cascade_thresholds(p_va[b_va], s_va[b_va], self.budget, self.share)
        self.val_benign_scores = dict(p=p_va[b_va].astype(np.float32), s=s_va[b_va].astype(np.float32))   # for baselines
        self.t1, self.t2 = self.thresholds['t1'], self.thresholds['t2']
        self.benign_median = np.median(split.X_train[split.y_train == 0], axis=0)   # for faithfulness / evasion tests
        self.fit_info = dict(n_train=len(split.y_train), n_fit=int(b_tr.sum()), n_val_benign=int(b_va.sum()),
                             split=split.name, **self.thresholds)
        return self

    # ------------------------------------------------------------------ score
    @property
    def explainer(self):
        if getattr(self, '_explainer', None) is None:
            self._explainer = _build_explainer(self.model, self.background)
            self._explainer._xz_rebuild = (self.model, self.background)
        return self._explainer

    def score(self, X):
        """p, phi, stage-2 score s and alert_type (0 benign / 1 KNOWN / 2 NOVEL) for each row."""
        p = self.model.predict_proba(X)[:, 1]
        phi = shap_values(self.explainer, X)
        Z = self.T.transform(X, phi)
        s = self.det.score(Z)
        return dict(p=p, phi=phi, Z=Z, s=s, alert_type=cascade_decision(p, s, self.t1, self.t2))

    def phi_hat(self, Z):
        """The detector's normal-looking reconstruction of each row's SHAP vector, in SHAP units (None if unavailable)."""
        if self.inp == 'raw' or not hasattr(self.det, 'reconstruct'):
            return None
        return self.T.inverse_shap(self.det.reconstruct(Z))

    def record(self, i, out, X, alert_id=None):
        """AlertRecord for flagged row i of a score() result."""
        t = int(out['alert_type'][i])
        if t == BENIGN:
            raise ValueError(f"row {i} is not an alert")
        ph = self.phi_hat(out['Z'][i:i + 1]) if t != 1 else None
        return build_record(alert_id if alert_id is not None else i, t, out['p'][i], out['s'][i],
                            self.t1, self.t2, out['phi'][i], None if ph is None else ph[0], X[i],
                            self.feature_names, self.categories)

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ex, self._explainer = getattr(self, '_explainer', None), None   # rebuilt on load from (model, background)
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        finally:
            self._explainer = ex

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
