"""
Evaluation metrics and the thresholding / fusion rules.
"""

import numpy as np
from sklearn import metrics


def binary_stats(y_true, y_pred, prefix=''):
    """Same confusion-matrix metrics as compute_performance_stats() in the original code, plus MCC."""
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a) / b if b else float('nan')
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {
        prefix + 'acc': div(tp + tn, tp + tn + fp + fn),
        prefix + 'precision': prec,
        prefix + 'recall': rec,
        prefix + 'f1': div(2 * prec * rec, prec + rec),
        prefix + 'fpr': div(fp, fp + tn),
        prefix + 'mcc': metrics.matthews_corrcoef(y_true, y_pred),
        prefix + 'tp': int(tp), prefix + 'fp': int(fp), prefix + 'tn': int(tn), prefix + 'fn': int(fn),
    }


def score_auc(scores, positive_mask, negative_mask=None, prefix=''):
    """AUROC / AUPR of a continuous anomaly score, positives vs negatives (default: everything else)."""
    if negative_mask is None:
        negative_mask = ~positive_mask
    keep = positive_mask | negative_mask
    y, s = positive_mask[keep].astype(int), scores[keep]
    if y.min() == y.max():
        return {prefix + 'auroc': float('nan'), prefix + 'aupr': float('nan')}
    return {prefix + 'auroc': metrics.roc_auc_score(y, s), prefix + 'aupr': metrics.average_precision_score(y, s)}


# ------------------------------------------------------------------ thresholds (never computed from test data)

def threshold_at_fpr(val_benign_scores, fpr):
    """Threshold so that a fraction `fpr` of benign VALIDATION flows would be flagged."""
    return float(np.quantile(val_benign_scores, 1.0 - fpr))


def threshold_percentile(fit_scores, pct=95):
    """Barnard et al.'s rule: the pct-th percentile of the detector's scores on its own training data."""
    return float(np.percentile(fit_scores, pct))


def or_fusion(p_attack, anomaly_flag):
    """Barnard et al.'s rule: alert if XGBoost says attack OR the anomaly detector says anomalous."""
    return ((p_attack >= 0.5) | anomaly_flag).astype(int)


# ------------------------------------------------------------------ cascade with an exact joint budget

BENIGN, KNOWN, NOVEL = 0, 1, 2


def budget_threshold(scores, k):
    """
    Largest-recall threshold that flags AT MOST k of `scores` with the rule `score >= thr`.

    Ties: if the k-th largest value is shared with unflagged rows, the threshold moves up to the next distinct
    value, so the budget is never exceeded (fewer than k flows may be flagged; report the achieved rate).
    k <= 0 returns +inf (nothing flagged).
    """
    scores = np.asarray(scores, dtype=np.float64)
    k = int(min(k, len(scores)))
    if k <= 0 or len(scores) == 0:
        return float('inf')
    desc = np.sort(scores)[::-1]
    t = desc[k - 1]
    if k < len(desc) and desc[k] == t:          # tie straddles the budget: step up to the next distinct value
        higher = desc[desc > t]
        return float(higher.min()) if len(higher) else float('inf')
    return float(t)


def cascade_thresholds(p_val_benign, s_val_benign, budget=0.02, share=0.5):
    """
    Thresholds (t1 for XGBoost's p, t2 for the stage-2 score s) from BENIGN VALIDATION flows only.

    budget: B, the fraction of benign validation flows the fused system may flag.
    share:  b2 / B, the part of the budget given to stage 2. b1 = (1 - share) * B goes to XGBoost.
    t1 flags at most floor(b1 * n) flows; t2 is set on the flows with p < t1 and gets whatever budget t1 left,
    so the total is floor(B * n) whenever ties allow it.
    Returns dict(t1, t2, val_fpr, val_fpr_stage1, val_fpr_stage2).
    """
    p, s = np.asarray(p_val_benign, np.float64), np.asarray(s_val_benign, np.float64)
    assert p.shape == s.shape and p.ndim == 1 and len(p) > 0
    assert 0 <= share <= 1 and 0 < budget < 1
    n = len(p)
    k_total = int(np.floor(budget * n))
    k1 = int(np.floor((1 - share) * budget * n))
    t1 = budget_threshold(p, k1)
    f1 = p >= t1
    t2 = budget_threshold(s[~f1], k_total - int(f1.sum()))
    f2 = ~f1 & (s >= t2)
    return dict(t1=t1, t2=t2, val_fpr=float((f1 | f2).mean()), val_fpr_stage1=float(f1.mean()), val_fpr_stage2=float(f2.mean()))


def cascade_decision(p, s, t1, t2):
    """alert_type per flow: KNOWN if p >= t1; NOVEL if p < t1 and s >= t2; else BENIGN."""
    p, s = np.asarray(p), np.asarray(s)
    known = p >= t1
    novel = ~known & (s >= t2)
    return np.where(known, KNOWN, np.where(novel, NOVEL, BENIGN))


def alert_stats(y_true, zd_mask, alert, prefix=''):
    """
    Metrics for any binary alert rule, with zero-day and known-attack recall reported separately.
    The test FPR is always included, because a validation budget drifts on test data.
    """
    y_true, zd_mask, alert = np.asarray(y_true), np.asarray(zd_mask, bool), np.asarray(alert, bool)
    known = (y_true == 1) & ~zd_mask
    row = binary_stats(y_true, alert.astype(int), prefix)
    row[prefix + 'zd_recall'] = float(alert[zd_mask].mean()) if zd_mask.any() else float('nan')
    row[prefix + 'known_recall'] = float(alert[known].mean()) if known.any() else float('nan')
    row[prefix + 'n_zd'], row[prefix + 'n_known'] = int(zd_mask.sum()), int(known.sum())
    return row


def compare_systems(p_vb, s_vb, p_test, s_test, y_test, zd_test, budget=0.02, shares=(0.25, 0.5, 0.75)):
    """
    Every alert rule we compare, all thresholds from the same BENIGN VALIDATION scores.
    s_vb / s_test may be None (XGBoost-only rows). Returns {system: (test alert mask, validation FPR, extra)}.

      xgb_0.5       p >= 0.5 (the pilot / Barnard)
      xgb_val       p >= threshold flagging `budget` of benign validation flows (= cascade with share 0)
      or_barnard    p >= 0.5  OR  s at `budget`            (Barnard's rule with a validation threshold for s)
      or_val        p at `budget`  OR  s at `budget`       (each stage its own budget; total can reach 2x budget)
      cascade_<sh>  joint budget, share sh to stage 2
    """
    n = len(p_vb)
    k = int(np.floor(budget * n))
    tp = budget_threshold(p_vb, k)
    out = {'xgb_0.5': (p_test >= 0.5, float(np.mean(p_vb >= 0.5)), {}),
           'xgb_val': (p_test >= tp, float(np.mean(p_vb >= tp)), {'t1': tp})}
    if s_vb is None:
        return out
    ts = budget_threshold(s_vb, k)
    out['or_barnard'] = ((p_test >= 0.5) | (s_test >= ts), float(np.mean((p_vb >= 0.5) | (s_vb >= ts))), {'t2': ts})
    out['or_val'] = ((p_test >= tp) | (s_test >= ts), float(np.mean((p_vb >= tp) | (s_vb >= ts))), {'t1': tp, 't2': ts})
    for sh in shares:
        th = cascade_thresholds(p_vb, s_vb, budget, sh)
        alert = cascade_decision(p_test, s_test, th['t1'], th['t2']) != BENIGN
        out[f'cascade_{sh:g}'] = (alert, th['val_fpr'], {'t1': th['t1'], 't2': th['t2']})
    return out
