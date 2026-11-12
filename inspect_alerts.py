"""
Fit the XplainIDS cascade on one split and look at what it produces (detection table, example alerts, example_alert.json).

    python inspect_alerts.py                                     # NSL-KDD official, seed 0, 2% budget split 1%/1%
    python inspect_alerts.py --dataset unsw_nb15 --protocol lofo --family Exploits --seed 1

Writes to results/pipeline_demo/<split>__seed<k>/:
    REPORT.md             detection table (cascade vs baselines at the same validation budget) + NOVEL/KNOWN examples
    detection.csv         the table as CSV
    example_alert.json    one NOVEL alert on a real zero-day flow and one KNOWN alert (AlertRecord dicts)
    alerts_sample.json    all printed examples
    run_meta.json         versions, seed, sizes and thresholds (thresholds come from validation data only)
and saves the fitted pipeline to cache/pipelines/ (gitignored) for the replay script later.
"""

import argparse
import json
import os
import platform
import time
import warnings

import numpy as np
import pandas as pd

from data import make_split
from evidence import evidence_line
from metrics import KNOWN, NOVEL, alert_stats, compare_systems
from pipeline import XplainIDSPipeline

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))


def versions():
    out = {'python': platform.python_version(), 'platform': platform.platform()}
    for m in ('numpy', 'pandas', 'sklearn', 'xgboost', 'shap', 'torch'):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def detection_table(pipe, split, out):
    """Cascade vs baselines, all thresholds from the same benign validation subsample (metrics.compare_systems)."""
    y, zd, normal = split.y_test, split.zd_test, split.y_test == 0
    p, vb = out['p'], pipe.val_benign_scores
    names = {'xgb_0.5': 'XGBoost alone, p >= 0.5 (pilot)', 'xgb_val': f'XGBoost alone, val {pipe.budget:.0%}',
             'or_barnard': f'OR fusion: p >= 0.5 OR s at val {pipe.budget:.0%} (Barnard rule)',
             'or_val': f'OR fusion: val {pipe.budget:.0%} + val {pipe.budget:.0%}'}
    systems = compare_systems(vb['p'], vb['s'], p, out['s'], y, zd, pipe.budget, shares=(pipe.share,))
    rows = []
    for key, (alert, val_fpr, _) in systems.items():
        r = alert_stats(y, zd, alert)
        name = names.get(key, f'Cascade, joint {pipe.budget:.0%} (share {pipe.share:g} to stage 2)')
        rows.append({'system': name, 'val FPR': val_fpr, 'test FPR': r['fpr'], 'zero-day recall': r['zd_recall'],
                     'known recall': r['known_recall'], 'precision': r['precision'], 'F1': r['f1'], 'MCC': r['mcc']})
    t = out['alert_type']
    missed_by_t1 = zd & (p < pipe.t1)
    extra = {'zero-day flows': int(zd.sum()), 'zero-days XGBoost passes (p < t1)': int(missed_by_t1.sum()),
             'of those caught as NOVEL': int((missed_by_t1 & (t == NOVEL)).sum()),
             'benign test flows flagged NOVEL': int((normal & (t == NOVEL)).sum())}
    return pd.DataFrame(rows), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='nsl_kdd')
    ap.add_argument('--protocol', default='official')
    ap.add_argument('--family', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--budget', type=float, default=0.02)
    ap.add_argument('--share', type=float, default=0.5, help='part of the budget given to stage 2')
    ap.add_argument('--detector', default='ae_small')
    ap.add_argument('--n', type=int, default=5, help='NOVEL examples to print')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    split = make_split(a.dataset, a.protocol, a.family, seed=a.seed)
    tag = f"{split.name.replace('/', '__')}__seed{a.seed}"
    out_dir = a.out or os.path.join(HERE, 'results', 'pipeline_demo', tag)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    pipe = XplainIDSPipeline(budget=a.budget, share=a.share, detector=a.detector, seed=a.seed).fit(split)
    t_fit = time.time() - t0
    print(f"fitted {split.name} seed {a.seed} in {t_fit:.0f}s; thresholds: {pipe.thresholds}", flush=True)
    pipe.save(os.path.join(HERE, 'cache', 'pipelines', f'{tag}__{a.detector}__b{a.budget}__s{a.share}.pkl'))

    t0 = time.time()
    out = pipe.score(split.X_test)
    t_score = time.time() - t0
    table, extra = detection_table(pipe, split, out)
    table.to_csv(os.path.join(out_dir, 'detection.csv'), index=False)

    # examples: NOVEL alerts on real zero-day flows (true positives), one NOVEL false alarm, one KNOWN alert
    rng = np.random.default_rng(a.seed)
    t, zd, normal = out['alert_type'], split.zd_test, split.y_test == 0
    novel_tp = np.where((t == NOVEL) & zd)[0]
    novel_fp = np.where((t == NOVEL) & normal)[0]
    known_tp = np.where((t == KNOWN) & (split.y_test == 1))[0]
    pick = lambda idx, k: list(rng.choice(idx, min(k, len(idx)), replace=False)) if len(idx) else []

    def pick_diverse(idx, k):
        """One flow per attack type first (rarest types first), then fill up at random."""
        types = split.attack_test[idx]
        chosen = []
        for at in sorted(set(types), key=lambda v: (types == v).sum()):
            if len(chosen) < k:
                chosen.append(rng.choice(idx[types == at]))
        rest = np.setdiff1d(idx, chosen)
        return chosen + pick(rest, k - len(chosen))

    examples = [('NOVEL, real zero-day', i) for i in pick_diverse(novel_tp, a.n)] + \
               [('NOVEL, false alarm on benign flow', i) for i in pick(novel_fp, 1)] + \
               [('KNOWN, real attack', i) for i in pick(known_tp, 1)]

    records, md_ex = [], []
    for label, i in examples:
        rec = pipe.record(int(i), out, split.X_test, alert_id=f"{tag}-test{int(i)}")
        rec['ground_truth'] = {'label': 'attack' if split.y_test[i] else 'benign', 'family': str(split.fam_test[i]),
                               'attack_type': str(split.attack_test[i]), 'zero_day': bool(zd[i])}   # for our evaluation only; never sent to the LLM
        records.append(rec)
        lines = [f"**{label}**: {rec['alert_id']} ({rec['ground_truth']['family']} / {rec['ground_truth']['attack_type']}) — p = {rec['p']:.4f} (t1 = {rec['t1']:.4f}), "
                 f"s = {rec['s']:.4f} (t2 = {rec['t2']:.4f}), type {rec['alert_type']}"]
        lines += [f"- {evidence_line(e)}" for e in rec['evidence']]
        md_ex.append('\n'.join(lines))

    with open(os.path.join(out_dir, 'alerts_sample.json'), 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    first = lambda kind: next((r for r in records if r['alert_type'].startswith(kind) and r['ground_truth']['label'] == 'attack'), None)
    with open(os.path.join(out_dir, 'example_alert.json'), 'w', encoding='utf-8') as f:
        json.dump({'novel': first('NOVEL'), 'known': first('KNOWN')}, f, indent=2)

    meta = dict(split=split.name, seed=a.seed, budget=a.budget, share=a.share, detector=a.detector, input='shap',
                thresholds_from='benign validation subsample only', fit=pipe.fit_info, test_rows=len(split.y_test),
                t_fit_s=round(t_fit, 1), t_score_ms_per_flow=round(1e3 * t_score / len(split.y_test), 3), versions=versions())
    with open(os.path.join(out_dir, 'run_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=float)

    md = [f"# XplainIDS cascade: {split.name}, seed {a.seed}\n",
          f"Budget {a.budget:.0%} of benign validation flows, {a.share:g} of it to stage 2 ({a.detector} on SHAP vectors). "
          f"Thresholds: t1 = {pipe.t1:.5f}, t2 = {pipe.t2:.5f}; achieved validation FPR {pipe.thresholds['val_fpr']:.4f} "
          f"(stage 1 {pipe.thresholds['val_fpr_stage1']:.4f}, stage 2 {pipe.thresholds['val_fpr_stage2']:.4f}).\n",
          "## Detection on the test set\n", table.to_markdown(index=False, floatfmt='.3f'), '',
          '\n'.join(f"- {k}: {v}" for k, v in extra.items()), '',
          "## Example alerts\n",
          "NOVEL evidence compares each feature's SHAP contribution (phi) with the autoencoder's normal-looking "
          "reconstruction (phi_hat). KNOWN evidence lists the features that pushed XGBoost towards 'attack'.\n",
          '\n\n'.join(md_ex), '']
    with open(os.path.join(out_dir, 'REPORT.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('\n'.join(md))
    print(f"\nWrote {out_dir}")


if __name__ == '__main__':
    main()
