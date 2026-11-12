"""
Run the controlled ablation grid and append results to results/results.csv (resumable).

    python run_experiments.py --quick          # smoke test, ~1-2 min
    python run_experiments.py --repro          # reproduce Barnard et al. 2022 on NSL-KDD (Table A)
    python run_experiments.py                  # full grid: 14 splits x 5 seeds x {benign,all} x {raw,shap,concat} x 4 detectors

One CSV row per (split, seed, train_set, input, detector, threshold policy). XGBoost-only rows have detector='none'.
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from data import all_splits, make_split
from detectors import InputTransform, make_detector
from explain import first_stage
from metrics import binary_stats, or_fusion, score_auc, threshold_at_fpr, threshold_percentile

warnings.filterwarnings('ignore')
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results.csv')
KEY = ['dataset', 'protocol', 'family', 'seed', 'train_set', 'input', 'detector']
POLICIES = {'val_fpr1': ('fpr', 0.01), 'val_fpr2': ('fpr', 0.02), 'val_fpr5': ('fpr', 0.05), 'p1_p95': ('pct', 95)}


def append_rows(rows, path):
    """Append rows; rows may have different columns (e.g. XGBoost-only rows), so keep the file's header a superset."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = pd.DataFrame(rows)
    if not os.path.exists(path):
        new.to_csv(path, index=False)
        return
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if set(new.columns) <= set(header):
        new.reindex(columns=header).to_csv(path, mode='a', header=False, index=False)
    else:
        old = pd.read_csv(path, keep_default_na=False, na_values=[''])
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path, usecols=KEY, keep_default_na=False)
    return set(map(tuple, df.astype(str).values))


def evaluate_scores(base, split, p_test, s_fit, s_val, s_test):
    """All metrics for one fitted detector; returns one row per threshold policy."""
    zd, normal = split.zd_test, split.y_test == 0
    missed = p_test < 0.5   # flows XGBoost lets through: the only place the 2nd stage can add true detections
    common = dict(base)
    common.update(score_auc(s_test, zd, prefix='zd_'))                          # zero-day vs everything else (as in P1)
    common.update(score_auc(s_test, zd, normal, prefix='zdvn_'))                # zero-day vs normal only
    common.update(score_auc(s_test, zd & missed, normal & missed, prefix='res_'))  # among flows XGBoost missed

    rows = []
    for policy, (kind, val) in POLICIES.items():
        thr = threshold_at_fpr(s_val[split.y_val == 0], val) if kind == 'fpr' else threshold_percentile(s_fit, val)
        flag = s_test > thr
        fused = or_fusion(p_test, flag)
        row = dict(common, policy=policy, threshold=thr,
                   ad_zd_recall=flag[zd].mean(), ad_test_fpr=flag[normal].mean(),
                   fused_zd_recall=fused[zd].mean())
        row.update(binary_stats(split.y_test, fused, prefix='fused_'))
        rows.append(row)
    return rows


def run_split(dataset, protocol, family, seed, args, done):
    split = make_split(dataset, protocol, family, seed=seed, max_train=args.max_train)
    fs = first_stage(split, seed)
    p_test = fs['p_test']
    base = dict(dataset=dataset, protocol=protocol, family=family or '', seed=seed,
                n_train=len(split.y_train), n_test=len(split.y_test), n_zd=int(split.zd_test.sum()),
                t_xgb_fit=float(fs['t_xgb_fit']), t_shap_per_flow=float(fs['t_shap_per_flow']),
                xgb_zd_recall=float((p_test[split.zd_test] >= 0.5).mean()),
                xgb_zd_auroc_vs_normal=score_auc(p_test, split.zd_test, split.y_test == 0)['auroc'])
    base.update(binary_stats(split.y_test, (p_test >= 0.5).astype(int), prefix='xgb_'))

    key = (dataset, protocol, family or '', str(seed), '-', '-', 'none')
    if key not in done:
        append_rows([dict(base, train_set='-', input='-', detector='none', policy='xgb_only')], args.out)

    for train_set in args.train_sets:
        fit_mask = split.y_train == 0 if train_set == 'benign' else np.ones(len(split.y_train), bool)
        es_mask = split.y_val == 0 if train_set == 'benign' else np.ones(len(split.y_val), bool)
        for inp in args.inputs:
            T = InputTransform(inp).fit(split.X_train[fit_mask], fs['shap_train'][fit_mask])
            Z_fit = T.transform(split.X_train[fit_mask], fs['shap_train'][fit_mask])
            Z_val = T.transform(split.X_val, fs['shap_val'])
            Z_test = T.transform(split.X_test, fs['shap_test'])
            for det_name in args.detectors:
                key = (dataset, protocol, family or '', str(seed), train_set, inp, det_name)
                if key in done:
                    continue
                t0 = time.time()
                det = make_detector(det_name, seed).fit(Z_fit, Z_val[es_mask])
                t_fit = time.time() - t0
                s_fit, s_val, s_test = det.score(Z_fit), det.score(Z_val), det.score(Z_test)
                b = dict(base, train_set=train_set, input=inp, detector=det_name, t_det_fit=t_fit,
                         det_epochs=getattr(det, 'epochs_', np.nan))
                rows = evaluate_scores(b, split, p_test, s_fit, s_val, s_test)
                append_rows(rows, args.out)
                r = rows[0]
                print(f"  {split.name:32s} s{seed} {train_set:6s} {inp:6s} {det_name:8s} zd_auroc={r['zd_auroc']:.3f} "
                      f"res_auroc={r['res_auroc']:.3f} fused_f1@fpr1={r['fused_f1']:.3f} ({t_fit:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['nsl_kdd', 'unsw_nb15'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--detectors', nargs='+', default=['ae_small', 'iforest', 'knn', 'pca'])
    ap.add_argument('--inputs', nargs='+', default=['raw', 'shap', 'concat'])
    ap.add_argument('--train-sets', nargs='+', default=['benign', 'all'])
    ap.add_argument('--families', nargs='+', default=None, help='restrict to these held-out families (use "official" for NSL-KDD official)')
    ap.add_argument('--max-train', type=int, default=None)
    ap.add_argument('--out', default=RESULTS)
    ap.add_argument('--quick', action='store_true', help='smoke test: 5k training rows, 1 seed, 1 split, IForest')
    ap.add_argument('--repro', action='store_true', help='reproduce Barnard et al. 2022 (NSL-KDD official, P1 autoencoder)')
    args = ap.parse_args()

    splits = all_splits(args.datasets)
    if args.quick:
        args.out = args.out.replace('results.csv', 'quick.csv')
        args.seeds, args.detectors, args.max_train = [0], ['iforest', 'ae_small'], 5000
        splits = [('nsl_kdd', 'lofo', 'Probe')]
    if args.repro:
        args.out = args.out.replace('results.csv', 'repro.csv')
        args.detectors, args.inputs, args.train_sets = ['ae_p1', 'ae_small'], ['shap', 'raw'], ['all', 'benign']
        splits = [('nsl_kdd', 'official', None)]
    if args.families:
        splits = [s for s in splits if (s[2] or s[1]) in args.families]

    done = done_keys(args.out)
    for dataset, protocol, family in splits:
        for seed in args.seeds:
            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] {dataset}/{protocol}/{family or ''} seed {seed}", flush=True)
            run_split(dataset, protocol, family, seed, args, done)
            print(f"  done in {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
