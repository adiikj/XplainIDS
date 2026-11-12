"""
Pre-implementation data checks B1-B3 (read-only: nothing here changes the pipeline).

B1  Where do missed zero-days sit in XGBoost's probability? Histogram + quantiles of p on missed zero-day test flows
    vs benign test flows, per split. Decides A9 (is an uncertainty band worth having?).
B2  Per-split table: zero-day test count, XGBoost zero-day recall and number missed, known-attack recall, test FPR,
    at three XGBoost thresholds (0.5 as in the pilot; 2% and 1% of benign validation flows, the cascade's t1).
    Shows which splits have real misses, which the recall-aware rule (C4) needs.
B3  Threshold ties: how many benign validation flows share the same p (or IsolationForest score), and what validation
    FPR each threshold rule actually achieves.

Only XGBoost (and, for B3, an IsolationForest on raw features) is fitted, so this takes minutes, not hours.

    python data_checks.py                       # all splits, seed 0 -> results/checks/
    python data_checks.py --seeds 0 1 2         # more seeds
    python data_checks.py --datasets nsl_kdd    # one dataset
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from data import NSL_FAMILIES, UNSW_FAMILIES, make_split
from explain import fit_xgb
from metrics import budget_threshold, threshold_at_fpr

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
BAND = (0.2, 0.7)   # the uncertainty band considered in the research plan


def splits_for(datasets):
    out = []
    if 'nsl_kdd' in datasets:
        out.append(('nsl_kdd', 'official', None))
        out += [('nsl_kdd', 'lofo', f) for f in NSL_FAMILIES]
    if 'unsw_nb15' in datasets:
        out += [('unsw_nb15', 'lofo', f) for f in UNSW_FAMILIES]
    return out


def describe_p(p, prefix):
    """Quantiles and band shares of a set of XGBoost probabilities."""
    if len(p) == 0:
        return {f'{prefix}n': 0}
    q = np.quantile(p, [0.1, 0.5, 0.9])
    return {f'{prefix}n': len(p), f'{prefix}p10': q[0], f'{prefix}median': q[1], f'{prefix}p90': q[2],
            f'{prefix}share_p<0.01': (p < 0.01).mean(), f'{prefix}share_in_band': ((p >= BAND[0]) & (p <= BAND[1])).mean()}


def tie_stats(scores_vb, fpr, prefix):
    """
    B3 for one score on benign validation flows:
      the old rule   thr = quantile(1-fpr), flag = s > thr   (metrics.threshold_at_fpr + run_experiments.evaluate_scores)
      the new rule   thr = budget_threshold(s, floor(fpr*n)), flag = s >= thr   (never exceeds the budget)
    """
    n = len(scores_vb)
    thr_q = threshold_at_fpr(scores_vb, fpr)
    thr_b = budget_threshold(scores_vb, int(np.floor(fpr * n)))
    vals, counts = np.unique(scores_vb, return_counts=True)
    return {f'{prefix}distinct_values': len(vals),
            f'{prefix}largest_tie_share': counts.max() / n,
            f'{prefix}tied_at_quantile_share': (scores_vb == thr_q).mean(),
            f'{prefix}val_fpr_old_rule(>)': (scores_vb > thr_q).mean(),
            f'{prefix}val_fpr_old_rule_if_>=': (scores_vb >= thr_q).mean(),
            f'{prefix}val_fpr_new_rule': (scores_vb >= thr_b).mean() if np.isfinite(thr_b) else 0.0}


def run(dataset, protocol, family, seed, iforest_n):
    t0 = time.time()
    s = make_split(dataset, protocol, family, seed=seed)
    model = fit_xgb(s.X_train, s.y_train, seed)
    p_val = model.predict_proba(s.X_val)[:, 1]
    p_test = model.predict_proba(s.X_test)[:, 1]
    vb, normal, zd = s.y_val == 0, s.y_test == 0, s.zd_test
    known = (s.y_test == 1) & ~zd

    thresholds = {'0.5': 0.5,
                  'val2%': budget_threshold(p_val[vb], int(np.floor(0.02 * vb.sum()))),
                  'val1%': budget_threshold(p_val[vb], int(np.floor(0.01 * vb.sum())))}
    base = dict(dataset=dataset, split=family or protocol, seed=seed, n_train=len(s.y_train), n_val_benign=int(vb.sum()),
                n_test_benign=int(normal.sum()), n_test_known_attack=int(known.sum()), n_zero_day=int(zd.sum()))

    b2 = dict(base)
    for name, t in thresholds.items():
        flag = p_test >= t
        b2.update({f'thr[{name}]': t, f'val_fpr[{name}]': (p_val[vb] >= t).mean(), f'test_fpr[{name}]': flag[normal].mean(),
                   f'known_recall[{name}]': flag[known].mean() if known.any() else np.nan,
                   f'zd_recall[{name}]': flag[zd].mean(), f'zd_missed[{name}]': int((~flag & zd).sum())})

    # B1: what the cascade's stage 2 would receive (p < t1 at 1%), and what the pilot's OR fusion passed (p < 0.5)
    b1 = dict(base)
    for name in ('0.5', 'val1%'):
        t = thresholds[name]
        b1.update(describe_p(p_test[zd & (p_test < t)], f'missed_zd[{name}]_'))
        b1.update(describe_p(p_test[normal & (p_test < t)], f'passed_benign[{name}]_'))

    # B3: ties in p, and in an IsolationForest score as a stand-in for the second-stage score
    b3 = dict(base)
    for fpr in (0.01, 0.02):
        b3.update(tie_stats(p_val[vb], fpr, f'p@{fpr:.0%}:'))
    if iforest_n:
        from sklearn.ensemble import IsolationForest
        from detectors import InputTransform
        rng = np.random.default_rng(seed)
        fit_idx = np.where(s.y_train == 0)[0]
        fit_idx = fit_idx[rng.permutation(len(fit_idx))[:iforest_n]]
        T = InputTransform('raw').fit(s.X_train[fit_idx], None)
        ifo = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(T.transform(s.X_train[fit_idx], None))
        s_vb = -ifo.score_samples(T.transform(s.X_val[vb], None))
        for fpr in (0.01, 0.02):
            b3.update(tie_stats(s_vb, fpr, f'iforest_raw@{fpr:.0%}:'))

    print(f"  {dataset}/{family or protocol} seed {seed}: zd={int(zd.sum())}, zd recall@0.5={b2['zd_recall[0.5]']:.3f}, "
          f"missed@val1%={b2['zd_missed[val1%]']} ({time.time() - t0:.0f}s)", flush=True)
    arrays = dict(p_val_benign=p_val[vb].astype(np.float32), p_test=p_test.astype(np.float32), y_test=s.y_test, zd_test=zd,
                  t_val1=thresholds['val1%'])
    return b1, b2, b3, arrays


def plot_histograms(arrays, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = list(arrays)
    cols = 5
    rows = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    bins = np.concatenate([[0], np.logspace(-6, 0, 37)])
    for ax, key in zip(axes.ravel(), keys):
        a = arrays[key]
        p, zd, normal, t1 = a['p_test'], a['zd_test'], a['y_test'] == 0, float(a['t_val1'])
        clip = lambda x: np.clip(x, 1e-6, 1)
        for sel, lab, col in ((normal & (p < t1), 'benign, passed', 'tab:blue'), (zd & (p < t1), 'zero-day, missed', 'tab:red')):
            x = clip(p[sel])
            if len(x):   # share of that group's flows per bin (log-spaced bins make density=True misleading)
                ax.hist(x, bins=bins, weights=np.full(len(x), 1 / len(x)), alpha=0.5, label=lab, color=col)
        ax.set_ylabel('share of group', fontsize=7)
        ax.axvspan(*BAND, color='grey', alpha=0.15, label='band 0.2-0.7')
        ax.axvline(t1, color='k', lw=0.8, ls='--', label='t1 (val 1%)')
        ax.set_xscale('log')
        ax.set_xlim(1e-6, 1)
        ax.set_title(f"{key}  (missed n={int((zd & (p < t1)).sum())})", fontsize=9)
        ax.set_xlabel('XGBoost p(attack), log scale (p<1e-6 shown at 1e-6)', fontsize=7)
    for ax in axes.ravel()[len(keys):]:
        ax.axis('off')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('B1: XGBoost probability of flows passed to stage 2 (test set, p < t1 at 1% benign-validation FPR)')
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['nsl_kdd', 'unsw_nb15'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--iforest-n', type=int, default=50000, help='benign rows to fit the B3 IsolationForest on (0 = skip)')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'checks'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    B1, B2, B3, arrays = [], [], [], {}
    for dataset, protocol, family in splits_for(args.datasets):
        for seed in args.seeds:
            b1, b2, b3, arr = run(dataset, protocol, family, seed, args.iforest_n)
            B1.append(b1), B2.append(b2), B3.append(b3)
            if seed == args.seeds[0]:
                arrays[f"{dataset}/{family or protocol}"] = arr

    for name, rows in (('b1_missed_probability', B1), ('b2_per_split', B2), ('b3_ties', B3)):
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f'{name}.csv'), index=False)
    plot_histograms(arrays, os.path.join(args.out, 'b1_histograms.png'))

    # short readable summary
    b1, b2, b3 = pd.DataFrame(B1), pd.DataFrame(B2), pd.DataFrame(B3)
    g = ['dataset', 'split']
    s2 = b2.groupby(g, sort=False)[['n_zero_day', 'zd_recall[0.5]', 'zd_recall[val1%]', 'zd_missed[val1%]',
                                    'test_fpr[0.5]', 'test_fpr[val1%]', 'known_recall[val1%]']].mean().reset_index()
    s1 = b1.groupby(g, sort=False)[['missed_zd[val1%]_n', 'missed_zd[val1%]_median', 'missed_zd[val1%]_share_p<0.01',
                                    'missed_zd[val1%]_share_in_band', 'passed_benign[val1%]_share_in_band']].mean().reset_index()
    cols3 = [c for c in b3.columns if c.endswith(('largest_tie_share', 'val_fpr_old_rule(>)', 'val_fpr_new_rule'))]
    s3 = b3.groupby(g, sort=False)[cols3].mean().reset_index()
    md = [f"# Data checks B1-B3 (seeds {args.seeds})\n",
          "Generated by `data_checks.py`. Every number below comes from the CSVs next to this file.\n",
          "## B2. Per-split zero-day recall and misses (XGBoost alone)\n", s2.to_markdown(index=False, floatfmt='.3f'),
          "\n\nthr[val1%] = cascade t1 (1% of benign validation flows flagged by XGBoost).\n",
          "## B1. Missed zero-days: where is their probability? (flows passed to stage 2 at t1 = val 1%)\n",
          s1.to_markdown(index=False, floatfmt='.3f'),
          f"\n\nIf `share_in_band` is small and `share_p<0.01` is large, an uncertainty band {BAND} would never send the misses "
          "to stage 2, so A9 (drop the band) stands. See `b1_histograms.png`.\n",
          "## B3. Threshold ties on benign validation flows\n", s3.to_markdown(index=False, floatfmt='.4f'),
          "\n\nOld rule: quantile threshold with `score > thr` (the pilot). New rule: `metrics.budget_threshold`, `score >= thr`, "
          "never above the budget. Target values are 0.01 and 0.02.\n"]
    open(os.path.join(args.out, 'SUMMARY.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
