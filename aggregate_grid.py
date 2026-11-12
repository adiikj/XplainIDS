"""
Summarise results/grid/grid.csv -> results/grid/SUMMARY.md and CSVs.

Statistics: the paired difference is taken per (split, seed), averaged over seeds within each split, and the Wilcoxon
signed-rank test runs ACROSS SPLITS (one unit per split, so correlated seeds and variants do not inflate n).
Holm correction across the metrics of each comparison. Per-split tables show mean ± std over seeds.

    python aggregate_grid.py [--csv results/grid/grid.csv]
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
UNIT = ['dataset', 'split', 'seed']
METRICS = ['zd_recall', 'test_fpr', 'f1', 'mcc']
HIGHER_IS_BETTER = {'zd_recall': True, 'test_fpr': False, 'f1': True, 'mcc': True, 'known_recall': True,
                    'res_auroc': True, 'zdvn_auroc': True}
MAIN = dict(detector='ae_small', input='shap', system='cascade_0.5')


def select(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    order = np.argsort(p[ok])
    adj = np.minimum(1, np.maximum.accumulate(p[ok][order] * (ok.sum() - np.arange(ok.sum()))))
    out[np.where(ok)[0][order]] = adj
    return out


def paired(df, a, b, metrics, name):
    """Compare configuration a vs b (dicts of column filters). Returns one row per metric."""
    A = select(df, **a).set_index(UNIT)[metrics]
    B = select(df, **b).set_index(UNIT)[metrics]
    common = A.index.intersection(B.index)
    if len(common) == 0:
        return []
    d = (A.loc[common] - B.loc[common]).groupby(level=['dataset', 'split']).mean()   # one unit per split
    rows = []
    for m in metrics:
        x = d[m].dropna()
        try:
            p = wilcoxon(x).pvalue if len(x) >= 2 and (x != 0).any() else np.nan
        except ValueError:
            p = np.nan
        better = (x > 0) if HIGHER_IS_BETTER[m] else (x < 0)
        rows.append({'comparison': name, 'metric': m, 'splits': len(x), 'seeds per split': len(common) / max(len(d), 1),
                     'mean diff (A - B)': x.mean(), 'splits where A better': f"{int(better.sum())}/{len(x)}", 'p': p})
    ps = holm([r['p'] for r in rows])
    for r, ph in zip(rows, ps):
        r['p (Holm)'] = ph
    return rows


def mean_std(g, cols):
    m, s = g[cols].mean(), g[cols].std()
    return pd.Series({c: f"{m[c]:.3f} ± {0 if np.isnan(s[c]) else s[c]:.3f}" for c in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=os.path.join(HERE, 'results', 'grid', 'grid.csv'))
    ap.add_argument('--detector', default='ae_small', help='stage-2 detector for the main tables')
    a = ap.parse_args()
    MAIN['detector'] = a.detector
    out_dir = os.path.dirname(a.csv)
    df = pd.read_csv(a.csv)
    df['input'] = df['input'].fillna('-').astype(str)
    seeds = sorted(int(x) for x in df['seed'].unique())

    # 1. per-split detection table, main stage-2 variant
    keep = (df['detector'] == 'none') | ((df['detector'] == MAIN['detector']) & (df['input'] == MAIN['input']))
    t1 = (df[keep].groupby(['dataset', 'split', 'system'], sort=False)
          .apply(lambda g: mean_std(g, METRICS + ['known_recall'])).reset_index())
    order = ['xgb_0.5', 'xgb_val', 'or_barnard', 'or_val', 'cascade_0.25', 'cascade_0.5', 'cascade_0.75']
    t1['o'] = t1['system'].map({s: i for i, s in enumerate(order)})
    t1 = t1.sort_values(['dataset', 'split', 'o'], kind='stable').drop(columns='o')
    t1.to_csv(os.path.join(out_dir, 'per_split.csv'), index=False)

    # 2. paired comparisons across splits
    det_metrics = METRICS + ['known_recall']
    cmp = []
    cmp += paired(df, MAIN, dict(detector='none', system='xgb_val'), det_metrics,
                  f"A cascade 1%/1% ({a.detector}, SHAP) vs B XGBoost alone at the same 2% budget")
    cmp += paired(df, MAIN, dict(MAIN, system='or_barnard'), det_metrics, f'A cascade vs B Barnard OR rule ({a.detector}, SHAP)')
    cmp += paired(df, MAIN, dict(MAIN, system='or_val'), det_metrics, f'A cascade vs B OR fusion 2% + 2% ({a.detector}, SHAP)')
    for det in sorted(set(df['detector']) - {'none'}):
        cmp += paired(df, dict(detector=det, input='shap', system='cascade_0.5'),
                      dict(detector=det, input='raw', system='cascade_0.5'),
                      det_metrics + ['res_auroc', 'zdvn_auroc'], f'A SHAP input vs B raw input ({det}, cascade 1%/1%)')
    cmp = pd.DataFrame(cmp)
    cmp.to_csv(os.path.join(out_dir, 'comparisons.csv'), index=False)

    # 3. stage-2 share: which share would have been best per split (descriptive only: chosen on TEST data)
    sh = select(df, detector=MAIN['detector'], input=MAIN['input'])
    sh = pd.concat([sh[sh['system'].str.startswith('cascade_')], select(df, detector='none', system='xgb_val').assign(share=0.0)])
    sh_t = sh.groupby(['dataset', 'split', 'share'])['zd_recall'].mean().unstack('share')
    sh_t['best share (test, oracle)'] = sh_t.idxmax(axis=1)
    miss = df.groupby(['dataset', 'split'])[['n_zd', 'n_zd_missed']].mean().round(0)
    sh_t = sh_t.join(miss)
    sh_t.to_csv(os.path.join(out_dir, 'share_oracle.csv'))

    # 4. missed-flow AUROC subset sizes (D2)
    res = (select(df, system='cascade_0.5').groupby(['dataset', 'split', 'detector', 'input'])
           .agg(res_auroc=('res_auroc', 'mean'), n_zd_missed=('n_zd_missed', 'mean'), n_res_benign=('n_res_benign', 'mean'))
           .reset_index())

    fmt = lambda d, f='.3f': d.to_markdown(index=False, floatfmt=f)
    md = [f"# Grid summary ({df[['dataset', 'split']].drop_duplicates().shape[0]} splits, seeds {seeds})\n",
          "Generated by `aggregate_grid.py` from `grid.csv`. All thresholds come from benign validation flows; "
          "the budget is 2% of them. Test FPR is shown next to every recall because the validation budget drifts on test.\n",
          "## Paired comparisons (one unit per split, seed-averaged; Wilcoxon across splits; Holm within each comparison)\n",
          fmt(cmp, '.4f'), "\n\nWith fewer than 6 splits no Wilcoxon p-value can fall below 0.05; read the per-split columns.\n",
          f"## Per split: XGBoost baselines and {a.detector} on SHAP as stage 2 (mean ± std over seeds)\n", fmt(t1),
          "\n## Stage-2 share of the budget: zero-day recall by share (0 = XGBoost alone)\n",
          sh_t.reset_index().to_markdown(index=False, floatfmt='.3f'),
          "\n\nThe 'best share' column is picked on TEST data, so it is an oracle for reference only. The recall-aware rule "
          "must choose the share from training data (inner hold-out), never from this table.\n",
          "## AUROC on zero-days XGBoost missed (p < cascade t1), with subset sizes\n", fmt(res),
          f"\n\nres_auroc is left empty when fewer than 30 zero-days were missed.\n"]
    with open(os.path.join(out_dir, 'SUMMARY.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
