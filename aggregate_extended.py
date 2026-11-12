"""
Aggregate results/extended/<dataset>__<protocol>__<family>__seed<k>/q*.csv into results/extended/summary.md
(mean +- std over seeds per split, plus Wilcoxon signed-rank tests of SHAP vs raw input).
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, 'results', 'extended')


def load(q):
    frames = []
    for d in sorted(glob.glob(os.path.join(ROOT, '*__seed*'))):
        f = os.path.join(d, f'{q}.csv')
        if not os.path.isdir(d) or not os.path.exists(f):
            continue
        ds, proto, fam, seed = os.path.basename(d).split('__')
        df = pd.read_csv(f)
        df.insert(0, 'split', f"{ds}/{fam if fam != '-' else proto}")
        df.insert(1, 'seed', int(seed.replace('seed', '')))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mean_std(df, keys, cols):
    g = df.groupby(keys, sort=False)[cols]
    m, s, n = g.mean(), g.std().fillna(0), g.size()
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(m[c], s[c])]
    out['seeds'] = n
    return out.reset_index()


def md(title, df, note=''):
    return f"\n### {title}\n\n" + df.to_markdown(index=False, floatfmt='.3f') + "\n" + (f"\n{note}\n" if note else '')


def main():
    parts = ["# Extended pilot: NSL-KDD official + UNSW-NB15 leave-one-family-out (DoS, Exploits, Reconnaissance)\n",
             "Mean ± std over seeds. Detectors/SHAP fit on a 50k-flow training subsample (20k validation); XGBoost on full training data.\n"]

    q1 = load('q1')
    cols = ['zero-day AUROC', 'AUROC among XGB-missed', 'fused zero-day recall', 'fused FPR', 'fused F1', 'fused MCC']
    parts.append(md('Q1. Raw vs SHAP input (2% validation FPR, OR fusion)', mean_std(q1, ['split', 'train data', 'input', 'detector'], cols)))
    pooled = q1[q1.detector != 'xgboost_only'].groupby(['train data', 'input', 'detector'])[cols].mean().reset_index()
    parts.append(md('Q1 pooled over all splits and seeds', pooled))

    rows = []
    for metric in ['AUROC among XGB-missed', 'zero-day AUROC', 'fused zero-day recall', 'fused FPR', 'fused F1', 'fused MCC']:
        for other in ['raw']:
            a = q1[q1.input == 'shap'].set_index(['split', 'seed', 'train data', 'detector'])[metric]
            b = q1[q1.input == other].set_index(['split', 'seed', 'train data', 'detector'])[metric]
            a, b = a.align(b, join='inner')
            diff = (a - b).dropna()
            p = wilcoxon(diff).pvalue if len(diff) > 5 and (diff != 0).any() else np.nan
            rows.append({'metric': metric, 'pairs': len(diff), 'mean(SHAP - raw)': diff.mean(), 'SHAP better in': (diff > 0).mean(), 'Wilcoxon p': p})
    parts.append(md('Q1 paired test: SHAP vs raw input (pairs = split x seed x train data x detector)', pd.DataFrame(rows),
                    "For fused FPR, a positive difference means SHAP raises more false alarms."))

    for q, title, keys in [('q2', 'Q2. Feature reduction', ['split', 'features']),
                           ('q3', 'Q3. SHAP vs LIME agreement', ['split']),
                           ('q4', 'Q4. Explaining the alarm (SHAP AE)', ['split']),
                           ('q5a', 'Q5a. SHAP-guided evasion of XGBoost', ['split']),
                           ('q5b', 'Q5b. Evading attacks still caught by 2nd stage', ['split', 'input', 'detector']),
                           ('q6', 'Q6. Cost', ['split', 'component'])]:
        df = load(q)
        if df.empty:
            continue
        num = [c for c in df.columns if c not in keys + ['seed'] and pd.api.types.is_numeric_dtype(df[c])]
        parts.append(md(title, mean_std(df, keys, num)))
        if q == 'q5b':
            parts.append(md('Q5b pooled over splits', df.groupby(['input', 'detector'])[num].mean().reset_index()))

    q4b = load('q4b')
    if not q4b.empty:
        top = q4b.groupby(['split', 'feature'])['times in top-5'].sum().reset_index().sort_values(['split', 'times in top-5'], ascending=[True, False])
        parts.append(md('Q4b. Most frequent abnormal-explanation features (summed over seeds)', top.groupby('split').head(5)))

    out = os.path.join(ROOT, 'summary.md')
    open(out, 'w').write(''.join(parts))
    print(''.join(parts))
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
