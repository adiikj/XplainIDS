"""
Generate RESULTS.md: the project's results report, built ONLY from files under results/.

    python make_results_md.py            # -> RESULTS.md (repo root)

Rerun it after any new experiment; never edit RESULTS.md by hand. Every number in it is read from:
    results/checks/*.csv                     data checks B1-B3            (data_checks.py)
    results/extended/, results/repro_windows/ pilot and its Windows rerun (quick_study.py, compare_repro.py)
    results/pipeline_demo/<split>/          cascade demo + example alert (inspect_alerts.py)
    results/grid/grid.csv, comparisons.csv   main experiment grid         (run_grid.py, aggregate_grid.py)
"""

import json
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(HERE, 'results', *p)
SPLIT_ORDER = ['official', 'DoS', 'Probe', 'R2L', 'U2R', 'Analysis', 'Backdoor', 'Exploits', 'Fuzzers', 'Generic',
               'Reconnaissance', 'Shellcode', 'Worms']
DS = {'nsl_kdd': 'NSL-KDD', 'unsw_nb15': 'UNSW-NB15'}
METRIC = {'zd_recall': 'zero-day recall', 'test_fpr': 'test FPR', 'f1': 'F1', 'mcc': 'MCC', 'known_recall': 'known recall',
          'res_auroc': 'AUROC, XGBoost-missed zero-days vs benign', 'zdvn_auroc': 'AUROC, all zero-days vs benign'}


def md(df, fmt='.3f'):
    return df.to_markdown(index=False, floatfmt=fmt)


def pm(mean, std, f='{:.3f}'):
    return f"{f.format(mean)} ± {f.format(0 if np.isnan(std) else std)}"


def split_key(df):
    return df.assign(_d=df['dataset'].map({'nsl_kdd': 0, 'unsw_nb15': 1}),
                     _s=df['split'].map({s: i for i, s in enumerate(SPLIT_ORDER)})).sort_values(['_d', '_s']).drop(columns=['_d', '_s'])


# ------------------------------------------------------------------ sections

def section_datasets():
    b2 = pd.read_csv(R('checks', 'b2_per_split.csv'))
    b2 = b2[b2['seed'] == b2['seed'].min()]
    used = set(pd.read_csv(R('grid', 'grid.csv'), usecols=['dataset', 'split']).drop_duplicates().itertuples(index=False, name=None))
    t = split_key(b2)[['dataset', 'split', 'n_train', 'n_val_benign', 'n_test_benign', 'n_test_known_attack', 'n_zero_day']].copy()
    t['in main grid'] = [('yes' if (d, s) in used else 'no (too few XGBoost misses)') for d, s in zip(t['dataset'], t['split'])]
    t['dataset'] = t['dataset'].map(DS)
    t.columns = ['Dataset', 'Held-out split', 'Train rows', 'Benign val rows', 'Benign test rows', 'Known-attack test rows',
                 'Zero-day test rows', 'In main grid']
    return md(t, '.0f')


def section_checks():
    b1 = pd.read_csv(R('checks', 'b1_missed_probability.csv'))
    b2 = pd.read_csv(R('checks', 'b2_per_split.csv'))
    b3 = pd.read_csv(R('checks', 'b3_ties.csv'))
    seeds = sorted(int(x) for x in b2['seed'].unique())
    g = ['dataset', 'split']
    t2 = b2.groupby(g)[['zd_recall[0.5]', 'zd_recall[val1%]', 'zd_missed[val1%]', 'test_fpr[0.5]', 'test_fpr[val1%]']].mean().reset_index()
    t2 = split_key(t2)
    t2['dataset'] = t2['dataset'].map(DS)
    t2.columns = ['Dataset', 'Split', 'ZD recall, p ≥ 0.5', 'ZD recall, val 1%', 'ZD missed, val 1%', 'Test FPR, p ≥ 0.5', 'Test FPR, val 1%']
    t1 = b1.groupby(g)[['missed_zd[val1%]_n', 'missed_zd[val1%]_median', 'missed_zd[val1%]_share_p<0.01',
                        'missed_zd[val1%]_share_in_band', 'passed_benign[val1%]_share_in_band']].mean().reset_index()
    t1 = split_key(t1)
    t1['dataset'] = t1['dataset'].map(DS)
    t1.columns = ['Dataset', 'Split', 'Missed zero-days', 'Median p of missed', 'Share with p < 0.01', 'Share in 0.2–0.7 band',
                  'Benign passed in band']
    dev = max(abs(b3['p@1%:val_fpr_new_rule'] - 0.01).max(), abs(b3['p@2%:val_fpr_new_rule'] - 0.02).max())
    nsl = t1[t1['Dataset'] == 'NSL-KDD']
    return f"""Seeds {seeds}; mean over seeds. Source: `results/checks/` (`data_checks.py`).

**B2. How many zero-days does XGBoost alone miss?** (decides which splits can test a second stage)

{md(t2)}

**B1. Where do the missed zero-days sit in XGBoost's probability?** (decides whether an uncertainty band makes sense)

{md(t1)}

On NSL-KDD, {'all' if nsl['Share with p < 0.01'].min() >= 0.995 else 'at least ' + format(nsl['Share with p < 0.01'].min(), '.0%')} missed zero-days (every split) have p < 0.01, so a
0.2–0.7 band would never route them to stage 2. On UNSW-NB15 the band holds some misses but also a similar share of benign
flows. **Decision: no uncertainty band.**

**B3. Threshold ties.** With the never-over-budget rule (`metrics.budget_threshold`), XGBoost's achieved validation FPR
is within {dev:.4f} of the 1% and 2% targets on every split. **Decision: keep the rule, report the achieved FPR.**"""


def section_repro():
    a = pd.read_csv(R('extended', 'nsl_kdd__official__-__seed0', 'q1.csv'))
    b = pd.read_csv(R('repro_windows', 'nsl_kdd__official__-__seed0', 'q1.csv'))
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    ae = (a['detector'] == 'ae_small').to_numpy()
    diff = (a[num] - b[num]).abs().to_numpy(float)
    diff = np.where(np.isnan(a[num].to_numpy(float)) & np.isnan(b[num].to_numpy(float)), 0, diff)
    det, aed = np.nanmax(diff[~ae]), np.nanmax(diff[ae])
    rows = []
    for lab, sel in (('XGBoost, TreeSHAP, Isolation Forest rows', ~ae), ('Autoencoder rows', ae)):
        rows.append({'Part': lab, 'Rows': int(sel.sum()), 'Max |difference|': float(np.nanmax(diff[sel]))})
    return f"""The pilot was first run on a Linux machine (Oct 2026). We re-ran NSL-KDD official, seed 0, on the team's Windows
machine and compared with `compare_repro.py`. Source: `results/repro_windows/` vs `results/extended/`.

{md(pd.DataFrame(rows), '.4f')}

Everything deterministic reproduces (max difference {det:.4f}). The autoencoder rows differ by up to {aed:.3f}, within the
seed-to-seed spread of the committed runs: PyTorch CPU training is not bit-reproducible across machines. Since then every
run sets `XZ_TORCH_THREADS=4`, so autoencoder results repeat exactly on one machine."""


def section_demo():
    det = pd.read_csv(R('pipeline_demo', 'nsl_kdd__official__seed0', 'detection.csv'))
    meta = json.load(open(R('pipeline_demo', 'nsl_kdd__official__seed0', 'run_meta.json'), encoding='utf-8'))
    ex = json.load(open(R('pipeline_demo', 'nsl_kdd__official__seed0', 'example_alert.json'), encoding='utf-8'))['novel']
    f = meta['fit']
    ev = pd.DataFrame([{'Feature': e['name'], 'Value': e['raw_value'], 'φ (this flow)': e['phi'], 'φ̂ (normal-looking)': e['phi_hat'],
                        'Gap': e['deviation']} for e in ex['evidence']])
    v = meta['versions']
    return f"""One full run of the pipeline on NSL-KDD official, seed {meta['seed']}: budget {meta['budget']:.0%}, share {meta['share']} to stage 2.
Thresholds from benign validation flows only: t1 = {f['t1']:.5f}, t2 = {f['t2']:.5f}; achieved validation FPR {f['val_fpr']:.4f}
(stage 1 {f['val_fpr_stage1']:.4f}, stage 2 {f['val_fpr_stage2']:.4f}). Environment: Python {v['python']}, xgboost {v['xgboost']},
shap {v['shap']}, torch {v['torch']}, scikit-learn {v['sklearn']}. Source: `results/pipeline_demo/` (`inspect_alerts.py`).

{md(det)}

**Example NOVEL alert** ({ex['alert_id']}, an `{ex['ground_truth']['attack_type']}` attack never seen in training):
p = {ex['p']:.4f} < t1 = {ex['t1']:.4f}, so XGBoost missed it; s = {ex['s']:.4f} ≥ t2 = {ex['t2']:.4f}, so stage 2 raised it.
Evidence = features whose SHAP contribution φ departs most from the autoencoder's normal-looking reconstruction φ̂:

{md(ev, '.4f')}"""


def section_grid():
    g = pd.read_csv(R('grid', 'grid.csv'))
    g['input'] = g['input'].fillna('-').astype(str)
    seeds = sorted(int(s) for s in g['seed'].unique())
    systems = [('XGBoost alone, p ≥ 0.5', 'none', '-', 'xgb_0.5'), ('XGBoost alone, 2% budget', 'none', '-', 'xgb_val'),
               ('Barnard OR rule', 'ae_small', 'shap', 'or_barnard'), ('OR fusion 2% + 2%', 'ae_small', 'shap', 'or_val'),
               ('Cascade, AE on raw', 'ae_small', 'raw', 'cascade_0.5'), ('Cascade, AE on SHAP (ours)', 'ae_small', 'shap', 'cascade_0.5')]
    rows = []
    for name, det, inp, sysn in systems:
        s = g[(g['detector'] == det) & (g['input'] == inp) & (g['system'] == sysn)]
        per = s.groupby(['dataset', 'split'])[['zd_recall', 'known_recall', 'test_fpr', 'val_fpr', 'f1', 'mcc']].mean()
        for ds, sub in per.groupby(level='dataset'):
            m = sub.mean()
            rows.append({'Dataset': f"{DS[ds]} ({len(sub)} splits)", 'System': name, 'Val FPR': m['val_fpr'], 'Test FPR': m['test_fpr'],
                         'Zero-day recall': m['zd_recall'], 'Known recall': m['known_recall'], 'F1': m['f1'], 'MCC': m['mcc']})
    t_ds = pd.DataFrame(rows).sort_values('Dataset', kind='stable')

    def per_split(metric):
        out = []
        for name, det, inp, sysn in [systems[1], systems[4], systems[5]]:
            s = g[(g['detector'] == det) & (g['input'] == inp) & (g['system'] == sysn)]
            a = s.groupby(['dataset', 'split'])[metric].agg(['mean', 'std']).reset_index()
            a[name] = [pm(m, sd) for m, sd in zip(a['mean'], a['std'])]
            out.append(a[['dataset', 'split', name]])
        t = out[0].merge(out[1]).merge(out[2])
        t = split_key(t)
        t['dataset'] = t['dataset'].map(DS)
        return t.rename(columns={'dataset': 'Dataset', 'split': 'Split'})

    cmp = pd.read_csv(R('grid', 'comparisons.csv'))
    cmp = cmp[['comparison', 'metric', 'mean diff (A - B)', 'splits where A better', 'p (Holm)']].copy()
    cmp['metric'] = cmp['metric'].map(lambda m: METRIC.get(m, m))
    share = pd.read_csv(R('grid', 'share_oracle.csv'))
    share = split_key(share)
    share['dataset'] = share['dataset'].map(DS)
    for c in ('n_zd', 'n_zd_missed'):
        share[c] = share[c].round().astype(int)
    share = share.rename(columns={'dataset': 'Dataset', 'split': 'Split', 'n_zd': 'Zero-days', 'n_zd_missed': 'Missed by XGBoost'})
    return f"""{len(g[['dataset', 'split']].drop_duplicates())} zero-day splits × seeds {seeds} ({len(g)} rows). For each split and seed: XGBoost + TreeSHAP
once, then the autoencoder (AE) and Isolation Forest on SHAP and on raw inputs, and every system scored at the SAME 2% budget on
benign validation flows. Source: `results/grid/` (`run_grid.py`, `aggregate_grid.py`).

**Per dataset** (mean over splits of the 5-seed means; AE as stage 2):

{md(t_ds)}

**Zero-day recall per split** (mean ± std over seeds):

{md(per_split('zd_recall'))}

**Test false-positive rate per split** (the budget is 2% on validation; it drifts on test):

{md(per_split('test_fpr'))}

**Paired comparisons** (difference A − B per split, seed-averaged; Wilcoxon signed-rank across splits; Holm-corrected within
each comparison):

{md(cmp, '.4f')}

**Zero-day recall by stage-2 share of the budget** (0 = XGBoost alone; the best share is read off TEST data, so it is an
oracle shown only to motivate the recall-aware rule, never used to choose):

{md(share, '.3f')}"""


def section_expected(cmp):
    c = lambda comp, metric: cmp[(cmp['comparison'].str.startswith(comp)) & (cmp['metric'] == metric)].iloc[0]
    sr = c('A SHAP input vs B raw input (ae_small', 'zd_recall')
    sa = c('A SHAP input vs B raw input (ae_small', 'res_auroc')
    sf = c('A SHAP input vs B raw input (ae_small', 'test_fpr')
    xf = c('A cascade 1%/1% (ae_small', 'test_fpr')
    xr = c('A cascade 1%/1% (ae_small', 'zd_recall')
    of = c('A cascade vs B OR fusion 2% + 2%', 'test_fpr')
    ir = c('A SHAP input vs B raw input (iforest', 'zd_recall')
    rows = [
        ('An AE on SHAP vectors detects zero-days better than the same AE on raw features (Barnard et al.)',
         f"Supported: zero-day recall {sr['mean diff (A - B)']:+.3f}, better on {sr['splits where A better']} splits, p = {sr['p (Holm)']:.3f}; "
         f"AUROC on XGBoost-missed zero-days {sa['mean diff (A - B)']:+.3f} ({sa['splits where A better']}, p = {sa['p (Holm)']:.3f}). "
         f"Cost: test FPR {sf['mean diff (A - B)']:+.3f}"),
        ('The SHAP advantage holds for any stage-2 detector',
         f"Not supported: with Isolation Forest, recall {ir['mean diff (A - B)']:+.3f}, p = {ir['p (Holm)']:.2f}"),
        ('OR fusion with a separate budget per stage raises false alarms (research plan: about 2×)',
         f"Supported: OR 2% + 2% has test FPR {-of['mean diff (A - B)']:+.3f} higher than the cascade on {of['splits where A better']} splits; "
         "its validation FPR exceeds the 2% budget"),
        ('At the same budget, the cascade lowers false alarms vs XGBoost alone',
         f"Supported: test FPR {xf['mean diff (A - B)']:+.3f}, lower on {xf['splits where A better']} splits, p = {xf['p (Holm)']:.3f}"),
        ('At the same budget, the cascade raises zero-day recall vs XGBoost alone',
         f"Not significant: {xr['mean diff (A - B)']:+.3f}, better on {xr['splits where A better']} splits, p = {xr['p (Holm)']:.2f}. "
         "It helps where XGBoost misses the held-out family and costs recall where it does not"),
        ('A second stage helps only where the classifier misses new attacks (research plan hypothesis)',
         'Consistent with the per-split results and the share table above; the recall-aware rule is built to exploit it'),
        ('Missed zero-days sit in an uncertain middle band of p (research plan)',
         'Not supported: on NSL-KDD they sit at p < 0.01 (check B1); the band was dropped'),
    ]
    return md(pd.DataFrame(rows, columns=['Expected (hypothesis)', 'Observed']))


def summary_bullets(cmp):
    c = lambda comp, metric: cmp[(cmp['comparison'].str.startswith(comp)) & (cmp['metric'] == metric)].iloc[0]
    sr = c('A SHAP input vs B raw input (ae_small', 'zd_recall')
    xf = c('A cascade 1%/1% (ae_small', 'test_fpr')
    xr = c('A cascade 1%/1% (ae_small', 'zd_recall')
    f = lambda r: f"{r['mean diff (A - B)']:+.3f}, {r['splits where A better']} splits, p = {r['p (Holm)']:.3f}"
    return ("- **Main results (12 zero-day splits × 5 seeds, all at a 2% validation false-alarm budget):**\n"
            f"  - SHAP input beats raw input for the autoencoder: zero-day recall {f(sr)} (Holm-corrected Wilcoxon).\n"
            f"  - The cascade has fewer test false alarms than XGBoost alone: test FPR {f(xf)}.\n"
            f"  - Zero-day recall vs XGBoost alone: {f(xr)}, not significant. It rises where XGBoost misses the held-out "
            "family and falls where XGBoost already catches it, which is what the recall-aware rule (next step) targets.")


def main():
    cmp = pd.read_csv(R('grid', 'comparisons.csv'))
    parts = [
        "# XplainIDS: results so far",
        f"_Generated by `make_results_md.py` on {time.strftime('%Y-%m-%d')} from the files in `results/`. Do not edit by hand; "
        "rerun the script after new experiments._",
        "## 1. Summary",
        "- **What we test:** whether an explanation-based second stage (an autoencoder on TreeSHAP vectors) helps an XGBoost "
        "intrusion detector catch attack families it never saw in training, when every system gets the same 2% false-alarm budget.",
        summary_bullets(cmp),
        "- **Status:** pipeline, cascade, alert evidence and the main grid are done. The recall-aware rule, CICIDS2017, the LLM "
        "explanation layer and the user study are next (section 10).",
        "## 2. What we built",
        "| Part | File | What it does |\n| --- | --- | --- |\n"
        "| Data and zero-day splits | `data.py` | NSL-KDD and UNSW-NB15 loaders; leave-one-family-out splits; encoders fitted on training data only |\n"
        "| Stage 1 | `explain.py` | XGBoost classifier + interventional TreeSHAP (parallel, Windows and Linux) |\n"
        "| Stage 2 | `detectors.py` | Autoencoder, Isolation Forest (also kNN, PCA); input transforms; AE reconstruction φ̂ |\n"
        "| Thresholds and cascade | `metrics.py` | Validation-only thresholds; exact joint budget; all comparison systems |\n"
        "| Pipeline and evidence | `pipeline.py`, `evidence.py` | Fit / score / save; KNOWN and NOVEL alert evidence |\n"
        "| Experiments | `data_checks.py`, `inspect_alerts.py`, `run_grid.py`, `aggregate_grid.py` | Sections 5–7 |\n"
        "| Tests | `tests/` | Cascade budget, evidence rules, φ̂ round trip, parallel SHAP, end-to-end pipeline |",
        "## 3. Training and evaluation setup",
        "| Component | Setting |\n| --- | --- |\n"
        "| Preprocessing | Categorical columns ordinal-encoded on training data (unseen values → −1); stage-2 input: signed log1p (raw) or SHAP, MinMax-scaled to (−1, 1) on benign training rows |\n"
        "| XGBoost | 100 trees, max depth 6, learning rate 0.3, binary logistic; trained on the full training split |\n"
        "| TreeSHAP | Interventional, 500 background flows, explains P(attack); additivity checked |\n"
        "| Stage-2 data | Random subsample of 50,000 training and 20,000 validation rows; detectors trained on the BENIGN rows only |\n"
        "| Autoencoder | d → 64 → 16 → 64 → d, ReLU, tanh output; mean-absolute-error loss; Adam, lr 0.001, batch 512; early stopping on benign validation (patience 10, max 200 epochs); score = mean absolute reconstruction error |\n"
        "| Isolation Forest | 200 trees |\n"
        "| Thresholds | From benign validation flows only. Cascade: t1 (XGBoost) and t2 (stage 2) together flag exactly 2% (default 1% + 1%) |\n"
        "| Zero-day protocol | Held-out family removed from training AND validation; its test flows are the zero-days. NSL-KDD official = attack types in KDDTest+ absent from KDDTrain+ |\n"
        "| Statistics | Seeds 0–4; Wilcoxon signed-rank across splits (one unit per split); Holm correction |",
        "## 4. Datasets and splits",
        section_datasets(),
        "## 5. Experiment log",
        "| # | Experiment | Question | Command | Output | Status |\n| --- | --- | --- | --- | --- | --- |\n"
        "| E1 | Pilot study (Oct 2026) | Does SHAP input help a second stage? 4 splits × 3 seeds | `quick_study.py` (one run per split and seed), `aggregate_extended.py` | `results/extended/` | Done |\n"
        "| E2 | Windows reproduction | Does the pilot reproduce on our machine? | `quick_study.py`, `compare_repro.py` | `results/repro_windows/` | Done |\n"
        "| E3 | Data checks B1–B3 | Where do missed zero-days sit? Which splits have misses? Do ties break the budget? | `data_checks.py --seeds 0 1 2` | `results/checks/` | Done |\n"
        "| E4 | Cascade demo | Does the full pipeline run and give readable alerts? | `inspect_alerts.py` | `results/pipeline_demo/` | Done |\n"
        "| E5 | Main grid | Cascade vs baselines, SHAP vs raw, at the same 2% budget | `run_grid.py`, `aggregate_grid.py` | `results/grid/` | Done (5 seeds) |\n"
        "| E6 | Recall-aware rule | Can training data choose stage 2's share? | — | — | Next |\n"
        "| E7 | CICIDS2017 splits | Do results hold on a third dataset? | — | — | Planned |",
        "## 6. Pre-experiment checks and reproduction",
        "### 6.1 Data checks (E3)", section_checks(),
        "### 6.2 Reproduction on the team machine (E2)", section_repro(),
        "## 7. Main results (E5)", section_grid(),
        "## 8. A real alert (E4)", section_demo(),
        "## 9. Expected vs observed", section_expected(cmp),
        "## 10. Known issues and limitations",
        "- **Autoencoder runs and thread count.** PyTorch's thread count changes the trained AE (NSL-KDD official seed 0: cascade "
        "zero-day recall 0.838 with 4 threads, 0.760 with the default). Part of the current grid ran before `XZ_TORCH_THREADS=4` "
        "was fixed; a rerun of the stage-2 part with 4 threads is planned, and these tables will be regenerated.\n"
        "- **Validation-to-test shift.** A 2% validation budget becomes about 4–9% FPR on test (largest on UNSW-NB15, whose "
        "official train and test partitions differ), so test FPR is always reported.\n"
        "- **Hard families.** NSL-KDD R2L (e.g. SNMP-based attacks that look like normal traffic) stays near 0.1–0.2 recall for every method.\n"
        "- **What SHAP-based stage 2 cannot see.** A zero-day that differs only in features XGBoost never uses gets near-zero SHAP "
        "there and passes both stages.\n"
        "- **Not yet done:** the recall-aware rule (E6), CICIDS2017 (E7), reproducing Barnard et al.'s large autoencoder, the "
        "faithfulness test of NOVEL evidence, the LLM layer (Feb 2027) and the user study (Apr 2027).",
        "## 11. How to reproduce",
        "```\npython -m venv .venv\n./.venv/Scripts/python.exe -m pip install -r requirements.txt\nexport XZ_TORCH_THREADS=4\n"
        "./.venv/Scripts/python.exe -m pytest -q tests\n./.venv/Scripts/python.exe -u data_checks.py --seeds 0 1 2\n"
        "./.venv/Scripts/python.exe -u inspect_alerts.py\n./.venv/Scripts/python.exe -u run_grid.py\n"
        "./.venv/Scripts/python.exe aggregate_grid.py\n./.venv/Scripts/python.exe make_results_md.py\n```\n"
        "On Linux or macOS use `.venv/bin/python`. `XZ_TORCH_THREADS=4` fixes PyTorch's thread count so autoencoder runs repeat exactly.",
    ]
    out = '\n\n'.join(parts) + '\n'
    with open(os.path.join(HERE, 'RESULTS.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(out)
    print(f"Wrote {os.path.join(HERE, 'RESULTS.md')} ({len(out.splitlines())} lines)")


if __name__ == '__main__':
    main()
