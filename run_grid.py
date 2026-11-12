"""
Main experiment grid (12 zero-day splits x 5 seeds). Resumable: rerun the same command to continue.

For every (split, seed): XGBoost + TreeSHAP once (cached in cache/grid_fs/), then for every stage-2 variant
(detector x input) fit on benign training SHAP/raw vectors and score every alert rule at the SAME benign-validation
budget (metrics.compare_systems): xgb_0.5, xgb_val, or_barnard, or_val, cascade_0.25 / 0.5 / 0.75.

    python run_grid.py --quick                    # smoke test: 1 split, 1 seed, 5k rows (~2-5 min)
    python run_grid.py --seeds 0                  # all default splits, seed 0 (a few hours on a laptop)
    python run_grid.py                            # 12 splits x 5 seeds (overnight, resumable)
    python aggregate_grid.py                      # -> results/grid/SUMMARY.md

Default splits: NSL-KDD official + leave-one-family-out for DoS, Probe, R2L, U2R; UNSW-NB15 leave-one-family-out for
the 7 families with >= 30 XGBoost misses at the 1% validation threshold (data check B2; Backdoor and Worms excluded).
One CSV row per (split, seed, detector, input, system). XGBoost-only systems have detector 'none'.
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from data import make_split
from detectors import InputTransform, make_detector
from explain import shap_values
from metrics import alert_stats, budget_threshold, compare_systems, score_auc
from pipeline import fit_first_stage

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'grid', 'grid.csv')
SPLITS = [('nsl_kdd', 'official', None)] + [('nsl_kdd', 'lofo', f) for f in ('DoS', 'Probe', 'R2L', 'U2R')] + \
         [('unsw_nb15', 'lofo', f) for f in ('Analysis', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Reconnaissance', 'Shellcode')]
KEY = ['dataset', 'split', 'seed', 'detector', 'input']
MIN_SUBSET = 30   # AUROC on XGBoost-missed flows is reported only with at least this many missed zero-days


def first_stage_cached(split, seed, max_fit, max_val):
    """p and SHAP for the train/val subsample and the full test set, cached per (split, seed, sizes)."""
    path = os.path.join(HERE, 'cache', 'grid_fs', f"{split.name.replace('/', '__')}__seed{seed}__n{max_fit}_{max_val}.npz")
    if os.path.exists(path):
        return dict(np.load(path)), True
    t0 = time.time()
    f = fit_first_stage(split, seed, max_fit, max_val)
    t_first = time.time() - t0
    t0 = time.time()
    phi_te = shap_values(f['explainer'], split.X_test)
    d = dict(tr=f['tr'], va=f['va'], phi_tr=f['phi_tr'], phi_va=f['phi_va'], p_va=f['p_va'].astype(np.float32),
             p_te=f['model'].predict_proba(split.X_test)[:, 1].astype(np.float32), phi_te=phi_te,
             t_first_s=np.array(t_first), t_shap_test_ms=np.array(1e3 * (time.time() - t0) / len(split.X_test)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **d)
    return d, False


def done_keys(path):
    if not os.path.exists(path):
        return set()
    return set(map(tuple, pd.read_csv(path, usecols=KEY, keep_default_na=False).astype(str).values))


def append(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if set(df.columns) <= set(header):
            df.reindex(columns=header).to_csv(path, mode='a', header=False, index=False)
            return
        df = pd.concat([pd.read_csv(path, keep_default_na=False, na_values=['']), df], ignore_index=True)
    df.to_csv(path, index=False)


def system_rows(base, split, fs, s_vb, s_te, budget, shares):
    y, zd, normal = split.y_test, split.zd_test, split.y_test == 0
    p_vb = fs['p_va'][split.y_val[fs['va']] == 0]
    systems = compare_systems(p_vb, s_vb, fs['p_te'], s_te, y, zd, budget, shares)
    if s_vb is not None:
        systems = {k: v for k, v in systems.items() if not k.startswith('xgb_')}   # XGBoost rows are written once
    rows = []
    for name, (alert, val_fpr, th) in systems.items():
        r = alert_stats(y, zd, alert)
        rows.append(dict(base, system=name, share=float(name.split('_')[1]) if name.startswith('cascade_') else np.nan,
                         val_fpr=val_fpr, test_fpr=r['fpr'], zd_recall=r['zd_recall'], known_recall=r['known_recall'],
                         precision=r['precision'], recall=r['recall'], f1=r['f1'], mcc=r['mcc'],
                         t1=th.get('t1', np.nan), t2=th.get('t2', np.nan)))
    return rows


def run_one(dataset, protocol, family, seed, a, done):
    split = make_split(dataset, protocol, family, seed=seed)
    sname = family or protocol
    todo = [(d, i) for d in a.detectors for i in a.inputs if (dataset, sname, str(seed), d, i) not in done]
    xgb_done = (dataset, sname, str(seed), 'none', '-') in done
    if not todo and xgb_done:
        return
    fs, cached = first_stage_cached(split, seed, a.max_fit, a.max_val)
    tr, va = fs['tr'], fs['va']
    y_tr, y_va = split.y_train[tr], split.y_val[va]
    b_tr, b_va = y_tr == 0, y_va == 0
    zd, normal = split.zd_test, split.y_test == 0
    p_vb = fs['p_va'][b_va]
    t1_ref = budget_threshold(p_vb, int(np.floor(a.budget / 2 * len(p_vb))))   # cascade 1%/1% t1: "XGBoost missed" below it
    missed = fs['p_te'] < t1_ref
    base0 = dict(dataset=dataset, split=sname, seed=seed, budget=a.budget, n_test=len(zd), n_zd=int(zd.sum()),
                 n_zd_missed=int((zd & missed).sum()), t_first_s=float(fs['t_first_s']),
                 t_shap_test_ms=float(fs['t_shap_test_ms']), max_fit=a.max_fit, max_val=a.max_val)

    if not xgb_done:
        b = dict(base0, detector='none', input='-',
                 zdvn_auroc=score_auc(fs['p_te'], zd, normal)['auroc'])
        append(system_rows(b, split, fs, None, None, a.budget, a.shares), a.out)
    X_tr, X_va = split.X_train[tr], split.X_val[va]
    for det_name, inp in todo:
        t0 = time.time()
        T = InputTransform(inp).fit(X_tr[b_tr], fs['phi_tr'][b_tr])
        Z_tr, Z_va, Z_te = T.transform(X_tr, fs['phi_tr']), T.transform(X_va, fs['phi_va']), T.transform(split.X_test, fs['phi_te'])
        det = make_detector(det_name, seed).fit(Z_tr[b_tr], Z_va[b_va])
        s_va, s_te = det.score(Z_va), det.score(Z_te)
        res = score_auc(s_te, zd & missed, normal & missed)['auroc'] if (zd & missed).sum() >= MIN_SUBSET else np.nan
        b = dict(base0, detector=det_name, input=inp, t_det_fit_s=time.time() - t0,
                 zdvn_auroc=score_auc(s_te, zd, normal)['auroc'], res_auroc=res,
                 n_res_benign=int((normal & missed).sum()))
        rows = system_rows(b, split, fs, s_va[b_va], s_te, a.budget, a.shares)
        append(rows, a.out)
        c = next(r for r in rows if r['system'] == 'cascade_0.5')
        print(f"  {split.name:28s} s{seed} {det_name:8s} {inp:5s} cascade_0.5: zd_recall={c['zd_recall']:.3f} "
              f"test_fpr={c['test_fpr']:.3f} | res_auroc={res:.3f} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--splits', nargs='+', default=None, help='e.g. nsl_kdd/official unsw_nb15/Exploits (default: all 12)')
    ap.add_argument('--detectors', nargs='+', default=['ae_small', 'iforest'])
    ap.add_argument('--inputs', nargs='+', default=['shap', 'raw'])
    ap.add_argument('--budget', type=float, default=0.02)
    ap.add_argument('--shares', nargs='+', type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument('--max-fit', type=int, default=50000)
    ap.add_argument('--max-val', type=int, default=20000)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--quick', action='store_true', help='smoke test: NSL-KDD official, seed 0, 5k/3k rows, own CSV')
    a = ap.parse_args()

    splits = SPLITS
    if a.splits:
        splits = [s for s in SPLITS if f"{s[0]}/{s[2] or s[1]}" in a.splits]
    if a.quick:
        splits, a.seeds, a.max_fit, a.max_val = SPLITS[:1], [0], 5000, 3000
        a.out = a.out.replace('grid.csv', 'grid_quick.csv')
    done = done_keys(a.out)
    for seed in a.seeds:            # seed-major, so a partial run already covers every split
        for dataset, protocol, family in splits:
            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] {dataset}/{family or protocol} seed {seed}", flush=True)
            run_one(dataset, protocol, family, seed, a, done)
            print(f"  done in {time.time() - t0:.0f}s", flush=True)
    print(f"Wrote {a.out}")


if __name__ == '__main__':
    main()
