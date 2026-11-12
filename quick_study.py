"""
Six-question study of the XGBoost -> TreeSHAP -> anomaly-detector pipeline.

Q1 raw vs SHAP input for the second-stage detector      (Barnard et al.)
Q2 feature reduction: top-10 SHAP features              (Nugraha et al.)
Q3 SHAP vs LIME agreement                               (Nugraha et al., E-XAI)
Q4 explaining the alarm: fidelity / stability / sparsity (E-XAI, X-IDS surveys)
Q5 robustness to SHAP-guided evasion                    (Khan survey / Alani et al., E-XAI)
Q6 cost per flow                                        (Nugraha, E-XAI)

    python quick_study.py                                   # pilot: NSL-KDD official, seed 0 -> results/quick_study.md
    python quick_study.py --dataset unsw_nb15 --protocol lofo --family DoS --seed 1 --max-fit 50000 --out results/extended/...
Each run writes report.md plus one CSV per table into --out (a directory) or a single .md file (pilot default).
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from lime.lime_tabular import LimeTabularExplainer

from data import make_split
from detectors import InputTransform, make_detector
from explain import first_stage, fit_xgb, make_explainer, shap_values
from metrics import binary_stats, threshold_at_fpr
from run_experiments import evaluate_scores

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
DETECTORS = ['ae_small', 'iforest']
FPR = 0.02          # alert budget: 2% of benign validation flows
CONTROLLABLE = {    # attacker-controllable traffic statistics used by the evasion attack (Q5)
    'nsl_kdd': ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'count', 'srv_count', 'serror_rate',
                'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate'],
    'unsw_nb15': ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sload', 'dload', 'sinpkt', 'dinpkt', 'smean', 'dmean',
                  'trans_depth', 'response_body_len'],
}


class Study:
    def __init__(self, args):
        self.args, self.seed = args, args.seed
        self.rng = np.random.default_rng(args.seed)
        self.tables = []

    def emit(self, qid, title, df, note=''):
        self.tables.append((qid, title, df, note))
        print(f"\n### {title}\n\n" + df.to_markdown(index=False, floatfmt='.3f') + (f"\n\n{note}" if note else ''), flush=True)

    def write(self):
        a = self.args
        head = f"# {a.dataset} / {a.protocol} / {a.family or '-'} / seed {a.seed}" + (f" / detector-fit subsample {a.max_fit}" if a.max_fit else '') + "\n"
        md = head + ''.join(f"\n### {t}\n\n" + d.to_markdown(index=False, floatfmt='.3f') + "\n" + (f"\n{n}\n" if n else '')
                            for _, t, d, n in self.tables)
        if a.out.endswith('.md'):
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            open(a.out, 'w').write(md)
        else:
            os.makedirs(a.out, exist_ok=True)
            open(os.path.join(a.out, 'report.md'), 'w').write(md)
            for qid, _, d, _ in self.tables:
                d.to_csv(os.path.join(a.out, f'{qid}.csv'), index=False)
        print(f"\nWrote {a.out}")

    # ------------------------------------------------------------ setup
    def prepare(self):
        a = self.args
        split = make_split(a.dataset, a.protocol, a.family, seed=self.seed)
        if a.max_fit:
            # XGBoost sees the full training set; SHAP and the 2nd-stage detectors use a fixed random subsample
            model = fit_xgb(split.X_train, split.y_train, self.seed)
            tr = self.rng.choice(len(split.y_train), min(a.max_fit, len(split.y_train)), replace=False)
            va = self.rng.choice(len(split.y_val), min(a.max_fit // 2 // 1, len(split.y_val), 20000), replace=False)
            for k, idx in (('train', tr), ('val', va)):
                for f in ('X', 'y', 'fam'):
                    setattr(split, f'{f}_{k}', getattr(split, f'{f}_{k}')[idx])
            key = os.path.join(HERE, 'cache', f"sub__{split.name.replace('/', '__')}__seed{self.seed}__n{a.max_fit}.npz")
            if os.path.exists(key):
                fs = dict(np.load(key))
            else:
                ex = make_explainer(model, split.X_train, self.seed)
                t0 = time.time()
                fs = {f'shap_{k}': shap_values(ex, getattr(split, f'X_{k}')) for k in ('train', 'val', 'test')}
                fs['t_shap_per_flow'] = np.array((time.time() - t0) / sum(len(getattr(split, f'X_{k}')) for k in ('train', 'val', 'test')))
                fs['t_xgb_fit'] = np.array(np.nan)
                os.makedirs(os.path.dirname(key), exist_ok=True)
                np.savez_compressed(key, **fs)
            fs['p_test'] = model.predict_proba(split.X_test)[:, 1]
        else:
            fs = first_stage(split, self.seed)
            model = fit_xgb(split.X_train, split.y_train, self.seed)
        self.split, self.fs, self.model = split, fs, model
        self.p_test = fs['p_test']
        self.names = np.array(split.feature_names)

    def fit_eval(self, split, fs, p_test, inp, det_name, train_set):
        fit_mask = split.y_train == 0 if train_set == 'benign' else np.ones(len(split.y_train), bool)
        es_mask = split.y_val == 0 if train_set == 'benign' else np.ones(len(split.y_val), bool)
        T = InputTransform(inp).fit(split.X_train[fit_mask], fs['shap_train'][fit_mask])
        Z = {k: T.transform(getattr(split, f'X_{k}'), fs[f'shap_{k}']) for k in ('train', 'val', 'test')}
        t0 = time.time()
        det = make_detector(det_name, self.seed).fit(Z['train'][fit_mask], Z['val'][es_mask])
        t_fit = time.time() - t0
        s_fit, s_val = det.score(Z['train'][fit_mask]), det.score(Z['val'])
        t0 = time.time()
        s_test = det.score(Z['test'])
        t_score = (time.time() - t0) / len(s_test)
        row = next(r for r in evaluate_scores({}, split, p_test, s_fit, s_val, s_test) if r['policy'] == 'val_fpr2')
        return dict(row=row, det=det, T=T, Z=Z, s_test=s_test, t_fit=t_fit, t_score=t_score,
                    thr=threshold_at_fpr(s_val[split.y_val == 0], FPR))

    # ------------------------------------------------------------ questions
    def q1(self):
        s, p = self.split, self.p_test
        zd, normal = s.zd_test, s.y_test == 0
        xgb = binary_stats(s.y_test, (p >= 0.5).astype(int))
        rows = [{'train data': '-', 'input': '-', 'detector': 'xgboost_only', 'zero-day AUROC': np.nan, 'AUROC among XGB-missed': np.nan,
                 'zero-day recall (2nd stage)': np.nan, 'fused zero-day recall': (p[zd] >= 0.5).mean(),
                 'fused FPR': xgb['fpr'], 'fused F1': xgb['f1'], 'fused MCC': xgb['mcc']}]
        self.keep = {}
        for train_set in ('benign', 'all'):
            for inp in ('raw', 'shap', 'concat'):
                for d in DETECTORS:
                    k = self.fit_eval(s, self.fs, p, inp, d, train_set)
                    self.keep[(train_set, inp, d)] = k
                    r = k['row']
                    rows.append({'train data': train_set, 'input': inp, 'detector': d, 'zero-day AUROC': r['zd_auroc'],
                                 'AUROC among XGB-missed': r['res_auroc'], 'zero-day recall (2nd stage)': r['ad_zd_recall'],
                                 'fused zero-day recall': r['fused_zd_recall'], 'fused FPR': r['fused_fpr'], 'fused F1': r['fused_f1'],
                                 'fused MCC': r['fused_mcc']})
        self.emit('q1', f"Q1. Raw vs SHAP input to the second stage (threshold = {FPR:.0%} FPR on validation, OR-fused with XGBoost)",
                  pd.DataFrame(rows),
                  "Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.")

    def q2(self):
        s, fs = self.split, self.fs
        zd = s.zd_test
        top = np.argsort(np.abs(fs['shap_train']).mean(0))[::-1][:10]
        full_split = make_split(self.args.dataset, self.args.protocol, self.args.family, seed=self.seed)   # XGB always on full train
        m10 = fit_xgb(full_split.X_train[:, top], full_split.y_train, self.seed)
        ex10 = make_explainer(m10, s.X_train[:, top], self.seed)
        t0 = time.time()
        fs10 = {f'shap_{k}': shap_values(ex10, getattr(s, f'X_{k}')[:, top]) for k in ('train', 'val', 'test')}
        t10 = (time.time() - t0) / sum(len(getattr(s, f'X_{k}')) for k in ('train', 'val', 'test'))
        p10 = m10.predict_proba(s.X_test[:, top])[:, 1]
        s10 = type('S', (), {})()
        s10.__dict__.update(s.__dict__)
        for k in ('train', 'val', 'test'):
            setattr(s10, f'X_{k}', getattr(s, f'X_{k}')[:, top])
        r10 = self.fit_eval(s10, fs10, p10, 'shap', 'ae_small', 'benign')['row']
        p = self.p_test
        rows = [{'features': f'all {len(self.names)}', 'XGB acc': ((p >= .5) == s.y_test).mean(), 'XGB zd recall': (p[zd] >= .5).mean(),
                 'SHAP ms/flow': float(fs['t_shap_per_flow']) * 1e3, 'AE(SHAP) zd AUROC': self.keep[('benign', 'shap', 'ae_small')]['row']['zd_auroc']},
                {'features': 'top 10 by mean|SHAP|', 'XGB acc': ((p10 >= .5) == s.y_test).mean(), 'XGB zd recall': (p10[zd] >= .5).mean(),
                 'SHAP ms/flow': t10 * 1e3, 'AE(SHAP) zd AUROC': r10['zd_auroc']}]
        self.t_shap10 = t10
        self.emit('q2', "Q2. Feature reduction (Nugraha et al.)", pd.DataFrame(rows), "Top-10 features: " + ', '.join(self.names[top]))

    def q3(self):
        s, fs = self.split, self.fs
        idx = self.rng.choice(len(s.X_test), 200, replace=False)
        lime = LimeTabularExplainer(s.X_train[self.rng.choice(len(s.X_train), min(20000, len(s.X_train)), replace=False)],
                                    feature_names=list(self.names), discretize_continuous=False, mode='classification', random_state=self.seed)
        t0 = time.time()
        ov, top1 = [], []
        for i in idx:
            e = lime.explain_instance(s.X_test[i], self.model.predict_proba, num_features=len(self.names), num_samples=1000, labels=(1,))
            w = np.zeros(len(self.names))
            for f, v in e.as_map()[1]:
                w[f] = v
            ov.append(len(set(np.argsort(-np.abs(fs['shap_test'][i]))[:5]) & set(np.argsort(-np.abs(w))[:5])) / 5)
            top1.append(np.argmax(np.abs(fs['shap_test'][i])) == np.argmax(np.abs(w)))
        self.t_lime = (time.time() - t0) / len(idx)
        ov = np.array(ov)
        self.emit('q3', "Q3. SHAP vs LIME agreement on 200 random test flows",
                  pd.DataFrame([{'mean top-5 overlap': ov.mean(), 'flows with overlap >= 0.6': (ov >= 0.6).mean(),
                                 'top-1 feature agrees': np.mean(top1), 'LIME s/flow': self.t_lime,
                                 'SHAP ms/flow': float(fs['t_shap_per_flow']) * 1e3}]))

    @staticmethod
    @torch.no_grad()
    def per_feature_err(model, X):
        t = torch.from_numpy(X)
        return (model(t) - t).abs().numpy()

    def q4(self):
        s = self.split
        k = self.keep[('benign', 'shap', 'ae_small')]
        det, Z = k['det'], k['Z']
        flagged = np.where((k['s_test'] > k['thr']) & s.zd_test)[0]
        sub = flagged[self.rng.permutation(len(flagged))[:500]]
        err = self.per_feature_err(det.model, Z['test'][sub])
        top5 = np.argsort(-err, axis=1)[:, :5]
        sparsity = np.take_along_axis(err, top5, 1).sum(1) / err.sum(1)
        mu = Z['train'][s.y_train == 0].mean(0)
        Zs = Z['test'][sub]
        Zd, Zr = Zs.copy(), Zs.copy()
        for j in range(len(sub)):
            Zd[j, top5[j]] = mu[top5[j]]
            r5 = self.rng.choice(Zs.shape[1], 5, replace=False)
            Zr[j, r5] = mu[r5]
        s0, sd, sr = det.score(Zs), det.score(Zd), det.score(Zr)
        det2 = make_detector('ae_small', self.seed + 100).fit(Z['train'][s.y_train == 0], Z['val'][s.y_val == 0])
        top5b = np.argsort(-self.per_feature_err(det2.model, Zs), axis=1)[:, :5]
        stab = np.array([len(set(a) & set(b)) / 5 for a, b in zip(top5, top5b)])
        self.emit('q4', "Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)",
                  pd.DataFrame([{'flows analysed': len(sub), 'score drop, top-5 abnormal feats': ((s0 - sd) / s0).mean(),
                                 'score drop, random 5 feats': ((s0 - sr) / s0).mean(), 'stability (top-5 overlap, 2 seeds)': stab.mean(),
                                 'sparsity (error share in top-5)': sparsity.mean()}]),
                  "Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than random ones.")
        common = pd.Series(self.names[top5.ravel()]).value_counts().head(6)
        self.emit('q4b', "Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows",
                  pd.DataFrame({'feature': common.index, 'times in top-5': common.values}))

    def q5(self):
        s, model = self.split, self.model
        hit = np.where((s.y_test == 1) & (self.p_test >= 0.5))[0]
        sel = self.rng.choice(hit, min(300, len(hit)), replace=False)
        ctrl = np.array([list(self.names).index(c) for c in CONTROLLABLE[self.args.dataset]])
        med = np.median(s.X_train[s.y_train == 0], axis=0)
        ex = make_explainer(model, s.X_train, self.seed)
        Xa = s.X_test[sel].copy()
        changed = np.zeros(len(sel), int)
        for _ in range(5):
            act = np.where(model.predict_proba(Xa)[:, 1] >= 0.5)[0]
            if len(act) == 0:
                break
            sv = shap_values(ex, Xa[act])
            for n, j in enumerate(act):
                for f in ctrl[np.argsort(-sv[n, ctrl])]:   # most attack-pushing controllable feature not yet at the benign median
                    if Xa[j, f] != med[f]:
                        Xa[j, f] = med[f]
                        changed[j] += 1
                        break
        ev = model.predict_proba(Xa)[:, 1] < 0.5
        self.emit('q5a', "Q5a. SHAP-guided evasion of XGBoost (300 caught attacks; up to 5 controllable features moved to benign median)",
                  pd.DataFrame([{'XGBoost evasion success': ev.mean(), 'median features changed (evaded)': np.median(changed[ev]) if ev.any() else np.nan}]))
        if ev.sum() == 0:
            return
        sh_ev = shap_values(ex, Xa[ev])
        rows = []
        for inp in ('raw', 'shap', 'concat'):
            for d in DETECTORS:
                k = self.keep[('benign', inp, d)]
                Zev = k['T'].transform(Xa[ev], sh_ev)
                Zor = k['T'].transform(s.X_test[sel][ev], self.fs['shap_test'][sel][ev])
                rows.append({'input': inp, 'detector': d, 'evaded attacks caught by 2nd stage': (k['det'].score(Zev) > k['thr']).mean(),
                             'same attacks before evasion': (k['det'].score(Zor) > k['thr']).mean(),
                             'benign flows flagged': (k['s_test'][s.y_test == 0] > k['thr']).mean()})
        self.emit('q5b', "Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)", pd.DataFrame(rows))

    def q6(self):
        rows = [{'component': 'TreeSHAP, all feat (ms/flow, 8 workers)', 'cost': float(self.fs['t_shap_per_flow']) * 1e3},
                {'component': 'TreeSHAP, top-10 feat (ms/flow, 8 workers)', 'cost': self.t_shap10 * 1e3},
                {'component': 'LIME (ms/flow, 1 core)', 'cost': self.t_lime * 1e3}]
        for d in DETECTORS:
            k = self.keep[('benign', 'shap', d)]
            rows += [{'component': f'{d} fit (s)', 'cost': k['t_fit']}, {'component': f'{d} score (ms/flow)', 'cost': k['t_score'] * 1e3}]
        self.emit('q6', "Q6. Cost", pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='nsl_kdd')
    ap.add_argument('--protocol', default='official')
    ap.add_argument('--family', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-fit', type=int, default=None, help='subsample train rows for SHAP/detectors (val capped at 20k)')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'quick_study.md'))
    args = ap.parse_args()
    st = Study(args)
    st.prepare()
    s = st.split
    print(f"{s.name} seed {args.seed}: {len(s.y_train)} train rows for detectors, {s.zd_test.sum()} zero-day / {(s.y_test == 0).sum()} normal test flows")
    for q in (st.q1, st.q2, st.q3, st.q4, st.q5, st.q6):
        q()
    st.write()


if __name__ == '__main__':
    main()
