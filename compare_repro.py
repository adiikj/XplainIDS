"""
Compare a re-run of quick_study.py with the committed results.

    python compare_repro.py results/repro_windows/nsl_kdd__official__-__seed0 results/extended/nsl_kdd__official__-__seed0

Two kinds of numbers are checked separately:
  deterministic  XGBoost, TreeSHAP, IsolationForest, LIME (fixed seeds): must match within ±0.01 on any machine.
  AE-dependent   anything involving the PyTorch autoencoder (q1/q2 ae_small values, q4, q4b) and q5, whose
                 300 attack flows are drawn from a random stream that q4 consumes first. PyTorch CPU training is
                 not bit-reproducible across machines (thread count, CPU instruction set), so these are only
                 reported, to be judged against the seed-to-seed spread in results/extended/summary.md.
Timing columns and q6 are skipped (they depend on the machine).
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

TOL = 0.01
TIMING = ('ms/flow', 's/flow')
AE_FILES = {'q4.csv', 'q4b.csv', 'q5a.csv', 'q5b.csv'}


def diffs(a, b, rows=None):
    """Max |diff| over shared numeric, non-timing columns (NaN == NaN counts as equal)."""
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])
           and not any(t in c for t in TIMING)]
    if rows is not None:
        a, b = a[rows], b[rows]
    if not num or len(a) == 0:
        return 0.0, '-'
    x, y = a[num].to_numpy(float), b[num].to_numpy(float)
    d = np.where(np.isnan(x) & np.isnan(y), 0, np.abs(x - y))
    d = np.nan_to_num(d, nan=np.inf)
    i = np.unravel_index(np.argmax(d), d.shape)
    return float(d[i]), num[i[1]]


def main(new_dir, ref_dir):
    det_ok, det_worst, ae_worst = True, 0.0, 0.0
    for ref in sorted(glob.glob(os.path.join(ref_dir, 'q*.csv'))):
        name = os.path.basename(ref)
        if name == 'q6.csv':
            continue
        new = os.path.join(new_dir, name)
        if not os.path.exists(new):
            print(f"{name}: MISSING in {new_dir}")
            det_ok = False
            continue
        a, b = pd.read_csv(ref), pd.read_csv(new)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            print(f"{name}: shape/columns differ {a.shape} vs {b.shape}")
            det_ok = False
            continue
        if name in AE_FILES:
            m, col = diffs(a, b) if name != 'q4b.csv' else (float('nan'), '-')
            same = name != 'q4b.csv' or (a.astype(str).values == b.astype(str).values).all()
            ae_worst = max(ae_worst, m if np.isfinite(m) else 0)
            print(f"{name}: AE-dependent, max |diff| = {m:.4f} ({col})" + ('' if same else ' [feature ranking differs]'))
            continue
        if name == 'q1.csv':
            ae = (a['detector'] == 'ae_small').to_numpy()
            m_det, c_det = diffs(a, b, ~ae)
            m_ae, c_ae = diffs(a, b, ae)
        elif name == 'q2.csv':   # XGBoost columns deterministic; the AE AUROC column is not
            m_det, c_det = diffs(a.drop(columns=['AE(SHAP) zd AUROC']), b.drop(columns=['AE(SHAP) zd AUROC']))
            m_ae, c_ae = diffs(a[['AE(SHAP) zd AUROC']], b[['AE(SHAP) zd AUROC']])
        else:
            (m_det, c_det), (m_ae, c_ae) = diffs(a, b), (0.0, '-')
        det_worst, ae_worst = max(det_worst, m_det), max(ae_worst, m_ae)
        det_ok &= m_det <= TOL
        print(f"{name}: deterministic max |diff| = {m_det:.4f} ({c_det}) {'OK' if m_det <= TOL else 'DIFFERS'}"
              + (f"; AE-dependent max |diff| = {m_ae:.4f} ({c_ae})" if c_ae != '-' else ''))
    print(f"\nDETERMINISTIC (XGBoost, SHAP, IForest, LIME): {'MATCH' if det_ok else 'MISMATCH'} "
          f"(worst {det_worst:.4f}, tolerance {TOL})")
    print(f"AE-DEPENDENT: worst |diff| {ae_worst:.4f} -- compare with the seed spread (std) in results/extended/summary.md")
    return 0 if det_ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
