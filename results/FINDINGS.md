# Pilot findings (NSL-KDD official split, one seed). Full tables: quick_study.md

1. **SHAP input helps most on flows XGBoost lets through.** AUROC among XGB-missed flows 0.94-0.98 (SHAP) vs 0.90-0.94 (raw).
   Best fused system (AE on SHAP, trained on all data, 2% validation FPR budget): zero-day recall 78.5%, F1 0.888 vs 60.0%, 0.849 for raw features
   and 37.9%, 0.786 for XGBoost alone. Overall zero-day AUROC is NOT consistently better for SHAP (benign-only AE: 0.671 SHAP vs 0.747 raw).
2. **SHAP pipelines raise more false alarms** than the budget: 3.9-4.8% fused FPR vs 3.1-3.4% raw and 2.7% XGBoost alone.
   The validation-calibrated 2% threshold does not transfer exactly to the test set (distribution shift).
3. **Top-10 SHAP features** keep most performance (XGB acc 0.786 vs 0.795; AE zero-day AUROC 0.676 vs 0.671) but recall of XGBoost drops 0.379 -> 0.335.
   SHAP time per flow did not fall (0.95 vs 0.84 ms; dominated by parallel overhead and the fixed 500-row background).
4. **SHAP and LIME agree only partly:** mean top-5 overlap 0.50; top-1 feature agrees for 81.5% of flows. LIME costs ~37 ms/flow (1 core, 1000 samples) vs 0.84 ms/flow for TreeSHAP (8 cores).
5. **Alarms can be explained cheaply:** replacing the top-5 abnormal SHAP entries (by per-feature reconstruction error) with benign means cuts the anomaly score by 45.9%
   vs 5.2% for random 5 entries; top-5 stability across two AE seeds 0.73; the top-5 carry 53% of the error. Most-flagged features: dst_host_rerror_rate, diff_srv_rate, srv_rerror_rate.
6. **SHAP-guided evasion works against XGBoost** (89% of 300 caught attacks evaded by moving <=5 controllable features to the benign median; median 2 features).
   The SHAP-based AE still flags 94.8% of the evaded attacks (raw AE: 82.4%), but SHAP IsolationForest only 69.3% (raw: 78.3%), so the benefit depends on the detector.

Caveats: one seed, one dataset split; thresholds from validation; evasion is a simple greedy attack on a hand-picked list of 14 "controllable" features.
