# unsw_nb15 / lofo / Reconnaissance / seed 2 / detector-fit subsample 50000

### Q1. Raw vs SHAP input to the second stage (threshold = 2% FPR on validation, OR-fused with XGBoost)

| train data   | input   | detector     |   zero-day AUROC |   AUROC among XGB-missed |   zero-day recall (2nd stage) |   fused zero-day recall |   fused FPR |   fused F1 |   fused MCC |
|:-------------|:--------|:-------------|-----------------:|-------------------------:|------------------------------:|------------------------:|------------:|-----------:|------------:|
| -            | -       | xgboost_only |          nan     |                  nan     |                       nan     |                   0.997 |       0.255 |      0.896 |       0.758 |
| benign       | raw     | ae_small     |            0.339 |                    0.680 |                         0.017 |                   0.997 |       0.269 |      0.892 |       0.749 |
| benign       | raw     | iforest      |            0.377 |                    0.509 |                         0.010 |                   0.997 |       0.278 |      0.888 |       0.740 |
| benign       | shap    | ae_small     |            0.326 |                    0.826 |                         0.001 |                   0.998 |       0.270 |      0.892 |       0.750 |
| benign       | shap    | iforest      |            0.480 |                    0.740 |                         0.000 |                   0.998 |       0.285 |      0.887 |       0.736 |
| benign       | concat  | ae_small     |            0.332 |                    0.751 |                         0.016 |                   0.997 |       0.278 |      0.889 |       0.741 |
| benign       | concat  | iforest      |            0.408 |                    0.652 |                         0.000 |                   0.998 |       0.286 |      0.886 |       0.734 |
| all          | raw     | ae_small     |            0.474 |                    0.419 |                         0.003 |                   0.997 |       0.265 |      0.893 |       0.751 |
| all          | raw     | iforest      |            0.311 |                    0.238 |                         0.003 |                   0.997 |       0.277 |      0.889 |       0.741 |
| all          | shap    | ae_small     |            0.522 |                    0.783 |                         0.001 |                   0.998 |       0.269 |      0.892 |       0.749 |
| all          | shap    | iforest      |            0.444 |                    0.719 |                         0.001 |                   0.998 |       0.310 |      0.879 |       0.716 |
| all          | concat  | ae_small     |            0.481 |                    0.703 |                         0.001 |                   0.998 |       0.279 |      0.889 |       0.742 |
| all          | concat  | iforest      |            0.385 |                    0.591 |                         0.000 |                   0.998 |       0.290 |      0.884 |       0.731 |

Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.

### Q2. Feature reduction (Nugraha et al.)

| features             |   XGB acc |   XGB zd recall |   SHAP ms/flow |   AE(SHAP) zd AUROC |
|:---------------------|----------:|----------------:|---------------:|--------------------:|
| all 42               |     0.874 |           0.997 |          1.022 |               0.326 |
| top 10 by mean|SHAP| |     0.867 |           0.988 |          1.067 |               0.534 |

Top-10 features: sttl, proto, service, sbytes, ct_dst_sport_ltm, dload, smean, ct_state_ttl, ct_srv_src, ct_srv_dst

### Q3. SHAP vs LIME agreement on 200 random test flows

|   mean top-5 overlap |   flows with overlap >= 0.6 |   top-1 feature agrees |   LIME s/flow |   SHAP ms/flow |
|---------------------:|----------------------------:|-----------------------:|--------------:|---------------:|
|                0.434 |                       0.300 |                  0.280 |         0.039 |          1.022 |

### Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)

|   flows analysed |   score drop, top-5 abnormal feats |   score drop, random 5 feats |   stability (top-5 overlap, 2 seeds) |   sparsity (error share in top-5) |
|-----------------:|-----------------------------------:|-----------------------------:|-------------------------------------:|----------------------------------:|
|            3.000 |                              0.368 |                       -0.084 |                                0.667 |                             0.418 |

Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than random ones.

### Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows

| feature        |   times in top-5 |
|:---------------|-----------------:|
| rate           |                2 |
| ackdat         |                2 |
| sjit           |                2 |
| ct_dst_src_ltm |                2 |
| dur            |                1 |
| stcpb          |                1 |

### Q5a. SHAP-guided evasion of XGBoost (300 caught attacks; up to 5 controllable features moved to benign median)

|   XGBoost evasion success |   median features changed (evaded) |
|--------------------------:|-----------------------------------:|
|                     0.023 |                              3.000 |

### Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)

| input   | detector   |   evaded attacks caught by 2nd stage |   same attacks before evasion |   benign flows flagged |
|:--------|:-----------|-------------------------------------:|------------------------------:|-----------------------:|
| raw     | ae_small   |                                0.143 |                         0.143 |                  0.019 |
| raw     | iforest    |                                0.000 |                         0.000 |                  0.026 |
| shap    | ae_small   |                                0.429 |                         0.000 |                  0.024 |
| shap    | iforest    |                                0.000 |                         0.000 |                  0.036 |
| concat  | ae_small   |                                0.143 |                         0.000 |                  0.040 |
| concat  | iforest    |                                0.000 |                         0.000 |                  0.037 |

### Q6. Cost

| component                                  |   cost |
|:-------------------------------------------|-------:|
| TreeSHAP, all feat (ms/flow, 8 workers)    |  1.022 |
| TreeSHAP, top-10 feat (ms/flow, 8 workers) |  1.067 |
| LIME (ms/flow, 1 core)                     | 38.533 |
| ae_small fit (s)                           | 27.220 |
| ae_small score (ms/flow)                   |  0.001 |
| iforest fit (s)                            |  1.337 |
| iforest score (ms/flow)                    |  0.006 |
