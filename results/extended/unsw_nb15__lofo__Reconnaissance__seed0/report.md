# unsw_nb15 / lofo / Reconnaissance / seed 0 / detector-fit subsample 50000

### Q1. Raw vs SHAP input to the second stage (threshold = 2% FPR on validation, OR-fused with XGBoost)

| train data   | input   | detector     |   zero-day AUROC |   AUROC among XGB-missed |   zero-day recall (2nd stage) |   fused zero-day recall |   fused FPR |   fused F1 |   fused MCC |
|:-------------|:--------|:-------------|-----------------:|-------------------------:|------------------------------:|------------------------:|------------:|-----------:|------------:|
| -            | -       | xgboost_only |          nan     |                  nan     |                       nan     |                   0.990 |       0.263 |      0.894 |       0.755 |
| benign       | raw     | ae_small     |            0.383 |                    0.547 |                         0.020 |                   0.990 |       0.279 |      0.889 |       0.742 |
| benign       | raw     | iforest      |            0.420 |                    0.517 |                         0.009 |                   0.990 |       0.288 |      0.886 |       0.735 |
| benign       | shap    | ae_small     |            0.437 |                    0.922 |                         0.010 |                   0.991 |       0.281 |      0.890 |       0.744 |
| benign       | shap    | iforest      |            0.688 |                    0.793 |                         0.002 |                   0.991 |       0.304 |      0.881 |       0.723 |
| benign       | concat  | ae_small     |            0.353 |                    0.829 |                         0.015 |                   0.990 |       0.276 |      0.891 |       0.746 |
| benign       | concat  | iforest      |            0.410 |                    0.684 |                         0.001 |                   0.990 |       0.294 |      0.885 |       0.732 |
| all          | raw     | ae_small     |            0.416 |                    0.352 |                         0.002 |                   0.990 |       0.276 |      0.890 |       0.745 |
| all          | raw     | iforest      |            0.347 |                    0.128 |                         0.003 |                   0.990 |       0.292 |      0.885 |       0.732 |
| all          | shap    | ae_small     |            0.596 |                    0.898 |                         0.002 |                   0.991 |       0.286 |      0.888 |       0.739 |
| all          | shap    | iforest      |            0.546 |                    0.759 |                         0.001 |                   0.991 |       0.323 |      0.875 |       0.707 |
| all          | concat  | ae_small     |            0.518 |                    0.666 |                         0.001 |                   0.991 |       0.299 |      0.883 |       0.727 |
| all          | concat  | iforest      |            0.496 |                    0.609 |                         0.000 |                   0.990 |       0.306 |      0.880 |       0.721 |

Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.

### Q2. Feature reduction (Nugraha et al.)

| features             |   XGB acc |   XGB zd recall |   SHAP ms/flow |   AE(SHAP) zd AUROC |
|:---------------------|----------:|----------------:|---------------:|--------------------:|
| all 42               |     0.872 |           0.990 |          1.037 |               0.437 |
| top 10 by mean|SHAP| |     0.874 |           0.991 |          1.080 |               0.480 |

Top-10 features: sttl, proto, service, sbytes, ct_dst_sport_ltm, smean, ct_srv_dst, ct_srv_src, dload, ct_dst_src_ltm

### Q3. SHAP vs LIME agreement on 200 random test flows

|   mean top-5 overlap |   flows with overlap >= 0.6 |   top-1 feature agrees |   LIME s/flow |   SHAP ms/flow |
|---------------------:|----------------------------:|-----------------------:|--------------:|---------------:|
|                0.452 |                       0.370 |                  0.090 |         0.040 |          1.037 |

### Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)

|   flows analysed |   score drop, top-5 abnormal feats |   score drop, random 5 feats |   stability (top-5 overlap, 2 seeds) |   sparsity (error share in top-5) |
|-----------------:|-----------------------------------:|-----------------------------:|-------------------------------------:|----------------------------------:|
|           35.000 |                              0.441 |                        0.079 |                                0.674 |                             0.520 |

Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than random ones.

### Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows

| feature        |   times in top-5 |
|:---------------|-----------------:|
| ct_dst_src_ltm |               33 |
| rate           |               29 |
| sload          |               26 |
| sbytes         |               26 |
| smean          |               26 |
| dinpkt         |                5 |

### Q5a. SHAP-guided evasion of XGBoost (300 caught attacks; up to 5 controllable features moved to benign median)

|   XGBoost evasion success |   median features changed (evaded) |
|--------------------------:|-----------------------------------:|
|                     0.023 |                              2.000 |

### Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)

| input   | detector   |   evaded attacks caught by 2nd stage |   same attacks before evasion |   benign flows flagged |
|:--------|:-----------|-------------------------------------:|------------------------------:|-----------------------:|
| raw     | ae_small   |                                0.143 |                         0.286 |                  0.034 |
| raw     | iforest    |                                0.000 |                         0.000 |                  0.027 |
| shap    | ae_small   |                                0.286 |                         0.286 |                  0.029 |
| shap    | iforest    |                                0.000 |                         0.000 |                  0.052 |
| concat  | ae_small   |                                0.429 |                         0.286 |                  0.033 |
| concat  | iforest    |                                0.000 |                         0.000 |                  0.037 |

### Q6. Cost

| component                                  |   cost |
|:-------------------------------------------|-------:|
| TreeSHAP, all feat (ms/flow, 8 workers)    |  1.037 |
| TreeSHAP, top-10 feat (ms/flow, 8 workers) |  1.080 |
| LIME (ms/flow, 1 core)                     | 39.612 |
| ae_small fit (s)                           | 13.813 |
| ae_small score (ms/flow)                   |  0.001 |
| iforest fit (s)                            |  1.325 |
| iforest score (ms/flow)                    |  0.006 |
