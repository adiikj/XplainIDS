# unsw_nb15 / lofo / DoS / seed 2 / detector-fit subsample 50000

### Q1. Raw vs SHAP input to the second stage (threshold = 2% FPR on validation, OR-fused with XGBoost)

| train data   | input   | detector     |   zero-day AUROC |   AUROC among XGB-missed |   zero-day recall (2nd stage) |   fused zero-day recall |   fused FPR |   fused F1 |   fused MCC |
|:-------------|:--------|:-------------|-----------------:|-------------------------:|------------------------------:|------------------------:|------------:|-----------:|------------:|
| -            | -       | xgboost_only |          nan     |                  nan     |                       nan     |                   0.999 |       0.261 |      0.894 |       0.755 |
| benign       | raw     | ae_small     |            0.639 |                    0.991 |                         0.104 |                   1.000 |       0.271 |      0.892 |       0.748 |
| benign       | raw     | iforest      |            0.621 |                    0.817 |                         0.043 |                   0.999 |       0.285 |      0.886 |       0.735 |
| benign       | shap    | ae_small     |            0.517 |                    0.977 |                         0.009 |                   1.000 |       0.311 |      0.879 |       0.717 |
| benign       | shap    | iforest      |            0.440 |                    0.969 |                         0.004 |                   1.000 |       0.299 |      0.882 |       0.725 |
| benign       | concat  | ae_small     |            0.632 |                    0.981 |                         0.146 |                   1.000 |       0.281 |      0.888 |       0.741 |
| benign       | concat  | iforest      |            0.566 |                    0.900 |                         0.005 |                   0.999 |       0.290 |      0.885 |       0.733 |
| all          | raw     | ae_small     |            0.387 |                    0.829 |                         0.016 |                   0.999 |       0.292 |      0.884 |       0.730 |
| all          | raw     | iforest      |            0.445 |                    0.557 |                         0.013 |                   0.999 |       0.283 |      0.887 |       0.737 |
| all          | shap    | ae_small     |            0.301 |                    0.971 |                         0.002 |                   1.000 |       0.284 |      0.887 |       0.738 |
| all          | shap    | iforest      |            0.270 |                    0.951 |                         0.001 |                   0.999 |       0.298 |      0.882 |       0.726 |
| all          | concat  | ae_small     |            0.371 |                    0.980 |                         0.012 |                   1.000 |       0.302 |      0.882 |       0.724 |
| all          | concat  | iforest      |            0.349 |                    0.876 |                         0.001 |                   0.999 |       0.299 |      0.882 |       0.725 |

Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.

### Q2. Feature reduction (Nugraha et al.)

| features             |   XGB acc |   XGB zd recall |   SHAP ms/flow |   AE(SHAP) zd AUROC |
|:---------------------|----------:|----------------:|---------------:|--------------------:|
| all 42               |     0.872 |           0.999 |          1.061 |               0.517 |
| top 10 by mean|SHAP| |     0.869 |           0.999 |          1.014 |               0.678 |

Top-10 features: sttl, proto, service, ct_dst_sport_ltm, ct_srv_dst, smean, ct_srv_src, sbytes, ct_dst_src_ltm, ct_state_ttl

### Q3. SHAP vs LIME agreement on 200 random test flows

|   mean top-5 overlap |   flows with overlap >= 0.6 |   top-1 feature agrees |   LIME s/flow |   SHAP ms/flow |
|---------------------:|----------------------------:|-----------------------:|--------------:|---------------:|
|                0.488 |                       0.460 |                  0.135 |         0.038 |          1.061 |

### Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)

|   flows analysed |   score drop, top-5 abnormal feats |   score drop, random 5 feats |   stability (top-5 overlap, 2 seeds) |   sparsity (error share in top-5) |
|-----------------:|-----------------------------------:|-----------------------------:|-------------------------------------:|----------------------------------:|
|           37.000 |                              0.348 |                        0.061 |                                0.476 |                             0.411 |

Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than random ones.

### Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows

| feature      |   times in top-5 |
|:-------------|-----------------:|
| ct_srv_dst   |               19 |
| dur          |               15 |
| sinpkt       |               15 |
| ct_state_ttl |               15 |
| dpkts        |               14 |
| sbytes       |               11 |

### Q5a. SHAP-guided evasion of XGBoost (300 caught attacks; up to 5 controllable features moved to benign median)

|   XGBoost evasion success |   median features changed (evaded) |
|--------------------------:|-----------------------------------:|
|                     0.037 |                              4.000 |

### Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)

| input   | detector   |   evaded attacks caught by 2nd stage |   same attacks before evasion |   benign flows flagged |
|:--------|:-----------|-------------------------------------:|------------------------------:|-----------------------:|
| raw     | ae_small   |                                0.455 |                         0.273 |                  0.015 |
| raw     | iforest    |                                0.000 |                         0.000 |                  0.027 |
| shap    | ae_small   |                                0.545 |                         0.091 |                  0.066 |
| shap    | iforest    |                                0.273 |                         0.091 |                  0.051 |
| concat  | ae_small   |                                0.545 |                         0.273 |                  0.041 |
| concat  | iforest    |                                0.000 |                         0.091 |                  0.036 |

### Q6. Cost

| component                                  |   cost |
|:-------------------------------------------|-------:|
| TreeSHAP, all feat (ms/flow, 8 workers)    |  1.061 |
| TreeSHAP, top-10 feat (ms/flow, 8 workers) |  1.014 |
| LIME (ms/flow, 1 core)                     | 37.624 |
| ae_small fit (s)                           | 28.521 |
| ae_small score (ms/flow)                   |  0.001 |
| iforest fit (s)                            |  1.228 |
| iforest score (ms/flow)                    |  0.006 |
