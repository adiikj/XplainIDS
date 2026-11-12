# unsw_nb15 / lofo / Reconnaissance / seed 1 / detector-fit subsample 50000

### Q1. Raw vs SHAP input to the second stage (threshold = 2% FPR on validation, OR-fused with XGBoost)

| train data   | input   | detector     |   zero-day AUROC |   AUROC among XGB-missed |   zero-day recall (2nd stage) |   fused zero-day recall |   fused FPR |   fused F1 |   fused MCC |
|:-------------|:--------|:-------------|-----------------:|-------------------------:|------------------------------:|------------------------:|------------:|-----------:|------------:|
| -            | -       | xgboost_only |          nan     |                  nan     |                       nan     |                   0.995 |       0.261 |      0.894 |       0.754 |
| benign       | raw     | ae_small     |            0.317 |                    0.650 |                         0.011 |                   0.996 |       0.281 |      0.888 |       0.738 |
| benign       | raw     | iforest      |            0.414 |                    0.419 |                         0.011 |                   0.995 |       0.285 |      0.886 |       0.734 |
| benign       | shap    | ae_small     |            0.301 |                    0.781 |                         0.002 |                   0.997 |       0.283 |      0.888 |       0.739 |
| benign       | shap    | iforest      |            0.466 |                    0.781 |                         0.001 |                   0.996 |       0.300 |      0.882 |       0.724 |
| benign       | concat  | ae_small     |            0.332 |                    0.790 |                         0.020 |                   0.996 |       0.278 |      0.889 |       0.741 |
| benign       | concat  | iforest      |            0.435 |                    0.691 |                         0.000 |                   0.995 |       0.285 |      0.886 |       0.735 |
| all          | raw     | ae_small     |            0.414 |                    0.402 |                         0.004 |                   0.996 |       0.284 |      0.887 |       0.736 |
| all          | raw     | iforest      |            0.369 |                    0.251 |                         0.003 |                   0.995 |       0.288 |      0.885 |       0.731 |
| all          | shap    | ae_small     |            0.576 |                    0.764 |                         0.001 |                   0.996 |       0.307 |      0.879 |       0.718 |
| all          | shap    | iforest      |            0.505 |                    0.756 |                         0.001 |                   0.996 |       0.297 |      0.883 |       0.726 |
| all          | concat  | ae_small     |            0.496 |                    0.717 |                         0.003 |                   0.996 |       0.305 |      0.880 |       0.720 |
| all          | concat  | iforest      |            0.470 |                    0.650 |                         0.000 |                   0.996 |       0.296 |      0.882 |       0.726 |

Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.

### Q2. Feature reduction (Nugraha et al.)

| features             |   XGB acc |   XGB zd recall |   SHAP ms/flow |   AE(SHAP) zd AUROC |
|:---------------------|----------:|----------------:|---------------:|--------------------:|
| all 42               |     0.872 |           0.995 |          1.089 |               0.301 |
| top 10 by mean|SHAP| |     0.876 |           0.996 |          1.034 |               0.519 |

Top-10 features: sttl, proto, service, ct_dst_sport_ltm, ct_state_ttl, sbytes, dload, smean, ct_srv_dst, ct_dst_src_ltm

### Q3. SHAP vs LIME agreement on 200 random test flows

|   mean top-5 overlap |   flows with overlap >= 0.6 |   top-1 feature agrees |   LIME s/flow |   SHAP ms/flow |
|---------------------:|----------------------------:|-----------------------:|--------------:|---------------:|
|                0.440 |                       0.330 |                  0.115 |         0.040 |          1.089 |

### Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)

|   flows analysed |   score drop, top-5 abnormal feats |   score drop, random 5 feats |   stability (top-5 overlap, 2 seeds) |   sparsity (error share in top-5) |
|-----------------:|-----------------------------------:|-----------------------------:|-------------------------------------:|----------------------------------:|
|            6.000 |                              0.323 |                       -0.066 |                                0.467 |                             0.463 |

Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than random ones.

### Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows

| feature          |   times in top-5 |
|:-----------------|-----------------:|
| rate             |                4 |
| dmean            |                3 |
| tcprtt           |                3 |
| ct_dst_sport_ltm |                2 |
| dinpkt           |                2 |
| ackdat           |                2 |

### Q5a. SHAP-guided evasion of XGBoost (300 caught attacks; up to 5 controllable features moved to benign median)

|   XGBoost evasion success |   median features changed (evaded) |
|--------------------------:|-----------------------------------:|
|                     0.043 |                              3.000 |

### Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)

| input   | detector   |   evaded attacks caught by 2nd stage |   same attacks before evasion |   benign flows flagged |
|:--------|:-----------|-------------------------------------:|------------------------------:|-----------------------:|
| raw     | ae_small   |                                0.077 |                         0.000 |                  0.037 |
| raw     | iforest    |                                0.000 |                         0.000 |                  0.027 |
| shap    | ae_small   |                                0.231 |                         0.000 |                  0.037 |
| shap    | iforest    |                                0.231 |                         0.000 |                  0.048 |
| concat  | ae_small   |                                0.538 |                         0.000 |                  0.037 |
| concat  | iforest    |                                0.000 |                         0.000 |                  0.027 |

### Q6. Cost

| component                                  |   cost |
|:-------------------------------------------|-------:|
| TreeSHAP, all feat (ms/flow, 8 workers)    |  1.089 |
| TreeSHAP, top-10 feat (ms/flow, 8 workers) |  1.034 |
| LIME (ms/flow, 1 core)                     | 39.680 |
| ae_small fit (s)                           | 26.893 |
| ae_small score (ms/flow)                   |  0.001 |
| iforest fit (s)                            |  1.125 |
| iforest score (ms/flow)                    |  0.006 |
