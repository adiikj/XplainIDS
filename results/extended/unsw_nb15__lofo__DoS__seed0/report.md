# unsw_nb15 / lofo / DoS / seed 0 / detector-fit subsample 50000

### Q1. Raw vs SHAP input to the second stage (threshold = 2% FPR on validation, OR-fused with XGBoost)

| train data   | input   | detector     |   zero-day AUROC |   AUROC among XGB-missed |   zero-day recall (2nd stage) |   fused zero-day recall |   fused FPR |   fused F1 |   fused MCC |
|:-------------|:--------|:-------------|-----------------:|-------------------------:|------------------------------:|------------------------:|------------:|-----------:|------------:|
| -            | -       | xgboost_only |          nan     |                  nan     |                       nan     |                   0.998 |       0.257 |      0.896 |       0.759 |
| benign       | raw     | ae_small     |            0.592 |                    0.944 |                         0.083 |                   0.999 |       0.276 |      0.890 |       0.746 |
| benign       | raw     | iforest      |            0.617 |                    0.824 |                         0.030 |                   0.998 |       0.284 |      0.887 |       0.738 |
| benign       | shap    | ae_small     |            0.629 |                    0.979 |                         0.016 |                   1.000 |       0.278 |      0.891 |       0.746 |
| benign       | shap    | iforest      |            0.401 |                    0.954 |                         0.002 |                   1.000 |       0.303 |      0.882 |       0.724 |
| benign       | concat  | ae_small     |            0.643 |                    0.972 |                         0.155 |                   0.999 |       0.281 |      0.889 |       0.742 |
| benign       | concat  | iforest      |            0.549 |                    0.897 |                         0.012 |                   0.999 |       0.288 |      0.887 |       0.736 |
| all          | raw     | ae_small     |            0.413 |                    0.886 |                         0.027 |                   0.999 |       0.275 |      0.891 |       0.746 |
| all          | raw     | iforest      |            0.420 |                    0.563 |                         0.013 |                   0.998 |       0.281 |      0.888 |       0.740 |
| all          | shap    | ae_small     |            0.292 |                    0.965 |                         0.003 |                   1.000 |       0.290 |      0.886 |       0.736 |
| all          | shap    | iforest      |            0.264 |                    0.909 |                         0.002 |                   0.999 |       0.306 |      0.880 |       0.720 |
| all          | concat  | ae_small     |            0.367 |                    0.970 |                         0.017 |                   0.999 |       0.281 |      0.889 |       0.743 |
| all          | concat  | iforest      |            0.322 |                    0.853 |                         0.001 |                   0.999 |       0.291 |      0.885 |       0.733 |

Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.

### Q2. Feature reduction (Nugraha et al.)

| features             |   XGB acc |   XGB zd recall |   SHAP ms/flow |   AE(SHAP) zd AUROC |
|:---------------------|----------:|----------------:|---------------:|--------------------:|
| all 42               |     0.874 |           0.998 |          0.943 |               0.629 |
| top 10 by mean|SHAP| |     0.863 |           0.997 |          0.957 |               0.716 |

Top-10 features: sttl, proto, service, ct_dst_sport_ltm, sbytes, smean, ct_srv_dst, ct_state_ttl, ct_srv_src, ct_dst_ltm

### Q3. SHAP vs LIME agreement on 200 random test flows

|   mean top-5 overlap |   flows with overlap >= 0.6 |   top-1 feature agrees |   LIME s/flow |   SHAP ms/flow |
|---------------------:|----------------------------:|-----------------------:|--------------:|---------------:|
|                0.427 |                       0.290 |                  0.550 |         0.038 |          0.943 |

### Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)

|   flows analysed |   score drop, top-5 abnormal feats |   score drop, random 5 feats |   stability (top-5 overlap, 2 seeds) |   sparsity (error share in top-5) |
|-----------------:|-----------------------------------:|-----------------------------:|-------------------------------------:|----------------------------------:|
|           64.000 |                              0.426 |                        0.015 |                                0.637 |                             0.468 |

Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than random ones.

### Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows

| feature          |   times in top-5 |
|:-----------------|-----------------:|
| sinpkt           |               40 |
| ct_dst_sport_ltm |               40 |
| proto            |               36 |
| sttl             |               36 |
| dtcpb            |               23 |
| sjit             |               19 |

### Q5a. SHAP-guided evasion of XGBoost (300 caught attacks; up to 5 controllable features moved to benign median)

|   XGBoost evasion success |   median features changed (evaded) |
|--------------------------:|-----------------------------------:|
|                     0.017 |                              3.000 |

### Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)

| input   | detector   |   evaded attacks caught by 2nd stage |   same attacks before evasion |   benign flows flagged |
|:--------|:-----------|-------------------------------------:|------------------------------:|-----------------------:|
| raw     | ae_small   |                                0.200 |                         0.000 |                  0.034 |
| raw     | iforest    |                                0.000 |                         0.000 |                  0.029 |
| shap    | ae_small   |                                0.400 |                         0.000 |                  0.040 |
| shap    | iforest    |                                0.000 |                         0.000 |                  0.051 |
| concat  | ae_small   |                                0.000 |                         0.000 |                  0.049 |
| concat  | iforest    |                                0.000 |                         0.000 |                  0.036 |

### Q6. Cost

| component                                  |   cost |
|:-------------------------------------------|-------:|
| TreeSHAP, all feat (ms/flow, 8 workers)    |  0.943 |
| TreeSHAP, top-10 feat (ms/flow, 8 workers) |  0.957 |
| LIME (ms/flow, 1 core)                     | 38.201 |
| ae_small fit (s)                           | 27.662 |
| ae_small score (ms/flow)                   |  0.001 |
| iforest fit (s)                            |  1.149 |
| iforest score (ms/flow)                    |  0.006 |
