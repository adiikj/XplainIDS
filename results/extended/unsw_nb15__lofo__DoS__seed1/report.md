# unsw_nb15 / lofo / DoS / seed 1 / detector-fit subsample 50000

### Q1. Raw vs SHAP input to the second stage (threshold = 2% FPR on validation, OR-fused with XGBoost)

| train data   | input   | detector     |   zero-day AUROC |   AUROC among XGB-missed |   zero-day recall (2nd stage) |   fused zero-day recall |   fused FPR |   fused F1 |   fused MCC |
|:-------------|:--------|:-------------|-----------------:|-------------------------:|------------------------------:|------------------------:|------------:|-----------:|------------:|
| -            | -       | xgboost_only |          nan     |                  nan     |                       nan     |                   0.998 |       0.260 |      0.896 |       0.759 |
| benign       | raw     | ae_small     |            0.598 |                    0.939 |                         0.074 |                   0.999 |       0.273 |      0.892 |       0.750 |
| benign       | raw     | iforest      |            0.627 |                    0.815 |                         0.037 |                   0.998 |       0.287 |      0.887 |       0.738 |
| benign       | shap    | ae_small     |            0.444 |                    0.979 |                         0.007 |                   0.999 |       0.278 |      0.891 |       0.747 |
| benign       | shap    | iforest      |            0.415 |                    0.951 |                         0.005 |                   0.999 |       0.290 |      0.887 |       0.736 |
| benign       | concat  | ae_small     |            0.621 |                    0.981 |                         0.162 |                   0.999 |       0.275 |      0.892 |       0.749 |
| benign       | concat  | iforest      |            0.584 |                    0.894 |                         0.003 |                   0.999 |       0.296 |      0.885 |       0.731 |
| all          | raw     | ae_small     |            0.400 |                    0.878 |                         0.023 |                   0.999 |       0.277 |      0.890 |       0.746 |
| all          | raw     | iforest      |            0.416 |                    0.582 |                         0.012 |                   0.998 |       0.287 |      0.887 |       0.737 |
| all          | shap    | ae_small     |            0.303 |                    0.928 |                         0.005 |                   0.999 |       0.306 |      0.881 |       0.724 |
| all          | shap    | iforest      |            0.294 |                    0.924 |                         0.001 |                   0.999 |       0.301 |      0.883 |       0.727 |
| all          | concat  | ae_small     |            0.385 |                    0.964 |                         0.018 |                   0.999 |       0.292 |      0.886 |       0.734 |
| all          | concat  | iforest      |            0.371 |                    0.852 |                         0.003 |                   0.999 |       0.294 |      0.885 |       0.732 |

Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.

### Q2. Feature reduction (Nugraha et al.)

| features             |   XGB acc |   XGB zd recall |   SHAP ms/flow |   AE(SHAP) zd AUROC |
|:---------------------|----------:|----------------:|---------------:|--------------------:|
| all 42               |     0.874 |           0.998 |          1.079 |               0.444 |
| top 10 by mean|SHAP| |     0.868 |           0.995 |          1.074 |               0.817 |

Top-10 features: sttl, proto, service, sbytes, ct_state_ttl, dload, ct_srv_dst, ct_dst_sport_ltm, smean, ct_srv_src

### Q3. SHAP vs LIME agreement on 200 random test flows

|   mean top-5 overlap |   flows with overlap >= 0.6 |   top-1 feature agrees |   LIME s/flow |   SHAP ms/flow |
|---------------------:|----------------------------:|-----------------------:|--------------:|---------------:|
|                0.419 |                       0.300 |                  0.190 |         0.038 |          1.079 |

### Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)

|   flows analysed |   score drop, top-5 abnormal feats |   score drop, random 5 feats |   stability (top-5 overlap, 2 seeds) |   sparsity (error share in top-5) |
|-----------------:|-----------------------------------:|-----------------------------:|-------------------------------------:|----------------------------------:|
|           29.000 |                              0.429 |                        0.033 |                                0.655 |                             0.446 |

Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than random ones.

### Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows

| feature      |   times in top-5 |
|:-------------|-----------------:|
| sbytes       |               16 |
| ct_state_ttl |               16 |
| dur          |               14 |
| dbytes       |                9 |
| dpkts        |                8 |
| dttl         |                7 |

### Q5a. SHAP-guided evasion of XGBoost (300 caught attacks; up to 5 controllable features moved to benign median)

|   XGBoost evasion success |   median features changed (evaded) |
|--------------------------:|-----------------------------------:|
|                     0.030 |                              2.000 |

### Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)

| input   | detector   |   evaded attacks caught by 2nd stage |   same attacks before evasion |   benign flows flagged |
|:--------|:-----------|-------------------------------------:|------------------------------:|-----------------------:|
| raw     | ae_small   |                                0.111 |                         0.111 |                  0.029 |
| raw     | iforest    |                                0.000 |                         0.000 |                  0.028 |
| shap    | ae_small   |                                0.444 |                         0.222 |                  0.029 |
| shap    | iforest    |                                0.000 |                         0.000 |                  0.044 |
| concat  | ae_small   |                                0.556 |                         0.111 |                  0.031 |
| concat  | iforest    |                                0.000 |                         0.000 |                  0.045 |

### Q6. Cost

| component                                  |   cost |
|:-------------------------------------------|-------:|
| TreeSHAP, all feat (ms/flow, 8 workers)    |  1.079 |
| TreeSHAP, top-10 feat (ms/flow, 8 workers) |  1.074 |
| LIME (ms/flow, 1 core)                     | 37.952 |
| ae_small fit (s)                           | 28.346 |
| ae_small score (ms/flow)                   |  0.001 |
| iforest fit (s)                            |  1.268 |
| iforest score (ms/flow)                    |  0.006 |
