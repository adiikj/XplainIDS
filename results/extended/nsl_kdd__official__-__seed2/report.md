# nsl_kdd / official / - / seed 2 / detector-fit subsample 50000

### Q1. Raw vs SHAP input to the second stage (threshold = 2% FPR on validation, OR-fused with XGBoost)

| train data   | input   | detector     |   zero-day AUROC |   AUROC among XGB-missed |   zero-day recall (2nd stage) |   fused zero-day recall |   fused FPR |   fused F1 |   fused MCC |
|:-------------|:--------|:-------------|-----------------:|-------------------------:|------------------------------:|------------------------:|------------:|-----------:|------------:|
| -            | -       | xgboost_only |          nan     |                  nan     |                       nan     |                   0.325 |       0.028 |      0.776 |       0.632 |
| benign       | raw     | ae_small     |            0.674 |                    0.904 |                         0.629 |                   0.661 |       0.033 |      0.853 |       0.727 |
| benign       | raw     | iforest      |            0.696 |                    0.930 |                         0.517 |                   0.617 |       0.032 |      0.842 |       0.712 |
| benign       | shap    | ae_small     |            0.638 |                    0.975 |                         0.741 |                   0.741 |       0.046 |      0.889 |       0.775 |
| benign       | shap    | iforest      |            0.666 |                    0.963 |                         0.732 |                   0.732 |       0.041 |      0.876 |       0.757 |
| benign       | concat  | ae_small     |            0.669 |                    0.959 |                         0.728 |                   0.728 |       0.038 |      0.874 |       0.754 |
| benign       | concat  | iforest      |            0.669 |                    0.951 |                         0.705 |                   0.706 |       0.032 |      0.866 |       0.745 |
| all          | raw     | ae_small     |            0.852 |                    0.929 |                         0.517 |                   0.643 |       0.033 |      0.847 |       0.718 |
| all          | raw     | iforest      |            0.837 |                    0.922 |                         0.419 |                   0.558 |       0.032 |      0.830 |       0.696 |
| all          | shap    | ae_small     |            0.912 |                    0.969 |                         0.665 |                   0.745 |       0.045 |      0.882 |       0.764 |
| all          | shap    | iforest      |            0.881 |                    0.959 |                         0.576 |                   0.658 |       0.037 |      0.856 |       0.728 |
| all          | concat  | ae_small     |            0.914 |                    0.960 |                         0.597 |                   0.704 |       0.036 |      0.866 |       0.743 |
| all          | concat  | iforest      |            0.858 |                    0.944 |                         0.570 |                   0.628 |       0.032 |      0.848 |       0.720 |

Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.

### Q2. Feature reduction (Nugraha et al.)

| features             |   XGB acc |   XGB zd recall |   SHAP ms/flow |   AE(SHAP) zd AUROC |
|:---------------------|----------:|----------------:|---------------:|--------------------:|
| all 41               |     0.787 |           0.325 |          1.053 |               0.638 |
| top 10 by mean|SHAP| |     0.768 |           0.254 |          1.101 |               0.596 |

Top-10 features: src_bytes, dst_bytes, dst_host_srv_count, service, count, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_srv_serror_rate, dst_host_same_src_port_rate, dst_host_serror_rate

### Q3. SHAP vs LIME agreement on 200 random test flows

|   mean top-5 overlap |   flows with overlap >= 0.6 |   top-1 feature agrees |   LIME s/flow |   SHAP ms/flow |
|---------------------:|----------------------------:|-----------------------:|--------------:|---------------:|
|                0.599 |                       0.775 |                  0.695 |         0.038 |          1.053 |

### Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)

|   flows analysed |   score drop, top-5 abnormal feats |   score drop, random 5 feats |   stability (top-5 overlap, 2 seeds) |   sparsity (error share in top-5) |
|-----------------:|-----------------------------------:|-----------------------------:|-------------------------------------:|----------------------------------:|
|          500.000 |                              0.469 |                        0.059 |                                0.768 |                             0.523 |

Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than random ones.

### Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows

| feature                |   times in top-5 |
|:-----------------------|-----------------:|
| dst_host_rerror_rate   |              262 |
| src_bytes              |              215 |
| serror_rate            |              211 |
| rerror_rate            |              199 |
| srv_rerror_rate        |              197 |
| dst_host_diff_srv_rate |              178 |

### Q5a. SHAP-guided evasion of XGBoost (300 caught attacks; up to 5 controllable features moved to benign median)

|   XGBoost evasion success |   median features changed (evaded) |
|--------------------------:|-----------------------------------:|
|                     0.880 |                              2.000 |

### Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)

| input   | detector   |   evaded attacks caught by 2nd stage |   same attacks before evasion |   benign flows flagged |
|:--------|:-----------|-------------------------------------:|------------------------------:|-----------------------:|
| raw     | ae_small   |                                0.765 |                         0.780 |                  0.013 |
| raw     | iforest    |                                0.731 |                         0.765 |                  0.020 |
| shap    | ae_small   |                                0.936 |                         1.000 |                  0.046 |
| shap    | iforest    |                                0.833 |                         1.000 |                  0.041 |
| concat  | ae_small   |                                0.841 |                         1.000 |                  0.038 |
| concat  | iforest    |                                0.780 |                         0.973 |                  0.032 |

### Q6. Cost

| component                                  |   cost |
|:-------------------------------------------|-------:|
| TreeSHAP, all feat (ms/flow, 8 workers)    |  1.053 |
| TreeSHAP, top-10 feat (ms/flow, 8 workers) |  1.101 |
| LIME (ms/flow, 1 core)                     | 37.573 |
| ae_small fit (s)                           | 25.515 |
| ae_small score (ms/flow)                   |  0.001 |
| iforest fit (s)                            |  1.849 |
| iforest score (ms/flow)                    |  0.009 |
