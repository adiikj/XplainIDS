# nsl_kdd / official / - / seed 0 / detector-fit subsample 50000

### Q1. Raw vs SHAP input to the second stage (threshold = 2% FPR on validation, OR-fused with XGBoost)

| train data   | input   | detector     |   zero-day AUROC |   AUROC among XGB-missed |   zero-day recall (2nd stage) |   fused zero-day recall |   fused FPR |   fused F1 |   fused MCC |
|:-------------|:--------|:-------------|-----------------:|-------------------------:|------------------------------:|------------------------:|------------:|-----------:|------------:|
| -            | -       | xgboost_only |          nan     |                  nan     |                       nan     |                   0.379 |       0.027 |      0.786 |       0.643 |
| benign       | raw     | ae_small     |            0.782 |                    0.891 |                         0.598 |                   0.638 |       0.032 |      0.843 |       0.714 |
| benign       | raw     | iforest      |            0.679 |                    0.917 |                         0.542 |                   0.633 |       0.032 |      0.837 |       0.705 |
| benign       | shap    | ae_small     |            0.668 |                    0.968 |                         0.778 |                   0.778 |       0.044 |      0.906 |       0.804 |
| benign       | shap    | iforest      |            0.779 |                    0.969 |                         0.764 |                   0.764 |       0.038 |      0.886 |       0.773 |
| benign       | concat  | ae_small     |            0.683 |                    0.963 |                         0.742 |                   0.742 |       0.038 |      0.879 |       0.763 |
| benign       | concat  | iforest      |            0.709 |                    0.961 |                         0.755 |                   0.759 |       0.032 |      0.867 |       0.748 |
| all          | raw     | ae_small     |            0.846 |                    0.845 |                         0.491 |                   0.619 |       0.034 |      0.846 |       0.716 |
| all          | raw     | iforest      |            0.828 |                    0.899 |                         0.426 |                   0.518 |       0.030 |      0.817 |       0.679 |
| all          | shap    | ae_small     |            0.915 |                    0.967 |                         0.711 |                   0.787 |       0.047 |      0.903 |       0.797 |
| all          | shap    | iforest      |            0.902 |                    0.964 |                         0.641 |                   0.741 |       0.039 |      0.866 |       0.742 |
| all          | concat  | ae_small     |            0.915 |                    0.961 |                         0.599 |                   0.705 |       0.034 |      0.865 |       0.743 |
| all          | concat  | iforest      |            0.872 |                    0.951 |                         0.519 |                   0.617 |       0.030 |      0.836 |       0.704 |

Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.

### Q2. Feature reduction (Nugraha et al.)

| features             |   XGB acc |   XGB zd recall |   SHAP ms/flow |   AE(SHAP) zd AUROC |
|:---------------------|----------:|----------------:|---------------:|--------------------:|
| all 41               |     0.795 |           0.379 |          0.853 |               0.668 |
| top 10 by mean|SHAP| |     0.768 |           0.265 |          0.984 |               0.617 |

Top-10 features: src_bytes, dst_bytes, service, dst_host_srv_count, dst_host_serror_rate, dst_host_same_srv_rate, count, dst_host_diff_srv_rate, dst_host_srv_serror_rate, dst_host_same_src_port_rate

### Q3. SHAP vs LIME agreement on 200 random test flows

|   mean top-5 overlap |   flows with overlap >= 0.6 |   top-1 feature agrees |   LIME s/flow |   SHAP ms/flow |
|---------------------:|----------------------------:|-----------------------:|--------------:|---------------:|
|                0.518 |                       0.580 |                  0.755 |         0.039 |          0.853 |

### Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)

|   flows analysed |   score drop, top-5 abnormal feats |   score drop, random 5 feats |   stability (top-5 overlap, 2 seeds) |   sparsity (error share in top-5) |
|-----------------:|-----------------------------------:|-----------------------------:|-------------------------------------:|----------------------------------:|
|          500.000 |                              0.467 |                        0.043 |                                0.754 |                             0.505 |

Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than random ones.

### Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows

| feature              |   times in top-5 |
|:---------------------|-----------------:|
| dst_host_rerror_rate |              266 |
| diff_srv_rate        |              214 |
| src_bytes            |              209 |
| srv_rerror_rate      |              207 |
| duration             |              190 |
| srv_serror_rate      |              176 |

### Q5a. SHAP-guided evasion of XGBoost (300 caught attacks; up to 5 controllable features moved to benign median)

|   XGBoost evasion success |   median features changed (evaded) |
|--------------------------:|-----------------------------------:|
|                     0.873 |                              2.000 |

### Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)

| input   | detector   |   evaded attacks caught by 2nd stage |   same attacks before evasion |   benign flows flagged |
|:--------|:-----------|-------------------------------------:|------------------------------:|-----------------------:|
| raw     | ae_small   |                                0.744 |                         0.752 |                  0.010 |
| raw     | iforest    |                                0.748 |                         0.786 |                  0.020 |
| shap    | ae_small   |                                0.966 |                         1.000 |                  0.044 |
| shap    | iforest    |                                0.855 |                         1.000 |                  0.038 |
| concat  | ae_small   |                                0.927 |                         1.000 |                  0.038 |
| concat  | iforest    |                                0.760 |                         0.939 |                  0.027 |

### Q6. Cost

| component                                  |   cost |
|:-------------------------------------------|-------:|
| TreeSHAP, all feat (ms/flow, 8 workers)    |  0.853 |
| TreeSHAP, top-10 feat (ms/flow, 8 workers) |  0.984 |
| LIME (ms/flow, 1 core)                     | 39.028 |
| ae_small fit (s)                           | 33.165 |
| ae_small score (ms/flow)                   |  0.001 |
| iforest fit (s)                            |  1.694 |
| iforest score (ms/flow)                    |  0.008 |
