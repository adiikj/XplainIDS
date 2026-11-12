# nsl_kdd / official / - / seed 1 / detector-fit subsample 50000

### Q1. Raw vs SHAP input to the second stage (threshold = 2% FPR on validation, OR-fused with XGBoost)

| train data   | input   | detector     |   zero-day AUROC |   AUROC among XGB-missed |   zero-day recall (2nd stage) |   fused zero-day recall |   fused FPR |   fused F1 |   fused MCC |
|:-------------|:--------|:-------------|-----------------:|-------------------------:|------------------------------:|------------------------:|------------:|-----------:|------------:|
| -            | -       | xgboost_only |          nan     |                  nan     |                       nan     |                   0.339 |       0.028 |      0.783 |       0.640 |
| benign       | raw     | ae_small     |            0.748 |                    0.936 |                         0.630 |                   0.664 |       0.034 |      0.859 |       0.734 |
| benign       | raw     | iforest      |            0.684 |                    0.926 |                         0.537 |                   0.621 |       0.032 |      0.843 |       0.712 |
| benign       | shap    | ae_small     |            0.631 |                    0.970 |                         0.793 |                   0.793 |       0.043 |      0.908 |       0.809 |
| benign       | shap    | iforest      |            0.691 |                    0.973 |                         0.799 |                   0.799 |       0.043 |      0.898 |       0.791 |
| benign       | concat  | ae_small     |            0.649 |                    0.973 |                         0.690 |                   0.690 |       0.032 |      0.863 |       0.741 |
| benign       | concat  | iforest      |            0.690 |                    0.964 |                         0.759 |                   0.760 |       0.034 |      0.879 |       0.764 |
| all          | raw     | ae_small     |            0.772 |                    0.905 |                         0.320 |                   0.481 |       0.030 |      0.813 |       0.675 |
| all          | raw     | iforest      |            0.829 |                    0.913 |                         0.409 |                   0.546 |       0.032 |      0.831 |       0.697 |
| all          | shap    | ae_small     |            0.922 |                    0.967 |                         0.704 |                   0.779 |       0.043 |      0.886 |       0.771 |
| all          | shap    | iforest      |            0.896 |                    0.969 |                         0.694 |                   0.752 |       0.038 |      0.880 |       0.764 |
| all          | concat  | ae_small     |            0.906 |                    0.958 |                         0.605 |                   0.712 |       0.036 |      0.875 |       0.757 |
| all          | concat  | iforest      |            0.871 |                    0.957 |                         0.544 |                   0.638 |       0.031 |      0.853 |       0.728 |

Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.

### Q2. Feature reduction (Nugraha et al.)

| features             |   XGB acc |   XGB zd recall |   SHAP ms/flow |   AE(SHAP) zd AUROC |
|:---------------------|----------:|----------------:|---------------:|--------------------:|
| all 41               |     0.793 |           0.339 |          1.019 |               0.631 |
| top 10 by mean|SHAP| |     0.779 |           0.304 |          1.079 |               0.597 |

Top-10 features: src_bytes, dst_bytes, dst_host_same_srv_rate, dst_host_srv_count, service, count, dst_host_serror_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_serror_rate

### Q3. SHAP vs LIME agreement on 200 random test flows

|   mean top-5 overlap |   flows with overlap >= 0.6 |   top-1 feature agrees |   LIME s/flow |   SHAP ms/flow |
|---------------------:|----------------------------:|-----------------------:|--------------:|---------------:|
|                0.545 |                       0.650 |                  0.055 |         0.040 |          1.019 |

### Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)

|   flows analysed |   score drop, top-5 abnormal feats |   score drop, random 5 feats |   stability (top-5 overlap, 2 seeds) |   sparsity (error share in top-5) |
|-----------------:|-----------------------------------:|-----------------------------:|-------------------------------------:|----------------------------------:|
|          500.000 |                              0.378 |                        0.052 |                                0.768 |                             0.496 |

Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than random ones.

### Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows

| feature                  |   times in top-5 |
|:-------------------------|-----------------:|
| dst_host_srv_serror_rate |              236 |
| dst_host_rerror_rate     |              210 |
| dst_host_diff_srv_rate   |              184 |
| srv_rerror_rate          |              175 |
| duration                 |              167 |
| rerror_rate              |              151 |

### Q5a. SHAP-guided evasion of XGBoost (300 caught attacks; up to 5 controllable features moved to benign median)

|   XGBoost evasion success |   median features changed (evaded) |
|--------------------------:|-----------------------------------:|
|                     0.933 |                              2.000 |

### Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)

| input   | detector   |   evaded attacks caught by 2nd stage |   same attacks before evasion |   benign flows flagged |
|:--------|:-----------|-------------------------------------:|------------------------------:|-----------------------:|
| raw     | ae_small   |                                0.782 |                         0.557 |                  0.024 |
| raw     | iforest    |                                0.757 |                         0.804 |                  0.019 |
| shap    | ae_small   |                                0.989 |                         1.000 |                  0.043 |
| shap    | iforest    |                                0.989 |                         1.000 |                  0.043 |
| concat  | ae_small   |                                0.825 |                         1.000 |                  0.032 |
| concat  | iforest    |                                0.896 |                         1.000 |                  0.034 |

### Q6. Cost

| component                                  |   cost |
|:-------------------------------------------|-------:|
| TreeSHAP, all feat (ms/flow, 8 workers)    |  1.019 |
| TreeSHAP, top-10 feat (ms/flow, 8 workers) |  1.079 |
| LIME (ms/flow, 1 core)                     | 40.024 |
| ae_small fit (s)                           | 43.202 |
| ae_small score (ms/flow)                   |  0.001 |
| iforest fit (s)                            |  1.758 |
| iforest score (ms/flow)                    |  0.008 |
