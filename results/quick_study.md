# Pilot study: NSL-KDD official split, seed 0

### Q1. Raw vs SHAP input to the second stage (threshold = 2% FPR on validation, OR-fused with XGBoost)

| train data   | input   | detector     |   zero-day AUROC |   AUROC among XGB-missed |   zero-day recall (2nd stage) |   fused zero-day recall |   fused FPR |   fused F1 |   fused MCC |
|:-------------|:--------|:-------------|-----------------:|-------------------------:|------------------------------:|------------------------:|------------:|-----------:|------------:|
| -            | -       | XGBoost only |          nan     |                  nan     |                       nan     |                   0.379 |       0.027 |      0.786 |       0.643 |
| benign       | raw     | ae_small     |            0.747 |                    0.937 |                         0.682 |                   0.737 |       0.033 |      0.866 |       0.745 |
| benign       | raw     | iforest      |            0.682 |                    0.913 |                         0.497 |                   0.626 |       0.032 |      0.836 |       0.703 |
| benign       | shap    | ae_small     |            0.671 |                    0.977 |                         0.766 |                   0.766 |       0.044 |      0.890 |       0.777 |
| benign       | shap    | iforest      |            0.736 |                    0.968 |                         0.753 |                   0.754 |       0.039 |      0.867 |       0.744 |
| benign       | concat  | ae_small     |            0.661 |                    0.963 |                         0.746 |                   0.746 |       0.044 |      0.880 |       0.762 |
| benign       | concat  | iforest      |            0.707 |                    0.955 |                         0.722 |                   0.729 |       0.031 |      0.859 |       0.735 |
| all          | raw     | ae_small     |            0.888 |                    0.931 |                         0.442 |                   0.600 |       0.034 |      0.849 |       0.720 |
| all          | raw     | iforest      |            0.821 |                    0.903 |                         0.429 |                   0.530 |       0.031 |      0.823 |       0.687 |
| all          | shap    | ae_small     |            0.895 |                    0.940 |                         0.709 |                   0.785 |       0.048 |      0.888 |       0.772 |
| all          | shap    | iforest      |            0.882 |                    0.962 |                         0.633 |                   0.704 |       0.039 |      0.867 |       0.744 |
| all          | concat  | ae_small     |            0.903 |                    0.939 |                         0.627 |                   0.720 |       0.036 |      0.872 |       0.753 |
| all          | concat  | iforest      |            0.860 |                    0.947 |                         0.443 |                   0.559 |       0.030 |      0.825 |       0.690 |

Zero-day AUROC = anomaly score, new attacks vs all other test flows. 'Among XGB-missed' restricts to flows XGBoost passes as normal.

### Q2. Feature reduction (Nugraha et al.)

| features             |   XGB acc |   XGB zd recall |   SHAP ms/flow |   AE(SHAP) zd AUROC |
|:---------------------|----------:|----------------:|---------------:|--------------------:|
| all 41               |     0.795 |           0.379 |          0.844 |               0.671 |
| top 10 by mean|SHAP| |     0.786 |           0.335 |          0.947 |               0.676 |

Top-10 features: src_bytes, dst_host_srv_count, dst_bytes, service, dst_host_serror_rate, dst_host_same_srv_rate, dst_host_diff_srv_rate, count, dst_host_srv_serror_rate, dst_host_rerror_rate

### Q3. SHAP vs LIME agreement on 200 random test flows

|   mean top-5 overlap |   flows with overlap >= 0.6 |   top-1 feature agrees |   LIME s/flow |   SHAP ms/flow |
|---------------------:|----------------------------:|-----------------------:|--------------:|---------------:|
|                0.501 |                       0.495 |                  0.815 |         0.037 |          0.844 |

Overlap < 0.5 is flagged for analyst review in Nugraha et al.; here we report the raw overlap.

### Q4. Explaining the alarm: SHAP-AE per-feature reconstruction error (zero-day flows flagged at 2% FPR)

|   flows analysed |   score drop, top-5 abnormal feats |   score drop, random 5 feats |   stability (top-5 overlap, 2 seeds) |   sparsity (error share in top-5 of 41) |
|-----------------:|-----------------------------------:|-----------------------------:|-------------------------------------:|----------------------------------------:|
|          500.000 |                              0.459 |                        0.052 |                                0.728 |                                   0.532 |

Deletion test: replacing the top-5 abnormal SHAP entries with benign means should lower the anomaly score much more than replacing random ones.

### Q4b. Features most often named as 'abnormal explanation' for flagged zero-day flows

| feature              |   times in top-5 |
|:---------------------|-----------------:|
| dst_host_rerror_rate |              277 |
| diff_srv_rate        |              213 |
| srv_rerror_rate      |              209 |
| srv_serror_rate      |              202 |
| src_bytes            |              189 |
| rerror_rate          |              188 |

### Q5a. SHAP-guided evasion of XGBoost (300 attacks XGBoost caught; up to 5 controllable features moved to benign median)

|   XGBoost evasion success |   median features changed (evaded) |
|--------------------------:|-----------------------------------:|
|                     0.890 |                              2.000 |

### Q5b. Fraction of evaded attacks still caught by the second stage (2% FPR budget)

| input   | detector   |   evaded attacks caught by 2nd stage |   same attacks before evasion |   benign flows flagged |
|:--------|:-----------|-------------------------------------:|------------------------------:|-----------------------:|
| raw     | ae_small   |                                0.824 |                         0.843 |                  0.020 |
| raw     | iforest    |                                0.783 |                         0.816 |                  0.018 |
| shap    | ae_small   |                                0.948 |                         1.000 |                  0.044 |
| shap    | iforest    |                                0.693 |                         0.996 |                  0.039 |
| concat  | ae_small   |                                0.906 |                         1.000 |                  0.044 |
| concat  | iforest    |                                0.757 |                         0.918 |                  0.026 |

### Q6. Cost

| component                                |   cost |
|:-----------------------------------------|-------:|
| XGBoost fit (s)                          |  1.709 |
| TreeSHAP, 41 feat (ms/flow, 8 cores)     |  0.844 |
| TreeSHAP, top-10 feat (ms/flow, 8 cores) |  0.947 |
| LIME (ms/flow, 1 core)                   | 37.037 |
| ae_small fit (s)                         | 68.738 |
| ae_small score (ms/flow)                 |  0.001 |
| iforest fit (s)                          |  1.509 |
| iforest score (ms/flow)                  |  0.008 |
