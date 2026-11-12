# Extended pilot: NSL-KDD official + UNSW-NB15 leave-one-family-out (DoS, Exploits, Reconnaissance)
Mean ± std over seeds. Detectors/SHAP fit on a 50k-flow training subsample (20k validation); XGBoost on full training data.

### Q1. Raw vs SHAP input (2% validation FPR, OR fusion)

| split                    | train data   | input   | detector     | zero-day AUROC   | AUROC among XGB-missed   | fused zero-day recall   | fused FPR     | fused F1      | fused MCC     |   seeds |
|:-------------------------|:-------------|:--------|:-------------|:-----------------|:-------------------------|:------------------------|:--------------|:--------------|:--------------|--------:|
| nsl_kdd/official         | -            | -       | xgboost_only | nan ± 0.000      | nan ± 0.000              | 0.348 ± 0.028           | 0.028 ± 0.000 | 0.782 ± 0.005 | 0.638 ± 0.006 |       3 |
| nsl_kdd/official         | benign       | raw     | ae_small     | 0.735 ± 0.055    | 0.910 ± 0.023            | 0.654 ± 0.014           | 0.033 ± 0.001 | 0.852 ± 0.008 | 0.725 ± 0.010 |       3 |
| nsl_kdd/official         | benign       | raw     | iforest      | 0.687 ± 0.009    | 0.924 ± 0.006            | 0.624 ± 0.008           | 0.032 ± 0.000 | 0.841 ± 0.003 | 0.710 ± 0.004 |       3 |
| nsl_kdd/official         | benign       | shap    | ae_small     | 0.645 ± 0.020    | 0.971 ± 0.004            | 0.770 ± 0.027           | 0.044 ± 0.002 | 0.901 ± 0.011 | 0.796 ± 0.018 |       3 |
| nsl_kdd/official         | benign       | shap    | iforest      | 0.712 ± 0.059    | 0.969 ± 0.005            | 0.765 ± 0.033           | 0.041 ± 0.003 | 0.887 ± 0.011 | 0.774 ± 0.017 |       3 |
| nsl_kdd/official         | benign       | concat  | ae_small     | 0.667 ± 0.017    | 0.965 ± 0.007            | 0.720 ± 0.027           | 0.036 ± 0.003 | 0.872 ± 0.008 | 0.752 ± 0.011 |       3 |
| nsl_kdd/official         | benign       | concat  | iforest      | 0.689 ± 0.020    | 0.959 ± 0.007            | 0.742 ± 0.031           | 0.033 ± 0.001 | 0.871 ± 0.007 | 0.752 ± 0.011 |       3 |
| nsl_kdd/official         | all          | raw     | ae_small     | 0.823 ± 0.045    | 0.893 ± 0.043            | 0.581 ± 0.087           | 0.032 ± 0.002 | 0.835 ± 0.019 | 0.703 ± 0.024 |       3 |
| nsl_kdd/official         | all          | raw     | iforest      | 0.831 ± 0.005    | 0.911 ± 0.012            | 0.541 ± 0.020           | 0.031 ± 0.001 | 0.826 ± 0.008 | 0.691 ± 0.010 |       3 |
| nsl_kdd/official         | all          | shap    | ae_small     | 0.916 ± 0.005    | 0.968 ± 0.001            | 0.770 ± 0.022           | 0.045 ± 0.002 | 0.890 ± 0.011 | 0.777 ± 0.017 |       3 |
| nsl_kdd/official         | all          | shap    | iforest      | 0.893 ± 0.011    | 0.964 ± 0.005            | 0.717 ± 0.051           | 0.038 ± 0.001 | 0.867 ± 0.012 | 0.744 ± 0.018 |       3 |
| nsl_kdd/official         | all          | concat  | ae_small     | 0.912 ± 0.005    | 0.960 ± 0.001            | 0.707 ± 0.004           | 0.035 ± 0.001 | 0.869 ± 0.006 | 0.748 ± 0.008 |       3 |
| nsl_kdd/official         | all          | concat  | iforest      | 0.867 ± 0.008    | 0.951 ± 0.006            | 0.628 ± 0.011           | 0.031 ± 0.001 | 0.846 ± 0.009 | 0.717 ± 0.012 |       3 |
| unsw_nb15/DoS            | -            | -       | xgboost_only | nan ± 0.000      | nan ± 0.000              | 0.998 ± 0.001           | 0.260 ± 0.002 | 0.895 ± 0.001 | 0.758 ± 0.002 |       3 |
| unsw_nb15/DoS            | benign       | raw     | ae_small     | 0.610 ± 0.026    | 0.958 ± 0.029            | 0.999 ± 0.000           | 0.273 ± 0.003 | 0.891 ± 0.001 | 0.748 ± 0.002 |       3 |
| unsw_nb15/DoS            | benign       | raw     | iforest      | 0.622 ± 0.005    | 0.818 ± 0.005            | 0.998 ± 0.001           | 0.285 ± 0.001 | 0.887 ± 0.001 | 0.737 ± 0.001 |       3 |
| unsw_nb15/DoS            | benign       | shap    | ae_small     | 0.530 ± 0.093    | 0.978 ± 0.001            | 1.000 ± 0.000           | 0.289 ± 0.019 | 0.887 ± 0.007 | 0.737 ± 0.017 |       3 |
| unsw_nb15/DoS            | benign       | shap    | iforest      | 0.419 ± 0.020    | 0.958 ± 0.010            | 0.999 ± 0.000           | 0.297 ± 0.007 | 0.883 ± 0.003 | 0.728 ± 0.007 |       3 |
| unsw_nb15/DoS            | benign       | concat  | ae_small     | 0.632 ± 0.011    | 0.978 ± 0.006            | 0.999 ± 0.000           | 0.279 ± 0.004 | 0.890 ± 0.002 | 0.744 ± 0.004 |       3 |
| unsw_nb15/DoS            | benign       | concat  | iforest      | 0.566 ± 0.017    | 0.897 ± 0.003            | 0.999 ± 0.000           | 0.291 ± 0.004 | 0.885 ± 0.001 | 0.734 ± 0.002 |       3 |
| unsw_nb15/DoS            | all          | raw     | ae_small     | 0.400 ± 0.013    | 0.864 ± 0.031            | 0.999 ± 0.000           | 0.282 ± 0.009 | 0.888 ± 0.004 | 0.740 ± 0.009 |       3 |
| unsw_nb15/DoS            | all          | raw     | iforest      | 0.427 ± 0.016    | 0.567 ± 0.013            | 0.998 ± 0.001           | 0.283 ± 0.003 | 0.887 ± 0.001 | 0.738 ± 0.002 |       3 |
| unsw_nb15/DoS            | all          | shap    | ae_small     | 0.299 ± 0.006    | 0.955 ± 0.023            | 1.000 ± 0.000           | 0.293 ± 0.011 | 0.885 ± 0.003 | 0.732 ± 0.008 |       3 |
| unsw_nb15/DoS            | all          | shap    | iforest      | 0.276 ± 0.016    | 0.928 ± 0.021            | 0.999 ± 0.000           | 0.302 ± 0.004 | 0.882 ± 0.001 | 0.724 ± 0.004 |       3 |
| unsw_nb15/DoS            | all          | concat  | ae_small     | 0.374 ± 0.009    | 0.971 ± 0.008            | 0.999 ± 0.000           | 0.292 ± 0.010 | 0.885 ± 0.004 | 0.733 ± 0.010 |       3 |
| unsw_nb15/DoS            | all          | concat  | iforest      | 0.347 ± 0.024    | 0.860 ± 0.013            | 0.999 ± 0.000           | 0.295 ± 0.004 | 0.884 ± 0.002 | 0.730 ± 0.004 |       3 |
| unsw_nb15/Exploits       | -            | -       | xgboost_only | nan ± 0.000      | nan ± 0.000              | 0.973 ± 0.004           | 0.247 ± 0.001 | 0.895 ± 0.002 | 0.755 ± 0.005 |       3 |
| unsw_nb15/Exploits       | benign       | raw     | ae_small     | 0.637 ± 0.028    | 0.797 ± 0.046            | 0.977 ± 0.003           | 0.262 ± 0.004 | 0.891 ± 0.002 | 0.745 ± 0.005 |       3 |
| unsw_nb15/Exploits       | benign       | raw     | iforest      | 0.523 ± 0.019    | 0.624 ± 0.015            | 0.973 ± 0.004           | 0.271 ± 0.005 | 0.887 ± 0.000 | 0.735 ± 0.001 |       3 |
| unsw_nb15/Exploits       | benign       | shap    | ae_small     | 0.677 ± 0.014    | 0.955 ± 0.011            | 0.989 ± 0.002           | 0.267 ± 0.003 | 0.892 ± 0.002 | 0.748 ± 0.005 |       3 |
| unsw_nb15/Exploits       | benign       | shap    | iforest      | 0.730 ± 0.008    | 0.930 ± 0.008            | 0.986 ± 0.004           | 0.281 ± 0.008 | 0.886 ± 0.003 | 0.734 ± 0.007 |       3 |
| unsw_nb15/Exploits       | benign       | concat  | ae_small     | 0.704 ± 0.034    | 0.938 ± 0.023            | 0.983 ± 0.004           | 0.264 ± 0.004 | 0.891 ± 0.003 | 0.747 ± 0.007 |       3 |
| unsw_nb15/Exploits       | benign       | concat  | iforest      | 0.558 ± 0.023    | 0.798 ± 0.026            | 0.974 ± 0.004           | 0.269 ± 0.003 | 0.888 ± 0.003 | 0.740 ± 0.007 |       3 |
| unsw_nb15/Exploits       | all          | raw     | ae_small     | 0.690 ± 0.025    | 0.665 ± 0.044            | 0.976 ± 0.004           | 0.259 ± 0.003 | 0.891 ± 0.002 | 0.746 ± 0.004 |       3 |
| unsw_nb15/Exploits       | all          | raw     | iforest      | 0.654 ± 0.015    | 0.423 ± 0.050            | 0.973 ± 0.004           | 0.273 ± 0.003 | 0.886 ± 0.001 | 0.734 ± 0.003 |       3 |
| unsw_nb15/Exploits       | all          | shap    | ae_small     | 0.668 ± 0.014    | 0.954 ± 0.006            | 0.989 ± 0.001           | 0.272 ± 0.004 | 0.890 ± 0.002 | 0.743 ± 0.005 |       3 |
| unsw_nb15/Exploits       | all          | shap    | iforest      | 0.618 ± 0.016    | 0.914 ± 0.023            | 0.986 ± 0.005           | 0.289 ± 0.009 | 0.883 ± 0.004 | 0.727 ± 0.010 |       3 |
| unsw_nb15/Exploits       | all          | concat  | ae_small     | 0.704 ± 0.006    | 0.927 ± 0.003            | 0.984 ± 0.005           | 0.271 ± 0.007 | 0.889 ± 0.004 | 0.741 ± 0.010 |       3 |
| unsw_nb15/Exploits       | all          | concat  | iforest      | 0.661 ± 0.015    | 0.805 ± 0.026            | 0.976 ± 0.004           | 0.280 ± 0.002 | 0.884 ± 0.002 | 0.730 ± 0.006 |       3 |
| unsw_nb15/Reconnaissance | -            | -       | xgboost_only | nan ± 0.000      | nan ± 0.000              | 0.994 ± 0.004           | 0.260 ± 0.004 | 0.895 ± 0.001 | 0.756 ± 0.002 |       3 |
| unsw_nb15/Reconnaissance | benign       | raw     | ae_small     | 0.346 ± 0.034    | 0.626 ± 0.070            | 0.995 ± 0.004           | 0.276 ± 0.006 | 0.889 ± 0.002 | 0.743 ± 0.005 |       3 |
| unsw_nb15/Reconnaissance | benign       | raw     | iforest      | 0.404 ± 0.024    | 0.482 ± 0.055            | 0.994 ± 0.004           | 0.284 ± 0.005 | 0.887 ± 0.001 | 0.736 ± 0.003 |       3 |
| unsw_nb15/Reconnaissance | benign       | shap    | ae_small     | 0.354 ± 0.073    | 0.843 ± 0.072            | 0.995 ± 0.003           | 0.278 ± 0.007 | 0.890 ± 0.002 | 0.744 ± 0.006 |       3 |
| unsw_nb15/Reconnaissance | benign       | shap    | iforest      | 0.545 ± 0.125    | 0.771 ± 0.028            | 0.995 ± 0.003           | 0.297 ± 0.010 | 0.883 ± 0.003 | 0.728 ± 0.007 |       3 |
| unsw_nb15/Reconnaissance | benign       | concat  | ae_small     | 0.339 ± 0.012    | 0.790 ± 0.039            | 0.995 ± 0.004           | 0.277 ± 0.001 | 0.889 ± 0.001 | 0.743 ± 0.003 |       3 |
| unsw_nb15/Reconnaissance | benign       | concat  | iforest      | 0.418 ± 0.015    | 0.676 ± 0.021            | 0.994 ± 0.004           | 0.289 ± 0.005 | 0.886 ± 0.001 | 0.733 ± 0.002 |       3 |
| unsw_nb15/Reconnaissance | all          | raw     | ae_small     | 0.435 ± 0.034    | 0.391 ± 0.035            | 0.994 ± 0.004           | 0.275 ± 0.009 | 0.890 ± 0.003 | 0.744 ± 0.007 |       3 |
| unsw_nb15/Reconnaissance | all          | raw     | iforest      | 0.342 ± 0.029    | 0.206 ± 0.068            | 0.994 ± 0.004           | 0.285 ± 0.008 | 0.886 ± 0.002 | 0.735 ± 0.006 |       3 |
| unsw_nb15/Reconnaissance | all          | shap    | ae_small     | 0.565 ± 0.038    | 0.815 ± 0.073            | 0.995 ± 0.004           | 0.287 ± 0.019 | 0.886 ± 0.007 | 0.735 ± 0.016 |       3 |
| unsw_nb15/Reconnaissance | all          | shap    | iforest      | 0.498 ± 0.051    | 0.745 ± 0.022            | 0.995 ± 0.004           | 0.310 ± 0.013 | 0.879 ± 0.004 | 0.716 ± 0.009 |       3 |
| unsw_nb15/Reconnaissance | all          | concat  | ae_small     | 0.498 ± 0.019    | 0.695 ± 0.026            | 0.995 ± 0.004           | 0.294 ± 0.014 | 0.884 ± 0.004 | 0.730 ± 0.011 |       3 |
| unsw_nb15/Reconnaissance | all          | concat  | iforest      | 0.451 ± 0.058    | 0.617 ± 0.030            | 0.995 ± 0.004           | 0.298 ± 0.008 | 0.882 ± 0.002 | 0.726 ± 0.005 |       3 |

### Q1 pooled over all splits and seeds

| train data   | input   | detector   |   zero-day AUROC |   AUROC among XGB-missed |   fused zero-day recall |   fused FPR |   fused F1 |   fused MCC |
|:-------------|:--------|:-----------|-----------------:|-------------------------:|------------------------:|------------:|-----------:|------------:|
| all          | concat  | ae_small   |            0.622 |                    0.888 |                   0.921 |       0.223 |      0.882 |       0.738 |
| all          | concat  | iforest    |            0.582 |                    0.808 |                   0.899 |       0.226 |      0.874 |       0.726 |
| all          | raw     | ae_small   |            0.587 |                    0.703 |                   0.888 |       0.212 |      0.876 |       0.733 |
| all          | raw     | iforest    |            0.564 |                    0.527 |                   0.877 |       0.218 |      0.871 |       0.724 |
| all          | shap    | ae_small   |            0.612 |                    0.923 |                   0.939 |       0.224 |      0.888 |       0.747 |
| all          | shap    | iforest    |            0.571 |                    0.888 |                   0.924 |       0.234 |      0.878 |       0.728 |
| benign       | concat  | ae_small   |            0.586 |                    0.918 |                   0.924 |       0.214 |      0.886 |       0.746 |
| benign       | concat  | iforest    |            0.558 |                    0.832 |                   0.927 |       0.220 |      0.883 |       0.740 |
| benign       | raw     | ae_small   |            0.582 |                    0.823 |                   0.906 |       0.211 |      0.881 |       0.740 |
| benign       | raw     | iforest    |            0.559 |                    0.712 |                   0.897 |       0.218 |      0.875 |       0.730 |
| benign       | shap    | ae_small   |            0.552 |                    0.937 |                   0.938 |       0.219 |      0.892 |       0.756 |
| benign       | shap    | iforest    |            0.601 |                    0.907 |                   0.936 |       0.229 |      0.885 |       0.741 |

### Q1 paired test: SHAP vs raw input (pairs = split x seed x train data x detector)

| metric                 |   pairs |   mean(SHAP - raw) |   SHAP better in |   Wilcoxon p |
|:-----------------------|--------:|-------------------:|-----------------:|-------------:|
| AUROC among XGB-missed |      48 |              0.222 |            0.979 |        0.000 |
| zero-day AUROC         |      48 |              0.011 |            0.542 |        0.525 |
| fused zero-day recall  |      48 |              0.042 |            0.938 |        0.000 |
| fused FPR              |      48 |              0.012 |            0.979 |        0.000 |
| fused F1               |      48 |              0.010 |            0.438 |        0.923 |
| fused MCC              |      48 |              0.011 |            0.458 |        0.835 |

For fused FPR, a positive difference means SHAP raises more false alarms.

### Q2. Feature reduction

| split                    | features             | XGB acc       | XGB zd recall   | SHAP ms/flow   | AE(SHAP) zd AUROC   |   seeds |
|:-------------------------|:---------------------|:--------------|:----------------|:---------------|:--------------------|--------:|
| nsl_kdd/official         | all 41               | 0.792 ± 0.004 | 0.348 ± 0.028   | 0.975 ± 0.107  | 0.645 ± 0.020       |       3 |
| nsl_kdd/official         | top 10 by mean|SHAP| | 0.772 ± 0.006 | 0.274 ± 0.026   | 1.055 ± 0.062  | 0.603 ± 0.012       |       3 |
| unsw_nb15/DoS            | all 42               | 0.874 ± 0.001 | 0.998 ± 0.001   | 1.028 ± 0.074  | 0.530 ± 0.093       |       3 |
| unsw_nb15/DoS            | top 10 by mean|SHAP| | 0.867 ± 0.004 | 0.997 ± 0.002   | 1.015 ± 0.058  | 0.737 ± 0.072       |       3 |
| unsw_nb15/Exploits       | all 42               | 0.874 ± 0.002 | 0.973 ± 0.004   | 1.024 ± 0.038  | 0.677 ± 0.014       |       3 |
| unsw_nb15/Exploits       | top 10 by mean|SHAP| | 0.870 ± 0.005 | 0.942 ± 0.025   | 1.058 ± 0.043  | 0.638 ± 0.043       |       3 |
| unsw_nb15/Reconnaissance | all 42               | 0.873 ± 0.001 | 0.994 ± 0.004   | 1.049 ± 0.035  | 0.354 ± 0.073       |       3 |
| unsw_nb15/Reconnaissance | top 10 by mean|SHAP| | 0.872 ± 0.004 | 0.992 ± 0.004   | 1.060 ± 0.024  | 0.511 ± 0.028       |       3 |

### Q3. SHAP vs LIME agreement

| split                    | mean top-5 overlap   | flows with overlap >= 0.6   | top-1 feature agrees   | LIME s/flow   | SHAP ms/flow   |   seeds |
|:-------------------------|:---------------------|:----------------------------|:-----------------------|:--------------|:---------------|--------:|
| nsl_kdd/official         | 0.554 ± 0.041        | 0.668 ± 0.099               | 0.502 ± 0.388          | 0.039 ± 0.001 | 0.975 ± 0.107  |       3 |
| unsw_nb15/DoS            | 0.445 ± 0.038        | 0.350 ± 0.095               | 0.292 ± 0.225          | 0.038 ± 0.000 | 1.028 ± 0.074  |       3 |
| unsw_nb15/Exploits       | 0.461 ± 0.026        | 0.397 ± 0.077               | 0.142 ± 0.023          | 0.038 ± 0.000 | 1.024 ± 0.038  |       3 |
| unsw_nb15/Reconnaissance | 0.442 ± 0.009        | 0.333 ± 0.035               | 0.162 ± 0.103          | 0.039 ± 0.001 | 1.049 ± 0.035  |       3 |

### Q4. Explaining the alarm (SHAP AE)

| split                    | flows analysed   | score drop, top-5 abnormal feats   | score drop, random 5 feats   | stability (top-5 overlap, 2 seeds)   | sparsity (error share in top-5)   |   seeds |
|:-------------------------|:-----------------|:-----------------------------------|:-----------------------------|:-------------------------------------|:----------------------------------|--------:|
| nsl_kdd/official         | 500.000 ± 0.000  | 0.438 ± 0.052                      | 0.051 ± 0.008                | 0.763 ± 0.008                        | 0.508 ± 0.014                     |       3 |
| unsw_nb15/DoS            | 43.333 ± 18.339  | 0.401 ± 0.046                      | 0.037 ± 0.023                | 0.589 ± 0.099                        | 0.442 ± 0.029                     |       3 |
| unsw_nb15/Exploits       | 500.000 ± 0.000  | 0.266 ± 0.062                      | -0.022 ± 0.011               | 0.638 ± 0.144                        | 0.449 ± 0.018                     |       3 |
| unsw_nb15/Reconnaissance | 14.667 ± 17.673  | 0.377 ± 0.059                      | -0.024 ± 0.089               | 0.603 ± 0.118                        | 0.467 ± 0.051                     |       3 |

### Q5a. SHAP-guided evasion of XGBoost

| split                    | XGBoost evasion success   | median features changed (evaded)   |   seeds |
|:-------------------------|:--------------------------|:-----------------------------------|--------:|
| nsl_kdd/official         | 0.896 ± 0.033             | 2.000 ± 0.000                      |       3 |
| unsw_nb15/DoS            | 0.028 ± 0.010             | 3.000 ± 1.000                      |       3 |
| unsw_nb15/Exploits       | 0.048 ± 0.023             | 2.833 ± 0.289                      |       3 |
| unsw_nb15/Reconnaissance | 0.030 ± 0.012             | 2.667 ± 0.577                      |       3 |

### Q5b. Evading attacks still caught by 2nd stage

| split                    | input   | detector   | evaded attacks caught by 2nd stage   | same attacks before evasion   | benign flows flagged   |   seeds |
|:-------------------------|:--------|:-----------|:-------------------------------------|:------------------------------|:-----------------------|--------:|
| nsl_kdd/official         | raw     | ae_small   | 0.764 ± 0.019                        | 0.696 ± 0.121                 | 0.015 ± 0.007          |       3 |
| nsl_kdd/official         | raw     | iforest    | 0.745 ± 0.013                        | 0.785 ± 0.019                 | 0.020 ± 0.001          |       3 |
| nsl_kdd/official         | shap    | ae_small   | 0.964 ± 0.027                        | 1.000 ± 0.000                 | 0.044 ± 0.002          |       3 |
| nsl_kdd/official         | shap    | iforest    | 0.893 ± 0.084                        | 1.000 ± 0.000                 | 0.041 ± 0.003          |       3 |
| nsl_kdd/official         | concat  | ae_small   | 0.864 ± 0.055                        | 1.000 ± 0.000                 | 0.036 ± 0.004          |       3 |
| nsl_kdd/official         | concat  | iforest    | 0.812 ± 0.074                        | 0.971 ± 0.031                 | 0.031 ± 0.003          |       3 |
| unsw_nb15/DoS            | raw     | ae_small   | 0.255 ± 0.178                        | 0.128 ± 0.137                 | 0.026 ± 0.010          |       3 |
| unsw_nb15/DoS            | raw     | iforest    | 0.000 ± 0.000                        | 0.000 ± 0.000                 | 0.028 ± 0.001          |       3 |
| unsw_nb15/DoS            | shap    | ae_small   | 0.463 ± 0.075                        | 0.104 ± 0.112                 | 0.045 ± 0.019          |       3 |
| unsw_nb15/DoS            | shap    | iforest    | 0.091 ± 0.157                        | 0.030 ± 0.052                 | 0.049 ± 0.005          |       3 |
| unsw_nb15/DoS            | concat  | ae_small   | 0.367 ± 0.318                        | 0.128 ± 0.137                 | 0.040 ± 0.009          |       3 |
| unsw_nb15/DoS            | concat  | iforest    | 0.000 ± 0.000                        | 0.030 ± 0.052                 | 0.039 ± 0.005          |       3 |
| unsw_nb15/Exploits       | raw     | ae_small   | 0.331 ± 0.080                        | 0.227 ± 0.092                 | 0.027 ± 0.012          |       3 |
| unsw_nb15/Exploits       | raw     | iforest    | 0.000 ± 0.000                        | 0.015 ± 0.026                 | 0.027 ± 0.006          |       3 |
| unsw_nb15/Exploits       | shap    | ae_small   | 0.594 ± 0.118                        | 0.423 ± 0.175                 | 0.042 ± 0.005          |       3 |
| unsw_nb15/Exploits       | shap    | iforest    | 0.315 ± 0.195                        | 0.361 ± 0.127                 | 0.063 ± 0.013          |       3 |
| unsw_nb15/Exploits       | concat  | ae_small   | 0.671 ± 0.221                        | 0.459 ± 0.216                 | 0.042 ± 0.003          |       3 |
| unsw_nb15/Exploits       | concat  | iforest    | 0.082 ± 0.073                        | 0.120 ± 0.112                 | 0.039 ± 0.008          |       3 |
| unsw_nb15/Reconnaissance | raw     | ae_small   | 0.121 ± 0.038                        | 0.143 ± 0.143                 | 0.030 ± 0.010          |       3 |
| unsw_nb15/Reconnaissance | raw     | iforest    | 0.000 ± 0.000                        | 0.000 ± 0.000                 | 0.027 ± 0.000          |       3 |
| unsw_nb15/Reconnaissance | shap    | ae_small   | 0.315 ± 0.102                        | 0.095 ± 0.165                 | 0.030 ± 0.006          |       3 |
| unsw_nb15/Reconnaissance | shap    | iforest    | 0.077 ± 0.133                        | 0.000 ± 0.000                 | 0.046 ± 0.008          |       3 |
| unsw_nb15/Reconnaissance | concat  | ae_small   | 0.370 ± 0.204                        | 0.095 ± 0.165                 | 0.037 ± 0.004          |       3 |
| unsw_nb15/Reconnaissance | concat  | iforest    | 0.000 ± 0.000                        | 0.000 ± 0.000                 | 0.034 ± 0.005          |       3 |

### Q5b pooled over splits

| input   | detector   |   evaded attacks caught by 2nd stage |   same attacks before evasion |   benign flows flagged |
|:--------|:-----------|-------------------------------------:|------------------------------:|-----------------------:|
| concat  | ae_small   |                                0.568 |                         0.420 |                  0.039 |
| concat  | iforest    |                                0.224 |                         0.280 |                  0.036 |
| raw     | ae_small   |                                0.368 |                         0.299 |                  0.025 |
| raw     | iforest    |                                0.186 |                         0.200 |                  0.025 |
| shap    | ae_small   |                                0.584 |                         0.406 |                  0.040 |
| shap    | iforest    |                                0.344 |                         0.348 |                  0.049 |

### Q6. Cost

| split                    | component                                  | cost           |   seeds |
|:-------------------------|:-------------------------------------------|:---------------|--------:|
| nsl_kdd/official         | TreeSHAP, all feat (ms/flow, 8 workers)    | 0.975 ± 0.107  |       3 |
| nsl_kdd/official         | TreeSHAP, top-10 feat (ms/flow, 8 workers) | 1.055 ± 0.062  |       3 |
| nsl_kdd/official         | LIME (ms/flow, 1 core)                     | 38.875 ± 1.232 |       3 |
| nsl_kdd/official         | ae_small fit (s)                           | 33.961 ± 8.870 |       3 |
| nsl_kdd/official         | ae_small score (ms/flow)                   | 0.001 ± 0.000  |       3 |
| nsl_kdd/official         | iforest fit (s)                            | 1.767 ± 0.078  |       3 |
| nsl_kdd/official         | iforest score (ms/flow)                    | 0.008 ± 0.001  |       3 |
| unsw_nb15/DoS            | TreeSHAP, all feat (ms/flow, 8 workers)    | 1.028 ± 0.074  |       3 |
| unsw_nb15/DoS            | TreeSHAP, top-10 feat (ms/flow, 8 workers) | 1.015 ± 0.058  |       3 |
| unsw_nb15/DoS            | LIME (ms/flow, 1 core)                     | 37.926 ± 0.290 |       3 |
| unsw_nb15/DoS            | ae_small fit (s)                           | 28.176 ± 0.454 |       3 |
| unsw_nb15/DoS            | ae_small score (ms/flow)                   | 0.001 ± 0.000  |       3 |
| unsw_nb15/DoS            | iforest fit (s)                            | 1.215 ± 0.061  |       3 |
| unsw_nb15/DoS            | iforest score (ms/flow)                    | 0.006 ± 0.000  |       3 |
| unsw_nb15/Exploits       | TreeSHAP, all feat (ms/flow, 8 workers)    | 1.024 ± 0.038  |       3 |
| unsw_nb15/Exploits       | TreeSHAP, top-10 feat (ms/flow, 8 workers) | 1.058 ± 0.043  |       3 |
| unsw_nb15/Exploits       | LIME (ms/flow, 1 core)                     | 38.171 ± 0.321 |       3 |
| unsw_nb15/Exploits       | ae_small fit (s)                           | 31.911 ± 0.931 |       3 |
| unsw_nb15/Exploits       | ae_small score (ms/flow)                   | 0.001 ± 0.000  |       3 |
| unsw_nb15/Exploits       | iforest fit (s)                            | 1.177 ± 0.072  |       3 |
| unsw_nb15/Exploits       | iforest score (ms/flow)                    | 0.006 ± 0.000  |       3 |
| unsw_nb15/Reconnaissance | TreeSHAP, all feat (ms/flow, 8 workers)    | 1.049 ± 0.035  |       3 |
| unsw_nb15/Reconnaissance | TreeSHAP, top-10 feat (ms/flow, 8 workers) | 1.060 ± 0.024  |       3 |
| unsw_nb15/Reconnaissance | LIME (ms/flow, 1 core)                     | 39.275 ± 0.644 |       3 |
| unsw_nb15/Reconnaissance | ae_small fit (s)                           | 22.642 ± 7.648 |       3 |
| unsw_nb15/Reconnaissance | ae_small score (ms/flow)                   | 0.001 ± 0.000  |       3 |
| unsw_nb15/Reconnaissance | iforest fit (s)                            | 1.262 ± 0.119  |       3 |
| unsw_nb15/Reconnaissance | iforest score (ms/flow)                    | 0.006 ± 0.000  |       3 |

### Q4b. Most frequent abnormal-explanation features (summed over seeds)

| split                    | feature                |   times in top-5 |
|:-------------------------|:-----------------------|-----------------:|
| nsl_kdd/official         | dst_host_rerror_rate   |              738 |
| nsl_kdd/official         | srv_rerror_rate        |              579 |
| nsl_kdd/official         | src_bytes              |              424 |
| nsl_kdd/official         | dst_host_diff_srv_rate |              362 |
| nsl_kdd/official         | duration               |              357 |
| unsw_nb15/DoS            | sinpkt                 |               55 |
| unsw_nb15/DoS            | ct_dst_sport_ltm       |               40 |
| unsw_nb15/DoS            | proto                  |               36 |
| unsw_nb15/DoS            | sttl                   |               36 |
| unsw_nb15/DoS            | ct_state_ttl           |               31 |
| unsw_nb15/Exploits       | dttl                   |             1295 |
| unsw_nb15/Exploits       | dmean                  |              849 |
| unsw_nb15/Exploits       | dpkts                  |              454 |
| unsw_nb15/Exploits       | dloss                  |              371 |
| unsw_nb15/Exploits       | dbytes                 |              277 |
| unsw_nb15/Reconnaissance | ct_dst_src_ltm         |               35 |
| unsw_nb15/Reconnaissance | rate                   |               35 |
| unsw_nb15/Reconnaissance | sbytes                 |               26 |
| unsw_nb15/Reconnaissance | sload                  |               26 |
| unsw_nb15/Reconnaissance | smean                  |               26 |
