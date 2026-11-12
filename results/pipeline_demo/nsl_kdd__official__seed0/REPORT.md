# XplainIDS cascade: nsl_kdd/official, seed 0

Budget 2% of benign validation flows, 0.5 of it to stage 2 (ae_small on SHAP vectors). Thresholds: t1 = 0.00722, t2 = 0.06902; achieved validation FPR 0.0200 (stage 1 0.0099, stage 2 0.0100).

## Detection on the test set

| system                                            |   val FPR |   test FPR |   zero-day recall |   known recall |   precision |    F1 |   MCC |
|:--------------------------------------------------|----------:|-----------:|------------------:|---------------:|------------:|------:|------:|
| XGBoost alone, p >= 0.5 (pilot)                   |     0.001 |      0.027 |             0.379 |          0.776 |       0.969 | 0.786 | 0.643 |
| XGBoost alone, val 2%                             |     0.020 |      0.048 |             0.764 |          0.875 |       0.959 | 0.897 | 0.787 |
| OR fusion: p >= 0.5 OR s at val 2% (Barnard rule) |     0.020 |      0.042 |             0.842 |          0.890 |       0.965 | 0.918 | 0.827 |
| OR fusion: val 2% + val 2%                        |     0.031 |      0.053 |             0.872 |          0.906 |       0.957 | 0.926 | 0.837 |
| Cascade, joint 2% (share 0.5 to stage 2)          |     0.020 |      0.042 |             0.838 |          0.886 |       0.965 | 0.916 | 0.822 |

- zero-day flows: 3750
- zero-days XGBoost passes (p < t1): 1178
- of those caught as NOVEL: 572
- benign test flows flagged NOVEL: 37

## Example alerts

NOVEL evidence compares each feature's SHAP contribution (phi) with the autoencoder's normal-looking reconstruction (phi_hat). KNOWN evidence lists the features that pushed XGBoost towards 'attack'.

**NOVEL, real zero-day**: nsl_kdd__official__seed0-test7634 (R2L / xlock) — p = 0.0023 (t1 = 0.0072), s = 0.0837 (t2 = 0.0690), type NOVEL
- duration = 41 contributed +0.028; normal-looking flows show about -0.001
- num_failed_logins = 2 contributed +0.022; normal-looking flows show about +0.000
- dst_host_count = 156 contributed -0.023; normal-looking flows show about -0.009
- service = telnet contributed -0.028; normal-looking flows show about -0.041
- dst_host_srv_count = 22 contributed -0.009; normal-looking flows show about +0.002

**NOVEL, real zero-day**: nsl_kdd__official__seed0-test5282 (R2L / xsnoop) — p = 0.0027 (t1 = 0.0072), s = 0.0817 (t2 = 0.0690), type NOVEL
- src_bytes = 1256 contributed -0.083; normal-looking flows show about -0.125
- dst_host_serror_rate = 0 contributed -0.059; normal-looking flows show about -0.033
- flag = S1 contributed -0.023; normal-looking flows show about -0.005
- count = 1 contributed -0.035; normal-looking flows show about -0.049
- service = X11 contributed -0.046; normal-looking flows show about -0.035

**NOVEL, real zero-day**: nsl_kdd__official__seed0-test18862 (U2R / xterm) — p = 0.0035 (t1 = 0.0072), s = 0.1234 (t2 = 0.0690), type NOVEL
- hot = 3 contributed +0.033; normal-looking flows show about +0.001
- src_bytes = 3750 contributed -0.109; normal-looking flows show about -0.078
- num_compromised = 23 contributed +0.026; normal-looking flows show about +0.000
- duration = 89 contributed +0.021; normal-looking flows show about -0.000
- dst_host_srv_diff_host_rate = 0.03 contributed -0.035; normal-looking flows show about -0.017

**NOVEL, real zero-day**: nsl_kdd__official__seed0-test13717 (U2R / httptunnel) — p = 0.0046 (t1 = 0.0072), s = 0.0698 (t2 = 0.0690), type NOVEL
- duration = 126 contributed +0.026; normal-looking flows show about -0.001
- dst_host_srv_rerror_rate = 0.7 contributed -0.023; normal-looking flows show about -0.001
- dst_host_srv_diff_host_rate = 0 contributed -0.004; normal-looking flows show about -0.016
- protocol_type = tcp contributed -0.015; normal-looking flows show about -0.003
- dst_host_rerror_rate = 0.11 contributed -0.001; normal-looking flows show about -0.013

**NOVEL, real zero-day**: nsl_kdd__official__seed0-test8876 (DoS / mailbomb) — p = 0.0070 (t1 = 0.0072), s = 0.0744 (t2 = 0.0690), type NOVEL
- dst_host_rerror_rate = 0.4 contributed +0.037; normal-looking flows show about -0.013
- src_bytes = 2599 contributed -0.200; normal-looking flows show about -0.167
- dst_host_count = 255 contributed +0.005; normal-looking flows show about -0.007

**NOVEL, false alarm on benign flow**: nsl_kdd__official__seed0-test6932 (normal / normal) — p = 0.0001 (t1 = 0.0072), s = 0.0719 (t2 = 0.0690), type NOVEL
- duration = 160 contributed +0.028; normal-looking flows show about -0.002
- src_bytes = 327 contributed -0.110; normal-looking flows show about -0.089
- num_access_files = 1 contributed +0.013; normal-looking flows show about +0.000
- dst_host_srv_diff_host_rate = 0.04 contributed -0.029; normal-looking flows show about -0.022

**KNOWN, real attack**: nsl_kdd__official__seed0-test6847 (DoS / back) — p = 0.9997 (t1 = 0.0072), s = 0.2738 (t2 = 0.0690), type KNOWN
- src_bytes = 54540 contributed +0.214
- hot = 2 contributed +0.148
- dst_host_rerror_rate = 0.01 contributed +0.049
