# XplainIDS: a stakeholder-tiered, explainable intrusion detection system

B.Tech project (GP035), Delhi Technological University.

XplainIDS detects network attacks, including ones it was never trained on ("zero-days"), and explains each alert.

```
flow -> XGBoost (p) -> TreeSHAP (phi) --p >= t1--> KNOWN alert   (evidence: top positive SHAP contributions)
                                      \-p <  t1--> autoencoder on SHAP vectors of benign traffic
                                                     s >= t2 --> NOVEL alert (evidence: phi vs the AE's normal-looking reconstruction phi_hat)
```

- **Joint false-alarm budget:** t1 and t2 are set on benign validation flows only, so the two stages together flag exactly 2% of them.
- **Planned next:** alert records go to an LLM layer that writes an analyst brief and a manager summary, with a grounding check.

**Results so far: [RESULTS.md](RESULTS.md)** (what we ran, how the models were trained, every result table, expected vs observed). It is generated from `results/` by `make_results_md.py`.

## Layout

| Path | What |
|---|---|
| `data.py` | NSL-KDD / UNSW-NB15 loaders, leave-one-family-out zero-day splits |
| `explain.py` | XGBoost + parallel interventional TreeSHAP (Windows and Linux) |
| `detectors.py` | autoencoder, Isolation Forest, kNN, PCA; input transforms |
| `metrics.py` | metrics, validation-only thresholds, cascade with a joint budget |
| `evidence.py`, `pipeline.py` | alert-type-aware evidence; the `XplainIDSPipeline` (fit / score / save / load) |
| `inspect_alerts.py` | fit the cascade on one split, print example alerts, write `example_alert.json` |
| `run_grid.py`, `aggregate_grid.py` | main experiment grid (12 splits x 5 seeds) and its statistics |
| `data_checks.py` | pre-implementation data checks (probability of missed zero-days, misses per split, threshold ties) |
| `quick_study.py`, `run_experiments.py`, `aggregate_extended.py` | the earlier pilot study (kept so its results stay reproducible) |
| `make_results_md.py` | builds `RESULTS.md` from the files in `results/` |
| `compare_repro.py` | compares a re-run of the pilot with the committed results |
| `tests/` | pytest suite |
| `data/` | dataset files go here (not committed); see `data/README.md` |
| `results/` | every number we report, each produced by a script above |

## Setup

Python 3.11. Commands below are for Windows (Git Bash); on Linux or macOS use `.venv/bin/python`.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pytest -q tests
```

Then reproduce every result:

```bash
export XZ_TORCH_THREADS=4          # fixed thread count: autoencoder runs repeat exactly
./.venv/Scripts/python.exe -u data_checks.py --seeds 0 1 2
./.venv/Scripts/python.exe -u inspect_alerts.py
./.venv/Scripts/python.exe -u run_grid.py
./.venv/Scripts/python.exe aggregate_grid.py
./.venv/Scripts/python.exe make_results_md.py
```

For the pilot harness, see `results/FINDINGS.md`.

## Rules we follow

- Encoders, scalers, thresholds and model choices are fitted on training or validation data only, never on test data.
- Every number we report comes from a file in `results/` produced by a script in this repo.
- Test FPR is reported next to every recall, because a validation budget drifts on test data.
- Several seeds per configuration. Paired tests use one unit per split (seed-averaged), with Holm correction.

## Data sources

- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html (mirror used: github.com/defcom17/NSL_KDD)
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset (mirror used: huggingface.co/datasets/Mireu-Lab/UNSW-NB15)
