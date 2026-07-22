# DA-MetaForecaster

Code accompanying the paper *"DA-MetaForecaster: A Drift-Adaptive Three-Phase Framework for
Regime-Aware Multi-Horizon S&P 500 Return Forecasting"* (submitted to IJIES).

This repository contains the training, evaluation, and reproduction scripts for every table
reported in the paper. It does not include the manuscript itself, only the code and result
files needed to reproduce the reported numbers.

## What's here

| File | Produces | Description |
|---|---|---|
| `patchtst_macro.py` | Phase 1 backbone | PatchTST backbone pre-training on 12 price/volatility + macro features (base architecture: L=42, P=16, S=8, 400,513 params). |
| `patchtst_macro_leakfix.py` | Table 4, Table 5 | Full 3-phase pipeline (backbone → ANIL pre-training → regime-gated crash-only TTA) with the label-timing leak fixed, single seed per horizon. |
| `patchtst_multiseed.py` | Limitations (seed-sensitivity) | 3-seed ensemble of the pipeline, used to characterize seed-to-seed variance. |
| `patchtst_walkforward.py` | **Table 6** (central claim) | Walk-forward validation of the adapt-once TTA policy across 5 historical folds spanning 2008–2023. This is the script behind the paper's main validated result. |
| `sota_baselines.py` | Table 3 | Baseline reimplementations (DLinear, iTransformer, N-HiTS, TimeMixer) under the identical protocol, feature set, and per-horizon seeds as the main model. |
| `results/predictions_adaptonce_F5_Original_H*d.csv` | Table 7 | Per-day predictions (baseline vs. adapt-once TTA) on the primary 2018–2023 test window, at each horizon. Source data for the trading-utility metrics. |
| `results/walkforward_fold_results_adaptonce.json` | Table 6 | Raw per-fold crash-DA results from the walk-forward validation. |
| `data_pipeline.py` | — | Consolidated data loading, feature construction, and regime-label generation, factored out of the ~80-line block previously duplicated across the four scripts above. `load_dataset()` and `assign_regimes()` are the entry points; the latter is what the Section 7.5 threshold-sensitivity check calls to re-label already-fetched data under different drawdown/momentum thresholds. |
| `baseline_reconcile.py` | Table 3 note | Independently retrains the PatchTST backbone under the SOTA-comparison protocol (same seed formula) and computes crash-regime DA, to directly check it against Table 4/5's "Phase 1 only" crash DA rather than just asserting seed consistency. Result saved in `results/baseline_reconcile_results.json`. |
| `splits.json` | — | Exact date-range and window-count boundaries for the primary train/val/test split (per horizon) and for all five walk-forward folds, as inspectable artifacts rather than only hardcoded constants. |
| `config.json` | — | Every architecture hyperparameter, training setting, TTA policy setting, regime-detection threshold, and per-horizon seed in one file. |

## The central finding

Repurposing ANIL as a backbone pre-trainer (rather than a direct meta-learner), followed by
regime-gated, crash-only, single-adaptation-per-episode test-time adaptation, improves crash-regime
directional accuracy at H=10d from 0.373 to 0.459 (+8.6pp) on the primary 2018–2023 test window.
This replicates at a mean of +8.0pp (p=0.0215, paired t-test) across three independent historical
crash episodes (2008–2010 GFC, 2019–2021 COVID, and the original 2018–2023 window) in the
walk-forward study — see `patchtst_walkforward.py`.

An earlier, uncorrected version of the TTA buffer leaked label information (appending an
H-day-forward label before it would actually be observable in production), which inflated the
originally measured gains. `patchtst_macro_leakfix.py` and `patchtst_walkforward.py` implement the
fix: a window's label is only added to the adaptation buffer once `H` trading days have elapsed.

## Reproducing the results

```bash
pip install -r requirements.txt

# Table 3 — SOTA baseline comparison
python sota_baselines.py

# Tables 4-5 — main results + component ablation (single seed per horizon)
python patchtst_macro_leakfix.py

# Table 6 — walk-forward validation across 5 historical folds (the central claim)
python patchtst_walkforward.py

# Seed-sensitivity check referenced in the Limitations section
python patchtst_multiseed.py
```

The Section 7.5 bootstrap CI (day-level, 5,000 resamples on the primary F5 window)
and the regime-threshold sensitivity grid (drawdown -8% to -15%, momentum 0-4%) are
both computed directly from the saved `results/predictions_adaptonce_F5_Original_H*.csv`
files with no retraining required — `assign_regimes()` in `data_pipeline.py` re-labels
the crash regime under alternate thresholds for the latter.

Data (S&P 500 prices, VIX, Treasury yields, credit spreads) is fetched live from Yahoo Finance
(`yfinance`) and FRED (`pandas_datareader`) on first run and cached locally; no raw price data is
committed to this repository. All experiments are seeded deterministically per horizon
(`seed = 42 + 100*H`, i.e. H=1d→142, H=5d→542, H=10d→1042, H=21d→2142) with `cudnn.deterministic=True`
for exact reproducibility given the same PyTorch/CUDA version.

## Walk-forward folds

| Fold | Train through | Validation | Test | Era |
|---|---|---|---|---|
| F1_GFC | 2006 | 2007 | 2008–2010 | Global Financial Crisis |
| F2_EuroDebt | 2010 | 2011 | 2012–2014 | Calm bull market (no qualifying crash) |
| F3_Q4-2018 | 2014 | 2015 | 2016–2018 | Q4 2018 correction |
| F4_COVID | 2017 | 2018 | 2019–2021 | COVID crash |
| F5_Original | 2015 | 2016–2017 | 2018–2023 | Primary test window used throughout the paper |

Each fold trains its own Phase 1 backbone and Phase 2 ANIL pre-training from scratch; only the
Phase 3 TTA policy is held fixed across folds.

## Requirements

See `requirements.txt`. Tested with PyTorch 2.x on both CUDA and CPU. `xgboost` is optional — if
unavailable, `sota_baselines.py` falls back to `sklearn.GradientBoostingRegressor` automatically.

## Citation

If you use this code, please cite the paper (citation details to be added on acceptance).

## License

MIT — see `LICENSE`. (Defaulted to MIT as the standard permissive choice for research code;
change it in `LICENSE` if you'd prefer something else.)
