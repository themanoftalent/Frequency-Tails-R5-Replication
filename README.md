# Frequency, Tails, and Market Channels: Replication Workspace

Empirical replication codebase for the study **“Frequency, Tails, and Market Channels: Reassessing the Uncertainty-Bank Equity Nexus in the United States and Türkiye”**.

The pipeline implements market price retrieval, return calculation, GEPU timing scenarios, HAC regressions, incremental $R^2$ comparisons with stationary bootstrap, quantile regressions, DCC-GARCH dynamic correlations, generalized VAR connectedness (FEVD), and expanding-window recursive forecasts.

## 1. Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test]'
pytest -q
```

## 2. Data Preparation

### GEPU Index
Download `Global_Policy_Uncertainty_Data.xlsx` from [policyuncertainty.com](https://www.policyuncertainty.com/global_monthly.html) and place it in `data/raw/` (or supply a CSV formatted with `date,GEPU_current` as `data/raw/gepu.csv`).

### Market Price Series
Download the required market series (`KBE`, `IXG`, `XBANK.IS`, `^GSPC`, `XU100.IS`, `^VIX`, `^OVX`, `^MOVE`) via Yahoo Finance:

```bash
frequency-tails download --root .
```

To construct the aligned daily and monthly dataset:

```bash
frequency-tails prepare --root . --gepu-file gepu.csv
```

## 3. Running the Models

Run core regressions, incremental $R^2$ bootstraps, subperiod stability tests, and quantile regressions:

```bash
frequency-tails core --root . --bootstrap 999
```

Run optional modules:

```bash
# Dynamic conditional correlations (DCC-GARCH)
frequency-tails dcc --root .

# Cross-quantile contrasts (tail vs median)
frequency-tails quantile_contrasts --root . --bootstrap 1999

# Recursive out-of-sample forecasts
frequency-tails forecasts --root .
```

Or run all components end-to-end:

```bash
frequency-tails all --root . --gepu-file gepu.csv --skip-download
```

## 4. Key Outputs

All generated tables and figures are written to `outputs/`:

- `outputs/tables/daily_ols.csv`, `monthly_ols.csv`: Baseline HAC regression estimates.
- `outputs/tables/nested_r2.csv`, `stationary_bootstrap_r2.csv`: Incremental $R^2$ tests and bootstrap intervals.
- `outputs/tables/quantile_pointwise.csv`, `joint_quantile_contrasts.csv`: Quantile regression estimates and contrasts.
- `outputs/tables/daily4_fevd.csv`, `daily6_fevd.csv`, `monthly_gepu_fevd.csv`: Generalized variance decompositions.
- `outputs/tables/dcc_paths.csv`, `dcc_marginals.csv`: DCC parameter estimates and correlation paths.
- `outputs/tables/forecast_scores.csv`: Recursive out-of-sample evaluation and Clark-West test statistics.
- `outputs/figures/`: Diagnostic plots (VIX stress periods, DCC correlation paths, FEVD heatmaps, quantile curves).

## 5. Documentation

- `docs/METHOD_TO_CODE_MAP.md`: Mapping between paper sections and codebase modules.
- `docs/REPLICATION_LIMITATIONS.md`: Notes on calendar alignment, timing assumptions, and estimation specifications.
