# Replication Notes and Caveats

This document outlines key methodological differences, data availability requirements, and areas requiring verification against original study sources.

## Data and Sample Notes

1. **Market price snapshots**: Price series downloaded via Yahoo Finance may exhibit minor retroactive adjustments compared to exact original retrieval dates.
2. **Calendar alignment**: Daily Turkish and US equity sessions overlap partially due to differing trading hours. Regressors for US uncertainty indices include a 1-day lag to align information sets.
3. **GEPU availability**: Monthly GEPU index values are released with a lag. Daily mapping scenarios (first session of month $m+1$, mid-month, $m+2$, etc.) are evaluated as sensitivity benchmarks.

## Econometric Implementation

1. **DCC-GARCH**: The univariate marginals are estimated using AR(p)-GJR-GARCH(1,1) with Student-t errors via `arch`. The second stage estimates the dynamic correlation path using Gaussian quasi-maximum likelihood.
2. **Quantile bootstrap**: Pointwise confidence intervals and cross-quantile contrasts use circular block bootstrapping across all quantiles jointly.
3. **Connectedness**: FEVD calculations implement the Pesaran-Shin generalized variance decomposition normalized across rows (Diebold-Yilmaz total connectedness index).
4. **Out-of-sample forecasts**: Evaluated using an expanding window with initial lag order selection by BIC and evaluated via the Clark-West test for nested models.
