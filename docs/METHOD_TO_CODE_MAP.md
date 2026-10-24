# Methodology to Code Mapping

| Manuscript Section | Implementation | Outputs |
|---|---|---|
| §3.1 Series and sources | `data.py:download_market_prices`, `load_gepu` | raw CSVs, retrieval manifest |
| §3.2 Exchange calendars and return alignment | `data.py:read_market_csv`, `log_change`, `construct_data` | daily, monthly, missing flags |
| §3.3 GEPU timing scenarios | `data.py:assign_gepu_scenario` | `gepu_first`, `gepu_mid`, `gepu_m2`, `gepu_m3` |
| §4.1 HAC contemporaneous regressions | `models.py:hac_ols`, `nested_increments` | `daily_ols.csv`, `nested_r2.csv` |
| §4.1 R² comparisons | `models.py:bootstrap_r2_difference` | `stationary_bootstrap_r2.csv` |
| §4.2 Monthly regressions | `cli.py:run_core` with `hac_monthly=3` | `monthly_ols.csv` |
| §4.3 DCC GJR-GARCH | `dcc.py:estimate_marginals`, `dcc_gaussian_qml` | DCC correlation paths and estimates |
| §4.4 Generalized variance decompositions | `models.py:normalized_generalized_fevd` | daily and monthly FEVD tables |
| §4.5 Quantile regressions | `models.py:quantile_bootstrap`, `cross_quantile_contrast` | quantile estimates and contrasts |
| §4.6 Subperiod stability tests | `models.py:period_slope_test` | `period_wald.jsonl.csv` |
| §4.7 VIX asymmetry | `models.py:asymmetry` | `asymmetry.csv` |
| §4.8 Recursive forecasting | `models.py:recursive_forecast` | forecast scores and paths |
| Appendix A8 Multiple testing corrections | `models.py:adjust_pvalues` | Holm and Benjamini-Hochberg adjustments |
