"""CLI commands for data pipeline and estimation."""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Settings, RETURNS
from .data import download_market_prices, construct_data, sha256
from .models import (hac_ols, tidy_ols, nested_increments, bootstrap_r2_difference,
                     quantile_bootstrap, cross_quantile_contrast, period_slope_test,
                     asymmetry, normalized_generalized_fevd, recursive_forecast, adjust_pvalues)
from .figures import plot_vix_stress, plot_connectedness, plot_quantile_coefficients, plot_dcc

LOG = logging.getLogger(__name__)


def write_table(path: Path, records):
    pd.DataFrame(records).to_csv(path, index=False)
    LOG.info("Wrote %s", path)


def run_core(s: Settings, daily: pd.DataFrame, monthly: pd.DataFrame, boot: int) -> None:
    all_ols, all_inc, all_boot, all_period, all_asym, all_quant, all_month = [], [], [], [], [], [], []
    for y in RETURNS:
        market = "turkish_market" if y == "turkish_banks" else "us_market"
        for label, cols in (("vix", ["vix"]), ("ovx", ["ovx"]), ("move", ["move"]),
                            ("gepu", ["gepu"]), ("market_only", [market]),
                            ("market_vix_gepu", [market, "vix", "gepu"]),
                            ("all_uncertainties", [market, "vix", "gepu", "ovx", "move"])):
            sample = daily
            if label in ("vix", "ovx", "move", "gepu"):
                # Use common sample across bivariate comparisons
                sample = daily.dropna(subset=[y, "vix", "ovx", "move", "gepu"])
            fit = hac_ols(sample, y, cols, s.hac_daily)
            all_ols.append(tidy_ols(fit, y, label, "daily"))
        all_inc.append(nested_increments(daily, y, market, lags=s.hac_daily))
        for a, b in (("vix", "ovx"), ("ovx", "move"), ("move", "gepu"), ("vix", "gepu")):
            extended = daily.dropna(subset=[y, "vix", "ovx", "move", "gepu"])
            all_boot.append(bootstrap_r2_difference(extended, y, a, b, reps=boot,
                           mean_block=s.block_length, seed=s.seed))
        info, _ = period_slope_test(daily, y, market)
        all_period.append(info)
        for include_market in (True, False):
            _, contrast = asymmetry(daily, y, market if include_market else None)
            all_asym.append(contrast)
        for label, cols in (("vix", ["vix"]), ("gepu", ["gepu"]),
                            ("vix_gepu", ["vix", "gepu"]), ("market_only", [market]),
                            ("market_vix_gepu", [market, "vix", "gepu"])):
            fit = hac_ols(monthly, y, cols, s.hac_monthly)
            all_month.append(tidy_ols(fit, y, label, "monthly"))
        q = quantile_bootstrap(daily, y, ["vix", "gepu", market], reps=boot,
                              block=s.block_length, seed=s.seed)
        all_quant.append(q)
        plot_quantile_coefficients(q, s.figures / f"quantile_{y}.png")
    pd.concat(all_ols).to_csv(s.tables / "daily_ols.csv", index=False)
    pd.concat(all_inc).to_csv(s.tables / "nested_r2.csv", index=False)
    write_table(s.tables / "stationary_bootstrap_r2.csv", all_boot)
    write_table(s.tables / "period_wald.jsonl.csv", all_period)
    write_table(s.tables / "asymmetry.csv", all_asym)
    pd.concat(all_quant).to_csv(s.tables / "quantile_pointwise.csv", index=False)
    pd.concat(all_month).to_csv(s.tables / "monthly_ols.csv", index=False)
    plot_vix_stress(daily, s.figures / "vix_stress.png")

    # VAR connectedness on daily systems
    for tag, cols in (("daily4", ["vix", *RETURNS]),
                      ("daily6", ["vix", "ovx", "move", *RETURNS])):
        matrix, directional, tci, _ = normalized_generalized_fevd(daily, cols)
        matrix.to_csv(s.tables / f"{tag}_fevd.csv")
        directional.to_csv(s.tables / f"{tag}_directional.csv")
        (s.tables / f"{tag}_tci.json").write_text(json.dumps({"tci_percent": tci, "variables": cols}, indent=2))
        plot_connectedness(matrix, s.figures / f"{tag}_fevd.png")
    matrix, directional, tci, _ = normalized_generalized_fevd(monthly,
                                            ["vix", "gepu", *RETURNS], p=1, horizon=10)
    matrix.to_csv(s.tables / "monthly_gepu_fevd.csv")
    directional.to_csv(s.tables / "monthly_gepu_directional.csv")
    (s.tables / "monthly_gepu_tci.json").write_text(json.dumps({"tci_percent": tci}, indent=2))


def run_optional(s: Settings, daily: pd.DataFrame, mode: str, reps: int) -> None:
    if mode == "dcc":
        from .dcc import estimate_marginals, dcc_gaussian_qml, path_stress_summary
        marg, parameters = estimate_marginals(daily[["vix", *RETURNS]].dropna())
        path, info = dcc_gaussian_qml(marg)
        marg.to_csv(s.processed / "dcc_standardized_residuals.csv")
        path.to_csv(s.tables / "dcc_paths.csv")
        parameters.to_csv(s.tables / "dcc_marginals.csv", index=False)
        (s.tables / "dcc_second_stage.json").write_text(json.dumps(info, indent=2))
        write_table(s.tables / "dcc_stress.csv", path_stress_summary(path, daily.vix_level))
        plot_dcc(path, s.figures / "dcc_paths.png")
    elif mode == "quantile_contrasts":
        results = []
        for y in RETURNS:
            m = "turkish_market" if y == "turkish_banks" else "us_market"
            results.append(cross_quantile_contrast(daily, y, ["vix", "gepu", m],
                                                  reps=reps))
        write_table(s.tables / "joint_quantile_contrasts.csv", results)
    elif mode == "forecasts":
        results = []
        for y in RETURNS:
            m = "turkish_market" if y == "turkish_banks" else "us_market"
            for predictor in ("vix", "gepu"):
                forecasts, score = recursive_forecast(daily, y, m, extra=predictor)
                forecasts.to_csv(s.tables / f"forecast_{y}_{predictor}.csv")
                results.append(score)
        write_table(s.tables / "forecast_scores.csv", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "prepare", "core", "dcc",
                                                "quantile_contrasts", "forecasts", "all"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--gepu-file", default="gepu.csv")
    parser.add_argument("--bootstrap", type=int, default=999,
                        help="Number of bootstrap iterations (default: 999)")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = Settings(args.root.resolve()); s.create_dirs()
    if args.action == "download":
        download_market_prices(s); return
    if args.action == "all" and not args.skip_download:
        download_market_prices(s)
    if args.action in ("prepare", "all"):
        daily, monthly = construct_data(s, args.gepu_file)
    else:
        if not (s.processed / "daily.csv").exists():
            raise FileNotFoundError("Run 'frequency-tails prepare' first")
        daily = pd.read_csv(s.processed / "daily.csv", index_col="date", parse_dates=True)
        monthly = pd.read_csv(s.processed / "monthly.csv", index_col="date", parse_dates=True)
    if args.action in ("core", "all"):
        run_core(s, daily, monthly, args.bootstrap)
    if args.action in ("dcc", "quantile_contrasts", "forecasts"):
        run_optional(s, daily, args.action, args.bootstrap)
    if args.action == "all":
        for component in ("dcc", "quantile_contrasts", "forecasts"):
            try:
                run_optional(s, daily, component, max(args.bootstrap, 1999) if component == "quantile_contrasts" else args.bootstrap)
            except Exception as exc:
                LOG.exception("Module %s failed: %s", component, exc)

if __name__ == "__main__":
    main()
