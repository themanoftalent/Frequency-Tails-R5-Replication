"""Econometric models, hypothesis tests, and resampling procedures."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.api import VAR
from statsmodels.regression.quantile_regression import QuantReg


def complete_case(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise KeyError(f"Missing model columns: {sorted(missing)}")
    return frame.loc[:, columns].replace([np.inf, -np.inf], np.nan).dropna()


def hac_ols(frame: pd.DataFrame, target: str, regressors: list[str], lags: int = 5):
    subset = complete_case(frame, [target] + regressors)
    if len(subset) <= len(regressors) + lags + 2:
        raise ValueError("Too few complete rows for HAC OLS")
    X = sm.add_constant(subset[regressors], has_constant="add")
    return sm.OLS(subset[target], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def tidy_ols(result, market: str, specification: str, sample_name: str) -> pd.DataFrame:
    ci = result.conf_int()
    return pd.DataFrame([{"market": market, "specification": specification,
                          "sample": sample_name, "term": name, "nobs": int(result.nobs),
                          "coefficient": float(result.params[name]),
                          "se_hac": float(result.bse[name]),
                          "p": float(result.pvalues[name]),
                          "ci_low": float(ci.loc[name, 0]),
                          "ci_high": float(ci.loc[name, 1]),
                          "r2": float(result.rsquared)} for name in result.params.index])


def nested_increments(frame: pd.DataFrame, y: str, market: str,
                      variables: tuple[str, str] = ("vix", "gepu"), lags: int = 5) -> pd.DataFrame:
    """Estimate nested specifications on a balanced sample and compute delta R2."""
    a, b = variables
    data = complete_case(frame, [y, market, a, b])
    specs = {"market_only": [market], "market_vix": [market, a],
             "market_gepu": [market, b], "market_both": [market, a, b]}
    fits = {name: hac_ols(data, y, cols, lags) for name, cols in specs.items()}
    base = fits["market_only"].rsquared
    return pd.DataFrame([{"target": y, "model": name, "nobs": len(data),
                          "r2": fit.rsquared, "delta_r2": fit.rsquared - base}
                         for name, fit in fits.items()])


def stationary_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Politis-Romano stationary bootstrap indices."""
    if n < 2 or mean_block < 1:
        raise ValueError("n >= 2 and mean_block >= 1 required")
    result = np.empty(n, dtype=int)
    result[0] = rng.integers(n)
    for i in range(1, n):
        result[i] = rng.integers(n) if rng.random() < 1 / mean_block else (result[i-1] + 1) % n
    return result


def block_indices(n: int, length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate circular block bootstrap indices of fixed block length."""
    pieces = []
    for _ in range(int(np.ceil(n / length))):
        start = rng.integers(n)
        pieces.extend((start + np.arange(length)) % n)
    return np.asarray(pieces[:n], dtype=int)


def bootstrap_r2_difference(frame: pd.DataFrame, y: str, a: str, b: str,
                            reps: int = 999, mean_block: int = 20,
                            seed: int = 20261024) -> dict:
    d = complete_case(frame, [y, a, b])
    def r2(x: pd.DataFrame, variable: str) -> float:
        return float(sm.OLS(x[y], sm.add_constant(x[[variable]])).fit().rsquared)
    original = r2(d, a) - r2(d, b)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for j in range(reps):
        sampled = d.iloc[stationary_indices(len(d), mean_block, rng)].reset_index(drop=True)
        draws[j] = r2(sampled, a) - r2(sampled, b)
    ci = np.quantile(draws, [0.025, 0.975])
    # Compute centered bootstrap p-value
    centered = draws - draws.mean()
    p = (1 + np.sum(np.abs(centered) >= abs(original))) / (reps + 1)
    return {"market": y, "contrast": f"{a} minus {b}", "nobs": len(d),
            "difference": original, "ci_low": ci[0], "ci_high": ci[1], "p_approx": p}


def quantile_model(frame: pd.DataFrame, target: str, regressors: list[str],
                   tau: float, max_iter: int = 10000):
    d = complete_case(frame, [target] + regressors)
    fit = QuantReg(d[target], sm.add_constant(d[regressors])).fit(q=tau, max_iter=max_iter)
    return fit, d


def quantile_bootstrap(frame: pd.DataFrame, target: str, regressors: list[str],
                       taus: tuple[float, ...] = (.05, .10, .25, .50, .75, .90, .95),
                       reps: int = 499, block: int = 20, seed: int = 20261024) -> pd.DataFrame:
    """Estimate quantile regressions across multiple quantiles with block bootstrap intervals."""
    d = complete_case(frame, [target] + regressors).reset_index(drop=True)
    X = sm.add_constant(d[regressors], has_constant="add")
    point = {}
    for tau in taus:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            point[tau] = QuantReg(d[target], X).fit(q=tau, max_iter=10000).params
    rng = np.random.default_rng(seed)
    rows = []
    simulated = {tau: [] for tau in taus}
    failures = 0
    for _ in range(reps):
        sample = d.iloc[block_indices(len(d), block, rng)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                batch = {tau: QuantReg(sample[target].to_numpy(),
                         sm.add_constant(sample[regressors].to_numpy(), has_constant="add"))
                         .fit(q=tau, max_iter=10000).params for tau in taus}
            for tau in taus:
                simulated[tau].append(np.asarray(batch[tau]))
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
    for tau in taus:
        estimates = np.asarray(simulated[tau])
        for i, term in enumerate(point[tau].index):
            lo, hi = np.quantile(estimates[:, i], [.025, .975]) if len(estimates) else (np.nan, np.nan)
            rows.append({"target": target, "tau": tau, "term": term,
                          "coefficient": float(point[tau].iloc[i]),
                          "ci_low": lo, "ci_high": hi, "valid_draws": len(estimates),
                          "failed_draws": failures})
    return pd.DataFrame(rows)


def cross_quantile_contrast(frame: pd.DataFrame, target: str, regressors: list[str],
                            term: str = "vix", lower: float = .10, reference: float = .50,
                            reps: int = 1999, block: int = 20, seed: int = 20261024) -> dict:
    d = complete_case(frame, [target] + regressors).reset_index(drop=True)
    if term not in regressors:
        raise ValueError("Contrasted term must be in model")
    def coeff(x, tau):
        return float(QuantReg(x[target], sm.add_constant(x[regressors], has_constant="add"))
                     .fit(q=tau, max_iter=10000).params[term])
    point = coeff(d, lower) - coeff(d, reference)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        x = d.iloc[block_indices(len(d), block, rng)].reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                draws.append(coeff(x, lower) - coeff(x, reference))
            except (ValueError, np.linalg.LinAlgError):
                continue
    ci = np.quantile(draws, [.025, .975]) if draws else [np.nan, np.nan]
    centered = np.asarray(draws) - np.mean(draws) if draws else np.array([])
    p = (1 + np.sum(np.abs(centered) >= abs(point))) / (len(draws) + 1) if draws else np.nan
    return {"target": target, "contrast": f"{lower} - {reference}", "term": term,
            "estimate": point, "ci_low": ci[0], "ci_high": ci[1],
            "p_approx": p, "valid_draws": len(draws)}


def period_slope_test(frame: pd.DataFrame, y: str, market: str,
                      boundaries=("2020-01-01", "2022-01-01"), lags=5):
    """Test parameter stability across subperiods with an interaction model and HAC Wald test."""
    d = complete_case(frame, [y, "vix", "gepu", market]).copy()
    d["period"] = np.searchsorted(pd.to_datetime(boundaries), d.index, side="right")
    X = pd.DataFrame(index=d.index)
    X["intercept"] = 1.
    for c in ("vix", "gepu", market):
        X[c] = d[c]
    for j in range(1, len(boundaries) + 1):
        flag = (d["period"] == j).astype(float)
        X[f"period_{j}"] = flag
        for c in ("vix", "gepu", market):
            X[f"{c}_p{j}"] = d[c] * flag
    fit = sm.OLS(d[y], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    restrictions = np.zeros((len(boundaries), len(fit.params)))
    for j in range(1, len(boundaries) + 1):
        restrictions[j-1, X.columns.get_loc(f"vix_p{j}")] = 1.
    test = fit.wald_test(restrictions, scalar=True)
    slopes = [float(fit.params["vix"])]
    slopes.extend(float(fit.params["vix"] + fit.params[f"vix_p{j}"]) for j in range(1, len(boundaries)+1))
    return {"target": y, "nobs": len(d), "wald": float(test.statistic),
            "p": float(test.pvalue), "slopes": slopes}, fit


def asymmetry(frame: pd.DataFrame, y: str, market: str | None = None, lags=5):
    d = frame.copy()
    d["vix_up"] = d["vix"].clip(lower=0)
    d["vix_down"] = d["vix"].clip(upper=0)
    regressors = ["vix_up", "vix_down", "gepu"] + ([market] if market else [])
    fit = hac_ols(d, y, regressors, lags)
    contrast = np.zeros(len(fit.params))
    contrast[list(fit.params.index).index("vix_up")] = 1
    contrast[list(fit.params.index).index("vix_down")] = -1
    t = fit.t_test(contrast)
    return fit, {"target": y, "market_control": bool(market), "nobs": int(fit.nobs),
                 "difference": float(t.effect.item()), "p": float(t.pvalue),
                 "ci_low": float(t.conf_int()[0, 0]), "ci_high": float(t.conf_int()[0, 1])}


def normalized_generalized_fevd(frame: pd.DataFrame, columns: list[str], p=2, horizon=10):
    """Compute Pesaran-Shin generalized FEVD and total connectedness index."""
    d = complete_case(frame, columns)
    fit = VAR(d).fit(p)
    sigma = np.asarray(fit.sigma_u)
    if fit.is_stable() is False:
        raise RuntimeError("Unstable VAR; generalized FEVD not reported")
    phi = fit.ma_rep(horizon - 1)
    k = len(columns)
    shares = np.zeros((k, k))
    for i in range(k):
        denom = sum((phi_h @ sigma @ phi_h.T)[i, i] for phi_h in phi)
        for j in range(k):
            numerator = sum((phi_h[i, :] @ sigma[:, j]) ** 2 / sigma[j, j] for phi_h in phi)
            shares[i, j] = numerator / denom
    shares = 100 * shares / shares.sum(axis=1, keepdims=True)
    table = pd.DataFrame(shares, index=columns, columns=columns)
    total = (np.sum(shares) - np.trace(shares)) / k
    to = pd.Series(shares.sum(axis=0) - np.diag(shares), index=columns, name="TO")
    from_ = pd.Series(shares.sum(axis=1) - np.diag(shares), index=columns, name="FROM")
    return table, pd.DataFrame({"TO": to, "FROM": from_, "NET": to - from_}), float(total), fit


def adjust_pvalues(table: pd.DataFrame, group_col="family", p_col="p") -> pd.DataFrame:
    """Apply Holm-Bonferroni and Benjamini-Hochberg corrections by family."""
    result = table.copy()
    result["holm_p"] = np.nan
    result["bh_q"] = np.nan
    for _, group in result.groupby(group_col):
        idx = group.index[group[p_col].notna()]
        if len(idx):
            vals = result.loc[idx, p_col].astype(float)
            result.loc[idx, "holm_p"] = multipletests(vals, method="holm")[1]
            result.loc[idx, "bh_q"] = multipletests(vals, method="fdr_bh")[1]
    return result


def recursive_forecast(frame: pd.DataFrame, y: str, market: str,
                       start: str = "2018-01-02", max_lag: int = 10,
                       extra: str = "vix", minimum_train: int = 300) -> tuple[pd.DataFrame, dict]:
    """Expanding-window recursive out-of-sample forecast comparison."""
    raw = frame[[y, market, extra]].copy()
    begin = pd.Timestamp(start)
    if raw.index.min() >= begin:
        raise ValueError("Training period must precede forecast start")
    # Build lag features
    lagged = pd.DataFrame(index=raw.index)
    lagged[y] = raw[y]
    for col in (y, market, extra):
        for lag in range(1, max_lag + 1):
            lagged[f"{col}_lag{lag}"] = raw[col].shift(lag)
    base_train = lagged.loc[lagged.index < begin]
    def predictors(p, include_extra):
        vars_ = [f"{c}_lag{i}" for c in (y, market) for i in range(1, p + 1)]
        if include_extra:
            vars_ += [f"{extra}_lag{i}" for i in range(1, p + 1)]
        return vars_
    candidates = []
    for p in range(1, max_lag + 1):
        features = predictors(p, True)
        d = complete_case(base_train, [y] + features)
        if len(d) >= minimum_train:
            model = sm.OLS(d[y], sm.add_constant(d[features], has_constant="add")).fit()
            candidates.append((model.bic, p))
    if not candidates:
        raise ValueError("Insufficient training data after lag construction")
    chosen = min(candidates)[1]
    columns = predictors(chosen, True)
    baseline = predictors(chosen, False)
    results = []
    for date in lagged.index[lagged.index >= begin]:
        past = lagged.loc[lagged.index < date]
        fits = {}
        valid = True
        for tag, cols in (("base", baseline), ("full", columns)):
            sample = complete_case(past, [y] + cols)
            row = lagged.loc[date, cols]
            if len(sample) < minimum_train or row.isna().any() or pd.isna(lagged.loc[date, y]):
                valid = False
                break
            fitted = sm.OLS(sample[y], sm.add_constant(sample[cols], has_constant="add")).fit()
            fits[tag] = float(np.r_[1., row.to_numpy(dtype=float)] @ fitted.params.to_numpy())
        if valid:
            results.append({"date": date, "actual": float(lagged.at[date, y]),
                            "base": fits["base"], "full": fits["full"]})
    result = pd.DataFrame(results)
    if result.empty:
        raise ValueError("No valid evaluation rows")
    result = result.set_index("date")
    eb = result.actual - result.base
    ef = result.actual - result.full
    score = 100 * (1 - np.sum(ef**2) / np.sum(eb**2))
    # Clark-West test for nested models
    adjusted = eb**2 - (ef**2 - (result.base - result.full)**2)
    cw = sm.OLS(adjusted, np.ones((len(adjusted), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})
    z = float(cw.tvalues.iloc[0])
    pval = float(stats.norm.sf(z))
    return result, {"target": y, "added": extra, "chosen_lag_initial_BIC": chosen,
                    "oos_n": len(result), "r2_oos_percent": float(score),
                    "cw_z": z, "cw_one_sided_p": pval}
