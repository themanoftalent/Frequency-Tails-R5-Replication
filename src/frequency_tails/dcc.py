"""Two-stage Dynamic Conditional Correlation (DCC-GARCH) model."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import slogdet, solve
from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg


def estimate_marginals(frame: pd.DataFrame, max_ar=10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit AR-GJR-GARCH(1,1) with Student-t errors to obtain standardized residuals."""
    residuals = {}
    metadata = []
    for column in frame:
        x = frame[column].dropna()
        chosen = min(range(1, max_ar + 1), key=lambda p: AutoReg(x, lags=p, old_names=False,
                                                                    trend="c").fit().bic)
        mean = AutoReg(x, lags=chosen, old_names=False, trend="c").fit()
        mean_resid = mean.resid
        variance = arch_model(mean_resid, mean="Zero", vol="GARCH", p=1, o=1, q=1,
                              dist="StudentsT", rescale=False).fit(disp="off")
        standardized = pd.Series(variance.std_resid, index=mean_resid.index).replace([np.inf, -np.inf], np.nan)
        residuals[column] = standardized
        metadata.append({"series": column, "ar_order": chosen, "ar_bic": mean.bic,
                         "garch_loglik": variance.loglikelihood,
                         "convergence_flag": variance.convergence_flag,
                         **{f"parameter_{k}": float(v) for k, v in variance.params.items()}})
    return pd.DataFrame(residuals).dropna(), pd.DataFrame(metadata)


def dcc_gaussian_qml(standardized: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Estimate DCC(1,1) correlation paths using Gaussian QML."""
    z = standardized.to_numpy(dtype=float)
    if len(z) < 100:
        raise ValueError("At least 100 observations required")
    qbar = np.cov(z, rowvar=False)
    n, k = z.shape
    def compute(params, retain=False):
        a, b = params
        if min(a, b) < 0 or a + b >= .999:
            return np.inf if not retain else None
        q = qbar.copy()
        paths = np.empty((n, k, k)) if retain else None
        ll = 0.
        for t in range(n):
            if t:
                q = (1-a-b)*qbar + a*np.outer(z[t-1], z[t-1]) + b*q
            diag = np.sqrt(np.diag(q))
            r = q / np.outer(diag, diag)
            if retain:
                paths[t] = r
            sign, logdet = slogdet(r)
            if sign <= 0:
                return np.inf if not retain else None
            ll += logdet + z[t] @ solve(r, z[t]) - z[t] @ z[t]
        return paths if retain else .5*ll
    fit = minimize(compute, x0=np.array([.035, .927]), method="SLSQP",
                   bounds=[(0, .998), (0, .998)],
                   constraints=[{"type": "ineq", "fun": lambda x: .999 - x.sum()}],
                   options={"maxiter": 500, "ftol": 1e-8})
    if not fit.success:
        raise RuntimeError(f"DCC optimization did not converge: {fit.message}")
    paths = compute(fit.x, retain=True)
    rows = {}
    for j in range(1, k):
        rows[f"{standardized.columns[0]}__{standardized.columns[j]}"] = paths[:, 0, j]
    output = pd.DataFrame(rows, index=standardized.index)
    details = {"a": float(fit.x[0]), "b": float(fit.x[1]),
               "a_plus_b": float(fit.x.sum()), "success": bool(fit.success),
               "likelihood": "Gaussian QML",
               "objective": float(fit.fun), "observations": n}
    return output, details


def path_stress_summary(paths: pd.DataFrame, vix_level: pd.Series, threshold=30) -> pd.DataFrame:
    """Summarize conditional correlation paths during normal vs high-stress periods."""
    aligned = paths.join(vix_level.rename("vix_level"), how="inner").dropna()
    stressed = aligned.vix_level > threshold
    rows = []
    for col in paths:
        x = aligned[col]
        rows.append({"pair": col, "nobs": len(x), "stress_count": int(stressed.sum()),
                     "threshold": threshold, "mean": x.mean(), "minimum": x.min(),
                     "maximum": x.max(), "share_positive": 100 * (x > 0).mean(),
                     "stress_mean": x[stressed].mean() if stressed.any() else np.nan,
                     "normal_mean": x[~stressed].mean() if (~stressed).any() else np.nan})
    return pd.DataFrame(rows)
