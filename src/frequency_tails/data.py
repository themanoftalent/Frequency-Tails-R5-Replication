"""Data ingestion, log return calculation, and calendar alignment."""
from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Settings, TICKERS

LOG = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_market_prices(settings: Settings) -> None:
    """Download daily close prices from Yahoo Finance and save raw CSV files."""
    import yfinance as yf
    settings.create_dirs()
    manifest = []
    for name, ticker in TICKERS.items():
        LOG.info("Downloading %s (%s)", name, ticker)
        result = yf.download(ticker, start=settings.start, end=settings.end_exclusive,
                             auto_adjust=True, actions=False, progress=False,
                             threads=False)
        if result.empty:
            LOG.warning("No data returned for %s", ticker)
            continue
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = result.columns.get_level_values(0)
        if "Close" not in result.columns:
            LOG.warning("Missing adjusted Close for %s", ticker)
            continue
        daily = pd.DataFrame({"date": pd.to_datetime(result.index).tz_localize(None).normalize(),
                              "close": pd.to_numeric(result["Close"], errors="coerce").to_numpy()})
        path = settings.raw / f"{name}.csv"
        daily.to_csv(path, index=False)
        manifest.append({"series": name, "ticker": ticker, "file": path.name,
                         "records": len(daily), "sha256": sha256(path),
                         "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                         "auto_adjust": True})
    (settings.raw / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2))


def read_market_csv(path: Path) -> pd.Series:
    """Load daily close prices from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Required price series missing: {path}")
    frame = pd.read_csv(path)
    if not {"date", "close"}.issubset(frame.columns):
        raise ValueError(f"{path.name} needs date,close columns")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    if frame.date.duplicated().any():
        raise ValueError(f"Duplicate source sessions in {path.name}")
    return pd.Series(pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
                     index=frame.date, name=path.stem).sort_index()


def log_change(series: pd.Series) -> pd.Series:
    """Compute 100 * log returns."""
    valid = series.where(series > 0)
    return 100 * (np.log(valid) - np.log(valid.shift(1)))


def load_gepu(path: Path) -> pd.Series:
    """Load monthly GEPU series from Excel or CSV."""
    if not path.exists():
        raise FileNotFoundError(f"GEPU file missing: {path}")
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    df.columns = [str(x).strip() for x in df.columns]
    candidate = next((col for col in df.columns if col.casefold() == "gepu_current"), None)
    if candidate is None:
        raise ValueError("GEPU_current column not found. Prepare data/raw/gepu.csv with date,GEPU_current")
    date_col = next((c for c in df.columns if c.lower() in ("date", "month", "time")), None)
    if date_col is not None:
        months = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M")
    elif {"Year", "Month"}.issubset(df.columns):
        months = pd.PeriodIndex(year=df["Year"].astype(int), month=df["Month"].astype(int), freq="M")
    else:
        raise ValueError("Supply date/month column or Year and Month for monthly GEPU")
    values = pd.to_numeric(df[candidate], errors="coerce")
    result = pd.Series(values.to_numpy(), index=pd.PeriodIndex(months, freq="M"), name="gepu")
    result = result[~result.index.isna()]
    if result.index.duplicated().any():
        raise ValueError("Duplicate GEPU reference months")
    return result.sort_index().where(lambda s: s > 0)


def geppu_monthly_changes(gepu: pd.Series) -> pd.Series:
    # Resample to ensure consecutive monthly periods before differencing
    contiguous = gepu.asfreq("M")
    return log_change(contiguous).rename("gepu")


def assign_gepu_scenario(monthly_changes: pd.Series, common_dates: pd.DatetimeIndex,
                         delay_months: int = 1, day_of_month: int = 1) -> pd.Series:
    """Map monthly GEPU changes to daily dates under a specified lag/day rule."""
    if delay_months < 1 or not 1 <= day_of_month <= 28:
        raise ValueError("Delay must be >= 1 month and day between 1 and 28")
    dates = pd.DatetimeIndex(common_dates).sort_values()
    assigned = pd.Series(0.0, index=dates, name="gepu_assigned")
    for month, change in monthly_changes.dropna().items():
        target = (month + delay_months).to_timestamp() + pd.Timedelta(days=day_of_month - 1)
        eligible = dates[dates >= target]
        if len(eligible) == 0:
            continue
        chosen = eligible[0]
        # Keep within the target calendar month
        if chosen.to_period("M") == (month + delay_months):
            assigned.loc[chosen] = float(change)
    return assigned


def construct_data(settings: Settings, gepu_file: str = "gepu.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct merged daily and monthly analysis datasets."""
    series = {name: read_market_csv(settings.raw / f"{name}.csv") for name in TICKERS}
    # Compute returns on each asset's own calendar first
    returns = {name: log_change(s).rename(name) for name, s in series.items()}
    # 1-day lag on US volatility indices to reflect market hours
    for name in ("vix", "ovx", "move"):
        returns[f"{name}_previous_us"] = returns[name].shift(1).rename(f"{name}_previous_us")
    common = returns["us_market"].index.intersection(returns["turkish_market"].index)
    common = common[common <= pd.Timestamp(settings.last_baseline)]
    daily = pd.DataFrame(index=common)
    for name, ret in returns.items():
        daily[name] = ret.reindex(common)
    daily["vix_level"] = series["vix"].reindex(common)

    # Monthly series resampled to month-end
    monthly_prices = {}
    for name, price in series.items():
        monthly_prices[name] = price.resample("ME").last().loc[:settings.last_month]
    monthly = pd.DataFrame({name: log_change(p) for name, p in monthly_prices.items()})
    gepu = load_gepu(settings.raw / gepu_file)
    monthly_changes = geppu_monthly_changes(gepu)
    monthly["gepu"] = monthly_changes.to_timestamp("M").reindex(monthly.index)
    monthly = monthly.loc["2010-02-01":]

    for delay, day, suffix in ((1, 1, "first"), (1, 15, "mid"), (2, 1, "m2"), (3, 1, "m3")):
        daily[f"gepu_{suffix}"] = assign_gepu_scenario(monthly_changes, common, delay, day)
    daily["gepu"] = daily["gepu_first"]
    daily.index.name = monthly.index.name = "date"
    settings.create_dirs()
    daily.to_csv(settings.processed / "daily.csv")
    monthly.to_csv(settings.processed / "monthly.csv")

    # Record missing session indicators for auditing
    pd.DataFrame({name: series[name].reindex(common).isna().astype(int) for name in TICKERS},
                 index=common).to_csv(settings.processed / "source_missing_flags.csv", index_label="date")
    return daily, monthly
