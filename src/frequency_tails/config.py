"""Configuration and parameters for the replication pipeline."""
from dataclasses import dataclass
from pathlib import Path

TICKERS = {
    "us_banks": "KBE", "global_financials": "IXG", "turkish_banks": "XBANK.IS",
    "us_market": "^GSPC", "turkish_market": "XU100.IS",
    "vix": "^VIX", "ovx": "^OVX", "move": "^MOVE",
}
RETURNS = ("us_banks", "global_financials", "turkish_banks")
UNCERTAINTY = ("vix", "ovx", "move", "gepu")

@dataclass(frozen=True)
class Settings:
    root: Path
    start: str = "2010-01-01"
    end_exclusive: str = "2026-09-09"
    last_baseline: str = "2026-08-31"
    last_month: str = "2026-07"
    hac_daily: int = 5
    hac_monthly: int = 3
    bootstrap_draws: int = 999
    block_length: int = 20
    seed: int = 20261024

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def tables(self) -> Path:
        return self.root / "outputs" / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "outputs" / "figures"

    def create_dirs(self) -> None:
        for directory in (self.raw, self.processed, self.tables, self.figures):
            directory.mkdir(parents=True, exist_ok=True)
