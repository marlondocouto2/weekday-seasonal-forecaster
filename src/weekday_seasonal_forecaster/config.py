"""All tunable settings in one place."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # Synthetic data
    start: str = "2021-01-01"
    end: str = "2025-12-31"
    seed: int = 42

    # Models
    ma_window: int = 28            # MovingAverage baseline: trailing days
    level_window: int = 28         # SeasonalWeekday: trailing days for the level
    weekday_window_weeks: int = 12  # SeasonalWeekday: trailing weeks for weekday factors
    month_window_days: int = 364   # SeasonalWeekday: trailing days for month-position factors

    # Backtest
    horizons: tuple[int, ...] = (1, 7, 30)
    backtest_days: int = 365       # length of the out-of-sample evaluation period
    buffer_quantile: float = 0.95  # "required buffer" = this quantile of under-forecast error

    # Output
    reports_dir: Path = field(default=PROJECT_ROOT / "reports")

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


CONFIG = Config()
