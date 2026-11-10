"""Forecast models. All share one interface: fit(df) then predict(horizon or dates).

`df` has columns `date` and `volume`, sorted by date, with no rows after the forecast
origin. `predict` returns a Series of forecasts indexed by date.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import CONFIG
from .features import MONTH_BUCKETS, future_dates, month_bucket


class BaseModel:
    name = "base"

    def fit(self, df: pd.DataFrame) -> "BaseModel":
        y = df.set_index("date")["volume"].astype(float).sort_index()
        self.y_ = y
        self.last_date_ = y.index[-1]
        self.business_days_ = bool((y.index.day_of_week < 5).all())
        self._fit(y)
        return self

    def predict(self, horizon_or_dates: int | Sequence) -> pd.Series:
        if isinstance(horizon_or_dates, (int, np.integer)):
            dates = future_dates(self.last_date_, int(horizon_or_dates), self.business_days_)
        else:
            dates = pd.DatetimeIndex(horizon_or_dates)
        return pd.Series(self._predict(dates), index=dates, name=self.name)

    def _fit(self, y: pd.Series) -> None:
        raise NotImplementedError

    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError


class NaiveWeekly(BaseModel):
    """Forecast = the most recent observed value on the same weekday."""

    name = "NaiveWeekly"

    def _fit(self, y: pd.Series) -> None:
        self.last_by_weekday_ = y.groupby(y.index.day_of_week).last()

    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return self.last_by_weekday_.reindex(dates.day_of_week).to_numpy()


class MovingAverage(BaseModel):
    """Forecast = trailing N-observation mean, flat over the horizon."""

    name = "MovingAverage"

    def __init__(self, window: int = CONFIG.ma_window):
        self.window = window

    def _fit(self, y: pd.Series) -> None:
        self.level_ = y.iloc[-self.window:].mean()

    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(dates), self.level_)


class SeasonalWeekday(BaseModel):
    """Forecast = level x weekday factor x month-position factor.

    - Weekday factor: each weekday's average / overall average over the last
      `weekday_window_weeks` weeks, after removing month-position effects and
      measuring each day against its centred one-week average (local level).
    - Month-position factor: same ratio for day-of-month buckets over the last
      `month_window_days` days, after removing weekday effects.
    - The two are estimated alternately for a few passes, since each one distorts
      the other (a payday that lands on a Friday would otherwise inflate Friday).
    - Level: trailing `level_window` mean of the fully deseasonalized series.
    """

    name = "SeasonalWeekday"

    def __init__(
        self,
        level_window: int = CONFIG.level_window,
        weekday_window_weeks: int = CONFIG.weekday_window_weeks,
        month_window_days: int = CONFIG.month_window_days,
        n_passes: int = 3,
    ):
        self.level_window = level_window
        self.weekday_window_weeks = weekday_window_weeks
        self.month_window_days = month_window_days
        self.n_passes = n_passes

    @staticmethod
    def _ratio_factors(values: pd.Series, keys: pd.Series | np.ndarray) -> pd.Series:
        """Group mean / overall mean, normalized so factors average 1.0 across days."""
        overall = values.mean()
        factors = values.groupby(np.asarray(keys)).mean() / overall
        day_weighted_mean = factors.reindex(np.asarray(keys)).mean()
        return factors / day_weighted_mean

    def _fit(self, y: pd.Series) -> None:
        # Weekday window in observations: 12 weeks = 84 calendar days or 60 business days.
        per_week = 5 if self.business_days_ else 7
        n_wd = self.weekday_window_weeks * per_week
        y_mp = y[y.index > y.index[-1] - pd.Timedelta(days=self.month_window_days)]

        weekday = pd.Series(1.0, index=range(7))
        month = pd.Series(1.0, index=MONTH_BUCKETS)
        for _ in range(self.n_passes):
            wd_of = lambda s: weekday.reindex(s.index.day_of_week).to_numpy()  # noqa: E731
            mb_of = lambda s: month.reindex(month_bucket(s.index)).to_numpy()  # noqa: E731
            month = self._ratio_factors(y_mp / wd_of(y_mp), month_bucket(y_mp.index))
            month = month.reindex(MONTH_BUCKETS).fillna(1.0)
            # Weekday: compare each day with its centred one-week average (which holds
            # exactly one of each weekday), so trend and slow seasonal bumps such as a
            # December peak cancel out. Only training data is used: the last few days
            # simply have no centred average yet and are skipped.
            y_adj = y / mb_of(y)
            local = y_adj / y_adj.rolling(per_week, center=True).mean()
            local = local.dropna().iloc[-n_wd:]
            weekday = self._ratio_factors(local, local.index.day_of_week)
            # Weekday factors average to 1 over the weekdays present.
            weekday = (weekday / weekday.mean()).reindex(range(7))

        self.weekday_factors_ = weekday
        self.month_factors_ = month
        recent = y.iloc[-self.level_window:]
        seasonal = weekday.reindex(recent.index.day_of_week).to_numpy() * month.reindex(
            month_bucket(recent.index)
        ).to_numpy()
        self.level_ = (recent / seasonal).mean()

    def _predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        wd = self.weekday_factors_.reindex(dates.day_of_week).fillna(1.0).to_numpy()
        mp = self.month_factors_.reindex(month_bucket(dates)).to_numpy()
        return self.level_ * wd * mp


def default_models() -> list[BaseModel]:
    """The three models compared in this project, baselines first."""
    return [NaiveWeekly(), MovingAverage(), SeasonalWeekday()]
