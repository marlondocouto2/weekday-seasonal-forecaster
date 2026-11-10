"""Calendar features shared by the models."""

from __future__ import annotations

import pandas as pd

MONTH_BUCKETS = ["d01_03", "d04_13", "d14_16", "d17_late", "last3"]


def month_bucket(dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
    """Day-of-month bucket: 1-3, 4-13, 14-16, 17 to month end minus 3, last 3 days."""
    dates = pd.DatetimeIndex(dates)
    day = dates.day
    days_in_month = dates.days_in_month
    bucket = pd.Series("d17_late", index=dates)
    bucket[day <= 3] = "d01_03"
    bucket[(day >= 4) & (day <= 13)] = "d04_13"
    bucket[(day >= 14) & (day <= 16)] = "d14_16"
    bucket[day > days_in_month - 3] = "last3"
    return bucket


def future_dates(last_date: pd.Timestamp, horizon: int, business_days: bool) -> pd.DatetimeIndex:
    """The next `horizon` dates after `last_date` (calendar or business days)."""
    freq = "B" if business_days else "D"
    return pd.date_range(last_date, periods=horizon + 1, freq=freq)[1:]
