"""Walk-forward backtest and error metrics."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .config import CONFIG
from .models import BaseModel

WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def walk_forward(
    df: pd.DataFrame,
    model_factory: Callable[[], BaseModel],
    horizons: Iterable[int] = CONFIG.horizons,
    test_periods: int = CONFIG.backtest_days,
) -> pd.DataFrame:
    """Rolling-origin backtest.

    At every origin the model is refit on rows up to and including the origin only,
    then forecasts the next max(horizons) periods. Rows are kept where the target date
    falls in the last `test_periods` observations and the step ahead is in `horizons`.
    Horizons are counted in rows (days for daily data, business days for business-day data).
    """
    df = df[["date", "volume"]].sort_values("date").reset_index(drop=True)
    horizons = sorted(set(horizons))
    h_max = max(horizons)
    n = len(df)
    test_start = n - test_periods
    first_origin = test_start - h_max

    records = []
    for origin in range(first_origin, n - 1):
        train = df.iloc[: origin + 1]
        model = model_factory().fit(train)
        for h in horizons:
            target = origin + h
            if target < test_start or target >= n:
                continue
            target_date = df.at[target, "date"]
            records.append(
                {
                    "model": model.name,
                    "origin": df.at[origin, "date"],
                    "date": target_date,
                    "horizon": h,
                    "forecast": float(model.predict([target_date]).iloc[0]),
                    "actual": float(df.at[target, "volume"]),
                }
            )
    out = pd.DataFrame(records)
    out["error"] = out["actual"] - out["forecast"]  # positive = under-forecast (shortfall)
    return out


def required_buffer(actual: np.ndarray, forecast: np.ndarray, q: float = CONFIG.buffer_quantile) -> float:
    """Cash held on top of the forecast to cover the shortfall on q of days.

    Under-forecast error = max(actual - forecast, 0); the buffer is its q-quantile.
    """
    shortfall = np.maximum(np.asarray(actual) - np.asarray(forecast), 0.0)
    return float(np.quantile(shortfall, q))


def metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    """MAE, MSE, RMSE, MAPE (%) and required buffer for one set of forecasts."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    err = a - f
    mse = float(np.mean(err**2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(np.mean(np.abs(err) / np.abs(a)) * 100),  # volumes are never near zero
        "required_buffer": required_buffer(a, f),
    }


def results_table(backtest: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, horizon), one column per metric."""
    rows = [
        {"model": m, "horizon": h, "n": len(g), **metrics(g["actual"], g["forecast"])}
        for (m, h), g in backtest.groupby(["model", "horizon"], sort=False)
    ]
    return pd.DataFrame(rows).sort_values(["horizon", "MAE"]).reset_index(drop=True)


def error_by_weekday(backtest: pd.DataFrame) -> pd.DataFrame:
    """MAE by weekday per model and horizon (long format)."""
    bt = backtest.assign(weekday=backtest["date"].dt.day_of_week)
    out = (
        bt.groupby(["model", "horizon", "weekday"])["error"]
        .apply(lambda e: e.abs().mean())
        .rename("MAE")
        .reset_index()
    )
    out["weekday_name"] = out["weekday"].map(dict(enumerate(WEEKDAY_NAMES)))
    return out
