"""Data sources: a synthetic generator with known structure, and an optional public loader.

The synthetic series is built multiplicatively from parts whose true values are kept in
extra columns, so tests can check that a model recovers them:

    volume = trend x weekday x month_position x annual x noise
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Raw weekday shape (Mon..Sun). Normalized to average 1.0 before use, so the
# "true" factors are directly comparable with what the model estimates.
RAW_WEEKDAY_SHAPE = np.array([1.10, 1.00, 0.95, 1.00, 1.25, 0.80, 0.70])
TRUE_WEEKDAY_FACTORS = RAW_WEEKDAY_SHAPE / RAW_WEEKDAY_SHAPE.mean()

BASE_DAILY_VOLUME = 50_000_000.0  # USD on the first day
ANNUAL_GROWTH = 0.15              # 15% a year
NOISE_SIGMA = 0.06                # ~6% multiplicative log-normal noise


def _last_business_day(dates: pd.DatetimeIndex) -> pd.Series:
    """For each date, the last Mon-Fri day of its month."""
    month_end = dates + pd.offsets.MonthEnd(0)
    # Roll back from month end to Friday when month end is on a weekend.
    shift = np.where(month_end.day_of_week == 5, 1, np.where(month_end.day_of_week == 6, 2, 0))
    return pd.Series(month_end - pd.to_timedelta(shift, unit="D"), index=dates)


def month_position_effect(dates: pd.DatetimeIndex) -> np.ndarray:
    """True payday effect: lifts around the 1st, the 15th and the last business day."""
    day = dates.day.to_numpy()
    effect = np.ones(len(dates))
    effect[day == 1] *= 1.25
    effect[day == 2] *= 1.12
    effect[day == 3] *= 1.05
    effect[day == 15] *= 1.20
    effect[(day == 14) | (day == 16)] *= 1.06
    lbd = _last_business_day(dates).to_numpy()
    days_to_lbd = (lbd - dates.to_numpy()).astype("timedelta64[D]").astype(int)
    effect[days_to_lbd == 0] *= 1.30
    effect[days_to_lbd == 1] *= 1.08
    return effect


def annual_effect(dates: pd.DatetimeIndex) -> np.ndarray:
    """Mild yearly sine wave plus a December peak centred on the 20th."""
    doy = dates.day_of_year.to_numpy()
    sine = 1 + 0.04 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    dec20 = pd.to_datetime([f"{y}-12-20" for y in dates.year]).day_of_year.to_numpy()
    december = 1 + 0.12 * np.exp(-0.5 * ((doy - dec20) / 6.0) ** 2)
    return sine * december


def generate_daily_flows(
    start: str = "2021-01-01", end: str = "2025-12-31", seed: int = 42
) -> pd.DataFrame:
    """Simulate daily outgoing payment volume (USD) with known components.

    Returns columns: date, volume, and the true components trend, weekday_factor,
    month_factor, annual_factor, noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    years = (dates - dates[0]).days.to_numpy() / 365.25

    trend = BASE_DAILY_VOLUME * (1 + ANNUAL_GROWTH) ** years
    weekday = TRUE_WEEKDAY_FACTORS[dates.day_of_week]
    month = month_position_effect(dates)
    annual = annual_effect(dates)
    noise = rng.lognormal(mean=0.0, sigma=NOISE_SIGMA, size=len(dates))

    return pd.DataFrame(
        {
            "date": dates,
            "volume": trend * weekday * month * annual * noise,
            "trend": trend,
            "weekday_factor": weekday,
            "month_factor": month,
            "annual_factor": annual,
            "noise": noise,
        }
    )


FISCAL_DATA_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
    "v1/accounting/dts/deposits_withdrawals_operating_cash"
)


def load_treasury_withdrawals(start: str = "2021-01-01", end: str | None = None) -> pd.DataFrame:
    """Daily U.S. Treasury operating-cash withdrawals (USD), business days only.

    Source: Daily Treasury Statement, Table II, via the Fiscal Data API. Sums every
    withdrawal category except public-debt redemptions (large, lumpy debt rollovers
    that are scheduled, not forecast). Requires network access and `requests`.
    """
    import requests

    flt = f"record_date:gte:{start},transaction_type:eq:Withdrawals"
    if end:
        flt += f",record_date:lte:{end}"
    rows, page = [], 1
    while True:
        resp = requests.get(
            FISCAL_DATA_URL,
            params={
                "filter": flt,
                "fields": "record_date,transaction_catg,transaction_today_amt",
                "page[size]": 10_000,
                "page[number]": page,
            },
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload["data"])
        if page >= payload["meta"]["total-pages"]:
            break
        page += 1

    raw = pd.DataFrame(rows)
    keep = (raw["transaction_catg"] != "null") & ~raw["transaction_catg"].str.startswith("Public Debt")
    raw = raw[keep].copy()
    raw["amount"] = pd.to_numeric(raw["transaction_today_amt"], errors="coerce") * 1e6  # $M -> $
    out = raw.groupby("record_date", as_index=False)["amount"].sum()
    out.columns = ["date", "volume"]
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)
