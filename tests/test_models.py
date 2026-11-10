import numpy as np
import pandas as pd
import pytest

from weekday_seasonal_forecaster.data import TRUE_WEEKDAY_FACTORS, generate_daily_flows
from weekday_seasonal_forecaster.evaluate import metrics, required_buffer, walk_forward
from weekday_seasonal_forecaster.models import (
    MovingAverage,
    NaiveWeekly,
    SeasonalWeekday,
    default_models,
)


@pytest.fixture(scope="module")
def flows() -> pd.DataFrame:
    return generate_daily_flows("2022-01-01", "2024-12-31", seed=7)


def test_generator_is_reproducible():
    a = generate_daily_flows("2023-01-01", "2023-03-31", seed=1)
    b = generate_daily_flows("2023-01-01", "2023-03-31", seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_weekday_factors_average_to_one(flows):
    model = SeasonalWeekday().fit(flows)
    assert model.weekday_factors_.mean() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("cutoff", ["2023-06-30", "2023-12-31", "2024-03-31", "2024-12-31"])
def test_recovers_true_weekday_factors_within_5pct(flows, cutoff):
    """With a one-year window the estimator is accurate: every factor within +/-5%."""
    model = SeasonalWeekday(weekday_window_weeks=52).fit(flows[flows["date"] <= cutoff])
    rel_err = model.weekday_factors_.to_numpy() / TRUE_WEEKDAY_FACTORS - 1
    assert np.abs(rel_err).max() < 0.05, rel_err


@pytest.mark.parametrize("cutoff", ["2023-06-30", "2023-12-31", "2024-03-31", "2024-12-31"])
def test_default_12_week_window_is_close_on_average(flows, cutoff):
    """The default 12-week window has only 12 observations per weekday, so it is
    noisier; on average the factors are still within 2.5% of the truth."""
    model = SeasonalWeekday().fit(flows[flows["date"] <= cutoff])
    rel_err = model.weekday_factors_.to_numpy() / TRUE_WEEKDAY_FACTORS - 1
    assert np.abs(rel_err).mean() < 0.025, rel_err


def test_predict_accepts_horizon_or_dates(flows):
    for model in default_models():
        model.fit(flows)
        by_h = model.predict(7)
        by_dates = model.predict(pd.date_range("2025-01-01", periods=7))
        assert len(by_h) == 7 and by_h.index[0] == pd.Timestamp("2025-01-01")
        pd.testing.assert_series_equal(by_h, by_dates, check_freq=False)
        assert np.isfinite(by_h).all()


def test_naive_weekly_repeats_last_week(flows):
    pred = NaiveWeekly().fit(flows).predict(7)
    last_week = flows.set_index("date")["volume"].iloc[-7:]
    np.testing.assert_allclose(pred.values, last_week.values)


def test_moving_average_is_trailing_mean(flows):
    pred = MovingAverage(window=28).fit(flows).predict(3)
    assert pred.iloc[0] == pytest.approx(flows["volume"].iloc[-28:].mean())


@pytest.mark.parametrize("model_cls", [NaiveWeekly, MovingAverage, SeasonalWeekday])
def test_no_leakage_future_data_does_not_change_forecasts(flows, model_cls):
    """Forecasts made at an origin must not change when data after the origin is altered."""
    df = flows[flows["date"] >= "2023-01-01"].reset_index(drop=True)
    cutoff = pd.Timestamp("2024-09-30")
    altered = df.copy()
    future = altered["date"] > cutoff
    altered.loc[future, "volume"] = altered.loc[future, "volume"] * 3.0 + 1e7

    a = walk_forward(df, model_cls, horizons=(1, 7, 30), test_periods=120)
    b = walk_forward(altered, model_cls, horizons=(1, 7, 30), test_periods=120)
    known = a["origin"] <= cutoff
    assert known.sum() > 0 and (~known).sum() > 0
    np.testing.assert_allclose(a.loc[known, "forecast"], b.loc[known, "forecast"])
    # Sanity check that the alteration was real: later forecasts do change.
    assert not np.allclose(a.loc[~known, "forecast"], b.loc[~known, "forecast"])


def test_walk_forward_only_trains_up_to_origin(flows):
    bt = walk_forward(flows, SeasonalWeekday, horizons=(1, 7), test_periods=30)
    assert (bt["date"] > bt["origin"]).all()
    assert set(bt["horizon"]) == {1, 7}
    assert bt.groupby("horizon").size().eq(30).all()


def test_required_buffer():
    actual = np.array([100.0] * 20)
    forecast = np.array([90.0] * 10 + [110.0] * 10)  # 10 under-forecasts of 10
    assert required_buffer(actual, forecast, q=0.95) == pytest.approx(10.0)
    assert required_buffer(actual, actual) == 0.0


def test_metrics_values():
    m = metrics([100.0, 200.0], [110.0, 180.0])
    assert m["MAE"] == pytest.approx(15.0)
    assert m["MSE"] == pytest.approx(250.0)
    assert m["RMSE"] == pytest.approx(np.sqrt(250.0))
    assert m["MAPE"] == pytest.approx(10.0)


def test_seasonal_beats_baselines(flows):
    """The project's headline claim, on a short backtest."""
    res = {}
    for cls in (NaiveWeekly, MovingAverage, SeasonalWeekday):
        bt = walk_forward(flows, cls, horizons=(7,), test_periods=180)
        res[cls.name] = metrics(bt["actual"], bt["forecast"])
    for base in ("NaiveWeekly", "MovingAverage"):
        assert res["SeasonalWeekday"]["MAE"] < res[base]["MAE"]
        assert res["SeasonalWeekday"]["required_buffer"] < res[base]["required_buffer"]
