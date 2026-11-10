"""One command: generate data -> fit -> backtest -> save figures and results.

    python scripts/run_pipeline.py                   # synthetic data (default, offline)
    python scripts/run_pipeline.py --source treasury # U.S. Treasury withdrawals (needs network)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from weekday_seasonal_forecaster.config import CONFIG  # noqa: E402
from weekday_seasonal_forecaster.data import (  # noqa: E402
    generate_daily_flows,
    load_treasury_withdrawals,
)
from weekday_seasonal_forecaster.evaluate import (  # noqa: E402
    error_by_weekday,
    results_table,
    walk_forward,
)
from weekday_seasonal_forecaster.models import (  # noqa: E402
    MovingAverage,
    NaiveWeekly,
    SeasonalWeekday,
)
from weekday_seasonal_forecaster import plots  # noqa: E402

MODELS = [NaiveWeekly, MovingAverage, SeasonalWeekday]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synthetic", "treasury"], default="synthetic")
    args = parser.parse_args()

    if args.source == "synthetic":
        df = generate_daily_flows(CONFIG.start, CONFIG.end, CONFIG.seed)
        out_dir, test_periods = CONFIG.reports_dir, CONFIG.backtest_days
    else:
        df = load_treasury_withdrawals(start=CONFIG.start)
        out_dir, test_periods = CONFIG.reports_dir / "treasury", 252  # ~12 months of business days
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data: {len(df):,} rows, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")

    t0 = time.time()
    backtest = pd.concat(
        [walk_forward(df, m, CONFIG.horizons, test_periods) for m in MODELS], ignore_index=True
    )
    print(f"Backtest: {len(backtest):,} forecasts in {time.time() - t0:.1f}s")

    results = results_table(backtest)
    by_weekday = error_by_weekday(backtest)
    results.to_csv(out_dir / "results.csv", index=False, float_format="%.4f")
    by_weekday.to_csv(out_dir / "error_by_weekday.csv", index=False, float_format="%.4f")
    backtest.to_csv(out_dir / "backtest_forecasts.csv.gz", index=False, float_format="%.2f",
                    compression={"method": "gzip", "mtime": 0})  # no timestamp: same data, same file

    # Charts: the last full calendar quarter of the test period.
    last = backtest["date"].max()
    q_end = last if last.is_quarter_end else (last - pd.offsets.QuarterEnd(1))
    q_start = q_end - pd.offsets.QuarterBegin(1, startingMonth=1)
    plots.plot_headline(results, fig_dir / "headline.png")
    plots.plot_actual_vs_forecast(backtest, 7, f"{q_start:%Y-%m-%d}", f"{q_end:%Y-%m-%d}",
                                  fig_dir / "actual_vs_forecast.png")
    plots.plot_error_by_weekday(by_weekday, 7, fig_dir / "error_by_weekday.png")
    plots.plot_results_table(results, fig_dir / "results_table.png")

    show = results.copy()
    for c in ["MAE", "RMSE", "required_buffer"]:
        show[c] = (show[c] / 1e6).round(2)
    print("\nResults ($M except MAPE %):")
    print(show.drop(columns=["MSE"]).round(2).to_string(index=False))
    print(f"\nSaved results and figures to {out_dir}")


if __name__ == "__main__":
    main()
