# Methodology

This document explains how the weekday-seasonal forecaster works: how the data is simulated,
how the model and its two baselines are estimated, how they are evaluated, and what the results
do and don't show. The [README](README.md) has the headline results; the
[walkthrough notebook](notebooks/walkthrough.ipynb) shows each step with charts. All numbers
below come from `python scripts/run_pipeline.py` with the default settings in
[`config.py`](src/weekday_seasonal_forecaster/config.py).

## 1. The problem and the idea

A treasury team has to fund each day's outgoing payments before the day starts. If the forecast
is too low, they are short and need expensive same-day funding. If it is too high, cash sits
idle. The goal is a forecast 1 to 30 days ahead that is accurate *and* easy to explain.

Daily payment volume has a strong, stable calendar shape: some weekdays are always busier than
others, and paydays lift volume around the same days every month. So the model is one
sentence:

```math
\hat{y}_t = L \times W_{\text{weekday}(t)} \times M_{\text{bucket}(t)}
```

That is, forecast = recent level × day-of-week factor × day-of-month (payday) factor. The point
of the project is to show that this small model, with the right seasonality, clearly beats the
naive rules of thumb teams often use.

This model is the base forecast that
[Project 2: cash-flow forecaster](https://github.com/marlondocouto2/cash-flow-forecaster) builds
on. There it is version 1, extended with spike detection, holiday effects and Fourier
seasonality.

## 2. Data

### 2.1 Where the data comes from

No data files are stored in the repo. Every run creates the data in memory with
`generate_daily_flows()` in [`data.py`](src/weekday_seasonal_forecaster/data.py). It takes
well under a second, and the fixed seed gives the same numbers on any machine.

### 2.2 Synthetic daily flows (default)

One series of daily outgoing payment volume in USD, from 2021-01-01 to 2025-12-31, with seed 42.
Each day is a product of known parts:

```math
y_t = T_t \times W_t \times M_t \times A_t \times \varepsilon_t
```

Every part is stored in its own column, so tests can check how well the model recovers it.

| Part | Definition |
|---|---|
| **Trend** $`T_t`$ | \$50M a day at the start, growing 15% a year: $`T_t = 50\text{M} \times 1.15^{\text{years}}`$ |
| **Weekday** $`W_t`$ | Mon 1.10, Tue 1.00, Wed 0.95, Thu 1.00, Fri 1.25, Sat 0.80, Sun 0.70, rescaled to average 1 |
| **Paydays** $`M_t`$ | ×1.25 on the 1st, ×1.12 on the 2nd, ×1.05 on the 3rd; ×1.20 on the 15th, ×1.06 on the 14th and 16th; ×1.30 on the last business day, ×1.08 the day before |
| **Annual** $`A_t`$ | $`\left(1 + 0.04 \sin\frac{2\pi(d-80)}{365.25}\right)\left(1 + 0.12\, e^{-\frac{1}{2}\left(\frac{d - d_{\text{Dec 20}}}{6}\right)^2}\right)`$, where $`d`$ is the day of the year: a mild yearly wave plus a December peak |
| **Noise** $`\varepsilon_t`$ | log-normal, $`\sigma = 0.06`$ (about 6%) |

The model deliberately has no annual component. The yearly wave and the December peak are left
in the data to show what a simple model misses (section 6).

### 2.3 Public data (optional)

`--source treasury` runs the same pipeline on U.S. Treasury operating-cash withdrawals from the
Daily Treasury Statement (Fiscal Data API, no key needed). The series is the sum of all
withdrawal categories except public-debt redemptions, which are large, scheduled debt
rollovers. It covers business days only, and amounts are converted from \$ millions to \$. The
true components are unknown for this series, so it can only be used to compare forecast
errors.

## 3. Models

Every model is a class with `fit(df)` and `predict(horizon or dates)`
([`models.py`](src/weekday_seasonal_forecaster/models.py)). `fit` receives only rows up to the
forecast origin.

### 3.1 Baseline: `NaiveWeekly`

The forecast for a day is the most recent observed value on the same weekday:

```math
\hat{y}_t = y_{s}, \quad s = \text{last observed day before the origin with } \text{weekday}(s) = \text{weekday}(t)
```

For horizons up to 7 days, this is "the same day last week". It captures the weekly shape for
free, but it copies one noisy observation, and it puts paydays on the wrong dates: last Tuesday
may have been the 1st, but this Tuesday is the 8th.

### 3.2 Baseline: `MovingAverage`

The forecast is the mean of the last $`N = 28`$ days, flat over the horizon:

```math
\hat{y}_t = \frac{1}{28}\sum_{i=0}^{27} y_{o-i}
```

where $`o`$ is the forecast origin. It averages the noise away but ignores the weekly and
payday patterns entirely.

### 3.3 Main model: `SeasonalWeekday`

The model estimates two sets of seasonal factors and a level, all from data up to the origin.

**Weekday factors $`W`$ (last 12 weeks).**

1. Divide each day by its current payday factor, to take out payday effects.
2. Divide by the **centred 7-day mean** around that day. The 7-day window contains exactly one of
   each weekday, so trend and slow seasonal bumps, such as the December peak, cancel out. The
   last three days of history have no centred mean yet and are simply skipped, so no future data
   is used.
3. Average the resulting ratio for each weekday over the last 12 weeks, then rescale so the
   seven factors average 1:

```math
W_d = \frac{\bar{r}_d}{\frac{1}{7}\sum_{d'} \bar{r}_{d'}}, \qquad r_t = \frac{y_t / M_t}{\frac{1}{7}\sum_{j=-3}^{3} y_{t+j} / M_{t+j}}
```

**Payday factors $`M`$ (last 364 days).** The same ratio-to-average idea, for five day-of-month
buckets:

| Bucket | Days |
|---|---|
| Start of month | 1–3 |
| Early | 4–13 |
| Mid-month payday | 14–16 |
| Late | 17 to month end minus 3 |
| Month end | last 3 days |

Each day is first divided by its weekday factor. Each bucket's factor is its average divided by
the overall average, normalized so the day-weighted average is 1. A full year is used so each
bucket appears in every month and every season.

**Alternating estimation.** The two sets of factors distort each other. A payday that lands on a
Friday inflates Friday, and a Friday that lands on a payday inflates that bucket. So they are
estimated in turn, each on data adjusted by the other's latest estimate, for **three passes**.

**Level $`L`$ (last 28 days).** The mean of the deseasonalized series:

```math
L = \frac{1}{28}\sum_{i=0}^{27} \frac{y_{o-i}}{W_{\text{weekday}(o-i)}\, M_{\text{bucket}(o-i)}}
```

**Forecast.** $`\hat{y}_t = L \cdot W_{\text{weekday}(t)} \cdot M_{\text{bucket}(t)}`$ for every
day in the horizon. The level is held flat. With 15% growth a year, the 14-day lag in a 28-day
mean costs about 0.5%, so it isn't worth a trend term.

## 4. Evaluation design

### 4.1 Walk-forward backtest

- **Test period:** the last 365 days (calendar 2025).
- **Origins:** every day from 30 days before the test period to its end. At each origin, every
  model is refit on data up to and including that day, and forecasts 1, 7 and 30 steps ahead. A
  forecast is kept when its target date falls inside the test period.
- **Scale:** 365 scored forecasts per model and horizon. The whole backtest takes about
  3 seconds.
- **Business-day data:** for Treasury data, horizons count business days, not calendar days.

### 4.2 Metrics

The error is $`e_t = y_t - \hat{y}_t`$, so a positive error means a shortfall.

- **MAE** $`= \overline{|e_t|}`$, **MSE** $`= \overline{e_t^2}`$, **RMSE** $`= \sqrt{\text{MSE}}`$.
- **MAPE** $`= \overline{|e_t| / y_t}`$. It is safe here because volumes are never near zero.
- **Required buffer**, the dollar metric:

```math
B = Q_{0.95}\left(\max(y_t - \hat{y}_t,\ 0)\right)
```

  This is the extra cash to hold on top of the forecast so that shortfalls are covered on 95% of
  days. Lower is better.
- **Error by weekday:** MAE grouped by the weekday of the target date. It shows *where* each
  model goes wrong.

### 4.3 Leakage controls

- `walk_forward` passes each model only `df.iloc[: origin + 1]`.
- The centred 7-day window inside `fit` runs on that slice only. Its last three values are
  undefined, never filled from the future.
- A test alters every value after a cutoff and checks that all forecasts made at or before the
  cutoff are unchanged, for all three models.

## 5. Validation of the method

Because the data is simulated, the estimated factors can be checked against the truth. These
checks run as tests (`pytest`):

| Check | Result on the default data |
|---|---|
| **Weekday factors average to 1** | exact |
| **Weekday factors, 12-week window**, across 46 monthly origins (2022–2025) | mean error 1.7%, worst 5.7% (test: average error within 2.5%) |
| **Weekday factors, 52-week window**, same origins | mean error 1.0%, worst 3.7% (test: every factor within ±5%) |
| **No leakage** | forecasts unchanged when future data is altered, for all models |
| **Headline claim** | the seasonal model beats both baselines on MAE and required buffer |

**Why the 12-week window is noisier.** It sees only 12 observations per weekday, each with 6%
noise. Paydays also land on some weekdays more than others in any 12-week stretch. A one-year
window is more accurate on this data because the simulated weekly pattern never changes. The
12-week default is kept because real weekly patterns drift, and a shorter window adapts faster.

At the last origin, the estimated factors vs the truth:

| | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|---|
| **True** | 1.132 | 1.029 | 0.978 | 1.029 | 1.287 | 0.824 | 0.721 |
| **Estimated** | 1.154 | 1.006 | 0.966 | 1.030 | 1.304 | 0.837 | 0.703 |

| | Days 1–3 | 4–13 | 14–16 | 17 to end−3 | Last 3 |
|---|---|---|---|---|---|
| **True** (2025 average) | 1.100 | 0.964 | 1.067 | 0.966 | 1.082 |
| **Estimated** | 1.072 | 0.959 | 1.072 | 0.981 | 1.068 |

The payday buckets are averages, so they can't reproduce the single-day lifts exactly. The 1st
of the month is ×1.25 in the data, but its bucket factor is about 1.07–1.10.

## 6. Results

Walk-forward over 2025. Volumes average \$97M a day.

| Horizon | Model | MAE | RMSE | MAPE | Required buffer |
|---|---|---|---|---|---|
| 7 days | **SeasonalWeekday** | **\$6.69M** | \$8.69M | 6.7% | **\$15.8M** |
| 7 days | NaiveWeekly | \$10.24M | \$14.33M | 10.1% | \$26.5M |
| 7 days | MovingAverage | \$15.77M | \$20.07M | 17.1% | \$33.8M |

The 1-day and 30-day horizons give the same ranking. The seasonal model's MAE is \$6.77M and
\$6.82M, against \$10.2–10.6M for `NaiveWeekly` and \$15.9–16.0M for `MovingAverage`; see
[`results.csv`](reports/results.csv).

**Why the seasonal model wins:**

- **Against `MovingAverage`:** the moving average predicts nearly the same number every day, so
  it is always low on Fridays and paydays and high on weekends. Its Friday and Sunday errors are
  \$29.6M and \$28.1M, against \$9.2M and \$4.8M for the seasonal model.
- **Against `NaiveWeekly`:** copying one day carries that day's noise into the forecast, so the
  error contains two days' worth of noise, and paydays are put on the wrong dates. The seasonal
  model estimates each effect from many days, so only today's noise remains. Its MAPE, 6.7%, is
  close to the 6% noise built into the data.
- **Bias:** none of the three is materially biased on average (−0.3% to +3.7%). The gains come
  from tracking the calendar shape, not from shifting the level.
- **Horizon:** accuracy barely changes from 1 to 30 days ahead. The weekly and payday shapes
  are known in advance, and the level drifts slowly.
- **Dollars:** a smaller and steadier error is what shrinks the buffer, from \$26.5M a day with
  the naive rule to \$15.8M.

**What it misses:** the December peak. There is no yearly component, so the model under-forecasts
the second half of December. Its error from 15 to 31 December is \$11.0M, against \$6.7M for the
year.

**Real data:** on U.S. Treasury withdrawals (`--source treasury`), the seasonal model has the
lowest MAE at every horizon, 26–36% below the baselines. At the 7-day horizon, though,
`NaiveWeekly` needs a smaller buffer (\$46B against \$53B). Treasury data has rare, very large
days (tax dates, benefit payments) that dominate the 95th percentile. Handling such days is the
subject of [Project 2](https://github.com/marlondocouto2/cash-flow-forecaster).

## 7. Assumptions and limitations

- **The data is simulated by design.** The true factors are known, so recovery is testable.
  Real flows also have spikes and holidays, which this model does not handle.
- **No yearly seasonality.** The December peak is under-forecast.
- **The level is a 28-day mean.** A single large spike inside the window inflates every
  forecast until it drops out.
- **The payday buckets are coarse.** They average single-day payday lifts over 3–10 days.
- **The windows are fixed.** 12 weeks, 364 days and 28 days were not tuned. Choosing them by
  backtest is a natural next step.

## 8. Reproducing

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py      # a few seconds
pytest
```

Seeds, windows and horizons are set in
[`config.py`](src/weekday_seasonal_forecaster/config.py), and package versions are pinned in
[`requirements.txt`](requirements.txt). The notebook version of this document,
[`notebooks/methodology.ipynb`](notebooks/methodology.ipynb), is generated from this file by
`python scripts/build_methodology_notebook.py`.
