# Data

No data files are committed. Both sources are produced on demand by
`src/weekday_seasonal_forecaster/data.py`.

## Synthetic (default)

```python
from weekday_seasonal_forecaster import generate_daily_flows
df = generate_daily_flows("2021-01-01", "2025-12-31", seed=42)
```

Columns: `date`, `volume` (USD), and the true components `trend`, `weekday_factor`,
`month_factor`, `annual_factor`, `noise`, so that `volume` is the product of the
last five. The same seed always produces the same series.

## U.S. Treasury (optional, public)

```python
from weekday_seasonal_forecaster import load_treasury_withdrawals
df = load_treasury_withdrawals(start="2021-01-01")
```

Source: Daily Treasury Statement, Table II "Deposits and Withdrawals of Operating Cash",
from the Fiscal Data API
(`https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/deposits_withdrawals_operating_cash`).
`volume` is the sum of every withdrawal category on a business day, converted from
$ millions to $, excluding "Public Debt Cash Redemp." (debt rollovers are scheduled, not
forecast) and the table's own total row. No API key is needed.
