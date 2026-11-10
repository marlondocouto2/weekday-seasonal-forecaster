"""Static charts for the README (PNG, matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import WEEKDAY_NAMES  # noqa: E402

# Fixed colour per model (colour follows the entity, never its rank).
MODEL_ORDER = ["SeasonalWeekday", "NaiveWeekly", "MovingAverage"]
COLORS = {"SeasonalWeekday": "#2a78d6", "NaiveWeekly": "#eb6834", "MovingAverage": "#1baf7a"}
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    }
)


def _millions(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v / 1e6:,.0f}M"))


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _grouped_bars(ax, table: pd.DataFrame, group_col: str, groups: list, value_col: str,
                  label_model: str | None = "SeasonalWeekday") -> None:
    models = [m for m in MODEL_ORDER if m in set(table["model"])]
    width = 0.8 / len(models)
    x = np.arange(len(groups))
    for i, m in enumerate(models):
        vals = table[table["model"] == m].set_index(group_col)[value_col].reindex(groups)
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width * 0.92,
                      color=COLORS[m], label=m, zorder=2)
        if m == label_model:
            for b, v in zip(bars, vals):
                ax.annotate(f"${v / 1e6:.1f}M", (b.get_x() + b.get_width() / 2, v),
                            xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.tick_params(axis="x", length=0)


def plot_headline(results: pd.DataFrame, path: Path) -> Path:
    """MAE and required buffer by horizon, one panel each."""
    horizons = sorted(results["horizon"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, col, title in [
        (axes[0], "MAE", "Average daily error (MAE)"),
        (axes[1], "required_buffer", "Required cash buffer (95th pct shortfall)"),
    ]:
        _grouped_bars(ax, results, "horizon", horizons, col)
        ax.set_xticklabels([f"{h} day{'s' if h > 1 else ''} ahead" for h in horizons])
        ax.set_title(title)
        _millions(ax)
    axes[0].legend(loc="upper left", ncols=3, bbox_to_anchor=(0, -0.08))
    fig.suptitle("Lower is better", x=0.125, y=1.0, ha="left", fontsize=9, color=INK_2)
    return _save(fig, path)


def plot_actual_vs_forecast(backtest: pd.DataFrame, horizon: int, start: str, end: str,
                            path: Path) -> Path:
    """Actual volume vs each model's forecast over one window (one horizon)."""
    bt = backtest[(backtest["horizon"] == horizon) & backtest["date"].between(start, end)]
    actual = bt.drop_duplicates("date").set_index("date")["actual"].sort_index()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(actual.index, actual.values, color=INK, lw=1.2, label="Actual", zorder=4)
    for m in reversed(MODEL_ORDER):
        s = bt[bt["model"] == m].set_index("date")["forecast"].sort_index()
        ax.plot(s.index, s.values, color=COLORS[m], lw=2 if m == MODEL_ORDER[0] else 1.4,
                label=m, zorder=3 if m == MODEL_ORDER[0] else 2)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(n) for n in ["Actual", *MODEL_ORDER]]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncols=4,
              loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.set_title(f"Actual vs {horizon}-day-ahead forecast, {pd.Timestamp(start):%b %d} to "
                 f"{pd.Timestamp(end):%b %d %Y}", pad=30)
    _millions(ax)
    ax.margins(x=0.01)
    return _save(fig, path)


def plot_error_by_weekday(by_weekday: pd.DataFrame, horizon: int, path: Path) -> Path:
    """MAE by weekday for each model at one horizon."""
    t = by_weekday[by_weekday["horizon"] == horizon]
    weekdays = sorted(t["weekday"].unique())
    fig, ax = plt.subplots(figsize=(11, 4))
    _grouped_bars(ax, t, "weekday", weekdays, "MAE", label_model=None)
    ax.set_xticklabels([WEEKDAY_NAMES[d] for d in weekdays])
    ax.set_title(f"Average error by weekday ({horizon}-day-ahead forecasts)")
    ax.legend(ncols=3, loc="upper left")
    _millions(ax)
    return _save(fig, path)


def plot_results_table(results: pd.DataFrame, path: Path) -> Path:
    """The results table rendered as an image for the README."""
    t = results.copy()
    for c in ["MAE", "RMSE", "required_buffer"]:
        t[c] = t[c].map(lambda v: f"${v / 1e6:,.2f}M")
    t["MSE"] = t["MSE"].map(lambda v: f"{v / 1e12:,.1f}")
    t["MAPE"] = t["MAPE"].map(lambda v: f"{v:.1f}%")
    t = t[["horizon", "model", "MAE", "RMSE", "MSE", "MAPE", "required_buffer"]]
    headers = ["Horizon (days)", "Model", "MAE", "RMSE", "MSE ($M²)", "MAPE", "Required buffer"]

    fig, ax = plt.subplots(figsize=(10, 0.36 * (len(t) + 1) + 0.3))
    ax.axis("off")
    table = ax.table(cellText=t.values, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.5)
    table.auto_set_column_width(list(range(len(headers))))
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(INK)
        if r == 0:
            cell.get_text().set_weight("bold")
            cell.get_text().set_color(INK_2)
        elif t.iloc[r - 1]["model"] == "SeasonalWeekday":
            cell.set_facecolor("#eef4fc")
    return _save(fig, path)
