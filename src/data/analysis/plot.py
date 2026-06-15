import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data.config import PREPARED_DIR, PREDICTIONS_DIR

MODELS = {
    "Chronos-2":   "chronos_2_predictions.parquet",
    "Transformer": "transformer_predictions.parquet",
    "XGBoost":     "xgboost_predictions.parquet",
    "kNN":         "knn_predictions.parquet",
    "ARIMA":       "arima_predictions.parquet",
}

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
DIAGRAMS_DIR = os.path.join("data", "diagrams")


def plot_series_week(series_id: str, week: int):
    actuals = pd.read_parquet(os.path.join(PREPARED_DIR, "test_windows_target.parquet"))
    actuals = actuals[actuals["id"] == str(series_id)].copy()
    if actuals.empty:
        print(f"Series {series_id} not found in actuals.")
        return

    actuals["timestamp"] = pd.to_datetime(actuals["timestamp"])
    test_start = actuals["timestamp"].min()
    week_start = test_start + pd.Timedelta(weeks=week)
    week_end = week_start + pd.Timedelta(weeks=1)

    mask = (actuals["timestamp"] >= week_start) & (actuals["timestamp"] < week_end)
    week_actuals = actuals[mask].drop_duplicates("timestamp").sort_values("timestamp")
    if week_actuals.empty:
        print(f"No actuals for series {series_id} in week {week} ({week_start.date()} – {week_end.date()}).")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(week_actuals["timestamp"], week_actuals["actual"], color="black", linewidth=1.5,
            label="Actual", zorder=5)

    for (label, filename), color in zip(MODELS.items(), COLORS):
        path = os.path.join(PREDICTIONS_DIR, filename)
        if not os.path.exists(path):
            continue
        preds = pd.read_parquet(path)
        preds = preds[preds["id"] == str(series_id)].copy()
        preds["timestamp"] = pd.to_datetime(preds["timestamp"])
        preds = preds[(preds["timestamp"] >= week_start) & (preds["timestamp"] < week_end)]
        preds = preds.drop_duplicates("timestamp").sort_values("timestamp")
        if preds.empty:
            continue
        ax.plot(preds["timestamp"], preds["prediction"], color=color, linewidth=1, label=label, alpha=0.85)
        if preds["uncertainty"].gt(0).any():
            ax.fill_between(
                preds["timestamp"],
                preds["prediction"] - preds["uncertainty"],
                preds["prediction"] + preds["uncertainty"],
                color=color, alpha=0.12,
            )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    ax.set_title(f"Series {series_id} — week {week} ({week_start.date()} to {week_end.date()})")
    ax.set_ylabel("Consumption")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)
    out = os.path.join(DIAGRAMS_DIR, f"plot_series{series_id}_week{week}.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_series_full(series_id: str):
    train = pd.read_parquet(os.path.join(PREPARED_DIR, "train_context.parquet"))
    val = pd.read_parquet(os.path.join(PREPARED_DIR, "val_context.parquet"))
    history = pd.concat([train, val])
    history = history[history["id"] == str(series_id)][["timestamp", "target"]].copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history = history.drop_duplicates("timestamp").sort_values("timestamp")

    test = pd.read_parquet(os.path.join(PREPARED_DIR, "test_windows_target.parquet"))
    test = test[test["id"] == str(series_id)][["timestamp", "actual"]].copy()
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    test = test.drop_duplicates("timestamp").sort_values("timestamp")

    if history.empty and test.empty:
        print(f"Series {series_id} not found.")
        return

    fig, ax = plt.subplots(figsize=(18, 4))
    if not history.empty:
        ax.plot(history["timestamp"], history["target"], color="black", linewidth=0.6, label="History")
    if not test.empty:
        ax.plot(test["timestamp"], test["actual"], color="tab:blue", linewidth=0.8, label="Test actuals")
        if not history.empty:
            ax.axvline(test["timestamp"].min(), color="gray", linestyle="--", linewidth=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    ax.set_title(f"Series {series_id} — full history")
    ax.set_ylabel("Consumption")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)
    out = os.path.join(DIAGRAMS_DIR, f"plot_series{series_id}_full.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_weekday_aggregate(series_id: str):
    train = pd.read_parquet(os.path.join(PREPARED_DIR, "train_context.parquet"))
    history = train[train["id"] == str(series_id)][["timestamp", "target", "day_of_week", "hour"]].copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history_agg = history.groupby(["day_of_week", "hour"])["target"].mean()

    train2 = pd.read_parquet(os.path.join(PREPARED_DIR, "val_context.parquet"))
    all_ctx = pd.concat([train, train2])
    recent = all_ctx[all_ctx["id"] == str(series_id)][["timestamp", "target", "day_of_week", "hour"]].copy()
    recent["timestamp"] = pd.to_datetime(recent["timestamp"])
    recent = recent.drop_duplicates("timestamp").sort_values("timestamp")
    end = recent["timestamp"].max()
    start = end - pd.Timedelta(weeks=2)
    recent = recent[recent["timestamp"] > start]

    if recent.empty:
        print(f"No context data for series {series_id}.")
        return

    recent_agg = recent.groupby(["day_of_week", "hour"])["target"].mean()

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = range(24)

    fig, axes = plt.subplots(1, 7, figsize=(18, 4), sharey=True)
    fig.suptitle(f"Series {series_id} — weekday pattern: all history vs last 2 weeks ({start.date()} to {end.date()})")
    for dow, (ax, label) in enumerate(zip(axes, day_labels)):
        hist_vals = [history_agg.get((dow, hr), float("nan")) for hr in hours]
        rec_vals  = [recent_agg.get((dow, hr), float("nan")) for hr in hours]
        ax.plot(hours, hist_vals, color="tab:blue", linewidth=1.2, label="All history")
        ax.plot(hours, rec_vals,  color="black",    linewidth=1.2, label="Last 2 weeks")
        ax.set_title(label)
        ax.set_xticks([0, 6, 12, 18])
        ax.grid(True, alpha=0.3)
        if dow == 0:
            ax.set_ylabel("Consumption")
        if dow == 6:
            ax.legend(fontsize=7)

    plt.tight_layout()
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)
    out = os.path.join(DIAGRAMS_DIR, f"plot_series{series_id}_weekday_agg.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_all_full():
    train = pd.read_parquet(os.path.join(PREPARED_DIR, "train_context.parquet"))
    val = pd.read_parquet(os.path.join(PREPARED_DIR, "val_context.parquet"))
    history = pd.concat([train, val])[["timestamp", "target"]].copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    agg = history.groupby(history["timestamp"].dt.date)["target"].agg(["mean", "min", "max"]).reset_index()
    agg.columns = ["date", "mean", "min", "max"]
    agg["date"] = pd.to_datetime(agg["date"])

    test = pd.read_parquet(os.path.join(PREPARED_DIR, "test_windows_target.parquet"))
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    test_agg = test.groupby(test["timestamp"].dt.date)["actual"].agg(["mean", "min", "max"]).reset_index()
    test_agg.columns = ["date", "mean", "min", "max"]
    test_agg["date"] = pd.to_datetime(test_agg["date"])

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.fill_between(agg["date"], agg["min"], agg["max"], color="black", alpha=0.12, label="History min–max")
    ax.plot(agg["date"], agg["mean"], color="black", linewidth=0.7, label="History mean")
    ax.fill_between(test_agg["date"], test_agg["min"], test_agg["max"], color="tab:blue", alpha=0.15, label="Test min–max")
    ax.plot(test_agg["date"], test_agg["mean"], color="tab:blue", linewidth=0.9, label="Test mean")
    ax.axvline(test_agg["date"].min(), color="gray", linestyle="--", linewidth=0.8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    ax.set_title("All series — daily average across all 370 series")
    ax.set_ylabel("Consumption (mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)
    out = os.path.join(DIAGRAMS_DIR, "plot_all_full.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_yearly(series_id: str):
    train = pd.read_parquet(os.path.join(PREPARED_DIR, "train_context.parquet"))
    val = pd.read_parquet(os.path.join(PREPARED_DIR, "val_context.parquet"))
    history = pd.concat([train, val])
    history = history[history["id"] == str(series_id)][["timestamp", "target"]].copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history = history.drop_duplicates("timestamp").sort_values("timestamp")
    history["year"] = history["timestamp"].dt.year
    history["day_of_year"] = history["timestamp"].dt.dayofyear

    daily = history.groupby(["year", "day_of_year"])["target"].mean().reset_index()
    years = sorted(daily["year"].unique())
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, year in enumerate(years):
        yr = daily[daily["year"] == year]
        ax.plot(yr["day_of_year"], yr["target"], linewidth=0.8,
                color=colors[i % len(colors)], label=str(year))

    ax.set_title(f"Series {series_id} — daily average by year")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Consumption")
    ax.legend(title="Year", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)
    out = os.path.join(DIAGRAMS_DIR, f"plot_series{series_id}_yearly.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


def plot_candles(series_id: str):
    import numpy as np
    train = pd.read_parquet(os.path.join(PREPARED_DIR, "train_context.parquet"))
    history = train[train["id"] == str(series_id)][["timestamp", "target", "hour"]].copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])

    fig, ax = plt.subplots(figsize=(14, 5))
    outlier_x, outlier_y = [], []

    for hr in range(24):
        vals = history[history["hour"] == hr]["target"].values
        if len(vals) == 0:
            continue
        q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        whisker_lo = vals[vals >= lower].min()
        whisker_hi = vals[vals <= upper].max()
        outliers = vals[(vals < lower) | (vals > upper)]

        # wick
        ax.plot([hr, hr], [whisker_lo, whisker_hi], color="steelblue", linewidth=1.2, zorder=1)
        # body (IQR box)
        ax.add_patch(plt.Rectangle((hr - 0.3, q1), 0.6, iqr,
                                   facecolor="steelblue", edgecolor="steelblue", alpha=0.6, zorder=2))
        # median
        median = np.median(vals)
        ax.plot([hr - 0.3, hr + 0.3], [median, median], color="white", linewidth=1.5, zorder=3)
        # outliers
        outlier_x.extend([hr] * len(outliers))
        outlier_y.extend(outliers.tolist())

    ax.scatter(outlier_x, outlier_y, color="crimson", s=6, zorder=4, label=f"Outliers (n={len(outlier_y)})")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Consumption")
    ax.set_title(f"Series {series_id} — hourly distribution (candlestick, IQR ± 1.5×IQR)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(DIAGRAMS_DIR, exist_ok=True)
    out = os.path.join(DIAGRAMS_DIR, f"plot_series{series_id}_candles.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    arg1 = sys.argv[1] if len(sys.argv) > 1 else "0"
    mode = sys.argv[2] if len(sys.argv) > 2 else ""
    if arg1 == "all" and not mode:
        plot_all_full()
    elif mode == "all":
        plot_series_full(arg1)
    elif mode == "weekday":
        plot_weekday_aggregate(arg1)
    elif mode == "yearly":
        plot_yearly(arg1)
    elif mode == "candles":
        plot_candles(arg1)
    else:
        week = int(mode) if mode else 0
        plot_series_week(arg1, week)