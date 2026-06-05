import os
import sys
import pandas as pd
from datasets import load_dataset
from config import PREDICTIONS_DIR

PREDICTION_LENGTH = 24


def load_actuals():
    dataset = load_dataset("LeoTungAnh/electricity_hourly")
    rows = []
    for i, row in enumerate(dataset["validation"]):
        start = pd.Timestamp(row["start"])
        for step in range(PREDICTION_LENGTH):
            rows.append({
                "id": str(i),
                "timestamp": start + pd.Timedelta(hours=step),
                "actual": row["target"][step],
            })
    return pd.DataFrame(rows)


def evaluate(predictions, actuals, label):
    df = predictions.merge(actuals, on=["id", "timestamp"])
    mae = (df["actual"] - df["prediction"]).abs().mean()
    rmse = ((df["actual"] - df["prediction"]) ** 2).mean() ** 0.5
    mean_uncertainty = df["uncertainty"].mean()
    print(f"[{label}]")
    print(f"  MAE:              {mae:.4f}")
    print(f"  RMSE:             {rmse:.4f}")
    print(f"  Mean uncertainty: {mean_uncertainty:.4f}")


if __name__ == "__main__":
    actuals = load_actuals()

    files = {
        "Chronos-2":    "predictions.parquet",
        "Transformer":  "transformer_predictions.parquet",
    }

    # allow overriding via CLI arg: python evaluate.py transformer_predictions.parquet
    if len(sys.argv) > 1:
        files = {"Model": sys.argv[1]}

    for label, filename in files.items():
        path = os.path.join(PREDICTIONS_DIR, filename)
        if not os.path.exists(path):
            print(f"[{label}] no predictions file found at {path}")
            continue
        evaluate(pd.read_parquet(path), actuals, label)