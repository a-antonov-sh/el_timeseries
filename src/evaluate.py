import os
import sys
import pandas as pd
from datasets import load_dataset
from config import PREDICTION_LEN
from data.config import PREDICTIONS_DIR
from common.command import Command


def load_actuals():
    dataset = load_dataset("LeoTungAnh/electricity_hourly")
    rows = []
    for i, row in enumerate(dataset["validation"]):
        start = pd.Timestamp(row["start"])
        for step in range(PREDICTION_LEN):
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


N_SERIES = 30  # set to None to evaluate all series


def _run_evaluate(file=None):
    actuals = load_actuals()
    if N_SERIES is not None:
        actuals = actuals[actuals["id"].astype(int) < N_SERIES]

    files = {
        "Chronos-2":   "chronos_2_predictions.parquet",
        "Transformer": "transformer_predictions.parquet",
        "XGBoost":     "xgboost_predictions.parquet",
        "kNN":         "knn_predictions.parquet",
        "ARIMA":       "arima_predictions.parquet",
    }
    if file:
        files = {"Model": file}

    for label, filename in files.items():
        path = os.path.join(PREDICTIONS_DIR, filename)
        if not os.path.exists(path):
            print(f"[{label}] no predictions file found at {path}")
            continue
        evaluate(pd.read_parquet(path), actuals, label)


class EvaluateCommand(Command):
    name = "evaluate"
    help = "Evaluate predictions against validation ground truth"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--file", type=str, default=None, help="Specific predictions file to evaluate")

    def __call__(self):
        _run_evaluate(self.args.file)


if __name__ == "__main__":
    _run_evaluate(sys.argv[1] if len(sys.argv) > 1 else None)