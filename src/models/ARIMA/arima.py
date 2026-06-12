import os
import numpy as np
import pandas as pd

from config import CONTEXT_LEN, PREDICTION_LEN
from data.config import PREPARED_DIR
from models.ARIMA.config import MODEL_PATH
from common.base_model import SeriesModel


class ARIMAModel(SeriesModel):
    name = "arima"
    help = "ARIMA model — train or predict"
    predictions_filename = "arima_predictions.parquet"

    def __init__(self):
        self._models = None

    def train(self):
        import joblib
        from pmdarima import auto_arima
        context_df = pd.read_parquet(os.path.join(PREPARED_DIR, "context.parquet"))
        series_ids = sorted(context_df["id"].unique(), key=int)
        print(f"Fitting auto_arima for {len(series_ids)} series...")
        models = {}
        for i, series_id in enumerate(series_ids):
            if i % 10 == 0:
                print(f"  {i}/{len(series_ids)}")
            s = context_df[context_df["id"] == series_id].sort_values("timestamp").tail(CONTEXT_LEN)
            y = s["target"].values.astype(np.float64)
            models[series_id] = auto_arima(
                y,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(models, MODEL_PATH)
        print(f"Saved {MODEL_PATH}")

    def _get_models(self):
        if self._models is None:
            import joblib
            self._models = joblib.load(MODEL_PATH)
        return self._models

    def _predict_series(self, series_id, context, future):
        model = self._get_models()[series_id]
        forecast, conf_int = model.predict(n_periods=PREDICTION_LEN, return_conf_int=True)
        return [
            {
                "id": series_id,
                "timestamp": frow["timestamp"],
                "target_name": "target",
                "prediction": float(forecast[step]),
                "uncertainty": float(conf_int[step, 1] - conf_int[step, 0]),
            }
            for step, (_, frow) in enumerate(future.iterrows())
        ]

