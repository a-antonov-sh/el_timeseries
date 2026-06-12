import os
import numpy as np
import pandas as pd

from config import CONTEXT_LEN, PREDICTION_LEN
from data.config import PREPARED_DIR
from models.xg_boost.config import MODEL_PATH, TRAIN_STRIDE, N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE
from common.base_model import SeriesModel


def _build_features(s: pd.DataFrame) -> np.ndarray:
    values = s["target"].values.astype(np.float32)
    return np.concatenate([
        values,
        [values.mean(), values.std(), values.min(), values.max(), values[-1]],
        [s["day_of_week"].iloc[-1], s["hour"].iloc[-1], s["month"].iloc[-1]],
    ])


def _build_Xy(context_df: pd.DataFrame):
    X, y = [], []
    for series_id in sorted(context_df["id"].unique(), key=int):
        s = context_df[context_df["id"] == series_id].sort_values("timestamp")
        values = s["target"].values.astype(np.float32)
        n = len(values)
        for start in range(0, n - CONTEXT_LEN - PREDICTION_LEN + 1, TRAIN_STRIDE):
            end = start + CONTEXT_LEN
            window = s.iloc[start:end]
            X.append(_build_features(window))
            y.append(values[end:end + PREDICTION_LEN])
    return np.array(X), np.array(y)


class XGBoostModel(SeriesModel):
    name = "xgboost"
    help = "XGBoost model — train or predict"
    predictions_filename = "xgboost_predictions.parquet"

    def __init__(self):
        self._model = None

    def train(self):
        import xgboost as xgb
        import joblib
        from sklearn.multioutput import MultiOutputRegressor
        context_df = pd.read_parquet(os.path.join(PREPARED_DIR, "context.parquet"))
        print(f"Building dataset from {len(context_df['id'].unique())} series...")
        X, y = _build_Xy(context_df)
        print(f"Training on {len(X)} examples, {X.shape[1]} features → {y.shape[1]} targets")
        model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                learning_rate=LEARNING_RATE,
                tree_method="hist",
                verbosity=0,
            ),
            n_jobs=-1,
        )
        model.fit(X, y)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        import joblib
        joblib.dump(model, MODEL_PATH)
        print(f"Saved {MODEL_PATH}")

    def _get_model(self):
        if self._model is None:
            import joblib
            self._model = joblib.load(MODEL_PATH)
        return self._model

    def _predict_series(self, series_id, context, future):
        model = self._get_model()
        s = context.tail(CONTEXT_LEN)
        X = _build_features(s).reshape(1, -1)
        predictions = model.predict(X)[0]
        return [
            {
                "id": series_id,
                "timestamp": frow["timestamp"],
                "target_name": "target",
                "prediction": float(predictions[step]),
                "uncertainty": 0.0,
            }
            for step, (_, frow) in enumerate(future.iterrows())
        ]


if __name__ == "__main__":
    import sys
    {"train": lambda: XGBoostModel().train(), "predict": lambda: XGBoostModel().predict()}.get(
        sys.argv[1] if len(sys.argv) > 1 else "", lambda: XGBoostModel().train()
    )()
