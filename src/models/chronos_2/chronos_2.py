import pandas as pd
from chronos import Chronos2Pipeline
from data.data import Data
from models.chronos_2.config import QUANTILE_LEVELS
from common.base_model import BaseModel

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


class Chronos2Model(BaseModel):
    name = "chronos2"
    help = "Chronos-2 zero-shot forecast"
    predictions_filename = "chronos_2_predictions.parquet"
    supports_training = False

    def predict(self):
        data = Data()
        data.load_data()
        context_df = data.to_chronos_2(split="train")
        future_df = data.to_chronos_2(split="test")
        predictions = pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=24,
            quantile_levels=QUANTILE_LEVELS,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        q_cols = [str(q) for q in QUANTILE_LEVELS]
        predictions["prediction"] = predictions[q_cols].mean(axis=1)
        predictions["uncertainty"] = predictions[q_cols[-1]] - predictions[q_cols[0]]
        predictions.drop(columns=q_cols, inplace=True)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        self._save(predictions)