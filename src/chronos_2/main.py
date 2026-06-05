import os
import pandas as pd
from data.data import Data
from chronos_2.model.chronos_2 import Chronos_2
from config import PREDICTIONS_DIR, QUANTILE_LEVELS


if __name__ == '__main__':
    data = Data()
    data.load_data()
    context_df = data.to_chronos_2(split="train")
    future_df = data.to_chronos_2(split="test")
    model = Chronos_2(context_df, future_df)
    q_cols = [str(q) for q in QUANTILE_LEVELS]
    model.predictions["prediction"] = model.predictions[q_cols].mean(axis=1)
    model.predictions["uncertainty"] = model.predictions[q_cols[-1]] - model.predictions[q_cols[0]]
    model.predictions.drop(columns=q_cols, inplace=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(model.predictions.to_string())
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    model.predictions.to_parquet(os.path.join(PREDICTIONS_DIR, "predictions.parquet"), index=False)


