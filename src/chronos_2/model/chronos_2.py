import pandas as pd  # requires: pip install 'pandas[pyarrow]'
from chronos import Chronos2Pipeline
from config import QUANTILE_LEVELS

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


# Generate predictions with covariates

class Chronos_2:
    def __init__(self, context_df, future_df):
        self.predictions = pipeline.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=24,  # Number of steps to forecast
            quantile_levels=QUANTILE_LEVELS,
            id_column="id",  # Column identifying different time series
            timestamp_column="timestamp",  # Column with datetime information
            target="target",  # Column(s) with time series values to predict
        )

