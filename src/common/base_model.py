from abc import ABC, abstractmethod
import os
import pandas as pd
from data.config import PREPARED_DIR, PREDICTIONS_DIR


class BaseModel(ABC):
    name: str
    help: str
    predictions_filename: str
    supports_training: bool = True

    def train(self):
        raise NotImplementedError(f"{type(self).__name__} does not support training")

    @abstractmethod
    def predict(self):
        ...

    @classmethod
    def add_parser(cls, subparsers):
        parser = subparsers.add_parser(cls.name, help=cls.help)
        sub = parser.add_subparsers(required=True)
        if cls.supports_training:
            cls._add_subcommand(sub, "train", "Train the model")
        cls._add_subcommand(sub, "predict", "Run inference")

    @classmethod
    def _add_subcommand(cls, sub, name, help_text):
        p = sub.add_parser(name, help=help_text)
        model_cls = cls
        method = name
        class Cmd:
            def __init__(self, args): pass
            def __call__(self): getattr(model_cls(), method)()
        p.set_defaults(script=Cmd)

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        context_df = pd.read_parquet(os.path.join(PREPARED_DIR, "context.parquet"))
        future_df = pd.read_parquet(os.path.join(PREPARED_DIR, "future.parquet"))
        return context_df, future_df

    def _save(self, df: pd.DataFrame):
        print(df.to_string())
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        df.to_parquet(os.path.join(PREDICTIONS_DIR, self.predictions_filename), index=False)


class SeriesModel(BaseModel):
    """Base for models that predict one series at a time."""

    def predict(self):
        context_df, future_df = self._load_data()
        rows = []
        for series_id in sorted(context_df["id"].unique(), key=int):
            context_s = context_df[context_df["id"] == series_id].sort_values("timestamp")
            future_s = future_df[future_df["id"] == series_id].sort_values("timestamp")
            rows.extend(self._predict_series(series_id, context_s, future_s))
        self._save(pd.DataFrame(rows))

    @abstractmethod
    def _predict_series(self, series_id: str, context: pd.DataFrame, future: pd.DataFrame) -> list[dict]:
        ...
