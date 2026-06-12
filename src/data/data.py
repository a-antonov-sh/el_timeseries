import os
import pandas as pd
from datasets import load_dataset
from data.config import PREPARED_DIR, TRAIN_HISTORY


class Data:
    def __init__(self):
        self.train = None
        self.test = None

    def load_data(self):
        dataset = load_dataset("LeoTungAnh/electricity_hourly")
        print(dataset)
        self.train = dataset["train"]
        self.test = dataset["test"]
        self.test = self.test.remove_columns("target")
        # print("Train:", self.train)
        # print("Test:", self.test)
        # print("Features:", self.train.features)
        # print("Train sample:", self.train[0])

    # def prepare(self, seq_len=168, forward=24):

    def _build_context_rows(self):
        rows = []
        total = len(self.train)
        for i, row in enumerate(self.train):
            if i % 50 == 0:
                print(f"  context: {i}/{total} series")
            target = row["target"] if TRAIN_HISTORY is None else row["target"][-TRAIN_HISTORY:]
            offset = 0 if TRAIN_HISTORY is None else max(0, len(row["target"]) - TRAIN_HISTORY)
            start = pd.Timestamp(row["start"]) + pd.Timedelta(hours=offset)
            for step, value in enumerate(target):
                ts = start + pd.Timedelta(hours=step)
                rows.append({
                    "id": str(i),
                    "timestamp": ts,
                    "target": value,
                    "day_of_week": ts.day_of_week,
                    "hour": ts.hour,
                    "month": ts.month,
                })
        return pd.DataFrame(rows)

    def _build_future_rows(self):
        rows = []
        total = len(self.train)
        for i, row in enumerate(self.train):
            if i % 50 == 0:
                print(f"  future: {i}/{total} series")
            last_ts = pd.Timestamp(row["start"]) + pd.Timedelta(hours=len(row["target"]))
            for step in range(24):
                ts = last_ts + pd.Timedelta(hours=step)
                rows.append({
                    "id": str(i),
                    "timestamp": ts,
                    "day_of_week": ts.day_of_week,
                    "hour": ts.hour,
                    "month": ts.month,
                })
        return pd.DataFrame(rows)

    def save_prepared(self):
        os.makedirs(PREPARED_DIR, exist_ok=True)
        print("Building context...")
        self._build_context_rows().to_parquet(os.path.join(PREPARED_DIR, "context.parquet"), index=False)
        print("Building future...")
        self._build_future_rows().to_parquet(os.path.join(PREPARED_DIR, "future.parquet"), index=False)
        print("Done.")

    def to_chronos_2(self, split="train"):
        if split == "train":
            return self._build_context_rows()
        else:
            return self._build_future_rows()

