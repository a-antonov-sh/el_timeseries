import numpy as np
import torch
from torch.utils.data import Dataset
from models.transform.config import CONTEXT_LEN, PREDICTION_LEN, TRAIN_STRIDE


class ElectricityDataset(Dataset):
    def __init__(self, context_df):
        self.examples = []
        ids = sorted(context_df["id"].unique(), key=int)
        total = len(ids)
        for idx, series_id in enumerate(ids):
            if idx % 50 == 0:
                print(f"  building dataset: {idx}/{total} series, {len(self.examples)} examples so far")
            s = context_df[context_df["id"] == series_id].sort_values("timestamp")
            values = s["target"].values.astype(np.float32)
            dow = s["day_of_week"].values.astype(np.int64)
            hour = s["hour"].values.astype(np.int64)
            month = s["month"].values.astype(np.int64)
            n = len(values)
            for start in range(0, n - CONTEXT_LEN - PREDICTION_LEN + 1, TRAIN_STRIDE):
                end = start + CONTEXT_LEN
                self.examples.append((
                    values[start:end],
                    dow[start:end],
                    hour[start:end],
                    month[start:end],
                    values[end:end + PREDICTION_LEN],
                ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        values, dow, hour, month, target = self.examples[idx]
        return (
            torch.tensor(values),
            torch.tensor(dow),
            torch.tensor(hour),
            torch.tensor(month),
            torch.tensor(target),
        )
