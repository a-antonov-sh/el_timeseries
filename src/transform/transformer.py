import os
import torch
import torch.nn as nn
import pandas as pd
from transform.bert import BERT
from config import CONTEXT_LEN, PREDICTION_LEN, HIDDEN_SIZE, BertConfig


class _ElectricityBert(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.value_proj = nn.Linear(1, HIDDEN_SIZE).to(device)
        self.day_emb = nn.Embedding(7, HIDDEN_SIZE).to(device)
        self.hour_emb = nn.Embedding(24, HIDDEN_SIZE).to(device)
        self.month_emb = nn.Embedding(13, HIDDEN_SIZE).to(device)  # months 1-12
        self.bert = BERT(array_size=CONTEXT_LEN, input_size=HIDDEN_SIZE, cfg=BertConfig(), device=device)
        self.head = nn.Linear(HIDDEN_SIZE, PREDICTION_LEN * 2).to(device)

    def forward(self, values, day_of_week, hour, month):
        emb = (self.value_proj(values.unsqueeze(-1)) +
               self.day_emb(day_of_week) +
               self.hour_emb(hour) +
               self.month_emb(month))
        out = self.bert(emb)[:, -1, :]
        mean, log_std = self.head(out).chunk(2, dim=-1)
        return mean, log_std.exp()


class Transformer:
    def __init__(self, context_df, future_df, model_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _ElectricityBert(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.predictions = self._predict(context_df, future_df)

    def _build_tensors(self, context_df):
        tensors = []
        for series_id in sorted(context_df["id"].unique(), key=int):
            s = context_df[context_df["id"] == series_id].sort_values("timestamp").tail(CONTEXT_LEN)
            pad = CONTEXT_LEN - len(s)
            values = torch.tensor(s["target"].values, dtype=torch.float32)
            dow = torch.tensor(s["day_of_week"].values, dtype=torch.long)
            hour = torch.tensor(s["hour"].values, dtype=torch.long)
            month = torch.tensor(s["month"].values, dtype=torch.long)
            if pad > 0:
                values = torch.cat([torch.zeros(pad), values])
                dow = torch.cat([torch.zeros(pad, dtype=torch.long), dow])
                hour = torch.cat([torch.zeros(pad, dtype=torch.long), hour])
                month = torch.cat([torch.ones(pad, dtype=torch.long), month])  # pad with month=1
            tensors.append((
                values.to(self.device),
                dow.to(self.device),
                hour.to(self.device),
                month.to(self.device),
            ))
        return tensors

    def _predict(self, context_df, future_df):
        self.model.eval()
        tensors = self._build_tensors(context_df)
        rows = []
        with torch.no_grad():
            for i, (values, dow, hour, month) in enumerate(tensors):
                mean, std = self.model(values.unsqueeze(0), dow.unsqueeze(0), hour.unsqueeze(0), month.unsqueeze(0))
                mean = mean.squeeze(0).cpu().numpy()
                std = std.squeeze(0).cpu().numpy()
                future = future_df[future_df["id"] == str(i)].sort_values("timestamp")
                for step, (_, frow) in enumerate(future.iterrows()):
                    rows.append({
                        "id": str(i),
                        "timestamp": frow["timestamp"],
                        "target_name": "target",
                        "prediction": float(mean[step]),
                        "uncertainty": float(std[step]),
                    })
        return pd.DataFrame(rows)
