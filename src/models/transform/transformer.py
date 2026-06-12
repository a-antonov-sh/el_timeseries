import os
import torch
import pandas as pd
from models.transform.electricity_bert import ElectricityBert
from models.transform.config import CONTEXT_LEN, MODEL_DIR
from common.base_model import BaseModel


class TransformerModel(BaseModel):
    name = "transformer"
    help = "Transformer model — train or predict"
    predictions_filename = "transformer_predictions.parquet"

    @classmethod
    def add_parser(cls, subparsers):
        parser = subparsers.add_parser(cls.name, help=cls.help)
        sub = parser.add_subparsers(required=True)
        train_p = sub.add_parser("train", help="Train the model")
        train_p.add_argument("--epochs", type=int, default=None)
        train_p.add_argument("--lr", type=float, default=None)
        train_p.add_argument("--batch-size", type=int, default=None)
        class TrainCmd:
            def __init__(self, args): self.args = args
            def __call__(self):
                cls().train(epochs=self.args.epochs, lr=self.args.lr, batch_size=self.args.batch_size)
        train_p.set_defaults(script=TrainCmd)
        cls._add_subcommand(sub, "predict", "Run inference")

    def train(self, epochs=None, lr=None, batch_size=None):
        import models.transform.config as cfg
        if epochs is not None: cfg.TRAIN_EPOCHS = epochs
        if lr is not None: cfg.TRAIN_LR = lr
        if batch_size is not None: cfg.TRAIN_BATCH_SIZE = batch_size
        from models.transform.train import _run_training
        _run_training()

    def predict(self):
        context_df, future_df = self._load_data()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ElectricityBert(device)
        model_path = os.path.join(MODEL_DIR, "transformer.pt")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        self._save(self._run_predict(model, device, context_df, future_df))

    def _run_predict(self, model, device, context_df, future_df):
        model.eval()
        tensors = self._build_tensors(device, context_df)
        rows = []
        with torch.no_grad():
            for i, (values, dow, hour, month) in enumerate(tensors):
                mean, std = model(values.unsqueeze(0), dow.unsqueeze(0), hour.unsqueeze(0), month.unsqueeze(0))
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

    def _build_tensors(self, device, context_df):
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
                month = torch.cat([torch.ones(pad, dtype=torch.long), month])
            tensors.append((
                values.to(device),
                dow.to(device),
                hour.to(device),
                month.to(device),
            ))
        return tensors
