import os
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transform.transformer import _ElectricityBert
from config import (
    CONTEXT_LEN, PREDICTION_LEN, MODEL_DIR, PREPARED_DIR,
    TRAIN_EPOCHS, TRAIN_LR, TRAIN_WEIGHT_DECAY, TRAIN_GRAD_CLIP, TRAIN_BATCH_SIZE, TRAIN_STRIDE, MAX_BATCHES_PER_EPOCH,
)


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


if __name__ == "__main__":
    context_df = pd.read_parquet(os.path.join(PREPARED_DIR, "context.parquet"))

    all_ids = sorted(context_df["id"].unique(), key=int)
    split = int(len(all_ids) * 0.9)
    train_df = context_df[context_df["id"].isin(all_ids[:split])]
    val_df = context_df[context_df["id"].isin(all_ids[split:])]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _ElectricityBert(device)
    path = os.path.join(MODEL_DIR, "transformer.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded existing model from {path}")

    print("Building train dataset...")
    train_dataset = ElectricityDataset(train_df)
    print("Building val dataset...")
    val_dataset = ElectricityDataset(val_df)

    loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE)
    print(f"Train: {len(train_dataset)} examples  Val: {len(val_dataset)} examples  device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
    criterion = nn.MSELoss()

    for epoch in range(TRAIN_EPOCHS):
        print(f"Starting epoch {epoch + 1}/{TRAIN_EPOCHS}")
        model.train()
        total_loss = 0.0
        total_time = 0.0
        batches = itertools.islice(loader, MAX_BATCHES_PER_EPOCH)
        for idx, (values, dow, hour, month, targets) in enumerate(batches):
            t0 = time.time()
            values, dow, hour, month, targets = (
                values.to(device), dow.to(device),
                hour.to(device), month.to(device), targets.to(device),
            )
            mean, _ = model(values, dow, hour, month)
            loss = criterion(mean, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()
            total_time += time.time() - t0
            if idx % 10 == 0:
                print(f"  batch {idx}  loss={loss.item():.4f}  avg_time={total_time / (idx + 1):.3f}s")
        n_batches = idx + 1
        train_loss = total_loss / n_batches

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for values, dow, hour, month, targets in itertools.islice(val_loader, MAX_BATCHES_PER_EPOCH):
                values, dow, hour, month, targets = (
                    values.to(device), dow.to(device),
                    hour.to(device), month.to(device), targets.to(device),
                )
                mean, _ = model(values, dow, hour, month)
                val_loss += criterion(mean, targets).item()
        val_loss /= min(MAX_BATCHES_PER_EPOCH or len(val_loader), len(val_loader))
        print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "transformer.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved {path}")