# Chronos-2 Electricity Forecasting

Hourly electricity consumption forecasting using two models:
- **Chronos-2** — Amazon's pretrained zero-shot time series foundation model
- **Transformer** — Custom BERT-based model trained from scratch on the same data

Dataset: [`LeoTungAnh/electricity_hourly`](https://huggingface.co/datasets/LeoTungAnh/electricity_hourly) — 370 series of hourly electricity consumption.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install chronos-forecasting torch transformers datasets pandas pyarrow
```

All paths and hyperparameters are in `src/config.py`.

---

## Pipeline

### 1. Prepare data

Builds sliding-window context and future DataFrames with `day_of_week` and `hour` covariates, saves to `data/prepared/`.

```bash
./exec/prepare
```

Output:
- `data/prepared/context.parquet` — full training history (370 series × ~26k steps)
- `data/prepared/future.parquet` — 24 future timestamps per series

---

### 2a. Chronos-2 forecast

Runs zero-shot inference using `amazon/chronos-2` on CPU.

```bash
./exec/chronos_2
```

Output: `data/predictions/predictions.parquet`

---

### 2b. Transformer — train

Trains the BERT-based model on sliding windows (context 168h → predict 24h).

```bash
./exec/train
```

Resumes from `data/models/transformer.pt` if it exists. Key config options:

| Config | Default | Description |
|--------|---------|-------------|
| `TRAIN_EPOCHS` | 10      | Number of epochs |
| `TRAIN_LR` | 1e-4    | Learning rate |
| `TRAIN_WEIGHT_DECAY` | 1e-3    | AdamW weight decay |
| `TRAIN_GRAD_CLIP` | 0.01    | Gradient clipping norm |
| `TRAIN_BATCH_SIZE` | 512     | Batch size |
| `TRAIN_STRIDE` | 24      | Sliding window stride (hours) |
| `MAX_BATCHES_PER_EPOCH` | 200     | Cap batches per epoch (`None` = all) |

---

### 2c. Transformer — predict

Runs inference with the saved model.

```bash
./exec/predict
```

Output: `data/predictions/transformer_predictions.parquet`

---

### 3. Evaluate

Evaluates against the validation split ground truth. Runs both models if both prediction files exist.

```bash
./exec/evaluate
```

Current results:

| Model        | MAE       | RMSE      | Mean uncertainty |
|--------------|-----------|-----------|------------------|
| Chronos-2    | 1.008     | 1.376     | 0.531            |
| Transformer  | **0.749** | **1.050** | 1.003            |

---

## Project structure

```
src/
  config.py                  — all hyperparameters and paths
  prepare.py                 — data preparation entry point
  evaluate.py                — evaluation entry point
  data/
    data.py                  — dataset loading and preprocessing
  chronos_2/
    main.py                  — Chronos-2 inference entry point
    model/chronos_2.py       — Chronos2Pipeline wrapper
  transform/
    bert.py                  — BERT encoder (TransformerEncoder + positional embeddings)
    transformer.py           — ElectricityBert model + Transformer inference class
    train.py                 — training loop with sliding-window dataset
    predict.py               — transformer inference entry point
exec/
  prepare                    — run data preparation
  chronos_2                  — run Chronos-2 forecast
  train                      — train the transformer
  predict                    — run transformer forecast
  evaluate                   — evaluate both models
data/
  prepared/                  — preprocessed parquet files
  predictions/               — model output parquet files
  models/                    — saved transformer weights
```