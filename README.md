# Chronos-2 Electricity Forecasting

Hourly electricity consumption forecasting using multiple models:
- **Chronos-2** — Amazon's pretrained zero-shot time series foundation model
- **Transformer** — Custom BERT-based model trained from scratch on the same data
- **XGBoost** — Multi-output gradient boosted trees (one regressor per forecast step)
- **kNN** — k-Nearest Neighbours regressor
- **ARIMA** — Per-series auto_arima with confidence interval uncertainty

Dataset: [`LeoTungAnh/electricity_hourly`](https://huggingface.co/datasets/LeoTungAnh/electricity_hourly) — 370 series of hourly electricity consumption.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Config is split per module under `src/` — shared globals in `src/config.py`, per-model configs in `src/models/<model>/config.py`.

---

## Pipeline

All commands are run through the single entry point `exec/run`:

```
exec/run <command> [subcommand] [options]
```

### 1. Prepare data

```bash
exec/run prepare
```

Builds context and future DataFrames with `day_of_week`, `hour`, and `month` covariates, saves to `data/prepared/`.

Output:
- `data/prepared/context.parquet` — training history per series
- `data/prepared/future.parquet` — 24 future timestamps per series

---

### 2. Train & predict

| Model       | Train                              | Predict                              |
|-------------|------------------------------------|--------------------------------------|
| Transformer | `exec/run transformer train`       | `exec/run transformer predict`       |
| XGBoost     | `exec/run xgboost train`           | `exec/run xgboost predict`           |
| kNN         | `exec/run knn train`               | `exec/run knn predict`               |
| ARIMA       | `exec/run arima train`             | `exec/run arima predict`             |
| Chronos-2   | *(zero-shot, no training)*         | `exec/run chronos2 predict`          |

Transformer training resumes from `data/models/transformer.pt` if it exists. CLI overrides:

```bash
exec/run transformer train --epochs 5 --lr 1e-5 --batch-size 512
```

Key transformer config (`src/models/transform/config.py`):

| Config | Default | Description |
|--------|---------|-------------|
| `TRAIN_EPOCHS` | 10 | Number of epochs |
| `TRAIN_LR` | 1e-4 | Learning rate |
| `TRAIN_WEIGHT_DECAY` | 1e-3 | AdamW weight decay |
| `TRAIN_GRAD_CLIP` | 0.01 | Gradient clipping norm |
| `TRAIN_BATCH_SIZE` | 512 | Batch size |
| `TRAIN_STRIDE` | 24 | Sliding window stride (hours) |
| `MAX_BATCHES_PER_EPOCH` | 200 | Cap batches per epoch (`None` = all) |

---

### 3. Evaluate

```bash
exec/run evaluate
```

Evaluates all available prediction files against the validation split ground truth.

Current results (first 30 series):

| Model        | MAE           | RMSE          | Mean uncertainty |
|--------------|---------------|---------------|------------------|
| Chronos-2    | 1.021         | 1.428         | 1.0633           |
| Transformer  | **0.828**     | **1.215**     | 0.998            |
| XGBoost      | 0.8524        | 1.2405        | N/A              |
| kNN          | 0.8795        | 1.2476        | N/A              |
| ARIMA        | 1.0148        | 1.3698        | 3.5019           |

---

## Project structure

```
src/
  config.py                        — shared globals (CONTEXT_LEN, PREDICTION_LEN)
  program.py                       — CLI dispatcher
  evaluate.py                      — evaluation command
  common/
    base_model.py                  — BaseModel / SeriesModel ABC
    command.py                     — Command ABC (prepare, evaluate)
  data/
    data.py                        — dataset loading and preprocessing
    config.py                      — data paths and TRAIN_HISTORY
  models/
    chronos_2/
      chronos_2.py                 — Chronos2Model (zero-shot via Chronos2Pipeline)
      config.py
    transform/
      transformer.py               — TransformerModel
      electricity_bert.py          — ElectricityBert (nn.Module)
      dataset.py                   — ElectricityDataset (sliding windows)
      train.py                     — training loop
      bert.py                      — BERT encoder backbone
      config.py
    xg_boost/
      xg_boost.py                  — XGBoostModel
      config.py
    kNN/
      kNN.py                       — KNNModel
      config.py
    ARIMA/
      arima.py                     — ARIMAModel (auto_arima per series)
      config.py
exec/
  run                              — unified pipeline entry point
data/
  prepared/                        — preprocessed parquet files
  predictions/                     — model output parquet files
  models/                          — saved model weights / joblib files
```
