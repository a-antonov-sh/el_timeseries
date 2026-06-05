import os
from dataclasses import dataclass

PREPARED_DIR = os.path.join("data", "prepared")
PREDICTIONS_DIR = os.path.join("data", "predictions")
QUANTILE_LEVELS = [0.2, 0.4, 0.6, 0.8]

CONTEXT_LEN = 168
PREDICTION_LEN = 24
HIDDEN_SIZE = 256
TRAIN_HISTORY = 8760  # 365 * 24: keep only last year of each series

TRAIN_EPOCHS = 10
TRAIN_LR = 1e-4
TRAIN_WEIGHT_DECAY = 5e-1
TRAIN_GRAD_CLIP = .001
TRAIN_BATCH_SIZE = 512
TRAIN_STRIDE = 24
MAX_BATCHES_PER_EPOCH = 200  # set to an int to cap batches per epoch

MODEL_DIR = os.path.join("data", "models")


@dataclass
class BertConfig:
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.5
