import os
from dataclasses import dataclass

PREPARED_DIR = os.path.join("data", "prepared")
PREDICTIONS_DIR = os.path.join("data", "predictions")
QUANTILE_LEVELS = [0.2, 0.4, 0.6, 0.8]

CONTEXT_LEN = 168
PREDICTION_LEN = 24
HIDDEN_SIZE = 256
TRAIN_HISTORY = 8760  # 365 * 24: keep only last year of each series

TRAIN_EPOCHS = 2
TRAIN_LR = 1e-6
TRAIN_WEIGHT_DECAY = .95
TRAIN_GRAD_CLIP = .001
TRAIN_BATCH_SIZE = 256
TRAIN_STRIDE = 95
MAX_BATCHES_PER_EPOCH = 20  # set to an int to cap batches per epoch
MAX_VAL_BATCHES = 10
TRAIN_L1_WEIGHT = 0.0  # weight for L1 term; 0 = pure MSE, 1 = equal L1+L2

MODEL_DIR = os.path.join("data", "models")


@dataclass
class BertConfig:
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.5
