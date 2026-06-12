import os
from dataclasses import dataclass
from config import CONTEXT_LEN, PREDICTION_LEN

HIDDEN_SIZE = 256
MODEL_DIR = os.path.join("data", "models")

TRAIN_EPOCHS = 10
TRAIN_LR = 1e-6
TRAIN_WEIGHT_DECAY = .95
TRAIN_GRAD_CLIP = .001
TRAIN_BATCH_SIZE = 1024
TRAIN_STRIDE = 1
MAX_BATCHES_PER_EPOCH = None  # set to an int to cap batches per epoch
MAX_VAL_BATCHES = None
TRAIN_L1_WEIGHT = 0.0  # weight for L1 term; 0 = pure MSE, 1 = pure MAE


@dataclass
class BertConfig:
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.5