import os

PREPARED_DIR = os.path.join("data", "prepared")
PREDICTIONS_DIR = os.path.join("data", "predictions")
TRAIN_HISTORY = None  # 365 * 24: keep only last year of each series