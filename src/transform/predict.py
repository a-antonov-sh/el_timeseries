import os
import pandas as pd
from transform.transformer import Transformer
from config import PREPARED_DIR, PREDICTIONS_DIR, MODEL_DIR

if __name__ == "__main__":
    context_df = pd.read_parquet(os.path.join(PREPARED_DIR, "context.parquet"))
    future_df = pd.read_parquet(os.path.join(PREPARED_DIR, "future.parquet"))

    model_path = os.path.join(MODEL_DIR, "transformer.pt")
    transformer = Transformer(context_df, future_df, model_path=model_path)

    print(transformer.predictions.to_string())
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    transformer.predictions.to_parquet(
        os.path.join(PREDICTIONS_DIR, "transformer_predictions.parquet"), index=False
    )