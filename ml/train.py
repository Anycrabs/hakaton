import os
import pickle
import pandas as pd
import time
from pathlib import Path
from .features import TRAIN_PATH, train_with_cv, load_train
from .explain import get_feature_importance

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
PLOT_PATH = Path(os.getenv("PLOT_PATH", MODEL_PATH.parent / "feature_importance.png"))
CSV_READ_KWARGS = {"sep": ";", "encoding": "cp1251"}


def main():
    t0 = time.perf_counter()
    df = load_train()
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    pipeline, scores = train_with_cv(df)
    print("CV WMAE per fold:", [-s for s in scores])
    print("CV WMAE mean:", -pd.Series(scores).mean())
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    get_feature_importance(pipeline.named_steps["model"], feature_names, plot_path=str(PLOT_PATH))
    for path in {MODEL_PATH, DEFAULT_MODEL_PATH}:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"Model saved to {path}")
    print(f"Feature importance plot saved to {PLOT_PATH}")
    print(f"Training finished in {(time.perf_counter()-t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
