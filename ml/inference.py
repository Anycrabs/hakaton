import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List
from .features import TRAIN_PATH, TARGET_COL, WEIGHT_COL, ID_COL, CSV_READ_KWARGS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))


def load_model(model_path: Path = MODEL_PATH):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _reference_columns() -> List[str]:
    ref = pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS)
    return [c for c in ref.columns if c not in {TARGET_COL, WEIGHT_COL, ID_COL}]


REFERENCE_COLUMNS = _reference_columns() if TRAIN_PATH.exists() else []


def preprocess_input(client_data: Dict[str, Any], reference_cols: List[str] = None) -> pd.DataFrame:
    if reference_cols is None:
        reference_cols = REFERENCE_COLUMNS or _reference_columns()
    df = pd.DataFrame([client_data])
    for col in reference_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[reference_cols]
    return df


def predict_income(client_data: Dict[str, Any], model=None) -> float:
    if model is None:
        model = load_model()
    X = preprocess_input(client_data)
    pred = model.predict(X)[0]
    return float(pred)
