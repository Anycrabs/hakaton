import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
from .metrics import wmae

TARGET_COL = "target"
ID_COL = "id"
DATE_COL = "dt"
WEIGHT_COL = "sample_weight"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_PATH = DATA_DIR / "hackathon_income_train.csv"
TEST_PATH = DATA_DIR / "hackathon_income_test.csv"
CSV_READ_KWARGS = {
    "sep": ";",
    "encoding": "latin1",
    "encoding_errors": "replace",
    "engine": "python",
}


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]
    return cat_cols, num_cols


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", cat_transformer, cat_cols),
            ("numerical", num_transformer, num_cols),
        ]
    )
    return preprocessor


def build_model(random_state: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=random_state,
    )


def build_pipeline(preprocessor: ColumnTransformer, model: XGBRegressor) -> Pipeline:
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    def coerce_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

    weight = coerce_numeric(df[WEIGHT_COL]) if WEIGHT_COL in df.columns else None
    y = coerce_numeric(df[TARGET_COL])
    exclude_cols = {TARGET_COL, WEIGHT_COL, ID_COL}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    return X, y, weight


def _read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, **CSV_READ_KWARGS)


def load_train() -> pd.DataFrame:
    return _read_csv_safe(TRAIN_PATH)


def load_test() -> pd.DataFrame:
    return _read_csv_safe(TEST_PATH)


def train_with_cv(df: pd.DataFrame, n_splits: int = 5):
    X, y, weight = prepare_xy(df)
    cat_cols, num_cols = get_column_types(X)
    preprocessor = build_preprocessor(cat_cols, num_cols)
    model = build_model()
    pipeline = build_pipeline(preprocessor, model)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if weight is not None:
            w_train = weight.iloc[train_idx]
            w_val = weight.iloc[val_idx]
            pipeline.fit(X_train, y_train, model__sample_weight=w_train)
            pred = pipeline.predict(X_val)
            score = -wmae(y_val, pred, sample_weight=w_val)
        else:
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_val)
            score = -wmae(y_val, pred)
        scores.append(score)
        print(f"[CV] Fold {fold}/{n_splits} done, WMAE={-score:.4f}")

    if weight is not None:
        pipeline.fit(X, y, model__sample_weight=weight)
    else:
        pipeline.fit(X, y)
    return pipeline, scores
