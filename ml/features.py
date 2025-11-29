import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.model_selection import KFold, cross_val_score
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
CSV_READ_KWARGS = {"sep": ";", "encoding": "cp1251"}


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
    weight = df[WEIGHT_COL] if WEIGHT_COL in df.columns else None
    y = df[TARGET_COL]
    exclude_cols = {TARGET_COL, WEIGHT_COL, ID_COL}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    return X, y, weight


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH, **CSV_READ_KWARGS)


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH, **CSV_READ_KWARGS)


def train_with_cv(df: pd.DataFrame, n_splits: int = 5):
    X, y, weight = prepare_xy(df)
    cat_cols, num_cols = get_column_types(X)
    preprocessor = build_preprocessor(cat_cols, num_cols)
    model = build_model()
    pipeline = build_pipeline(preprocessor, model)
    scorer = make_scorer(wmae, greater_is_better=False)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fit_params = {"model__sample_weight": weight} if weight is not None else None
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer, fit_params=fit_params)
    if weight is not None:
        pipeline.fit(X, y, model__sample_weight=weight)
    else:
        pipeline.fit(X, y)
    return pipeline, scores
