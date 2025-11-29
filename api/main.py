import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List
import pandas as pd
from income_service.ml.inference import load_model, predict_income
from income_service.ml.explain import get_feature_importance
from income_service.ml.features import TRAIN_PATH
from .models import ClientRequest, PredictionResponse, FeatureImportanceItem
from .recommender import recommend_products

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "ml" / "model.pkl"
MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

app = FastAPI(title="Income Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
FEATURE_NAMES: List[str] = []
FEATURE_IMPORTANCE: List[dict] = []


@app.on_event("startup")
def startup_event():
    global MODEL, FEATURE_NAMES, FEATURE_IMPORTANCE
    MODEL = load_model(MODEL_PATH)
    ref = pd.read_csv(TRAIN_PATH)
    pre = MODEL.named_steps["preprocess"]
    FEATURE_NAMES = pre.get_feature_names_out().tolist()
    FEATURE_IMPORTANCE = get_feature_importance(
        MODEL.named_steps["model"], FEATURE_NAMES, plot_path=str(BASE_DIR / "ml" / "feature_importance.png")
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_income", response_model=PredictionResponse)
def predict_income_endpoint(payload: ClientRequest):
    client_dict = payload.dict()
    pred = predict_income(client_dict, model=MODEL)
    products = recommend_products(pred, client_dict)
    return PredictionResponse(
        predicted_income=pred,
        feature_importance=[FeatureImportanceItem(**fi) for fi in FEATURE_IMPORTANCE[:10]],
        recommended_products=products,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
