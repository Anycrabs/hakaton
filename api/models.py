from pydantic import BaseModel, Field
from typing import List


class ClientRequest(BaseModel):
    age: float = Field(..., example=35)
    employment_type: str = Field(..., example="salaried")
    credit_score: float = Field(..., example=720)
    savings_balance: float = Field(..., example=15000)
    city: str = Field(..., example="Moscow")
    dependents: int = Field(..., example=1)


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class Product(BaseModel):
    product_id: str
    name: str
    description: str
    reason: str


class PredictionResponse(BaseModel):
    predicted_income: float
    feature_importance: List[FeatureImportanceItem]
    recommended_products: List[Product]
