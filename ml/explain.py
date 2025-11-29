import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def _extract_importance(model) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        importance = getattr(model, "feature_importances_", None)
        if importance is not None:
            return np.asarray(importance, dtype=float)
    booster = getattr(model, "get_booster", lambda: None)()
    if booster is not None:
        score = booster.get_score(importance_type="weight")
        if score:
            # Map to array ordered by model.feature_names_in_ if present
            if hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
                return np.asarray([score.get(n, 0.0) for n in names], dtype=float)
            else:
                values = list(score.values())
                return np.asarray(values + [0.0] * (len(score) - len(values)), dtype=float)
    return np.array([])


def get_feature_importance(model, feature_names: List[str], top_n: int = 10, plot_path: str = "feature_importance.png"):
    importance = _extract_importance(model)
    if importance.size == 0:
        importance = np.zeros(len(feature_names), dtype=float)
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    df = df.sort_values(by="importance", ascending=False).head(top_n)
    plt.figure(figsize=(8, 5))
    plt.barh(df["feature"], df["importance"], color="#1f3c88")
    plt.gca().invert_yaxis()
    plt.title("Top feature importance")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return df.to_dict(orient="records")
