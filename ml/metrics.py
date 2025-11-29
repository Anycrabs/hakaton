import numpy as np


def wmae(y_true, y_pred, sample_weight=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
    return float(np.sum(sample_weight * np.abs(y_true - y_pred)) / np.sum(sample_weight))
