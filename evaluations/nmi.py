import numpy as np
from sklearn import metrics


def scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_1d = np.argmax(y_pred, axis=1)
    return float(metrics.cluster.normalized_mutual_info_score(y_true, y_pred_1d))
