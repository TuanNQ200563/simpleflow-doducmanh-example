import numpy as np
from sklearn import metrics


def scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_1d = np.argmax(y_pred, axis=1)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred_1d)
    return float(np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix))
