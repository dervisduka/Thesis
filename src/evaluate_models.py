from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _positive_scores(model, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        scores = np.asarray(scores, dtype=float)
        # Min-max scaling for consistent risk-score interpretation.
        denom = np.max(scores) - np.min(scores)
        if denom == 0:
            return np.zeros_like(scores)
        return (scores - np.min(scores)) / denom
    raise ValueError("Model does not support probability-like scoring.")


def evaluate_trained_models(
    trained_models: Dict[str, object],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rows = []
    confusion_matrices: Dict[str, np.ndarray] = {}
    scores_by_model: Dict[str, np.ndarray] = {}

    unique_labels = set(np.unique(y_test))

    for model_name, model in trained_models.items():
        y_pred = model.predict(x_test)
        y_scores = _positive_scores(model, x_test)
        scores_by_model[model_name] = y_scores
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        confusion_matrices[model_name] = cm

        roc_auc = np.nan
        if unique_labels == {0, 1}:
            roc_auc = roc_auc_score(y_test, y_scores)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("Recall", ascending=False).reset_index(drop=True)
    return metrics_df, confusion_matrices, scores_by_model


def save_metrics(metrics_df: pd.DataFrame, output_csv: str | Path) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)

