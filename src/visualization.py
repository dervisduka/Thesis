from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve


sns.set_theme(style="whitegrid")


def _ensure_dir(path: str | Path) -> Path:
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def plot_class_distribution(y: pd.Series, dataset_name: str, output_dir: str | Path) -> Path:
    out_dir = _ensure_dir(output_dir)
    out_path = out_dir / f"{dataset_name}_class_distribution.png"
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title(f"{dataset_name}: Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray], dataset_name: str, output_dir: str | Path
) -> None:
    out_dir = _ensure_dir(output_dir)
    for model_name, cm in confusion_matrices.items():
        safe = model_name.lower().replace(" ", "_")
        out_path = out_dir / f"{dataset_name}_{safe}_confusion_matrix.png"
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=[0, 1], yticklabels=[0, 1])
        plt.title(f"{dataset_name}: {model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_roc_curves(
    y_test: pd.Series,
    scores_by_model: Dict[str, np.ndarray],
    dataset_name: str,
    output_dir: str | Path,
) -> Path | None:
    if set(np.unique(y_test)) != {0, 1}:
        return None
    out_dir = _ensure_dir(output_dir)
    out_path = out_dir / f"{dataset_name}_roc_curves.png"
    plt.figure(figsize=(7, 5))
    for model_name, y_score in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset_name}: ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_model_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), constrained_layout=True)
    for idx, metric in enumerate(metrics):
        sns.barplot(data=metrics_df, x="Model", y=metric, ax=axes[idx])
        axes[idx].set_title(metric)
        axes[idx].tick_params(axis="x", rotation=45)
    fig.suptitle("Model Comparison")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def plot_feature_importance(model_pipeline, feature_names: list[str], model_name: str, output_dir: str | Path) -> Path | None:
    model = model_pipeline.named_steps.get("model")
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None

    out_dir = _ensure_dir(output_dir)
    out_path = out_dir / f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
    names = [p[0] for p in pairs]
    scores = [p[1] for p in pairs]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=scores, y=names, orient="h")
    plt.title(f"{model_name}: Top Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

