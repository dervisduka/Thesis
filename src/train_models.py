from __future__ import annotations

from time import perf_counter
from typing import Dict, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_model_pipeline

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def get_models(random_state: int = 42, fast: bool = False) -> Dict[str, object]:
    if fast:
        models: Dict[str, object] = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state
            ),
            "Decision Tree": DecisionTreeClassifier(
                class_weight="balanced", random_state=random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=120,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        }
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
                verbosity=0,
            )
        return models

    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        )
    return models


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    use_smote: bool = True,
    fast: bool = False,
    progress: bool = True,
) -> Dict[str, object]:
    fitted = {}
    models = get_models(random_state=random_state, fast=fast)
    total = len(models)
    for index, (name, model) in enumerate(models.items(), start=1):
        if progress:
            print(f"[train {index}/{total}] {name} started...", flush=True)
        started = perf_counter()
        pipeline = build_model_pipeline(
            clone(model),
            x_train=x_train,
            use_smote=use_smote,
            random_state=random_state,
        )
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        if progress:
            elapsed = perf_counter() - started
            print(f"[train {index}/{total}] {name} done in {elapsed:.1f}s", flush=True)
    return fitted


def select_best_model(metrics_df: pd.DataFrame, trained_models: Dict[str, object]) -> Tuple[str, object]:
    ordered = metrics_df.sort_values(["Recall", "ROC-AUC"], ascending=[False, False]).reset_index(drop=True)
    best_name = str(ordered.loc[0, "Model"])
    return best_name, trained_models[best_name]
