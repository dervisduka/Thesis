from __future__ import annotations

from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_data(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    try:
        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in x.columns if col not in numeric_cols]

    num_pipe = SkPipeline(steps=[("scaler", StandardScaler())])
    cat_pipe = SkPipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def build_model_pipeline(model, x_train: pd.DataFrame, use_smote: bool = False, random_state: int = 42):
    preprocessor = build_preprocessor(x_train)
    steps = [("preprocess", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
        steps.append(("model", model))
        return ImbPipeline(steps=steps)

    steps.append(("model", model))
    return SkPipeline(steps=steps)

