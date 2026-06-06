from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

import pandas as pd


DatasetKind = Literal["fraud", "phishing"]

PHISHING_TARGET_CANDIDATES = ("status", "label", "class", "result")
PHISHING_POSITIVE_TOKENS = {
    "1",
    "phishing",
    "phish",
    "malicious",
    "bad",
    "fraud",
    "true",
    "yes",
}
PHISHING_NEGATIVE_TOKENS = {
    "0",
    "legitimate",
    "legit",
    "safe",
    "good",
    "normal",
    "false",
    "no",
}


def load_csv(path: str | Path) -> pd.DataFrame:
    requested = Path(path)
    project_root = Path(__file__).resolve().parents[1]
    candidates = []

    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.append(Path.cwd() / requested)
        candidates.append(project_root / requested)

    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)

    data_dir = project_root / "data"
    available_csvs = sorted(p.name for p in data_dir.glob("*.csv")) if data_dir.exists() else []
    available_text = ", ".join(available_csvs) if available_csvs else "none"
    raise FileNotFoundError(
        "Dataset not found.\n"
        f"Requested path: {requested}\n"
        f"Checked paths: {', '.join(str(p) for p in candidates)}\n"
        f"Available CSV files in data/: {available_text}\n"
        "Put datasets in data/ (for example: data/creditcard.csv, data/phishing.csv)."
    )


def detect_phishing_target(df: pd.DataFrame) -> str:
    lowered = {col.lower(): col for col in df.columns}
    for candidate in PHISHING_TARGET_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(
        "Phishing target column not found. Expected one of: "
        f"{', '.join(PHISHING_TARGET_CANDIDATES)}"
    )


def normalize_binary_target(series: pd.Series) -> pd.Series:
    if series.dropna().isin([0, 1]).all():
        return series.astype(int)

    mapped = []
    for raw_value in series:
        token = str(raw_value).strip().lower()
        if token in PHISHING_POSITIVE_TOKENS:
            mapped.append(1)
            continue
        if token in PHISHING_NEGATIVE_TOKENS:
            mapped.append(0)
            continue
        raise ValueError(f"Unexpected target value during normalization: {raw_value!r}")
    return pd.Series(mapped, index=series.index, dtype="int64")


def prepare_dataset(dataset_kind: DatasetKind, csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series, str]:
    df = load_csv(csv_path)
    if dataset_kind == "fraud":
        target_col = "Class"
        if target_col not in df.columns:
            raise ValueError("Fraud dataset must contain target column 'Class'.")
        y = normalize_binary_target(df[target_col])
    else:
        target_col = detect_phishing_target(df)
        y = normalize_binary_target(df[target_col])

    x = df.drop(columns=[target_col])
    return x, y, target_col
