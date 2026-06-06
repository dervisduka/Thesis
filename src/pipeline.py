from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data_loader import DatasetKind, prepare_dataset
from .evaluate_models import evaluate_trained_models, save_metrics
from .preprocessing import split_data
from .train_models import select_best_model, train_models
from .visualization import (
    plot_class_distribution,
    plot_confusion_matrices,
    plot_model_comparison,
    plot_roc_curves,
)


def run_dataset_pipeline(
    dataset_kind: DatasetKind,
    input_csv: str,
    output_root: str = "outputs",
    random_state: int = 42,
) -> pd.DataFrame:
    x, y, _ = prepare_dataset(dataset_kind=dataset_kind, csv_path=input_csv)
    x_train, x_test, y_train, y_test = split_data(
        x, y, test_size=0.2, random_state=random_state
    )

    trained = train_models(
        x_train=x_train,
        y_train=y_train,
        random_state=random_state,
        use_smote=True,
    )
    metrics_df, cms, scores = evaluate_trained_models(
        trained_models=trained, x_test=x_test, y_test=y_test
    )

    root = Path(output_root)
    results_dir = root / "results"
    figures_dir = root / "figures"
    models_dir = root / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    result_csv = results_dir / f"{dataset_kind}_results.csv"
    save_metrics(metrics_df, result_csv)

    plot_class_distribution(y, dataset_kind, figures_dir)
    plot_confusion_matrices(cms, dataset_kind, figures_dir)
    plot_roc_curves(y_test, scores, dataset_kind, figures_dir)
    plot_model_comparison(
        metrics_df,
        figures_dir / f"{dataset_kind}_model_comparison.png",
    )

    best_name, best_model = select_best_model(metrics_df, trained)
    model_path = models_dir / f"{dataset_kind}_best_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"[{dataset_kind}] Saved metrics: {result_csv}")
    print(f"[{dataset_kind}] Saved best model ({best_name}): {model_path}")
    return metrics_df


def run_full_comparison(
    fraud_csv: str,
    phishing_csv: str,
    output_root: str = "outputs",
    random_state: int = 42,
) -> pd.DataFrame:
    fraud_metrics = run_dataset_pipeline(
        dataset_kind="fraud",
        input_csv=fraud_csv,
        output_root=output_root,
        random_state=random_state,
    ).assign(Dataset="Credit Card Fraud")

    phishing_metrics = run_dataset_pipeline(
        dataset_kind="phishing",
        input_csv=phishing_csv,
        output_root=output_root,
        random_state=random_state,
    ).assign(Dataset="Phishing Detection")

    combined = pd.concat([fraud_metrics, phishing_metrics], ignore_index=True)
    final_csv = Path(output_root) / "results" / "final_model_comparison.csv"
    combined.to_csv(final_csv, index=False)
    print(f"[combined] Saved comparison: {final_csv}")
    return combined


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run digital-risk model training pipelines.")
    parser.add_argument(
        "--dataset",
        choices=["fraud", "phishing", "both"],
        required=True,
        help="Dataset pipeline to run.",
    )
    parser.add_argument("--input", help="Input CSV for single-dataset mode.")
    parser.add_argument("--fraud-input", help="Fraud CSV path for --dataset both.")
    parser.add_argument("--phishing-input", help="Phishing CSV path for --dataset both.")
    parser.add_argument("--output-root", default="outputs", help="Output root directory.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.dataset in {"fraud", "phishing"}:
        if not args.input:
            raise ValueError("--input is required for single-dataset mode.")
        run_dataset_pipeline(
            dataset_kind=args.dataset,
            input_csv=args.input,
            output_root=args.output_root,
            random_state=args.random_state,
        )
        return

    if not args.fraud_input or not args.phishing_input:
        raise ValueError("--fraud-input and --phishing-input are required for --dataset both.")
    run_full_comparison(
        fraud_csv=args.fraud_input,
        phishing_csv=args.phishing_input,
        output_root=args.output_root,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()

