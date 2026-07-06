# Copilot Instructions for This Repository

## Current repository state

This repository is a Python ML thesis project for digital risk analysis across two binary-classification tasks:

1. Credit card fraud detection
2. Phishing website detection

## Build, test, and lint commands

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run notebooks:

```powershell
python -m notebook
```

Run Streamlit dashboard:

```powershell
python -m streamlit run app/streamlit_app.py
```

Run pipeline for one dataset (single-run validation):

```powershell
python -m src.pipeline --dataset fraud --input data/creditcard.csv
```

Run full comparison pipeline:

```powershell
python -m src.pipeline --dataset both --fraud-input data/creditcard.csv --phishing-input data/phishing.csv
```

There is currently no dedicated lint or automated unit-test suite in the repository.

## High-level architecture

Architecture is a two-track training flow (`fraud` and `phishing`) converging into shared artifacts and dashboard display:

1. `src.pipeline` is the entrypoint. It orchestrates load -> split -> train -> evaluate -> save artifacts.
2. `src.data_loader` handles CSV loading and dataset-specific target normalization.
3. `src.preprocessing` builds `ColumnTransformer` preprocessing and optional SMOTE-enabled training pipeline.
4. `src.train_models` trains a fixed model set and picks the best model by `Recall`, then `ROC-AUC`.
5. `src.evaluate_models` computes metrics and confusion matrices and writes per-dataset results.
6. `src.visualization` persists charts (class distribution, confusion matrices, ROC, metric comparison) in `outputs/figures/`.
7. `src.pipeline --dataset both` merges per-dataset metrics into `outputs/results/final_model_comparison.csv`.
8. `app/streamlit_app.py` reads generated CSV/PNG artifacts and exposes dashboard + risk scoring UI.

## Key conventions specific to this project

1. **Dataset mode is strict**: CLI only accepts `--dataset fraud|phishing|both`.
2. **Fraud target contract**: fraud data must include `Class` with binary 0/1 values.
3. **Phishing target detection is dynamic**: target is auto-detected from `status|label|class|result` (case-insensitive), then normalized to 0/1 using token mapping in `src.data_loader`.
4. **Class imbalance treatment is built in**: training uses class-weighted models plus SMOTE pipeline by default (`use_smote=True` in `run_dataset_pipeline`).
5. **Default model set excludes SVM** for practical runtime: Logistic Regression, Decision Tree, Random Forest, XGBoost (when installed).
6. **Best model selection rule is explicit**: sort by `Recall` descending, tie-break by `ROC-AUC`.
7. **Metric and artifact naming is fixed**:
   - `outputs/results/fraud_results.csv`
   - `outputs/results/phishing_results.csv`
   - `outputs/results/final_model_comparison.csv`
   - `outputs/models/fraud_best_model.pkl`
   - `outputs/models/phishing_best_model.pkl`
8. **Risk scoring bands are centralized in code** (`src.risk_scoring.classify_risk`):
   - `0.00-0.30` -> Low Risk / Allow
   - `0.31-0.70` -> Medium Risk / Require additional verification
   - `0.71-1.00` -> High Risk / Block or manual review
9. **Streamlit depends on generated artifacts**: UI reads only saved CSV/PNG files from `outputs/`; missing files show warnings instead of crashing.

# Copilot response style

Use caveman mode when answering:

- Be very concise.
- Avoid long explanations.
- No filler like “Sure”, “Certainly”, or “Here’s a breakdown”.
- Give the direct answer first.
- Prefer commands, code, and short notes.
- Explain only when needed.
- For code fixes, show the changed code and a short reason.
- For debugging, give likely cause + fix.
- For PR/code review, use short actionable comments.

Style:
- Few words.
- Useful words.
- No essay.