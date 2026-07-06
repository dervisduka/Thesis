from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.risk_scoring import classify_risk


st.set_page_config(page_title="Digital Risk ML Dashboard", layout="wide")
st.title("Digital Risk ML Dashboard")

results_dir = Path("outputs") / "results"
figures_dir = Path("outputs") / "figures"


def load_results(csv_name: str) -> pd.DataFrame | None:
    path = results_dir / csv_name
    if not path.exists():
        return None
    return pd.read_csv(path)


fraud_df = load_results("fraud_results.csv")
phishing_df = load_results("phishing_results.csv")
combined_df = load_results("final_model_comparison.csv")

dataset_choice = st.selectbox(
    "Choose dataset view",
    ["Credit Card Fraud", "Phishing Detection", "Combined Comparison"],
)

if dataset_choice == "Credit Card Fraud":
    if fraud_df is None:
        st.warning("Missing outputs/results/fraud_results.csv. Run the fraud pipeline first.")
    else:
        st.subheader("Credit Card Fraud Results")
        st.dataframe(fraud_df, use_container_width=True)
        st.bar_chart(fraud_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]])
elif dataset_choice == "Phishing Detection":
    if phishing_df is None:
        st.warning("Missing outputs/results/phishing_results.csv. Run the phishing pipeline first.")
    else:
        st.subheader("Phishing Detection Results")
        st.dataframe(phishing_df, use_container_width=True)
        st.bar_chart(phishing_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]])
else:
    if combined_df is None:
        st.warning(
            "Missing outputs/results/final_model_comparison.csv. "
            "Run the combined pipeline first."
        )
    else:
        st.subheader("Final Model Comparison")
        st.dataframe(combined_df, use_container_width=True)

st.divider()
st.subheader("Risk Scoring")
probability = st.slider("Fraud/Phishing probability", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
score = classify_risk(probability)
st.metric("Risk Level", score.level)
st.write(f"**Action:** {score.action}")

st.divider()
st.subheader("Generated figures")
if figures_dir.exists():
    image_paths = sorted(figures_dir.glob("*.png"))
    if not image_paths:
        st.info("No figures yet. Run a pipeline to generate charts.")
    else:
        for img in image_paths:
            st.image(str(img), caption=img.name, use_column_width=True)
else:
    st.info("No figures directory yet. Run a pipeline first.")
