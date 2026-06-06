# Digital Risk ML Thesis Project

Python ML project for two binary-classification risk tasks:

1. Credit card fraud detection
2. Phishing website detection

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data placement

Put datasets here:

```text
data/creditcard.csv
data/phishing.csv
```

## Run

Start notebooks:

```bash
python -m notebook
```

Run Streamlit app:

```bash
python -m streamlit run app/streamlit_app.py
```

## Programmatic training

```bash
python -m src.pipeline --dataset fraud --input data/creditcard.csv
python -m src.pipeline --dataset phishing --input data/phishing.csv
python -m src.pipeline --dataset both --fraud-input data/creditcard.csv --phishing-input data/phishing.csv
```

Outputs are written to:

- `outputs/results/`
- `outputs/figures/`
- `outputs/models/`
