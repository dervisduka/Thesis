# Digital Risk Machine Learning Application

## Titulli i Projektit

**Vlerësimi i Modeleve të Machine Learning në Analizën dhe Menaxhimin e Riskut Digjital**

Ky projekt ka si qëllim ndërtimin e një aplikacioni praktik me Python për analizën, trajnimin dhe krahasimin e modeleve të ndryshme Machine Learning në dy raste studimore të riskut digjital:

1. **Credit Card Fraud Detection**
   Zbulimi i transaksioneve mashtruese në pagesat digjitale.

2. **Phishing Website Detection**
   Zbulimi i faqeve phishing dhe klasifikimi i tyre si faqe të sigurta ose të rrezikshme.

Projekti do të përdoret si pjesë praktike për temën e diplomës dhe do të ndihmojë në analizimin e mënyrës se si modelet Machine Learning mund të mbështesin menaxhimin e riskut digjital.

---

## Objektivi Kryesor

Objektivi kryesor i projektit është të ndërtohen, trajnohen dhe krahasohen disa modele Machine Learning për të identifikuar raste me risk të lartë në dy kategori të ndryshme:

* risk financiar digjital;
* risk kibernetik.

Në fund, modelet do të krahasohen për të përcaktuar cili prej tyre performon më mirë sipas metrikave të vlerësimit.

---

## Dataset-et e Përdorura

### 1. Credit Card Fraud Detection

Ky dataset përmban transaksione me kartë krediti dhe përdoret për të identifikuar nëse një transaksion është normal apo mashtrues.

Target variable:

```text
Class
```

Ku:

```text
0 = Normal Transaction
1 = Fraud Transaction
```

Ky dataset është shumë i pabalancuar, pasi numri i transaksioneve normale është shumë më i madh se numri i transaksioneve mashtruese.

---

### 2. Phishing Website Detection

Ky dataset përdoret për të identifikuar nëse një website është phishing apo legitimate.

Target variable mund të jetë:

```text
status
label
class
result
```

Në varësi të dataset-it të shkarkuar, kolona target duhet të kontrollohet dhe të konvertohet në format numerik:

```text
0 = Legitimate Website
1 = Phishing Website
```

---

## Teknologjitë dhe Libraritë

Projekti do të ndërtohet me Python dhe do të përdorë libraritë:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
jupyter
```

Opsionale:

```text
streamlit
joblib
plotly
```

---

## Struktura e Projektit

```text
digital-risk-ml-thesis/
│
├── data/
│   ├── creditcard.csv
│   └── phishing.csv
│
├── notebooks/
│   ├── 01_credit_card_fraud.ipynb
│   ├── 02_phishing_detection.ipynb
│   └── 03_model_comparison.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── visualization.py
│
├── outputs/
│   ├── figures/
│   ├── results/
│   └── models/
│
├── app/
│   └── streamlit_app.py
│
├── thesis/
│   └── notes.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Hapat që Duhen Ndjekur

## Hapi 1: Krijimi i Projektit

Krijo folderin kryesor të projektit:

```bash
mkdir digital-risk-ml-thesis
cd digital-risk-ml-thesis
```

Krijo strukturën e folderave:

```bash
mkdir data notebooks src outputs app thesis
mkdir outputs/figures outputs/results outputs/models
```

---

## Hapi 2: Krijimi i Virtual Environment

```bash
python -m venv .venv
```

Aktivizo environment-in.

Në Windows:

```bash
.venv\Scripts\activate
```

Në Mac/Linux:

```bash
source .venv/bin/activate
```

---

## Hapi 3: Instalimi i Librarive

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter streamlit joblib
```

Krijo file `requirements.txt`:

```bash
pip freeze > requirements.txt
```

---

## Hapi 4: Shtimi i Dataset-eve

Vendos dataset-et në folderin `data/`:

```text
data/creditcard.csv
data/phishing.csv
```

Kontrollo që emrat e file-ve të jenë të saktë, sepse do përdoren në notebook dhe në kod.

---

## Hapi 5: Analiza e Credit Card Fraud Dataset

Krijo notebook:

```text
notebooks/01_credit_card_fraud.ipynb
```

Detyrat që duhet të bëhen:

* Leximi i dataset-it me pandas.
* Kontrolli i strukturës së të dhënave.
* Kontrolli i missing values.
* Analiza e shpërndarjes së klasave.
* Vizualizimi i class imbalance.
* Standardizimi i kolonave `Time` dhe `Amount`.
* Ndarja e dataset-it në train/test.
* Aplikimi i SMOTE ose `class_weight`.
* Trajnimi i modeleve.
* Vlerësimi i modeleve.
* Ruajtja e rezultateve në CSV.

Modelet që duhet të testohen:

```text
Logistic Regression
Decision Tree
Random Forest
XGBoost
```

Metrikat:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
```

---

## Hapi 6: Analiza e Phishing Website Detection Dataset

Krijo notebook:

```text
notebooks/02_phishing_detection.ipynb
```

Detyrat që duhet të bëhen:

* Leximi i dataset-it.
* Kontrolli i kolonave.
* Identifikimi i target variable.
* Konvertimi i target variable në format numerik.
* Kontrolli i missing values.
* Analiza e shpërndarjes së klasave.
* Ndarja në train/test.
* Standardizimi i features.
* Trajnimi i të njëjtave modele si te Credit Card Fraud.
* Vlerësimi i modeleve.
* Ruajtja e rezultateve në CSV.

---

## Hapi 7: Krahasimi Final i Modeleve

Krijo notebook:

```text
notebooks/03_model_comparison.ipynb
```

Ky notebook duhet të lexojë rezultatet nga:

```text
outputs/results/credit_card_results.csv
outputs/results/phishing_results.csv
```

Pastaj duhet të krijojë:

* tabelë finale krahasuese;
* grafik për Accuracy;
* grafik për Precision;
* grafik për Recall;
* grafik për F1-score;
* grafik për ROC-AUC.

Shembull tabele finale:

| Model               | Dataset            | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | ------------------ | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | Credit Card Fraud  | ...      | ...       | ...    | ...      | ...     |
| Random Forest       | Credit Card Fraud  | ...      | ...       | ...    | ...      | ...     |
| XGBoost             | Phishing Detection | ...      | ...       | ...    | ...      | ...     |

---

## Hapi 8: Risk Scoring

Pas trajnimit të modeleve, duhet të krijohet një logjikë e thjeshtë risk scoring.

Shembull:

| Fraud/Phishing Probability | Risk Level  | Action                          |
| -------------------------- | ----------- | ------------------------------- |
| 0.00 - 0.30                | Low Risk    | Allow                           |
| 0.31 - 0.70                | Medium Risk | Require additional verification |
| 0.71 - 1.00                | High Risk   | Block or manual review          |

Kjo pjesë e lidh Machine Learning me menaxhimin e riskut digjital.

---

## Hapi 9: Ruajtja e Modeleve

Modelet më të mira duhet të ruhen në folderin:

```text
outputs/models/
```

Përdor `joblib`:

```python
import joblib

joblib.dump(model, "../outputs/models/best_model.pkl")
```

---

## Hapi 10: Ndërtimi i Aplikacionit me Streamlit

Opsionalisht, projekti mund të ketë një aplikacion të thjeshtë web me Streamlit.

File:

```text
app/streamlit_app.py
```

Funksionalitete të aplikacionit:

* zgjedhja e dataset-it;
* shfaqja e rezultateve të modeleve;
* krahasimi i metrikave;
* shfaqja e grafikëve;
* parashikim i riskut për një input të ri;
* klasifikimi i riskut si Low, Medium ose High.

Për ta ekzekutuar:

```bash
streamlit run app/streamlit_app.py
```

---

## Hapi 11: Visualizations

Grafikët që duhet të krijohen:

* Class distribution chart.
* Confusion Matrix për çdo model.
* ROC Curve.
* Model comparison bar chart.
* Feature importance për Random Forest ose XGBoost.
* Risk level distribution.

Grafikët duhet të ruhen në:

```text
outputs/figures/
```

---

## Hapi 12: Çfarë duhet të kërkohet nga GitHub Copilot

Gjatë ndërtimit të projektit, përdor Copilot për detyra të vogla dhe të qarta.

Shembuj prompt-esh për Copilot:

```text
Create a Python function that loads a CSV dataset and returns a pandas DataFrame.
```

```text
Create a function that checks missing values and class distribution for a binary classification dataset.
```

```text
Create a function that trains Logistic Regression, Decision Tree, Random Forest and XGBoost models.
```

```text
Create a function that evaluates classification models using accuracy, precision, recall, F1-score and ROC-AUC.
```

```text
Create a function that plots a confusion matrix using seaborn.
```

```text
Create a Streamlit dashboard that displays model comparison results from a CSV file.
```

```text
Create a risk scoring function that converts predicted probability into Low, Medium or High risk.
```

---

## Hapi 13: Rezultatet që Duhet të Dalin

Në fund projekti duhet të prodhojë:

```text
credit_card_results.csv
phishing_results.csv
final_model_comparison.csv
confusion_matrix charts
ROC curves
feature importance charts
best trained model
Streamlit dashboard
```

---

## Hapi 14: Përfundimi Akademik

Në diplomë duhet të shpjegohet:

* cili model performoi më mirë për Credit Card Fraud;
* cili model performoi më mirë për Phishing Detection;
* si ndikoi class imbalance te Credit Card Fraud;
* pse Recall është metrikë shumë e rëndësishme për risk management;
* pse XGBoost ose Random Forest mund të jenë më të përshtatshme për risk detection;
* si mund të përdoret risk scoring në praktikë;
* cilat janë kufizimet e studimit;
* çfarë mund të përmirësohet në të ardhmen.

---

## Metrikat Kryesore

### Accuracy

Tregon sa parashikime totale janë të sakta.

### Precision

Tregon sa nga rastet e parashikuara si risk ishin realisht risk.

### Recall

Tregon sa nga rastet reale me risk u kapën nga modeli.

Në risk management, Recall është shumë i rëndësishëm sepse një rast fraud ose phishing që nuk kapet mund të shkaktojë humbje financiare ose dëme sigurie.

### F1-score

Balancë midis Precision dhe Recall.

### ROC-AUC

Tregon aftësinë e modelit për të dalluar mes klasave normale dhe atyre me risk.

---

## Komandat Kryesore

Instalimi:

```bash
pip install -r requirements.txt
```

Hapja e Jupyter Notebook:

```bash
jupyter notebook
```

Ekzekutimi i aplikacionit Streamlit:

```bash
streamlit run app/streamlit_app.py
```

---

## Statusi i Projektit

* [ ] Krijimi i strukturës së projektit
* [ ] Shkarkimi i dataset-it Credit Card Fraud
* [ ] Shkarkimi i dataset-it Phishing Website Detection
* [ ] Analiza eksploruese e dataset-it të parë
* [ ] Analiza eksploruese e dataset-it të dytë
* [ ] Përpunimi i të dhënave
* [ ] Trajnimi i modeleve ML
* [ ] Vlerësimi i modeleve
* [ ] Krahasimi final i rezultateve
* [ ] Krijimi i risk scoring
* [ ] Ruajtja e modeleve
* [ ] Ndërtimi i dashboard-it me Streamlit
* [ ] Përgatitja e grafikëve për diplomë
* [ ] Përgatitja e përfundimeve
* [ ] Përgatitja e prezantimit

---

## Përfundim

Ky projekt demonstron përdorimin e Machine Learning për analizën dhe menaxhimin e riskut digjital në dy fusha të rëndësishme: zbulimin e transaksioneve mashtruese dhe identifikimin e faqeve phishing. Duke krahasuar disa modele të ndryshme, projekti synon të tregojë se cilat algoritme janë më efektive për identifikimin e rasteve me risk të lartë dhe si mund të përdoren rezultatet për vendimmarrje në menaxhimin e riskut.
