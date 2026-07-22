<div align="center">

# 💳 Credit Card Fraud Detection And Analysis

### *Catching financial crime before it costs a cent — powered by XGBoost & SHAP*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://credit-card-fraud-detection-analysis.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](#)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-8E44AD?style=for-the-badge&logo=buffer&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-284%2C807%20Transactions-2ECC71?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
[![Status](https://img.shields.io/badge/Status-✅%20Live-2ECC71?style=for-the-badge)](https://credit-card-fraud-detection-analysis.streamlit.app/)

<br/>

> **284,807 real transactions** · **492 fraud cases (0.17%)** · **SMOTE + XGBoost pipeline** · **SHAP explainability** · **real-time transaction scorer** · **live streaming monitor** · **business impact analysis**

<br/>

### 🚀 [**Try the live app →**](https://credit-card-fraud-detection-analysis.streamlit.app/)

</div>

---

## 🔬 See It In Action

<div align="center">

**Real-time transaction scorer** — score any transaction instantly and see the model's reasoning:

![Real-time scorer](screenshots/realtime_scorer_demo.png)

**Live transaction monitor** — streams real held-out transactions one at a time and flags fraud as it happens:

![Live monitor](screenshots/live_monitor_demo.png)

</div>

---

## 🎯 The Problem

Credit card fraud is one of the **hardest real-world machine learning problems** — not because detection algorithms are weak, but because of the extreme data challenge hiding beneath the surface.

<div align="center">

### 📊 The Scale of Financial Fraud

| 🔢 Statistic | 💡 Fact | 😟 Impact |
|-------------|---------|----------|
| **$32.3 billion** | Lost to card fraud globally every year | Growing 14% annually |
| **0.17%** | Fraud rate in real transaction data | 1 fraud per 590 transactions |
| **577:1** | Ratio of normal to fraud in this dataset | Extreme class imbalance |
| **< 2 seconds** | Time available to approve or block a transaction | Real-time decision required |
| **False positives** | Blocking a legitimate customer costs loyalty + revenue | Balance precision vs recall |

</div>

Standard classifiers trained on imbalanced data simply learn to predict "Normal" every time and still achieve 99.8% accuracy — while **catching zero frauds**. This project solves that with SMOTE, threshold tuning, and the right evaluation metrics.

> *A model with 99.8% accuracy that never catches fraud is worse than useless — it's expensive.*

---

## ✨ What Makes This Unique

Most fraud detection projects stop at "train a classifier and show accuracy." This system goes several layers deeper — and ships as a live, interactive app instead of a static notebook:

```
TYPICAL PROJECT              THIS PROJECT
───────────────              ────────────────────────────────────────────
Train model          →       ✅ SMOTE-balanced XGBoost, benchmarked vs LR/RF
Show accuracy         →       ✅ Correct metrics: ROC-AUC + Precision-Recall
                     →       ✅ Threshold tuning — maximise F1 not accuracy
                     →       ✅ SHAP explainability — WHY was this fraud?
                     →       ✅ Business impact — exact € saved by the model
                     →       ✅ Real-time scorer with 5 risk tiers
                     →       ✅ Live streaming monitor for continuous scoring
                     →       ✅ Deployed as a working app, not just a notebook
```

<br/>

### 🏗️ The Core Components

| # | Component | Description | Tech Used |
|---|-----------|-------------|-----------|
| 1 | 📊 **Deep EDA** | Class imbalance, amount distribution, hourly fraud patterns | Matplotlib |
| 2 | ⚙️ **Feature Engineering** | Log_Amount, Hour, High_Amount, Zero_Amount, V1×V2 and V14×V17 interactions | Pandas, NumPy |
| 3 | ⚖️ **SMOTE Balancing** | Oversamples minority class on training data only | imbalanced-learn |
| 4 | 🤖 **Model Benchmark** | XGBoost (primary) vs Logistic Regression & Random Forest | Scikit-learn, XGBoost |
| 5 | 🧠 **SHAP XAI** | Global importance + per-transaction explanation | SHAP TreeExplainer |
| 6 | 💰 **Business Impact** | € caught, € missed, false alarm cost, net savings at scale | Custom analysis |
| 7 | 🔮 **Real-Time Scorer** | Score any transaction instantly with risk tier + SHAP reason | Full inference pipeline |
| 8 | 🔴 **Live Monitor** | Streams real held-out transactions and flags fraud as it arrives | Streamlit session state |
| 9 | 🌐 **Deployed App** | Interactive Streamlit app, live on the web | Streamlit Cloud / Replit |

---

## 📊 Dataset

<div align="center">

### 📦 Dataset At A Glance

| 🏷️ Property | 📋 Details |
|-------------|-----------|
| **Name** | Credit Card Fraud Detection |
| **Source** | Real European cardholders, September 2013 |
| **Total Transactions** | 284,807 |
| **Fraud Cases** | 492 (0.172%) |
| **Features** | 30 (V1–V28 via PCA + Time + Amount) |
| **Labels** | `Class` — 0 = Normal, 1 = Fraud |
| **Missing Values** | None |
| **File Size** | ~150 MB (full) |
| **Kaggle Link** | [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

</div>

### 📋 Class Distribution

```
Transaction Type    Count      Share    Visual Distribution
────────────────────────────────────────────────────────────
✅  Normal         284,315    99.83%   ████████████████████████████████████████
🚨  Fraud              492     0.17%   ▏
────────────────────────────────────────────────────────────
    TOTAL          284,807   100.00%

Imbalance Ratio: 577 normal transactions for every 1 fraud
```

### 📁 Feature Reference

| Feature | Type | Description |
|---------|------|-------------|
| `V1` – `V28` | `float` | PCA-transformed features (original features anonymised for privacy) |
| `Time` | `float` | Seconds elapsed from the first transaction in the dataset |
| `Amount` | `float` | Transaction amount in Euros |
| `Class` | `int` | Target label — `0` Normal · `1` Fraud |
| `Log_Amount` *(engineered)* | `float` | `log1p(Amount)` — normalises right-skewed distribution |
| `Hour` *(engineered)* | `int` | Hour of day (0–23) derived from `Time` |
| `High_Amount` *(engineered)* | `int` | `1` if Amount > 95th percentile, else `0` |
| `Zero_Amount` *(engineered)* | `int` | `1` if Amount == 0, else `0` |
| `V1_V2_interact` *(engineered)* | `float` | `V1 × V2` interaction term |
| `V14_V17_interact` *(engineered)* | `float` | `V14 × V17` interaction term |

> ⚠️ **Dataset note:** The full `creditcard.csv` (~150 MB) exceeds GitHub's file size limit and isn't bundled in this repo. The live app ships with a small synthetic demo dataset (same schema) so it runs instantly with zero setup — upload the real Kaggle CSV in the app's sidebar for results that match the numbers below.

---

## ⚙️ Feature Engineering

Six features were engineered from the raw dataset columns:

| Feature | Formula | Why It Helps |
|---------|---------|-------------|
| `Log_Amount` | `log1p(Amount)` | Transaction amounts are heavily right-skewed. Log-transform normalises this so models don't overweight large amounts. |
| `Hour` | `(Time // 3600) % 24` | Fraud peaks sharply overnight when monitoring is minimal. Hour captures this temporal signal directly. |
| `High_Amount` | `Amount > 95th percentile` | Flags unusually large transactions as a binary risk signal. |
| `Zero_Amount` | `Amount == 0` | Zero-value transactions are a known fraud-testing pattern (card validation probes). |
| `V1_V2_interact` | `V1 × V2` | Captures a non-linear interaction between two of the strongest PCA components. |
| `V14_V17_interact` | `V14 × V17` | `V14` and `V17` are consistently the top two SHAP features; their product sharpens the signal further. |

---

## 🤖 ML Models Compared

Models are trained on SMOTE-balanced data and evaluated on the **original imbalanced test split** (the real-world distribution):

| Model | Why Included |
|-------|-------------|
| **Logistic Regression** | Interpretable baseline; shows if the problem is linearly separable |
| **Random Forest** | Strong ensemble, handles non-linearity, robust to outliers |
| **XGBoost** | Primary model — best-in-class for tabular fraud data; native `scale_pos_weight` for imbalance |

> XGBoost is used for the live scorer and SHAP explanations. Logistic Regression and Random Forest are available as an optional benchmark comparison inside the app.

### ⚖️ Why NOT Accuracy?

On a 577:1 imbalanced dataset, a model that predicts "Normal" for every transaction scores **99.83% accuracy** while catching **zero frauds**. The correct metrics are:

- **ROC-AUC** — overall discrimination ability
- **Average Precision (PR-AUC)** — most informative for imbalanced classes
- **F1 Score** — harmonic mean of precision and recall
- **MCC** — Matthews Correlation Coefficient (single balanced metric)

---

## 📈 Results & Performance

> Computed live inside the app after training — exact numbers depend on whether you're using the bundled synthetic demo data or the real Kaggle dataset.

```
Model                  ROC-AUC    Avg Precision    F1       Recall    Precision
────────────────────────────────────────────────────────────────────────────────
🏆 XGBoost             ~98–99%       ~85–90%      ~85%      ~88%        ~82%
   Random Forest       ~96–98%       ~80–86%      ~80%      ~82%        ~78%
   Logistic Regression ~95–97%       ~72–78%      ~73%      ~76%        ~70%
────────────────────────────────────────────────────────────────────────────────
* Real Kaggle dataset. Numbers on the synthetic demo data are illustrative only.
```

### 🎯 Threshold Tuning

The default 0.5 classification threshold is not optimal for fraud. The pipeline sweeps thresholds from 0.1 → 0.9 and selects the value that **maximises F1** — typically landing between 0.35–0.45, boosting recall without sacrificing too much precision.

---

## 🧠 SHAP Explainability

SHAP (SHapley Additive exPlanations) answers the critical compliance question: **"Why did you flag this transaction?"**

### Global Feature Importance

```
Most Important Fraud Features (SHAP):
──────────────────────────────────────
V14   ████████████████████   Strongest fraud signal
V12   ████████████████
V10   ██████████████
V17   █████████████
V4    ████████
Amount ██████
Hour   ████
```

### Per-Transaction Explanation

For every scored transaction, the app shows a SHAP bar chart of the top 6 features that pushed the prediction toward — or away from — fraud, as seen in the screenshot above.

---

## 💰 Business Impact Analysis

The model's value is measured in euros, not just percentages. The **Business Impact** tab computes, live, on your test set:

<div align="center">

| Metric | Value |
|--------|-------|
| Average fraud transaction | configurable (default ~€122) |
| Frauds correctly caught | TP transactions |
| Fraud value recovered | TP × avg fraud value |
| Frauds missed | FN transactions |
| False alarm investigation cost | FP × cost per alarm |
| **Net savings vs no model** | **Caught value − False alarm cost** |

</div>

### 📈 Scaling Impact

The app also projects savings at scale (10K / 100K / 1M / 10M transactions per day) based on your test-set performance.

---

## 🔴 Live Monitor

Beyond one-off scoring, the app includes a **live streaming monitor** that replays real held-out transactions one at a time — like a live transaction feed — scoring each instantly and surfacing fraud alerts as they happen, with running totals for transactions scanned, frauds flagged, and € value blocked. See the screenshot above.

> This simulates what a production integration into a live payment stream (e.g. via Stripe/Plaid) would look like at the model-inference layer.

---

## 🗂️ Project Structure

```
📦 credit-card-fraud-detection/
│
├── 🖥️  app.py                     # Streamlit app — all 5 tabs, training pipeline, scorer
├── 🛠️  generate_sample_dataset.py  # Generates the synthetic demo dataset
├── 📄 requirements.txt             # Python dependencies
├── ⚙️  run.sh                      # Install + launch script (used by Replit)
├── ⚙️  .replit                     # Replit run/deploy configuration
├── ⚙️  .streamlit/config.toml      # Streamlit server settings (port, theme, upload size)
├── 🖼️  screenshots/                # App screenshots used in this README
└── 📄 README.md                   # This file
```

**Note:** the original notebook-based pipeline (`.ipynb`) that this app was built from is kept for reference but is not required to run the app — `app.py` reproduces the full pipeline (feature engineering → RobustScaler → SMOTE → XGBoost → SHAP) natively in Python.

---

## 🚀 Run It

### Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Live app
👉 **[credit-card-fraud-detection-analysis.streamlit.app](https://credit-card-fraud-detection-analysis.streamlit.app/)**

Works instantly with the bundled synthetic demo dataset. Upload the real `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) in the sidebar for results matching the numbers in this README.

---

### 🎚️ Risk Tier Reference

| Tier | Threshold | Action |
|------|-----------|--------|
| 🚨 CRITICAL | ≥ 80% | Block transaction immediately |
| 🔴 HIGH | 60–80% | Flag for manual review |
| 🟠 MEDIUM | 40–60% | Monitor closely |
| 🟡 LOW | 20–40% | Minor alert logged |
| 🟢 SAFE | < 20% | Approve transaction |

---

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **App / UI** | Streamlit |
| **Data** | Pandas, NumPy |
| **ML Models** | Scikit-learn, XGBoost |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP (TreeExplainer) |
| **Preprocessing** | RobustScaler, Stratified Split |
| **Visualization** | Matplotlib |
| **Deployment** | Streamlit Community Cloud / Replit |
| **Version Control** | Git + GitHub |

---

## 👨‍💻 Author

<div align="center">

**Arushi Garg**

*B.Tech Computer Science (AI and Data Science)*

## 🔗 Let's Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Arushi%20Garg-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/arushi-garg525/)
[![GitHub](https://img.shields.io/badge/GitHub-Shinearushi525-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Shinearushi525)
[![Email](https://img.shields.io/badge/Email-arushigarg525@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:Arushigarg525@gmail.com)

---
</div>
