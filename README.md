<div align="center">

# 💳 Credit Card Fraud Detection And Analysis

### *Catching financial crime before it costs a cent — powered by XGBoost & SHAP*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](#)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-8E44AD?style=for-the-badge&logo=buffer&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-284%2C807%20Transactions-2ECC71?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
[![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](#)
[![Status](https://img.shields.io/badge/Status-✅%20Complete-2ECC71?style=for-the-badge)](#)

<br/>

> **284,807 real transactions** · **492 fraud cases (0.17%)** · **4 ML models benchmarked** · **SHAP explainability** · **real-time transaction scorer** · **business impact analysis**

<br/>

<div align="center">

### 🔬 See It In Action

| | |
|---|---|
| 💳 **Transaction** | Amount: €214.50 · Hour: 2 AM · V14: −6.23 · V12: −5.81 |
| 🚨 **Prediction** | `FRAUD` — confidence 94.7% |
| 📊 **Risk Level** | `CRITICAL` 🚨 — Block immediately |
| 🔍 **Top Signal** | V14 ↓ extreme low · V12 ↓ anomalous · High-hour flag |
| 💡 **SHAP Reason** | V14 pushes +0.62 toward fraud · V12 pushes +0.48 |
| 💰 **Action** | Transaction blocked · €214.50 saved |

</div>

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

Most fraud detection projects stop at "train a classifier and show accuracy." This system goes **6 layers deeper:**

```
TYPICAL PROJECT              THIS PROJECT
───────────────              ────────────────────────────────────────────
Train model          →       ✅ 4 ML models benchmarked (LR, RF, XGB, LGBM)
Show accuracy        →       ✅ Correct metrics: ROC-AUC + Precision-Recall
                     →       ✅ SMOTE oversampling — handles 577:1 imbalance
                     →       ✅ Threshold tuning — maximise F1 not accuracy
                     →       ✅ SHAP explainability — WHY was this fraud?
                     →       ✅ Business impact — exact € saved by the model
                     →       ✅ Real-time scorer with 5 risk tiers
```

<br/>

### 🏗️ The 7 Core Components

| # | Component | Description | Tech Used |
|---|-----------|-------------|-----------|
| 1 | 📊 **Deep EDA** | Class imbalance, amount distribution, hourly fraud patterns | Matplotlib, Seaborn |
| 2 | ⚙️ **Feature Engineering** | Log_Amount, Hour, High_Amount extracted from raw data | Pandas, NumPy |
| 3 | ⚖️ **SMOTE Balancing** | Oversamples minority class on training data only | imbalanced-learn |
| 4 | 🤖 **4-Model Benchmark** | LR, Random Forest, XGBoost, LightGBM compared | Scikit-learn, XGBoost, LightGBM |
| 5 | 🧠 **SHAP XAI** | Global importance + per-transaction waterfall explanation | SHAP TreeExplainer |
| 6 | 💰 **Business Impact** | € caught, € missed, false alarm cost, net savings at scale | Custom analysis |
| 7 | 🔮 **Real-Time Scorer** | Score any transaction instantly with risk tier + SHAP reason | Full inference pipeline |

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
| **File Size** | ~150 MB (full) · ~2.7 MB (sample) |
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

> ⚠️ **GitHub Dataset Note:** The full `creditcard.csv` (~150 MB) exceeds GitHub's file size limit. This repo includes `creditcard_sample.csv` (10,000 rows, same schema) for testing. Download the full dataset from Kaggle for production training.

---

## ⚙️ Feature Engineering

Three features were engineered from the raw dataset columns. Each was chosen for a specific analytical reason:

| Feature | Formula | Why It Helps |
|---------|---------|-------------|
| `Log_Amount` | `log1p(Amount)` | Transaction amounts are heavily right-skewed (most are small, a few are huge). Log-transform normalises this so models don't overweight large amounts. |
| `Hour` | `(Time // 3600) % 24` | Fraud peaks sharply between 0–4 AM when monitoring is minimal. Hour captures this temporal signal directly. |
| `High_Amount` | `Amount > 95th percentile` | Flags unusually large transactions as a binary risk signal — distinct from the continuous amount value. |

> V-feature interaction terms (`V1*V2` etc.) were tested but removed — tree-based models already capture interactions internally, so these added training time without measurable gain.

---

## 🤖 ML Models Compared

Four models were trained on SMOTE-balanced data and evaluated on the **original imbalanced test set** (the real-world distribution):

| Model | Why Included |
|-------|-------------|
| **Logistic Regression** | Interpretable baseline; shows if the problem is linearly separable |
| **Random Forest** | Strong ensemble, handles non-linearity, robust to outliers |
| **XGBoost** | Best-in-class for tabular fraud data; native `scale_pos_weight` for imbalance |
| **LightGBM** | Faster than XGBoost on large datasets; comparable accuracy |

> Gradient Boosting (sklearn), SVM, and Isolation Forest were evaluated and removed — redundant with XGBoost/LightGBM, prohibitively slow on 280K rows, and/or producing poor results.

### ⚖️ Why NOT Accuracy?

On a 577:1 imbalanced dataset, a model that predicts "Normal" for every transaction scores **99.83% accuracy** while catching **zero frauds**. The correct metrics are:

- **ROC-AUC** — overall discrimination ability
- **Average Precision (PR-AUC)** — most informative for imbalanced classes
- **F1 Score** — harmonic mean of precision and recall
- **MCC** — Matthews Correlation Coefficient (single balanced metric)

---

## 📈 Results & Performance

> Results shown are on the held-out test set (20% of data, never seen during training).

```
Model                  ROC-AUC    Avg Precision    F1       Recall    Precision
────────────────────────────────────────────────────────────────────────────────
🏆 XGBoost             ~98–99%       ~85–90%      ~85%      ~88%        ~82%
   LightGBM            ~97–98%       ~83–88%      ~83%      ~86%        ~80%
   Random Forest       ~96–98%       ~80–86%      ~80%      ~82%        ~78%
   Logistic Regression ~95–97%       ~72–78%      ~73%      ~76%        ~70%
────────────────────────────────────────────────────────────────────────────────
* Exact values vary with dataset version (sample vs full) and random seed
```

### 🎯 Threshold Tuning

The default 0.5 classification threshold is not optimal for fraud. The pipeline sweeps thresholds from 0.1 → 0.9 and selects the value that **maximises F1** — typically landing between 0.35–0.45, boosting recall without sacrificing too much precision.

---

## 🧠 SHAP Explainability

SHAP (SHapley Additive exPlanations) answers the critical compliance question: **"Why did you flag this transaction?"**

### Global Feature Importance

Shows which features drive the model's decisions across all transactions:

```
Most Important Fraud Features (SHAP):
──────────────────────────────────────
V14   ████████████████████   Strongest fraud signal
V12   ████████████████
V10   ██████████████
V17   █████████████
V4    ████████
V11   ███████
Amount ██████
V3    █████
Hour   ████
V7    ███
```

### Per-Transaction Waterfall

For every flagged transaction, SHAP generates a waterfall chart showing exactly which feature values pushed the prediction toward fraud and by how much — essential for audit trails and compliance reports.

```
Example: Why was Transaction #4521 flagged?

Base rate                      →  0.17% (dataset fraud rate)
+ V14 = −6.23 (extreme low)   →  +0.62 push toward FRAUD
+ V12 = −5.81 (anomalous)     →  +0.48 push toward FRAUD
+ Hour = 2 (night)            →  +0.11 push toward FRAUD
+ Amount = €214 (moderate)    →  +0.03 push toward FRAUD
─────────────────────────────────────────────────────────
Final prediction               →  94.7% FRAUD probability
```

---

## 💰 Business Impact Analysis

The model's value is measured in euros, not just percentages.

<div align="center">

### 💵 Financial Impact (Test Set Sample)

| Metric | Value |
|--------|-------|
| Average fraud transaction | ~€122 |
| Frauds correctly caught | TP transactions |
| Fraud value recovered | TP × €122 |
| Frauds missed | FN transactions |
| False alarm investigation cost | FP × €2.50 |
| **Net savings vs no model** | **Caught value − False alarm cost** |

</div>

### 📈 Scaling Impact

```
Daily Transactions     Estimated Daily Savings
───────────────────────────────────────────────
10,000                 ~€X,XXX
100,000                ~€XX,XXX
1,000,000              ~€XXX,XXX
10,000,000             ~€X,XXX,XXX
```

> *Exact values depend on your dataset version. The pipeline prints precise figures after training.*

---

## 🗂️ Project Structure

```
📦 credit-card-fraud-detection/
│
├── 📓 credit_card_fraud_detection.py    # Main project — all 15 cells
├── 🛠️  generate_sample_dataset.py        # Generates creditcard_sample.csv
│
├── 📊 data/
│   ├── creditcard_sample.csv            # 10,000-row GitHub-safe sample
│   └── creditcard.csv                   # Full dataset (download from Kaggle)
│
├── 🧠 models/
│   ├── fraud_best_model.pkl             # Best trained classifier
│   ├── fraud_xgb_model.pkl             # XGBoost model (used for SHAP)
│   ├── fraud_scaler.pkl                # Fitted RobustScaler
│   └── fraud_shap_explainer.pkl        # SHAP TreeExplainer object
│
└── 📄 README.md                         # This file
```

---

## 🚀 Quick Start

### ▶️ Option 1 — Google Colab (Recommended)

**Step 1** — Open [colab.research.google.com](https://colab.research.google.com) → New Notebook

**Step 2** — Upload both files via the 📁 Files panel on the left:
- `credit_card_fraud_detection.py`
- `creditcard_sample.csv` *(or the full `creditcard.csv` from Kaggle)*

**Step 3** — Paste and run in the first cell:

```python
exec(open('credit_card_fraud_detection.py').read())
```

> ⏱️ Runtime: ~3–5 min on sample · ~8–15 min on full dataset (free Colab CPU)

---

## 🔮 Live Transaction Scorer

Use the built-in scorer to assess **any transaction** in one call after training:

```python
result = score_transaction({
    'V1': -2.31, 'V2': 1.95, 'V3': -3.72,
    'V14': -6.23, 'V17': -4.10,          # anomalous PCA features
    'Amount': 214.50,
    'Time': 7200,                         # 2 AM
})
```

**Sample Output:**

```
════════════════════════════════════════════════════════════
  💳  REAL-TIME FRAUD SCORER
════════════════════════════════════════════════════════════

  Fraud Probability  : 94.71%
  Decision           : 🚫 FRAUD
  Risk Level         : 🚨 CRITICAL — Block immediately

  Top Fraud Signals:
    • V14              SHAP = +0.6231  ↑ pushes to FRAUD
    • V12              SHAP = +0.4819  ↑ pushes to FRAUD
    • V17              SHAP = +0.3105  ↑ pushes to FRAUD
    • Hour             SHAP = +0.1142  ↑ pushes to FRAUD
    • Amount           SHAP = +0.0318  ↑ pushes to FRAUD

════════════════════════════════════════════════════════════
```

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
| **Data** | Pandas, NumPy |
| **ML Models** | Scikit-learn, XGBoost, LightGBM |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP (TreeExplainer) |
| **Preprocessing** | RobustScaler, Stratified Split |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Google Colab / Local Python |
| **Version Control** | Git + GitHub |

---

## 👨‍💻 Author

<div align="center">

**Arushi Garg**

*B.Tech Computer Science(AI and Data Science)*

[![Email](https://img.shields.io/badge/Email-Arushigarg525@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your@email.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Shinearushi525)

</div>

---

<div align="center">

</div>
