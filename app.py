"""
Credit Card Fraud Detection — Streamlit App
Rebuilt from: https://github.com/Shinearushi525/Credit_Card_Fraud_Detection

Reproduces the notebook's pipeline:
  - Feature engineering (Hour, Log_Amount, High_Amount, Zero_Amount, V1*V2, V14*V17)
  - RobustScaler on Amount/Time-derived features (V1-V28 already PCA-scaled)
  - SMOTE (sampling_strategy=0.1) on training data only
  - XGBoost as the primary model (+ optional Logistic Regression / Random Forest benchmark)
  - F1-optimal threshold tuning
  - SHAP TreeExplainer for per-transaction explanations
  - Real-time transaction scorer with risk tiers
  - Business impact (€ saved) calculator

No creditcard.csv ships with the original repo (file is ~150MB, over GitHub's
limit) so this app includes a small synthetic demo dataset that mimics the real
schema/imbalance. Upload the real Kaggle dataset (mlg-ulb/creditcardfraud) in the
sidebar for results that match the README's reported numbers.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, confusion_matrix, roc_curve,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

FRAUD_COLOR, NORMAL_COLOR, ACCENT_COLOR = "#E74C3C", "#2ECC71", "#3498DB"

V_COLS = [f"V{i}" for i in range(1, 29)]
MODEL_FEATURES = V_COLS + [
    "Amount", "Time", "Log_Amount", "Hour",
    "High_Amount", "Zero_Amount", "V1_V2_interact", "V14_V17_interact",
]
N_V = len(V_COLS)

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_sample_data(n_normal=9000, n_fraud=160, seed=42):
    """Small synthetic dataset with the same 31 columns / rough imbalance as the
    real Kaggle creditcard.csv, so the app works instantly with no download."""
    rng = np.random.default_rng(seed)

    def make_rows(n, fraud):
        data = {}
        for v in V_COLS:
            shift = 0.0
            if fraud and v in ("V14", "V12", "V10", "V17"):
                shift = -4.0
            elif fraud and v == "V4":
                shift = 3.0
            data[v] = rng.normal(shift, 1.4 if fraud else 1.0, n)
        data["Amount"] = np.round(rng.exponential(35 if fraud else 60, n), 2)
        data["Time"] = rng.integers(0, 172800, n).astype(float)  # 48h window
        data["Class"] = 1 if fraud else 0
        return pd.DataFrame(data)

    df = pd.concat([make_rows(n_normal, False), make_rows(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df[V_COLS + ["Time", "Amount", "Class"]]


@st.cache_data(show_spinner=False)
def load_uploaded(file_bytes):
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    missing = set(V_COLS + ["Time", "Amount", "Class"]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    return df


def engineer_features(df):
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Log_Amount"] = np.log1p(df["Amount"])
    df["High_Amount"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
    df["Zero_Amount"] = (df["Amount"] == 0).astype(int)
    df["V1_V2_interact"] = df["V1"] * df["V2"]
    df["V14_V17_interact"] = df["V14"] * df["V17"]
    return df


# ----------------------------------------------------------------------------
# Training pipeline
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_pipeline(_df, data_key, run_full_benchmark):
    df_feat = engineer_features(_df)

    df_train_raw, df_test_raw = train_test_split(
        df_feat, test_size=0.2, random_state=42, stratify=df_feat["Class"]
    )

    X_train = df_train_raw[MODEL_FEATURES].values.astype(float)
    X_test = df_test_raw[MODEL_FEATURES].values.astype(float)
    y_train = df_train_raw["Class"].values
    y_test = df_test_raw["Class"].values

    scaler = RobustScaler()
    X_train[:, N_V:] = scaler.fit_transform(X_train[:, N_V:])
    X_test[:, N_V:] = scaler.transform(X_test[:, N_V:])

    n_fraud_train = int((y_train == 1).sum())
    k_neighbors = max(1, min(5, n_fraud_train - 1)) if n_fraud_train > 1 else 1
    smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=k_neighbors)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = XGBClassifier(
        n_estimators=150, max_depth=6, scale_pos_weight=pos_weight,
        eval_metric="aucpr", verbosity=0, random_state=42,
    )
    xgb_model.fit(X_train_sm, y_train_sm)

    results, trained, proba_map = {}, {"XGBoost": xgb_model}, {}

    def evaluate(name, model):
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        results[name] = {
            "ROC-AUC": roc_auc_score(y_test, proba),
            "Avg Precision": average_precision_score(y_test, proba),
            "F1": f1_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred),
        }
        return proba

    proba_map["XGBoost"] = evaluate("XGBoost", xgb_model)

    if run_full_benchmark:
        lr = LogisticRegression(C=0.01, max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(X_train_sm, y_train_sm)
        trained["Logistic Regression"] = lr
        proba_map["Logistic Regression"] = evaluate("Logistic Regression", lr)

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train_sm, y_train_sm)
        trained["Random Forest"] = rf
        proba_map["Random Forest"] = evaluate("Random Forest", rf)

    thresholds = np.linspace(0.1, 0.9, 80)
    f1s = [f1_score(y_test, (proba_map["XGBoost"] >= t).astype(int)) for t in thresholds]
    best_thresh = float(thresholds[int(np.argmax(f1s))])

    explainer = shap.TreeExplainer(xgb_model)

    return {
        "scaler": scaler, "xgb_model": xgb_model, "trained": trained,
        "results": results, "proba_map": proba_map,
        "y_test": y_test, "X_test": X_test, "df_test_raw": df_test_raw.reset_index(drop=True),
        "best_thresh": best_thresh, "explainer": explainer,
        "amount_95": float(_df["Amount"].quantile(0.95)),
    }


def risk_tier(prob):
    if prob >= 0.80:
        return "🚨 CRITICAL — Block immediately", FRAUD_COLOR
    if prob >= 0.60:
        return "🔴 HIGH — Flag for review", "#E67E22"
    if prob >= 0.40:
        return "🟠 MEDIUM — Monitor closely", "#F39C12"
    if prob >= 0.20:
        return "🟡 LOW — Minor alert", "#F1C40F"
    return "🟢 SAFE — Approve transaction", NORMAL_COLOR


def score_transaction(pipeline, transaction):
    v_vals = [float(transaction.get(f"V{i}", 0.0)) for i in range(1, 29)]
    amount = float(transaction.get("Amount", 0.0))
    time_val = float(transaction.get("Time", 0.0))
    log_amt = np.log1p(amount)
    hour = (time_val // 3600) % 24
    high_amt = int(amount > pipeline["amount_95"])
    zero_amt = int(amount == 0)
    v1v2 = v_vals[0] * v_vals[1]
    v14v17 = v_vals[13] * v_vals[16]

    raw = np.array([v_vals + [amount, time_val, log_amt, hour, high_amt, zero_amt, v1v2, v14v17]])
    scaled = raw.copy()
    scaled[:, N_V:] = pipeline["scaler"].transform(raw[:, N_V:])
    X_df = pd.DataFrame(scaled, columns=MODEL_FEATURES)

    prob = float(pipeline["xgb_model"].predict_proba(X_df)[0][1])
    pred = int(prob >= pipeline["best_thresh"])
    risk, color = risk_tier(prob)

    sv = pipeline["explainer"].shap_values(X_df)
    pairs = sorted(zip(MODEL_FEATURES, sv[0]), key=lambda x: abs(x[1]), reverse=True)[:6]

    return prob, pred, risk, color, pairs


# ----------------------------------------------------------------------------
# Sidebar — data source
# ----------------------------------------------------------------------------
st.sidebar.title("💳 Fraud Detection")
st.sidebar.caption("Rebuilt from Shinearushi525/Credit_Card_Fraud_Detection")

data_source = st.sidebar.radio(
    "Dataset",
    ["Synthetic demo data (instant)", "Upload creditcard.csv (Kaggle)"],
    help="The real Kaggle dataset (284,807 rows) gives results matching the README. "
         "The demo dataset is synthetic and only for trying the app instantly.",
)

if data_source == "Upload creditcard.csv (Kaggle)":
    uploaded = st.sidebar.file_uploader("creditcard.csv", type=["csv"])
    if uploaded is not None:
        try:
            df = load_uploaded(uploaded.getvalue())
            data_key = f"upload-{uploaded.name}-{len(uploaded.getvalue())}"
            st.sidebar.success(f"Loaded {len(df):,} rows")
        except Exception as e:
            st.sidebar.error(str(e))
            st.stop()
    else:
        st.sidebar.info("Get the dataset from Kaggle: mlg-ulb/creditcardfraud, then upload it here.")
        df = load_sample_data()
        data_key = "sample-default"
        st.sidebar.warning("Using synthetic demo data until you upload a file.")
else:
    df = load_sample_data()
    data_key = "sample-default"

run_full_benchmark = st.sidebar.checkbox(
    "Run full 3-model benchmark (slower)", value=False,
    help="Off = XGBoost only (fast, used for scoring). On = also trains Logistic Regression and Random Forest for comparison.",
)

st.sidebar.metric("Transactions", f"{len(df):,}")
st.sidebar.metric("Fraud rate", f"{df['Class'].mean()*100:.3f}%")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("💳 Credit Card Fraud Detection & Analysis")
st.caption(
    "SMOTE-balanced XGBoost pipeline with SHAP explainability, threshold tuning, "
    "and business-impact analysis — Streamlit rebuild of the original notebook."
)

tab_overview, tab_train, tab_score, tab_impact = st.tabs(
    ["📊 Overview", "🤖 Train & Benchmark", "🔍 Real-Time Scorer", "💰 Business Impact"]
)

# ----------------------------------------------------------------------------
# Tab: Overview
# ----------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", f"{len(df):,}")
    c2.metric("Fraud cases", f"{int(df['Class'].sum()):,}")
    c3.metric("Normal cases", f"{int((df['Class']==0).sum()):,}")
    ratio = int((df['Class']==0).sum() / max(int(df['Class'].sum()), 1))
    c4.metric("Imbalance ratio", f"{ratio}:1")

    st.subheader("Dataset overview")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    counts = df["Class"].value_counts().sort_index()
    axes[0].bar(["Normal (0)", "Fraud (1)"], counts.values,
                color=[NORMAL_COLOR, FRAUD_COLOR], edgecolor="white", linewidth=2, width=0.5)
    axes[0].set_title("Class distribution")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for cls, color, label in [(0, NORMAL_COLOR, "Normal"), (1, FRAUD_COLOR, "Fraud")]:
        axes[1].hist(np.log1p(df[df["Class"] == cls]["Amount"]), bins=40, alpha=0.6,
                     color=color, label=label)
    axes[1].set_title("Amount distribution (log scale)")
    axes[1].legend()

    df_feat_preview = engineer_features(df)
    hourly = df_feat_preview.groupby("Hour")["Class"].mean() * 100
    axes[2].fill_between(hourly.index, hourly.values, alpha=0.4, color=FRAUD_COLOR)
    axes[2].plot(hourly.index, hourly.values, color=FRAUD_COLOR, marker="o", markersize=3)
    axes[2].set_title("Fraud rate by hour of day")
    axes[2].set_xlabel("Hour")
    axes[2].set_ylabel("Fraud rate (%)")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Preview raw data"):
        st.dataframe(df.head(20), use_container_width=True)

# ----------------------------------------------------------------------------
# Tab: Train & Benchmark
# ----------------------------------------------------------------------------
with tab_train:
    st.subheader("Train the model")
    st.write(
        "Applies feature engineering (Hour, Log_Amount, High/Zero-amount flags, V1×V2, "
        "V14×V17 interactions), a RobustScaler, SMOTE oversampling on the training split "
        "only, and trains XGBoost (plus Logistic Regression / Random Forest if the full "
        "benchmark is enabled)."
    )
    train_clicked = st.button("🚀 Train model", type="primary")

    if train_clicked or "pipeline_ready" in st.session_state:
        with st.spinner("Training..."):
            t0 = time.time()
            pipeline = train_pipeline(df, data_key, run_full_benchmark)
            elapsed = time.time() - t0
        st.session_state["pipeline_ready"] = True
        st.session_state["pipeline"] = pipeline
        st.success(f"Done in {elapsed:.1f}s — best F1 threshold = {pipeline['best_thresh']:.2f}")

        results_df = pd.DataFrame(pipeline["results"]).T
        results_df = results_df.sort_values("ROC-AUC", ascending=False)
        st.subheader("Model comparison")
        st.dataframe(
            results_df.style.format("{:.4f}").background_gradient(cmap="Greens", axis=0),
            use_container_width=True,
        )

        best_name = results_df.index[0]
        best_model = pipeline["trained"][best_name]
        y_test = pipeline["y_test"]
        best_proba = pipeline["proba_map"][best_name]
        best_pred = best_model.predict(pipeline["X_test"])

        colA, colB = st.columns(2)
        with colA:
            fig, ax = plt.subplots(figsize=(5, 4.2))
            for name, proba in pipeline["proba_map"].items():
                fpr, tpr, _ = roc_curve(y_test, proba)
                auc = roc_auc_score(y_test, proba)
                ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_title("ROC curve")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)
        with colB:
            fig, ax = plt.subplots(figsize=(5, 4.2))
            for name, proba in pipeline["proba_map"].items():
                prec, rec, _ = precision_recall_curve(y_test, proba)
                ap = average_precision_score(y_test, proba)
                ax.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")
            ax.set_title("Precision-Recall curve (better for imbalanced data)")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

        st.subheader(f"Confusion matrix — {best_name} @ threshold {pipeline['best_thresh']:.2f}")
        cm = confusion_matrix(y_test, (best_proba >= pipeline["best_thresh"]).astype(int))
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Normal", "Pred Fraud"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Normal", "Actual Fraud"])
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Global SHAP feature importance (XGBoost)")
        with st.spinner("Computing SHAP values..."):
            sample_n = min(300, pipeline["X_test"].shape[0])
            X_test_df = pd.DataFrame(pipeline["X_test"], columns=MODEL_FEATURES).sample(
                sample_n, random_state=42
            )
            shap_vals = pipeline["explainer"].shap_values(X_test_df)
        importance = pd.Series(np.abs(shap_vals).mean(axis=0), index=MODEL_FEATURES).sort_values(
            ascending=True
        ).tail(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance.index, importance.values, color=FRAUD_COLOR)
        ax.set_title("Mean |SHAP value| — top 12 features")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Click **Train model** to run the pipeline.")

# ----------------------------------------------------------------------------
# Tab: Real-Time Scorer
# ----------------------------------------------------------------------------
with tab_score:
    st.subheader("Score a transaction")
    if "pipeline" not in st.session_state:
        st.warning("Train the model first in the **Train & Benchmark** tab.")
    else:
        pipeline = st.session_state["pipeline"]
        df_test_raw = pipeline["df_test_raw"]

        mode = st.radio(
            "Choose input", ["Random known fraud", "Random known normal", "Manual entry"],
            horizontal=True,
        )

        if mode == "Manual entry":
            with st.expander("Transaction fields", expanded=True):
                c1, c2 = st.columns(2)
                amount = c1.number_input("Amount (€)", min_value=0.0, value=100.0, step=1.0)
                time_val = c2.number_input("Time (seconds since first txn)", min_value=0.0, value=50000.0, step=100.0)
                st.caption("V1–V28 are anonymized PCA features. Leave at 0 for a typical value, "
                           "or tweak V14/V12/V10/V17/V4 (the strongest fraud signals) to see the effect.")
                v_cols_ui = st.columns(7)
                v_input = {}
                for i in range(1, 29):
                    default = -6.0 if i == 14 else (-5.5 if i == 12 else 0.0)
                    v_input[f"V{i}"] = v_cols_ui[(i - 1) % 7].number_input(
                        f"V{i}", value=float(default), step=0.5, key=f"v{i}"
                    )
            transaction = {**v_input, "Amount": amount, "Time": time_val}
        else:
            subset = df_test_raw[df_test_raw["Class"] == (1 if mode == "Random known fraud" else 0)]
            if st.button("🎲 Draw a new example"):
                st.session_state["example_idx"] = int(np.random.randint(len(subset)))
            idx = st.session_state.get("example_idx", 0) % max(len(subset), 1)
            row = subset.iloc[idx]
            transaction = row[V_COLS + ["Amount", "Time"]].to_dict()
            st.caption(f"Actual label: {'FRAUD' if row['Class']==1 else 'NORMAL'} · "
                       f"Amount €{row['Amount']:.2f} · Time {row['Time']:.0f}s")

        if st.button("🔎 Score transaction", type="primary"):
            prob, pred, risk, color, pairs = score_transaction(pipeline, transaction)

            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud probability", f"{prob*100:.2f}%")
            c2.metric("Decision", "🚫 FRAUD" if pred else "✅ LEGITIMATE")
            c3.markdown(f"**Risk level**  \n<span style='color:{color}; font-weight:700'>{risk}</span>",
                        unsafe_allow_html=True)

            st.progress(min(max(prob, 0.0), 1.0))

            st.subheader("Why this decision — top SHAP factors")
            names = [p[0] for p in pairs][::-1]
            vals = [p[1] for p in pairs][::-1]
            colors = [FRAUD_COLOR if v > 0 else NORMAL_COLOR for v in vals]
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.barh(names, vals, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (→ pushes toward FRAUD, ← pushes toward NORMAL)")
            st.pyplot(fig)
            plt.close(fig)

# ----------------------------------------------------------------------------
# Tab: Business Impact
# ----------------------------------------------------------------------------
with tab_impact:
    st.subheader("Business impact of the model")
    if "pipeline" not in st.session_state:
        st.warning("Train the model first in the **Train & Benchmark** tab.")
    else:
        pipeline = st.session_state["pipeline"]
        y_test = pipeline["y_test"]
        proba = pipeline["proba_map"]["XGBoost"]
        thresh = pipeline["best_thresh"]

        c1, c2, c3 = st.columns(3)
        avg_fraud_value = c1.number_input("Average fraud transaction value (€)", value=122.0, min_value=0.0)
        false_alarm_cost = c2.number_input("Cost per false alarm (€)", value=2.50, min_value=0.0)
        daily_txns = c3.number_input("Daily transactions to project", value=1_000_000, min_value=1000, step=1000)

        pred = (proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        caught_value = tp * avg_fraud_value
        missed_value = fn * avg_fraud_value
        false_alarm_total = fp * false_alarm_cost
        net_savings = caught_value - false_alarm_total
        no_model_loss = (tp + fn) * avg_fraud_value

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frauds caught", f"{tp:,}")
        m2.metric("Frauds missed", f"{fn:,}")
        m3.metric("False alarms", f"{fp:,}")
        m4.metric("Net savings (test set)", f"€{net_savings:,.0f}")

        st.metric(
            f"Projected daily savings at {daily_txns:,.0f} txns/day",
            f"€{net_savings/len(y_test)*daily_txns:,.0f}",
        )

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        cats = ["Fraud\ncaught", "Fraud\nmissed", "False alarm\ncost", "Net\nsavings"]
        vals = [caught_value, -missed_value, -false_alarm_total, net_savings]
        colors = [NORMAL_COLOR, FRAUD_COLOR, "#F39C12", NORMAL_COLOR if net_savings > 0 else FRAUD_COLOR]
        axes[0].bar(cats, vals, color=colors, edgecolor="white")
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set_title("Financial impact breakdown (€)")

        scale = [10_000, 100_000, 1_000_000, 10_000_000]
        savings_scale = [net_savings / len(y_test) * s for s in scale]
        axes[1].bar([f"{s//1000}K" if s < 1_000_000 else f"{s//1_000_000}M" for s in scale],
                    savings_scale, color=ACCENT_COLOR, edgecolor="white")
        axes[1].set_title("Projected daily savings at scale")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "⚠️ With the synthetic demo dataset these numbers are illustrative only. "
            "Upload the real Kaggle creditcard.csv for meaningful business figures."
        )
