"""
Generates creditcard_sample.csv — a synthetic dataset with the same schema and
rough class imbalance as the real Kaggle creditcard.csv (mlg-ulb/creditcardfraud).

This is NOT real data. It exists so the app/repo can be tried without downloading
the ~150MB Kaggle file. For real fraud-detection results, download the actual
dataset from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and use the
"Upload creditcard.csv" option in the app instead.

Usage:
    python generate_sample_dataset.py
"""

import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]


def make_rows(n, fraud, rng):
    data = {}
    for v in V_COLS:
        shift = 0.0
        if fraud and v in ("V14", "V12", "V10", "V17"):
            shift = -4.0
        elif fraud and v == "V4":
            shift = 3.0
        data[v] = rng.normal(shift, 1.4 if fraud else 1.0, n)
    data["Amount"] = np.round(rng.exponential(35 if fraud else 60, n), 2)
    data["Time"] = rng.integers(0, 172800, n).astype(float)
    data["Class"] = 1 if fraud else 0
    return pd.DataFrame(data)


def generate(n_normal=9000, n_fraud=160, seed=42, out_path="creditcard_sample.csv"):
    rng = np.random.default_rng(seed)
    df = pd.concat([make_rows(n_normal, False, rng), make_rows(n_fraud, True, rng)], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df[V_COLS + ["Time", "Amount", "Class"]]
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote {len(df):,} rows ({n_fraud} fraud) to {out_path}")


if __name__ == "__main__":
    generate()
