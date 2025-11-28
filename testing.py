# src/test.py

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    matthews_corrcoef
)

from src.config import TARGET_COL
from src.data_prep import replace_zeros_with_nan, feature_engineering, preprocess_for_testing


def main(input_csv: str, artifacts_dir: str):
    # --- Load artifacts ---
    model = joblib.load(os.path.join(artifacts_dir, "model.pkl"))
    imputer = joblib.load(os.path.join(artifacts_dir, "imputer.pkl"))
    scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))

    df = pd.read_csv(input_csv)

    # --- Separate label if exists ---
    if TARGET_COL in df.columns:
        y_true = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
    else:
        y_true = None
        X = df.copy()

    # --- Pre-clean ---
    X = replace_zeros_with_nan(X)
    X = feature_engineering(X)

    # --- Apply imputer + scaler ---
    X_scaled = preprocess_for_testing(X, imputer, scaler)

    # --- Predictions ---
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # --- Save predictions ---
    pred_path = os.path.join(artifacts_dir, "predictions.csv")
    pd.DataFrame({"Prediction": y_pred, "Probability": y_prob}).to_csv(pred_path, index=False)
    print(f"Predictions saved to {pred_path}")

    # --- If ground truth exists → compute metrics ---
    if y_true is not None:
        print("\n=== TEST METRICS ===")

        from sklearn.metrics import f1_score

        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred, average="binary")  # F1 pro diabetiky (třída 1)
        f1_macro = f1_score(y_true, y_pred, average="macro")  # F1 průměr pro obě třídy

        print(f"Accuracy (ACC):   {acc:.4f}")
        print(f"ROC-AUC:          {auc:.4f}")
        print(f"MCC:              {mcc:.4f}")
        print(f"F1-score (class 1):     {f1:.4f}")
        print(f"F1-score (macro avg):   {f1_macro:.4f}")

        # --- Confusion Matrix ---
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # --- Save confusion matrix ---
        pd.DataFrame(cm).to_csv(os.path.join(artifacts_dir, "confusion_matrix_test.csv"), index=False)

        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC Curve – Test dataset")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True)
        plt.legend()
        roc_path = os.path.join(artifacts_dir, "roc_curve_test.png")
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()

        print(f"ROC curve saved to {roc_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.test <input_csv> <artifacts_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
