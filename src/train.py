# src/train.py

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold

from src.config import TARGET_COL, N_SPLITS, RANDOM_STATE
from src.data_prep import replace_zeros_with_nan, feature_engineering, preprocess_for_training, preprocess_for_testing
from src.model import get_rf_model


def train_main(input_csv: str, artifacts_dir: str):
    os.makedirs(artifacts_dir, exist_ok=True)

    # === LOAD DATA ===
    df = pd.read_csv(input_csv)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # === Pre-clean ===
    X = replace_zeros_with_nan(X)

    # === FE ===
    X = feature_engineering(X)

    # === Train-test split ===
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # === Imputace + scaling ===
    X_train, imputer, scaler = preprocess_for_training(X_train_df)
    X_test = scaler.transform(imputer.transform(X_test_df))

    # === Model ===
    model = get_rf_model()
    model.fit(X_train, y_train)

    # === Eval ===
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    from sklearn.metrics import f1_score
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")  # F1 pro diabetiky (třída 1)
    f1_macro = f1_score(y_test, y_pred, average="macro")  # F1 průměr pro obě třídy

    # === OUTPUT ===
    print("\n=== RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"MCC:      {mcc:.4f}\n")
    print(f"F1-score (class 1):     {f1:.4f}")
    print(f"F1-score (macro avg):   {f1_macro:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Uložení CM
    pd.DataFrame(cm).to_csv(os.path.join(artifacts_dir, "confusion_matrix.csv"), index=False)

    # === Learning Curve ===
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        pd.concat([X_train_df, X_test_df])[X.columns],
        pd.concat([y_train, y_test]),
        cv=N_SPLITS,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy"
    )

    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation')
    plt.title("Learning Curve – Random Forest")
    plt.xlabel("Training samples")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "learning_curve.png"))
    plt.close()

    # === CV REPORT ===
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []

    for train_idx, test_idx in cv.split(X, y):
        X_tr_df, X_te_df = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        X_tr_df = replace_zeros_with_nan(X_tr_df)
        X_te_df = replace_zeros_with_nan(X_te_df)

        X_tr_df = feature_engineering(X_tr_df)
        X_te_df = feature_engineering(X_te_df)

        X_tr, imp, sc = preprocess_for_training(X_tr_df)
        X_te = preprocess_for_testing(X_te_df, imp, sc)

        m = get_rf_model()
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        cv_results.append(matthews_corrcoef(y_te, pred))

    with open(os.path.join(artifacts_dir, "cv_report.json"), "w") as f:
        json.dump({
            "cv_mcc_mean": float(np.mean(cv_results)),
            "cv_mcc_std": float(np.std(cv_results)),
            "cv_scores": list(map(float, cv_results))
        }, f, indent=2)

    # === Save artifacts ===
    import joblib
    joblib.dump(model, os.path.join(artifacts_dir, "model.pkl"))
    joblib.dump(imputer, os.path.join(artifacts_dir, "imputer.pkl"))
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
    X.columns.to_series().to_json(os.path.join(artifacts_dir, "feature_names.json"))

    print("\nTraining finished. Artifacts saved.")


if __name__ == "__main__":
    import sys
    train_main(sys.argv[1], sys.argv[2])
