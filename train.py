import os
import json

import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    StratifiedKFold,
    cross_val_score,
)



from data_prep.cleaning import *
from pipeline.build_pipeline import build_pipeline


def train_main(input_csv: str, artifacts_dir: str):
    """
    Hlavní trénovací funkce:
    - načte dataset,
    - rozdělí na train/test,
    - natrénuje pipeline,
    - spočítá metriky (Accuracy, AUC, MCC, F1),
    - vykreslí learning curve,
    - provede cross-validaci,
    - uloží model a artefakty do zadaného adresáře.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # NAČTENÍ A PŘÍPRAVA DAT
    df = pd.read_csv(input_csv)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # ROZDĚLENÍ DAT NA TRÉNOVACÍ A TESTOVACÍ SADU
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # TRÉNOVÁNÍ MODELU (PIPELINE)
    # Sestavení kompletní pipeline (preprocessing + model)
    pipe = build_pipeline()
    # Natrénování pipeline na trénovacích datech
    pipe.fit(X_train, y_train)

    # PREDIKCE NA TESTOVACÍ SADĚ
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # VÝPOČET KLASIFIKAČNÍCH METRIK
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    f1_pos = f1_score(y_test, y_pred, average="binary")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print("\n=== RESULTS (hold-out test set) ===")
    print(f"Accuracy:           {acc:.4f}")
    print(f"AUC:                {auc:.4f}")
    print(f"MCC:                {mcc:.4f}")
    print(f"F1-score (class 1): {f1_pos:.4f}")
    print(f"F1-score (macro):   {f1_macro:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # VÝPOČET A VYKRESLENÍ LEARNING CURVE
    print("\n[train] Computing learning curve...")
    lc_pipe = build_pipeline()
    mcc_scorer = make_scorer(matthews_corrcoef)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=lc_pipe,
        X=X,
        y=y,
        cv=N_SPLITS,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring=mcc_scorer,
        n_jobs=-1,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        train_sizes,
        train_scores.mean(axis=1),
        "o-",
        label="Train MCC",
    )
    plt.plot(
        train_sizes,
        val_scores.mean(axis=1),
        "o-",
        label="Validation MCC",
    )
    plt.title("Learning Curve – Random Forest (full pipeline)")
    plt.xlabel("Training samples")
    plt.ylabel("MCC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "learning_curve.png"))
    plt.close()

    # CROSS-VALIDACE PRO ROBUSTNÍ ODHAD VÝKONU
    print("[train] Running cross-validation (MCC)...")
    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    cv_pipe = build_pipeline()
    cv_scores = cross_val_score(
        cv_pipe,
        X,
        y,
        cv=cv,
        scoring=mcc_scorer,
        n_jobs=-1,
    )

    cv_report = {
        "cv_mcc_mean": float(np.mean(cv_scores)),
        "cv_mcc_std": float(np.std(cv_scores)),
        "cv_scores": list(map(float, cv_scores)),
    }

    with open(os.path.join(artifacts_dir, "cv_report.json"), "w") as f:
        json.dump(cv_report, f, indent=2)

    print(
        f"[CV] MCC: {cv_report['cv_mcc_mean']:.4f} "
        f"+/- {cv_report['cv_mcc_std']:.4f}"
    )

    # ULOŽENÍ MODELU A ARTEFAKTŮ
    joblib.dump(pipe, os.path.join(artifacts_dir, "model.pkl"))

    try:
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
        pd.Series(feature_names).to_json(
            os.path.join(artifacts_dir, "feature_names.json")
        )
    except Exception:
        pass

    print("\nTraining finished. Pipeline model saved to model.pkl.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.train <input_csv> <artifacts_dir>")
        raise SystemExit(1)

    input_csv = sys.argv[1]
    artifacts_dir = sys.argv[2]
    train_main(input_csv, artifacts_dir)
