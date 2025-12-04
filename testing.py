import os
import sys
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    f1_score,
    classification_report,
)

from config import TARGET_COL


def test_main(input_csv: str, artifacts_dir: str) -> None:
    """
    Načte uložený model, provede inference nad testovacím souborem
    a uloží predikce do predictions.csv. Vypíše klasifikační metriky,
    confusion matrix a případně srovná výkon s očekáváním.
    Slouží pro evaluaci na externí sadě dat dodané zadavatelem.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # Načtení již natrénovaného modelu
    model_path = os.path.join(artifacts_dir, "model.pkl")
    model = joblib.load(model_path)

    # Načtení CSV souboru s testovacími daty
    df = pd.read_csv(input_csv)

    # Oddělení cílové proměnné (y) od vstupních příznaků (X)
    y_true = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Predikce
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Ulozeni predikci
    pred_path = os.path.join(artifacts_dir, "predictions.csv")
    pd.DataFrame(
        {
            "Prediction": y_pred,
            "Probability": y_prob,
        }
    ).to_csv(pred_path, index=False)
    print(f"[test] Predictions saved to {pred_path}")

    print("\n=== RESULTS (external test set) ===")

    # VÝPOČET A ZOBRAZENÍ KLASIFIKAČNÍCH METRIK
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred, average="binary")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print(f"Accuracy:           {acc:.4f}")
    print(f"AUC:                {auc:.4f}")
    print(f"MCC:                {mcc:.4f}")
    print(f"F1-score (class 1): {f1_pos:.4f}")
    print(f"F1-score (macro):   {f1_macro:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    print("[test] done.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.test <input_csv> <artifacts_dir>")
        sys.exit(1)

    input_csv = sys.argv[1]
    artifacts_dir = sys.argv[2]
    test_main(input_csv, artifacts_dir)
