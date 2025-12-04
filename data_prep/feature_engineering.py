import numpy as np
import pandas as pd


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provádí rozšíření datasetu o nové klinické příznaky. Vypočítává metabolické indexy
    (HOMA-IR, GI_Ratio), interakční a mocninné rysy, binární rizikové příznaky
    a log-transformace. Výsledkem je rozšířená množina rysů pro model.

    Input: DataFrame s původními příznaky.
    Output: DataFrame s přidanými novými příznaky.
    """

    df.copy()

    if {"Glucose", "Insulin"}.issubset(df.columns):
        mask = df["Glucose"].notna() & df["Insulin"].notna() & (df["Insulin"] >= 0)
        df.loc[mask, "HOMA_IR"] = (
            df.loc[mask, "Glucose"] * df.loc[mask, "Insulin"]
        ) / 405
        df.loc[mask, "GI_Ratio"] = df.loc[mask, "Glucose"] / (
            df.loc[mask, "Insulin"] + 1
        )

    if {"BMI", "Age"}.issubset(df.columns):
        mask = df["BMI"].notna() & df["Age"].notna()
        df.loc[mask, "BMI_Age"] = df.loc[mask, "BMI"] * df.loc[mask, "Age"]

    if {"Pregnancies", "Age"}.issubset(df.columns):
        mask = (
            df["Pregnancies"].notna()
            & df["Age"].notna()
            & (df["Age"] > 0)
        )
        df.loc[mask, "Preg_Age_Ratio"] = df.loc[mask, "Pregnancies"] / (
            df.loc[mask, "Age"] + 1
        )

    for col in ["Glucose", "BMI", "Age"]:
        if col in df.columns:
            df[f"{col}_sq"] = df[col] ** 2

    risk_flags: list[str] = []

    if "Glucose" in df.columns:
        df["High_Glucose"] = (df["Glucose"] >= 140).astype(float)
        risk_flags.append("High_Glucose")

    if "BMI" in df.columns:
        df["Obese"] = (df["BMI"] >= 30).astype(float)
        risk_flags.append("Obese")

    if "Age" in df.columns:
        df["Senior_Flag"] = (df["Age"] >= 45).astype(float)
        risk_flags.append("Senior_Flag")

    if "BloodPressure" in df.columns:
        df["Hypertension"] = (df["BloodPressure"] >= 90).astype(float)
        risk_flags.append("Hypertension")

    if risk_flags:
        df["Risk_Score"] = df[risk_flags].sum(axis=1)

    for col in ["Insulin", "DiabetesPedigreeFunction"]:
        if col in df.columns:
            mask = df[col].notna() & (df[col] >= 0)
            df.loc[mask, f"{col}_log"] = np.log1p(df.loc[mask, col])

    if "HOMA_IR" in df.columns:
        mask = df["HOMA_IR"].notna() & (df["HOMA_IR"] >= 0)
        df.loc[mask, "HOMA_IR_log"] = np.log1p(df.loc[mask, "HOMA_IR"])

    return df
