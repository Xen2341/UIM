# src/data_prep.py

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from src.config import ZERO_NOT_POSSIBLE


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nahrazuje nereálné nuly v klinických proměnných NaN.
    """
    df = df.copy()
    for col in ZERO_NOT_POSSIBLE:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering přesně podle tvého skriptu.
    """
    df = df.copy()

    if {"Glucose", "Insulin"}.issubset(df.columns):
        df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1)

    if {"BMI", "Age"}.issubset(df.columns):
        df["BMI_Age"] = df["BMI"] * df["Age"]
        df["BMI_sq"] = df["BMI"] ** 2

    if {"Pregnancies", "Age"}.issubset(df.columns):
        df["Pregnancies_Age"] = df["Pregnancies"] * df["Age"]

    if {"Glucose", "BMI"}.issubset(df.columns):
        df["Glucose_x_BMI"] = df["Glucose"] * df["BMI"]
        df["Glucose_sq"] = df["Glucose"] ** 2

    return df


def preprocess_for_training(X: pd.DataFrame) -> tuple[np.ndarray, KNNImputer, StandardScaler]:
    """
    1) imputace KNN
    2) scaling
    """
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()

    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, imputer, scaler


def preprocess_for_testing(X, imputer, scaler):
    X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns)
    X_scaled = scaler.transform(X_imp)
    return X_scaled
