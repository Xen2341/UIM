# src/model.py

from sklearn.ensemble import RandomForestClassifier
from src.config import RF_PARAMS


def get_rf_model():
    """
    Vrací RandomForest model podle tvého původního nastavení.
    """
    return RandomForestClassifier(**RF_PARAMS)
