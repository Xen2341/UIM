import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import *

def replace_zeros_and_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nahradí neplatné klinické hodnoty nulou (tam, kde nula není možná)
    a negativní hodnoty označí jako NaN, aby byly následně imputovány.
    """

    df.copy()

    # Přepis nul na NaN v klinických atributech, kde nula fyzicky nedává smysl.
    for col in ZERO_NOT_POSSIBLE:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Eliminace chybně měřených negativních hodnot – vstup pro imputaci.

    df[NUM_COLS] = df[NUM_COLS].mask(df[NUM_COLS] < 0, np.nan)

    return df


class ClinicalCleaner(BaseEstimator, TransformerMixin):
    """
        Transformer pro klinické čištění dat:
        - nahrazuje neplatné hodnoty (0, negativní),
        - detekuje extrémní hodnoty podle IQR a ořezává je,
        - zachovává stabilní chování i při malém množství dat (min_valid_samples).
        Používá se jako první krok pipeline.
        """

    # INICIALIZACE TRANSFORMERU
    def __init__(self, iqr_factor: float = 1.5, min_valid_samples: int = 10):
        self.iqr_factor = iqr_factor
        self.min_valid_samples = min_valid_samples
        self.num_cols_: list[str] | None = None
        self.clip_bounds_: dict[str, tuple[float, float]] | None = None

    # METODA FIT - UČENÍ HRANIC PRO OŘEZÁVÁNÍ OUTLIERŮ
    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        d = replace_zeros_and_negatives(X)

        self.num_cols_ = d.select_dtypes(include=[np.number]).columns.tolist()
        self.clip_bounds_ = {}


        # VÝPOČET IQR HRANIC PRO KAŽDÝ NUMERICKÝ SLOUPEC
        for col in self.num_cols_:
            # Odstranění NaN hodnot pro výpočet statistik
            series = d[col].dropna()

            # Kontrola minimálního počtu vzorků pro spolehlivý výpočet
            if len(series) < self.min_valid_samples:
                print(
                    f"Varování: Sloupec '{col}' má pouze {len(series)} "
                    f"validních hodnot, přeskakuji detekci outlierů"
                )
                continue

            # Výpočet kvartilů a mezikvartilového rozpětí (IQR)
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            # Kontrola nulového IQR (všechny hodnoty jsou stejné)
            if iqr == 0:
                print(
                    f"Varování: Sloupec '{col}' má nulový IQR, "
                    f"přeskakuji detekci outlierů"
                )
                continue

            # Výpočet dolní a horní hranice pro ořezávání
            # Standardní Tukeyho metoda: Q1 - 1.5*IQR a Q3 + 1.5*IQR
            low = q1 - self.iqr_factor * iqr
            high = q3 + self.iqr_factor * iqr
            self.clip_bounds_[col] = (low, high)

        return self

    # METODA TRANSFORM - APLIKACE NAUČENÝCH HRANIC NA DATA
    def transform(self, X: pd.DataFrame):
        # Konverze na DataFrame s použitím naučených názvů sloupců
        if not isinstance(X, pd.DataFrame):
            if self.num_cols_ is not None:
                X = pd.DataFrame(X, columns=self.num_cols_)
            else:
                X = pd.DataFrame(X)

        # Předčištění dat - nahrazení nul a negativních hodnot
        d = replace_zeros_and_negatives(X)

        # OŘEZÁNÍ OUTLIERŮ PODLE NAUČENÝCH HRANIC
        # Aplikace clip() na každý sloupec s definovanými hranicemi
        if self.clip_bounds_:
            for col, (low, high) in self.clip_bounds_.items():
                if col in d.columns:
                    d[col] = d[col].clip(lower=low, upper=high)

        return d
