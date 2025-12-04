from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from data_prep.cleaning import ClinicalCleaner
from data_prep.feature_engineering import feature_engineering
from data_prep.build_preprocessor import build_preprocessor


from model.random_forest import random_forest


def build_pipeline() -> Pipeline:
    """
    Sestavuje kompletní ML pipeline složenou z:
    1) ClinicalCleaner – základní čištění,
    2) feature_engineering – tvorba nových příznaků,
    3) build_preprocessor – imputace + škálování,
    4) RandomForestClassifier – finální model.

    Pipeline je plně kompatibilní s cross-validací a joblib uložením.
    """
    cleaner = ClinicalCleaner(iqr_factor=2.5)
    fe_transformer = FunctionTransformer(feature_engineering, validate=False)
    preprocessor = build_preprocessor()
    clf = random_forest()

    pipe = Pipeline(
        steps=[
            ("clean", cleaner),
            ("fe", fe_transformer),
            ("prep", preprocessor),
            ("clf", clf),
        ]
    )

    return pipe
