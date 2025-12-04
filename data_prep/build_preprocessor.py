import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> ColumnTransformer:
    """
    Vytváří ColumnTransformer pro numerické sloupce:
        - KNN imputace,
        - StandardScaler.

    Output: Objekt pre použitie v plnej pipeline.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                make_column_selector(dtype_include=np.number),
            ),
        ],
        remainder="drop",
    )

    return preprocessor
