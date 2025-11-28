# src/config.py

# Cílová proměnná
TARGET_COL = "Outcome"

# Sloupce, kde 0 znamená chybějící hodnotu
ZERO_NOT_POSSIBLE = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Hyperparametry modelu dle tvého originálního algoritmu
RF_PARAMS = dict(
    n_estimators=600,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# CV nastavení
N_SPLITS = 5
RANDOM_STATE = 42
