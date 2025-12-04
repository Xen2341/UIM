from sklearn.ensemble import RandomForestClassifier
from config import  *

def random_forest() -> RandomForestClassifier:
    """
        Vytváří a vrací nakonfigurovaný RandomForestClassifier pro binární predikci diabetu.
        Model používá vyvážení tříd, omezenou hloubku stromů, omezené množství rysů
        a větší počet stromů, což zlepšuje stabilitu predikce i MCC.

        Output: Nakonfigurovaný RandomForestClassifier pro binární klasifikaci.
        """
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight="balanced",
        random_state= RANDOM_STATE,
        n_jobs=-1,
    )
