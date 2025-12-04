from sklearn.ensemble import RandomForestClassifier
from config import  *

def random_forest() -> RandomForestClassifier:
    """
        Vytváří a vrací nakonfigurovaný RandomForestClassifier pro binární predikci diabetu.
        Model používá vyvážení tříd, omezenou hloubku stromů, omezené množství rysů
        a větší počet stromů, což zlepšuje stabilitu predikce i MCC.
        """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=1,
        max_features=0.6,
        class_weight="balanced",
        random_state= RANDOM_STATE,
        n_jobs=-1,
    )
