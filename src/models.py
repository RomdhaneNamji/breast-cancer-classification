"""Model definitions. Scaling lives inside the Pipeline so that, during
cross-validation, it is fitted on the training folds only (no leakage)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config


def get_models(seed: int = config.SEED) -> dict:
    """The three classifiers compared in the notebook (hyperparameters unchanged)."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=7)),
        ]),
        # Tree ensembles are scale-invariant, so no scaler is needed.
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed),
    }
