"""Repeated stratified cross-validation on the TRAINING set only, model selection
by mean ROC-AUC, and fitting of the selected model on the full training set.

Usage: python -m src.train
The held-out test set is never touched here (see evaluate.py).
"""
from __future__ import annotations

import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src import config, plots
from src.data import load_dataset, split_data
from src.models import get_models

SCORING = {m: m for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}


def run_cross_validation(models: dict, X, y, seed: int = config.SEED, n_jobs: int = -1) -> pd.DataFrame:
    """Return one row per model with mean/SD of every metric and the runtime."""
    cv = RepeatedStratifiedKFold(n_splits=config.CV_SPLITS, n_repeats=config.CV_REPEATS,
                                 random_state=seed)
    rows = []
    for name, estimator in models.items():
        t0 = time.perf_counter()
        scores = cross_validate(estimator, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs)
        wall = time.perf_counter() - t0
        row = {"model": name}
        for metric in SCORING:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        row["cv_wall_time_s"] = round(wall, 3)
        row["fit_time_mean_s"] = scores["fit_time"].mean()
        rows.append(row)
        print(f"  {name:<20} AUC={row['roc_auc_mean']:.4f}  F1={row['f1_mean']:.4f}  [{wall:.1f}s]")
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def main() -> None:
    plots.set_style()
    df = load_dataset()
    X_train, _, y_train, _ = split_data(df)          # test set deliberately ignored
    print(f"Training set: {len(X_train)} samples")

    cv_results = run_cross_validation(get_models(), X_train, y_train)
    config.METRICS.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(config.METRICS / "cv_results.csv", index=False)

    best_name = cv_results.loc[0, "model"]
    best_model = get_models()[best_name].fit(X_train, y_train)
    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.MODELS / "best_model.joblib")
    (config.METRICS / "best_model.json").write_text(json.dumps({"best_model": best_name}, indent=2))

    plots.plot_model_comparison(cv_results, config.FIGURES / "model_comparison.png")
    plots.plot_cv_runtime(cv_results, config.FIGURES / "cv_runtime.png")
    print(f"\nSelected model (highest mean CV ROC-AUC): {best_name}")


if __name__ == "__main__":
    main()
