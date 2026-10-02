"""One-time evaluation of the selected model on the held-out test set.

Usage: python -m src.evaluate      (run after `python -m src.train`)
"""
from __future__ import annotations

import json

import joblib
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src import config, plots
from src.data import load_dataset, split_data


def main() -> None:
    plots.set_style()
    _, X_test, _, y_test = split_data(load_dataset())
    best_name = json.loads((config.METRICS / "best_model.json").read_text())["best_model"]
    model = joblib.load(config.MODELS / "best_model.joblib")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]          # P(malignant)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "model": best_name, "n_test": int(len(y_test)),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_malignant": precision_score(y_test, y_pred),
        "recall_malignant": recall_score(y_test, y_pred),
        "f1_malignant": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    (config.METRICS / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    report = classification_report(y_test, y_pred, target_names=list(config.CLASS_NAMES.values()))
    (config.METRICS / "classification_report.txt").write_text(report)

    plots.plot_confusion_matrix(y_test, y_pred, best_name, config.FIGURES / "confusion_matrix.png")
    plots.plot_roc(y_test, y_prob, best_name, config.FIGURES / "roc_curve.png")
    print(f"Best model: {best_name}\nTest ROC-AUC: {metrics['roc_auc']:.4f}\n\n{report}")


if __name__ == "__main__":
    main()
