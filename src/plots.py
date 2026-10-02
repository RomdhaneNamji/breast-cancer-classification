"""Figure functions with one consistent, colour-blind-safe style.

Every function draws one figure, saves it (if `path` is given) and returns it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src import config

# Okabe-Ito colour-blind-safe palette
CLASS_COLORS = {"Benign": "#0072B2", "Malignant": "#D55E00"}
MODEL_COLORS = {"Logistic Regression": "#009E73", "KNN": "#E69F00", "Random Forest": "#CC79A7"}
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1", "roc_auc": "ROC-AUC"}


def set_style() -> None:
    """Apply the shared plotting style (call once per script / notebook)."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.titleweight": "bold", "axes.titlesize": 13, "grid.alpha": 0.4,
    })


def _save(fig: plt.Figure, path: Path | None) -> plt.Figure:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def plot_class_distribution(df: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    counts = df[config.TARGET].value_counts().sort_index()
    labels = [config.CLASS_NAMES[i] for i in counts.index]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=[CLASS_COLORS[l] for l in labels], width=0.6)
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 4,
                f"{n} ({n / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Class distribution (n = {counts.sum()})")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def plot_correlation(df: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Pearson correlation of the 10 '*_mean' features and the diagnosis."""
    cols = [c for c in df.columns if c.endswith("_mean")] + [config.TARGET]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # keep the diagonal
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
                linewidths=0.4, annot_kws={"size": 8},
                cbar_kws={"shrink": 0.75, "label": "Pearson r"}, ax=ax)
    ax.grid(False)
    ax.set_title("Correlation of mean nuclear features and diagnosis")
    return _save(fig, path)


def plot_feature_distributions(df: pd.DataFrame, features: list[str] | None = None,
                               path: Path | None = None) -> plt.Figure:
    features = features or config.EDA_BOXPLOT_FEATURES
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    labels = [config.CLASS_NAMES[i] for i in (0, 1)]
    for ax, feature in zip(axes.ravel(), features):
        sns.boxplot(data=df, x=config.TARGET, y=feature, hue=config.TARGET, legend=False,
                    palette=[CLASS_COLORS[l] for l in labels], width=0.5, ax=ax,
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel(feature)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
    fig.suptitle("Selected features by diagnosis", fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, path)


def plot_model_comparison(cv: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    """Mean +/- SD of each metric over the repeated CV folds (dot plot).

    A dot plot avoids the truncated-baseline problem of zoomed bar charts.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8))
    offsets = np.linspace(-0.22, 0.22, len(cv))
    for offset, (_, row) in zip(offsets, cv.iterrows()):
        y = np.arange(len(METRICS)) + offset
        ax.errorbar([row[f"{m}_mean"] for m in METRICS], y,
                    xerr=[row[f"{m}_std"] for m in METRICS], fmt="o", capsize=3,
                    color=MODEL_COLORS[row["model"]], label=row["model"])
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.invert_yaxis()
    ax.set_xlabel("Score (mean ± SD over 5×10 repeated stratified CV folds, training set)")
    ax.set_xlim(right=1.0)  # scores cannot exceed 1; SD bars are clipped at the axis edge
    ax.set_title("Cross-validated model comparison")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    return _save(fig, path)


def plot_cv_runtime(cv: pd.DataFrame, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    order = cv.sort_values("cv_wall_time_s")
    ax.barh(order["model"], order["cv_wall_time_s"],
            color=[MODEL_COLORS[m] for m in order["model"]])
    ax.set_xlabel("Wall-clock time for 50 CV fits (s) — machine dependent")
    ax.set_title("Cross-validation runtime")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=list(config.CLASS_NAMES.values()),
        cmap="Blues", values_format="d", colorbar=False, ax=ax)
    ax.grid(False)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{title}\nConfusion matrix (test set)")
    return _save(fig, path)


def plot_roc(y_true, y_prob, title: str, path: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_true, y_prob, name=title, color="#D55E00", lw=2.2,
                                     plot_chance_level=True, ax=ax)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (malignant = positive)")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    return _save(fig, path)
