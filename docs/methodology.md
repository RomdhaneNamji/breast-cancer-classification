# Methodology

This document describes exactly what the code in `src/` does. Everything here
mirrors the original notebook; nothing has been tuned or added.

## 1. Data
* WDBC dataset, 569 samples × 30 numeric features (see `data/README.md`).
* Columns `id` (identifier) and `Unnamed: 32` (empty) are dropped.
* Target encoded `M → 1` (malignant, the *positive* class) and `B → 0`.
* Validation: 569 × 31 columns, no missing values, binary target.

## 2. Train/test split
* 80 % / 20 % (455 / 114 samples), **stratified** on the diagnosis, `random_state=42`.
* The test set is used **once**, in `src/evaluate.py`, after model selection.

## 3. Models (`src/models.py`)
| Model | Preprocessing | Hyperparameters (all as in the notebook) |
|---|---|---|
| Logistic Regression | `StandardScaler` | `max_iter=2000`, otherwise scikit-learn defaults (L2, C=1) |
| k-Nearest Neighbours | `StandardScaler` | `n_neighbors=7`, otherwise defaults |
| Random Forest | none (scale-invariant) | `n_estimators=300`, otherwise defaults |

Scaling is part of a scikit-learn `Pipeline`, so in every cross-validation fold the
scaler is fitted on the training folds only. This prevents information from the
validation fold leaking into preprocessing.

No hyperparameter search was performed; values were fixed a priori.

## 4. Model comparison (`src/train.py`)
* Repeated stratified k-fold CV on the **training set only**: 5 folds × 10 repeats = 50 fits per model, `random_state=42` (identical folds for every model).
* Metrics per fold: accuracy, precision, recall, F1, ROC-AUC (malignant = positive class, default 0.5 decision threshold for the threshold-based metrics).
* Reported as mean and standard deviation over the 50 folds.
  The folds overlap, so they are **not independent**: the SD describes fold-to-fold variability and must not be used for a naive significance test.
* Selection rule: highest mean ROC-AUC. The selected model is refitted on the full training set.

## 5. Final evaluation (`src/evaluate.py`)
The selected model predicts the held-out test set once. Reported: accuracy,
precision / recall / F1 for the malignant class, ROC-AUC, confusion matrix, ROC curve.

## 6. Reproducibility controls
* One seed (`config.SEED = 42`) for the split, the CV folds and the stochastic models.
* Paths are relative to the repository root.
* Tests (`tests/`) check data integrity, split stratification, absence of scaler leakage and determinism.

## 7. Known limitations
See the *Limitations* section of the README.
