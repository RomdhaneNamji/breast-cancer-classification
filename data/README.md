# Data

The dataset is **not stored in this repository**. It is small, but it belongs to
its original authors, so the scripts download it for you (see below).

## Dataset

| | |
|---|---|
| **Name** | Breast Cancer Wisconsin (Diagnostic) — "WDBC" |
| **Original source** | UCI Machine Learning Repository, <https://doi.org/10.24432/C5DW2B> |
| **Copy used in this project** | Kaggle mirror: <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data> |
| **Samples** | 569 (357 benign, 212 malignant) |
| **Features** | 30 numeric features, computed from digitised images of fine-needle aspirates (FNA) of breast masses |
| **Target** | `diagnosis`: `M` = malignant → `1`, `B` = benign → `0` |
| **Missing values** | none (after dropping an empty trailing column, see below) |
| **Licence (UCI original)** | CC BY 4.0 — attribution required |

> The licence of the *Kaggle mirror* may differ from the UCI original. Check the
> Kaggle dataset page before redistributing the CSV. This repository avoids the
> question by not committing the file.

## Features

Ten nuclear characteristics (radius, texture, perimeter, area, smoothness,
compactness, concavity, concave points, symmetry, fractal dimension), each
reported three ways: `_mean`, `_se` (standard error) and `_worst` (mean of the
three largest values) → 10 × 3 = 30 columns.

## How to get the data

**Option A — automatic (recommended)**

```bash
python -m src.download_data
```

This uses `kagglehub` to download the CSV and copies it to `data/raw/data.csv`.

**Option B — manual**

1. Download `data.csv` from the Kaggle page above.
2. Place it at `data/raw/data.csv`.

## Expected file

```
data/raw/data.csv      # columns: id, diagnosis, 30 feature columns, "Unnamed: 32" (empty)
```

`src/data.py` drops `id` and the empty `Unnamed: 32` column, encodes the target
and validates the result (569 rows × 31 columns, no missing values, binary target).

## Citation

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer
Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5DW2B
