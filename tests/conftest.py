"""Shared fixtures.

The tests must run without internet access or a Kaggle download, so they use
scikit-learn's bundled copy of the same WDBC dataset, reshaped to look exactly
like the Kaggle CSV (id + diagnosis M/B + 30 features + empty 'Unnamed: 32').
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer

_SUFFIX = {"mean": "_mean", "error": "_se", "worst": "_worst"}


def make_kaggle_style_frame() -> pd.DataFrame:
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()

    def to_kaggle_name(col: str) -> str:
        if col.startswith("mean "):
            base, kind = col[len("mean "):], "mean"
        elif col.startswith("worst "):
            base, kind = col[len("worst "):], "worst"
        else:
            base, kind = col[: -len(" error")], "error"
        return base.replace("fractal dimension", "fractal_dimension") + _SUFFIX[kind]

    X.columns = [to_kaggle_name(c) for c in X.columns]
    df = X.copy()
    df.insert(0, "diagnosis", np.where(bunch.target == 0, "M", "B"))  # sklearn: 0 = malignant
    df.insert(0, "id", np.arange(1, len(df) + 1))
    df["Unnamed: 32"] = np.nan
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_kaggle_style_frame()
