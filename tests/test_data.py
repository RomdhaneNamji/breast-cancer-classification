import pytest

from src import config
from src.data import clean, split_data, validate


def test_clean_drops_columns_and_encodes_target(raw_df):
    df = clean(raw_df.copy())
    assert df.shape == (569, 31)
    assert not set(config.DROP_COLUMNS) & set(df.columns)
    assert set(df[config.TARGET].unique()) == {0, 1}
    assert df[config.TARGET].sum() == 212          # 212 malignant, 357 benign


def test_validate_accepts_clean_data(raw_df):
    validate(clean(raw_df.copy()))


def test_validate_rejects_missing_values(raw_df):
    df = clean(raw_df.copy())
    df.iloc[0, 1] = None
    with pytest.raises(ValueError):
        validate(df)


def test_validate_rejects_unmapped_target(raw_df):
    df = raw_df.copy()
    df.loc[0, "diagnosis"] = "X"                    # unknown label -> NaN after mapping
    with pytest.raises(ValueError):
        validate(clean(df))


def test_split_is_stratified_and_disjoint(raw_df):
    df = clean(raw_df.copy())
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) == 455 and len(X_test) == 114
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.02   # class ratio preserved
    assert list(X_train.columns) == list(X_test.columns)
