import numpy as np

from src.data import clean, split_data
from src.models import get_models


def _split(raw_df):
    return split_data(clean(raw_df.copy()))


def test_scaler_is_fitted_on_training_data_only(raw_df):
    """Leakage guard: scaling statistics must come from the training set."""
    X_train, X_test, y_train, _ = _split(raw_df)
    pipe = get_models()["Logistic Regression"].fit(X_train, y_train)
    np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, X_train.mean().values)
    assert not np.allclose(pipe.named_steps["scaler"].mean_, X_test.mean().values)


def test_predictions_and_probabilities_are_valid(raw_df):
    X_train, X_test, y_train, y_test = _split(raw_df)
    model = get_models()["Logistic Regression"].fit(X_train, y_train)
    assert len(model.predict(X_test)) == len(y_test)
    prob = model.predict_proba(X_test)[:, 1]
    assert np.all((prob >= 0) & (prob <= 1))


def test_training_is_reproducible(raw_df):
    X_train, X_test, y_train, _ = _split(raw_df)
    p1 = get_models()["Random Forest"].fit(X_train, y_train).predict_proba(X_test)
    p2 = get_models()["Random Forest"].fit(X_train, y_train).predict_proba(X_test)
    np.testing.assert_array_equal(p1, p2)
