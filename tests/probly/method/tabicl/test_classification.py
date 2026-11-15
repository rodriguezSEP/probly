"""Classification: probly should read TabICL's predict_proba directly."""

from __future__ import annotations

import numpy as np
import pytest

from probly.predictor import predict

tabicl = pytest.importorskip("tabicl")
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


def test_classifier_predict_gives_categorical() -> None:
    features, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=0)

    model = tabicl.TabICLClassifier(n_estimators=1)
    model.fit(x_train, y_train)
    dist = predict(model, x_test)

    probs = dist.probabilities
    assert probs.shape == (len(x_test), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    acc = np.mean(np.argmax(probs, axis=1) == y_test)
    assert acc > 0.8
