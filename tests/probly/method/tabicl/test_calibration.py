"""Calibration check over several random splits of sklearn's diabetes data."""

from __future__ import annotations

import numpy as np
import pytest

from probly.method.tabicl import TabICLGaussianPredictor
from probly.predictor import predict

tabicl = pytest.importorskip("tabicl")
from sklearn.datasets import load_diabetes  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402


def test_interval_coverage_over_splits() -> None:
    features, y = load_diabetes(return_X_y=True)
    coverages = []

    for seed in range(5):
        x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=seed)
        model = tabicl.TabICLRegressor(n_estimators=1, random_state=seed)
        model.fit(x_train, y_train)
        dist = predict(TabICLGaussianPredictor(model), x_test)

        low = dist.quantile(0.1)
        high = dist.quantile(0.9)
        coverages.append(np.mean((y_test >= low) & (y_test <= high)))

    assert 0.7 <= np.mean(coverages) <= 0.9
