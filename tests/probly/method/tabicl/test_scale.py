"""Does the adapter still work when the training set gets larger?"""

from __future__ import annotations

import numpy as np
import pytest

from probly.method.tabicl import TabICLGaussianPredictor
from probly.predictor import predict

tabicl = pytest.importorskip("tabicl")


def _make_data(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    true_std = 0.07 * np.exp(-((x - 0.5) ** 2) / 0.9)
    y = rng.normal(loc=np.sin(x), scale=true_std)
    return x.reshape(-1, 1), y, true_std


@pytest.mark.parametrize("n_train", [500, 5000])
def test_adapter_scales(n_train: int) -> None:
    features, y, true_std = _make_data(n_train + 300)
    x_train, y_train = features[:n_train], y[:n_train]
    x_test, y_test = features[n_train:], y[n_train:]
    true_test_std = true_std[n_train:]

    model = tabicl.TabICLRegressor(n_estimators=1)
    model.fit(x_train, y_train)
    dist = predict(TabICLGaussianPredictor(model), x_test)

    corr = np.corrcoef(dist.std, true_test_std)[0, 1]
    coverage = np.mean((y_test >= dist.quantile(0.1)) & (y_test <= dist.quantile(0.9)))

    assert np.all(dist.var > 0)
    assert corr > 0.5
    assert 0.6 <= coverage <= 0.95
