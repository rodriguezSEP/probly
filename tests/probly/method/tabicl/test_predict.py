"""Test that probly's predict returns a Gaussian for the TabICL wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from probly.method.tabicl import TabICLGaussianPredictor
from probly.predictor import predict

tabicl = pytest.importorskip("tabicl")


def test_predict_returns_gaussian() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, size=600)
    features = x.reshape(-1, 1)
    y = rng.normal(loc=np.sin(x), scale=0.07 * np.exp(-((x - 0.5) ** 2) / 0.9))

    model = tabicl.TabICLRegressor(n_estimators=1)
    model.fit(features[:300], y[:300])

    dist = predict(TabICLGaussianPredictor(model), features[300:])

    assert dist.mean.shape == (300,)
    assert dist.std.shape == (300,)
    assert dist.quantile([0.1, 0.9]).shape == (300, 2)
