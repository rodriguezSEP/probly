"""Test the TabICL adapter with a real TabICL model on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from probly.method.tabicl import tabicl_mean_var

tabicl = pytest.importorskip("tabicl")


def _make_data(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    true_std = 0.07 * np.exp(-((x - 0.5) ** 2) / 0.9)
    y = rng.normal(loc=np.sin(x), scale=true_std)
    return x.reshape(-1, 1), y, true_std


def test_tabicl_adapter_tracks_true_spread() -> None:
    features, y, true_std = _make_data()
    x_train, x_test = features[:300], features[300:]
    y_train = y[:300]
    true_test_std = true_std[300:]

    model = tabicl.TabICLRegressor(n_estimators=1)
    model.fit(x_train, y_train)
    mean, var = tabicl_mean_var(model, x_test)

    assert mean.shape == (300,)
    assert var.shape == (300,)
    assert np.all(var > 0)

    corr = np.corrcoef(np.sqrt(var), true_test_std)[0, 1]
    assert corr > 0.5
