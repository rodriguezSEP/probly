"""Tests for the TabICL predictor dispatch."""

from __future__ import annotations

import numpy as np
import pytest

tabicl = pytest.importorskip("tabicl")

from probly.predictor import predict  # noqa: E402
from probly.representation.distribution import GaussianDistribution  # noqa: E402


def _fake_predict(x, /, output_type="mean", alphas=None):  # noqa: ARG001
    """Stand-in for TabICLRegressor.predict with fixed quantiles, so no model weights are needed."""
    n = len(x)
    if output_type == "quantiles":
        return np.tile([-1.0, 1.0], (n, 1))
    return np.zeros(n)


def test_regressor_returns_gaussian_without_wrapper() -> None:
    model = tabicl.TabICLRegressor()
    model.predict = _fake_predict
    x = np.zeros((4, 1))

    dist = predict(model, x)

    assert isinstance(dist, GaussianDistribution)
    assert dist.mean.shape == (4,)
    np.testing.assert_allclose(dist.var, (2 / (2 * 1.2815515655446004)) ** 2)


def test_zero_width_quantiles_give_positive_variance() -> None:
    model = tabicl.TabICLRegressor()

    def flat_predict(x, /, output_type="mean", alphas=None):  # noqa: ARG001
        if output_type == "quantiles":
            return np.zeros((len(x), 2))
        return np.zeros(len(x))

    model.predict = flat_predict
    dist = predict(model, np.zeros((3, 1)))

    assert np.all(dist.var > 0)


def _make_data(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    true_std = 0.07 * np.exp(-((x - 0.5) ** 2) / 0.9)
    y = rng.normal(loc=np.sin(x), scale=true_std)
    return x.reshape(-1, 1), y, true_std


@pytest.mark.parametrize("n_train", [300, 3000])
def test_real_regressor_tracks_true_noise(n_train: int) -> None:
    features, y, true_std = _make_data(n_train + 300)
    model = tabicl.TabICLRegressor(n_estimators=1)
    model.fit(features[:n_train], y[:n_train])

    dist = predict(model, features[n_train:])
    y_test = y[n_train:]

    corr = np.corrcoef(dist.std, true_std[n_train:])[0, 1]
    coverage = np.mean((y_test >= dist.quantile(0.1)) & (y_test <= dist.quantile(0.9)))
    assert corr > 0.5
    assert 0.6 <= coverage <= 0.95


def test_real_classifier_returns_probabilities() -> None:
    from sklearn.datasets import load_breast_cancer  # noqa: PLC0415
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    features, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=0)
    model = tabicl.TabICLClassifier(n_estimators=1)
    model.fit(x_train, y_train)

    probs = predict(model, x_test).probabilities

    assert probs.shape == (len(x_test), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.mean(np.argmax(probs, axis=1) == y_test) > 0.8
