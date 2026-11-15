"""Tests for the TabICL adapter with a fake model."""

from __future__ import annotations

import numpy as np

from probly.method.tabicl import tabicl_mean_var


class FakeModel:
    def predict(self, x, /, output_type="mean", alphas=None):  # noqa: ARG002
        if output_type == "quantiles":
            return np.array([[-1.0, 0.0, 1.0]] * len(x))
        return np.zeros(len(x))


def test_tabicl_mean_var_shapes_and_values() -> None:
    x = np.zeros((4, 1))
    mean, var = tabicl_mean_var(FakeModel(), x)
    assert mean.shape == (4,)
    assert var.shape == (4,)
    np.testing.assert_allclose(var, (2 / 2.563) ** 2)
    assert np.all(var > 0)
