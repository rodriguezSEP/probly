"""TabICL adapter: turns TabICL outputs into a Gaussian distribution in probly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from probly.predictor._common import GaussianDistributionPredictor
import probly.representation.distribution.numpy_gaussian  # noqa: F401  registers the NumPy version

if TYPE_CHECKING:
    import numpy as np


class QuantileRegressor(Protocol):
    """A fitted model with a TabICL-style ``predict`` method."""

    def predict(self, x: np.ndarray, /, output_type: str = "mean", alphas: list[float] | None = None) -> np.ndarray:
        """Predict means, variances or quantiles."""


def tabicl_mean_var(model: QuantileRegressor, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and variance from the 10 % and 90 % quantiles of TabICL."""
    mean = model.predict(x)
    q = model.predict(x, output_type="quantiles", alphas=[0.1, 0.5, 0.9])
    var = ((q[:, 2] - q[:, 0]) / 2.563) ** 2
    return mean, var


class TabICLGaussianPredictor:
    """Wrap a fitted TabICL regressor so that probly treats it as a Gaussian predictor."""

    def __init__(self, model: QuantileRegressor) -> None:
        """Store the fitted TabICL regressor."""
        self.model = model

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return mean and variance for the inputs."""
        return tabicl_mean_var(self.model, x)


GaussianDistributionPredictor.register(TabICLGaussianPredictor)
