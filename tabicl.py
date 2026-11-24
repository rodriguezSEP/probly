git"""TabICL-specific predictor dispatch.

TabICL's regressor describes its predictive distribution through quantiles. probly represents
regression outputs as Gaussian distributions, so the 10 % and 90 % quantiles are converted into a
variance. Classification needs no extra code: ``TabICLClassifier`` exposes ``predict_proba`` and is
picked up as a categorical distribution predictor automatically.
"""

from __future__ import annotations

import numpy as np
from tabicl import TabICLRegressor

from ._common import GaussianDistributionPredictor, predict_raw

# Distance between the 10 % and the 90 % quantile of a Gaussian, in standard deviations.
_GAUSSIAN_80_INTERVAL_WIDTH = 2.0 * 1.2815515655446004

GaussianDistributionPredictor.register(TabICLRegressor)


@predict_raw.register(TabICLRegressor)
def tabicl_regressor_predict_raw(predictor: TabICLRegressor, x: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and variance of a fitted TabICL regressor.

    The variance is derived from the 10 % and 90 % quantiles under a Gaussian assumption and is
    clipped to a tiny positive value, because probly requires strictly positive variances.
    """
    mean = np.asarray(predictor.predict(x, output_type="mean"), dtype=float)
    quantiles = np.asarray(predictor.predict(x, output_type="quantiles", alphas=[0.1, 0.9]), dtype=float)
    std = (quantiles[..., 1] - quantiles[..., 0]) / _GAUSSIAN_80_INTERVAL_WIDTH
    var = np.maximum(np.square(std), np.finfo(float).tiny)
    return mean, var
