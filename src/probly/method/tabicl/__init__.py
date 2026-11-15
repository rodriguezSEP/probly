"""TabICL integration: mean and variance from TabICL predictions."""

from __future__ import annotations

from probly.method.tabicl._common import TabICLGaussianPredictor, tabicl_mean_var

__all__ = ["TabICLGaussianPredictor", "tabicl_mean_var"]
