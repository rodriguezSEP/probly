import numpy as np
from tabpfn import TabPFNRegressor
from sklearn.model_selection import train_test_split
from probly.predictor import predict
from probly.quantification.decomposition.variance import SecondOrderVarianceDecomposition
from probly.representation.distribution import (
    NumpyGaussianDistribution,
    NumpyGaussianDistributionSample,
)
from experiments.tabpfn._common import fit_resampled_ensemble

def to_np(t):
    return t.cpu().numpy() if hasattr(t, "cuda") else np.asarray(t)

def member_gaussian(m, X):
    out = m.predict(X, output_type="full")
    crit, logits = out["criterion"], out["logits"]
    var = to_np(crit.variance(logits))
    return np.asarray(out["mean"]), var

def regressor_decomposition(x_train, y_train, x_test, k=5, resample="none", n_estimators=1, device="cpu", seed=0, frac=0.8):
    x_train, y_train = np.asarray(x_train), np.asarray(y_train)

    ens = fit_resampled_ensemble(
        TabPFNRegressor(n_estimators=n_estimators, device=device),
        x_train, y_train, k, resample, seed, frac
    )

    parts = [member_gaussian(m, x_test) for m in ens]
    means = np.stack([p[0] for p in parts])
    vars_ = np.stack([p[1] for p in parts])

    dist = NumpyGaussianDistribution(mean=means, var=vars_)
    sample = NumpyGaussianDistributionSample(dist, sample_axis=0)
    return SecondOrderVarianceDecomposition(sample)

