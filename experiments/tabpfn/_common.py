import numpy as np
from probly.method.ensemble import ensemble

def _groups(strata):
    return [np.flatnonzero(strata == s) for s in np.unique(strata)]

def bootstrap_indices(n, rng, strata=None):
    if strata is None:
        return rng.integers(0, n, size=n)
    return np.concatenate(
        [rng.choice(idx, size=len(idx), replace=True) for idx in _groups(strata)]
    )

def subsample_indices(n, rng, frac=0.8, strata=None):
    if strata is None:
        return rng.choice(n, size=max(1, round(frac * n)), replace=False)
    return np.concatenate(
        [rng.choice(idx, size=max(1, round(frac * len(idx))), replace=False)
         for idx in _groups(strata)]
    )

def fit_resampled_ensemble(estimator, x_train, y_train, k=5, resample="none",seed=0, frac=0.8, strata=None):
    x_train, y_train = np.asarray(x_train), np.asarray(y_train)
    rng = np.random.default_rng(seed)
    n = len(y_train)

    ens = ensemble(estimator, num_members=k)
    for i, m in enumerate(ens):
        m.set_params(random_state=seed + i)
        if resample == "bootstrap":
            idx = bootstrap_indices(n, rng, strata)
        elif resample == "subsample":
            idx = subsample_indices(n, rng, frac, strata)
        else:
            idx = np.arange(n)
        m.fit(x_train[idx], y_train[idx])
    return ens
