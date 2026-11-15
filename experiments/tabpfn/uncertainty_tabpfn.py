import numpy as np
from tabpfn import TabPFNClassifier
from probly.method.ensemble import ensemble
from probly.representation.sample import create_sample
from probly.predictor import predict
from probly.quantification.decomposition.entropy import SecondOrderEntropyDecomposition

def _stratified_bootstrap(y, rng):
    parts = [
        rng.choice(np.flatnonzero(y == n), size=int((y == n).sum()), replace=True)
        for n in np.unique(y)
    ]

    return np.concatenate(parts)

def _stratified_subsample(y, rng, frac=0.8):
    parts = [
        rng.choice(np.flatnonzero(y == n), size=max(1, round(frac * (y == n).sum())), replace=False)
        for n in np.unique(y)
    ]

    return np.concatenate(parts)

def ensemble_decomposition(x_train, y_train, x_test, k=5, resample="none", n_estimators=1, device="cpu", seed=0):
    x_train, y_train = np.asarray(x_train), np.asarray(y_train)
    rng = np.random.default_rng(seed)

    ens = ensemble(TabPFNClassifier(n_estimators=n_estimators, device=device), num_members=k)
    for i, en in enumerate(ens):
        en.set_params(random_state=seed + i)
        if resample == "bootstrap":
            bs = _stratified_bootstrap(y_train, rng)
            en.fit(x_train[bs], y_train[bs])
        elif resample == "subsample":
            ss = _stratified_subsample(y_train, rng)
            en.fit(x_train[ss], y_train[ss])
        else:
            en.fit(x_train, y_train)

    pred = predict(ens, x_test)

    sample = create_sample(pred, sample_axis=0)
    dec = SecondOrderEntropyDecomposition(sample)

    return dec
