import numpy as np
from tabpfn import TabPFNClassifier
from probly.representation.sample import create_sample
from probly.predictor import predict
from probly.quantification.decomposition.entropy import SecondOrderEntropyDecomposition
from _common import fit_resampled_ensemble

def classifier_decomposition(x_train, y_train, x_test, k=5, resample="none", n_estimators=1, device="cpu", seed=0, frac=0.8):
    x_train, y_train = np.asarray(x_train), np.asarray(y_train)

    ens = fit_resampled_ensemble(
        TabPFNClassifier(n_estimators=n_estimators, device=device),
        x_train, y_train, k, resample, seed, frac, strata=y_train,
        )

    pred = predict(ens, x_test)
    sample = create_sample(pred, sample_axis=0)
    return SecondOrderEntropyDecomposition(sample)

