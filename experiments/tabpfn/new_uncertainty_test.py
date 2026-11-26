import matplotlib as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import pairwise_distances
from tabpfn import TabPFNClassifier
from experiments.tabpfn.classifier_decomposition import classifier_decomposition

SEED = 0
MODE = ["none", "bootstrap", "subsample"]
NOISE = 0.3
K = 5

def decompose(estimators, x_train, y_train, x_test, mode, seed=SEED):
    dec = classifier_decomposition(
        x_train, y_train, x_test, n_estimators,   k=K, resample=mode, device="cuda", seed=seed,
    )
    total = np.asarray(dec.total)
    alea = np.asarray(dec.aleatoric)
    epi = np.asarray(dec.epistemic)
    assert np.allclose(total, alea + epi, atol=1e-5), "total != alea + epi"
    assert (epi >= -1e-6).all(), "neg epi!"
    return alea, epi
