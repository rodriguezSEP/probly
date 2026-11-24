from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import pairwise_distances

from uncertainty_tabpfn import ensemble_decomposition

SEEDS = range(8)
SEED = 0
MODES = ["none", "bootstrap", "subsample"]
K = 5
NOISE = 0.3
TRAIN_SIZES = [20, 50, 100, 300]


def make_grid(n: int = 40):
    xs = np.linspace(-2.5, 3.5, n)
    ys = np.linspace(-1.8, 2.3, n)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def decompose(x_train, y_train, x_test, mode, seed=SEED):
    dec = ensemble_decomposition(
        x_train, y_train, x_test, k=K, resample=mode, device="cuda", seed=seed,
    )
    total = np.asarray(dec.total)
    alea = np.asarray(dec.aleatoric)
    epi = np.asarray(dec.epistemic)
    assert np.allclose(total, alea + epi, atol=1e-5), "total != aleatorisch + epistemisch"
    assert (epi >= -1e-6).all(), "negative Epistemik"
    return alea, epi



# Heatmaps von aleatorischer/epistemischer Unsicherheit je Resampling-Modus.

x_train, y_train = make_moons(n_samples=100, noise=NOISE, random_state=SEED)
xx, yy, grid = make_grid()

fig, axes = plt.subplots(len(MODES), 2, figsize=(10, 4 * len(MODES)))
for row, mode in zip(axes, MODES):
    alea, epi = decompose(x_train, y_train, grid, mode)
    for ax, values, title in zip(row, (alea, epi), ("aleatorisch", "epistemisch")):
        im = ax.contourf(xx, yy, values.reshape(xx.shape), levels=20)
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="coolwarm",
                   edgecolor="k", s=18)
        ax.set_title(f"{title} ({mode})")
        fig.colorbar(im, ax=ax)
fig.tight_layout()
plt.show()


# Verhalten der Epistemik bei wachsender Trainingsmenge (nah vs. fern der Daten).

x_ref, _ = make_moons(n_samples=1000, noise=NOISE, random_state=999)
far = pairwise_distances(grid, x_ref).min(axis=1) > 1.0
near = ~far

res = {m: {"epi_near": [], "epi_far": [], "alea_near": []} for m in MODES}
for mode in MODES:
    for seed in SEEDS:
        x_big, y_big = make_moons(n_samples=max(TRAIN_SIZES), noise=NOISE, random_state=seed)
        epi_near, epi_far, alea_near = [], [], []
        for n in TRAIN_SIZES:
            xs, ys = x_big[:n], y_big[:n]
            alea, epi = decompose(xs, ys, grid, mode, seed=seed)
            epi_near.append(epi[near].mean())
            epi_far.append(epi[far].mean())
            alea_near.append(alea[near].mean())
        res[mode]["epi_near"].append(epi_near)
        res[mode]["epi_far"].append(epi_far)
        res[mode]["alea_near"].append(alea_near)

def errbar(ax, key, mode, **kw):
    arr = np.array(res[mode][key])
    se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return ax.errorbar(TRAIN_SIZES, arr.mean(axis=0), yerr=se, capsize=3, **kw)

fig, (ax_epi, ax_alea) = plt.subplots(1, 2, figsize=(12, 4.5))
for mode, color in zip(MODES, ["C0", "C1", "C2"]):
    errbar(ax_epi, "epi_near", mode, marker="o", color=color, label=f"{mode} - nah")
    errbar(ax_epi, "epi_far", mode, marker="s", ls="--", color=color, label=f"{mode} - fern")
    errbar(ax_alea, "alea_near", mode, marker="o", color=color, label=mode)

ax_epi.set(xlabel="Trainingsgroesse", ylabel="mittlere Epistemik", title="Epistemik")
ax_alea.set(xlabel="Trainingsgroesse", ylabel="mittlere Aleatorik (nah)", title="Aleatorik")
for ax in (ax_epi, ax_alea):
    ax.legend()
    ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()
