from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import pairwise_distances, roc_auc_score
from uncertainty_tabpfn import ensemble_decomposition

DEVICE = "cuda"
MODES = ["none", "bootstrap", "subsample"]
SEEDS = range(8)
K = 20
NOISE = 0.3
N_TRAIN = 100
N_ID = 300
N_OOD = 300
SCORES = ["epistemisch", "aleatorisch", "total"]
CENTERS = {
    "weit_ro": [4.0, 3.0],
    "weit_lu": [-3.0, -2.0],
    "weit_um": [0.5, -3.0],
    "weit_lo": [-3.0, 3.0],
}
OOD_SETS = [*CENTERS, "mittel"]

# Referenz fuer mittel-OOD: fest, unabhaengig vom Seed.
x_ref, _ = make_moons(n_samples=1000, noise=NOISE, random_state=999)


def decompose(x_train, y_train, x_test, mode, seed):
    dec = ensemble_decomposition(
        x_train, y_train, x_test, k=K, resample=mode, seed=seed, device=DEVICE,
    )
    alea = np.asarray(dec.aleatoric)
    epi = np.asarray(dec.epistemic)
    return {"epistemisch": epi, "aleatorisch": alea, "total": alea + epi}


def make_ood(kind, rng):
    if kind in CENTERS:
        return rng.normal(loc=CENTERS[kind], scale=0.5, size=(N_OOD, 2))
    # mittel: Gleichverteilung in der Box, nur Punkte weit weg von den Daten
    cand = rng.uniform([-2.5, -1.8], [3.5, 2.3], size=(20 * N_OOD, 2))
    far = pairwise_distances(cand, x_ref).min(axis=1) > 1.0
    return cand[far][:N_OOD]


# auroc[mode][ood][score] -> Liste ueber Seeds
auroc = {m: {o: {s: [] for s in SCORES} for o in OOD_SETS} for m in MODES}

for mode in MODES:
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x_train, y_train = make_moons(n_samples=N_TRAIN, noise=NOISE, random_state=seed)
        x_id, _ = make_moons(n_samples=N_ID, noise=NOISE, random_state=1000 + seed)
        x_oods = {o: make_ood(o, rng) for o in OOD_SETS}

        # Ein Durchlauf pro Seed fuer alle Testpunkte (ID + beide OOD-Mengen).
        x_all = np.vstack([x_id, *x_oods.values()])
        sc = decompose(x_train, y_train, x_all, mode, seed)

        n_id = len(x_id)
        start = n_id
        for o in OOD_SETS:
            n_o = len(x_oods[o])
            labels = np.r_[np.zeros(n_id), np.ones(n_o)]
            for s in SCORES:
                scores = np.r_[sc[s][:n_id], sc[s][start:start + n_o]]
                auroc[mode][o][s].append(roc_auc_score(labels, scores))
            start += n_o

# Tabelle
print(f"{'Modus':10s} {'OOD':7s} " + " ".join(f"{s:>16s}" for s in SCORES))
for mode in MODES:
    for o in OOD_SETS:
        cells = []
        for s in SCORES:
            a = np.array(auroc[mode][o][s])
            cells.append(f"{a.mean():.3f} +- {a.std(ddof=1):.3f}")
        print(f"{mode:10s} {o:7s} " + " ".join(f"{c:>16s}" for c in cells))

# Plot
fig, axes = plt.subplots(1, len(OOD_SETS), figsize=(4 * len(OOD_SETS), 4.5), sharey=True)
width = 0.25
for ax, o in zip(axes, OOD_SETS):
    for j, s in enumerate(SCORES):
        means = [np.mean(auroc[m][o][s]) for m in MODES]
        errs = [np.std(auroc[m][o][s], ddof=1) / np.sqrt(len(SEEDS)) for m in MODES]
        ax.bar(np.arange(len(MODES)) + (j - 1) * width, means, width,
               yerr=errs, capsize=3, label=s)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(MODES)
    ax.set_title(f"OOD: {o}")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, axis="y")
axes[0].set_ylabel("AUROC (ID vs. OOD)")
axes[0].legend()
fig.tight_layout()
plt.show()
