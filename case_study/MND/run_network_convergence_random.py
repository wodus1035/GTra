"""Order-independent network bootstrap-convergence (random-subset resampling).

Reuses edge_iterations.pkl (100 per-iteration edge sets; no re-bootstrap). For each
N in 10..90, draws B random size-N subsets of the 100 iterations, builds the
>=90%-reproducible network of each subset, and scores coverage / Jaccard /
#connections vs the full N=100 network -> mean +/- sd bands (removes the
iteration-order dependence of the nested 'first-N' version).
"""
import pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "NETWORK_CONVERGENCE_figs"
THR, B, SEED = 0.90, 200, 0


def network(sets, idx):
    cnt = {}
    for i in idx:
        for e in sets[i]:
            cnt[e] = cnt.get(e, 0) + 1
    n = len(idx)
    return {e for e, c in cnt.items() if c >= THR * n}


def main():
    sets = pickle.load(open(OUT / "edge_iterations.pkl", "rb"))
    M = len(sets)
    truth = network(sets, range(M))
    rng = np.random.default_rng(SEED)
    Ns = list(range(10, M, 10))
    rows = []
    for N in Ns:
        cov, jac, nc = [], [], []
        for _ in range(B):
            idx = rng.choice(M, size=N, replace=False)
            net = network(sets, idx)
            inter = len(net & truth)
            cov.append(inter / len(truth) if truth else np.nan)
            jac.append(inter / len(net | truth) if (net | truth) else np.nan)
            nc.append(len(net))
        rows.append({"N": N,
                     "n_conn_mean": np.mean(nc), "n_conn_sd": np.std(nc),
                     "coverage_mean": np.mean(cov), "coverage_sd": np.std(cov),
                     "jaccard_mean": np.mean(jac), "jaccard_sd": np.std(jac)})
    rows.append({"N": M, "n_conn_mean": len(truth), "n_conn_sd": 0.0,
                 "coverage_mean": 1.0, "coverage_sd": 0.0, "jaccard_mean": 1.0, "jaccard_sd": 0.0})
    df = pd.DataFrame(rows); df.to_csv(OUT / "network_convergence_random.csv", index=False)
    print(f"N=100 truth network = {len(truth)} connections; B={B} random subsets per N")
    print(df.round(3).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(df["N"], df["n_conn_mean"], yerr=df["n_conn_sd"], fmt="-o", capsize=3)
    ax[0].axhline(len(truth), ls="--", c="grey", label="N=100")
    ax[0].set_xlabel("bootstrap iterations N"); ax[0].set_ylabel("# connections (>=90% reproducible)")
    ax[0].set_title("Network size vs N (mean±sd, 200 random subsets)"); ax[0].legend()
    for col, mk, lab in [("coverage", "-o", "coverage of N=100"), ("jaccard", "-s", "Jaccard vs N=100")]:
        ax[1].errorbar(df["N"], df[f"{col}_mean"], yerr=df[f"{col}_sd"], fmt=mk, capsize=3, label=lab)
    ax[1].set_xlabel("bootstrap iterations N"); ax[1].set_ylabel("fraction"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title("Convergence to N=100 network (mean±sd)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "NC2_network_convergence_random.pdf", dpi=200)
    print("saved NETWORK_CONVERGENCE_figs/NC2_network_convergence_random.pdf\nDONE")


if __name__ == "__main__":
    main()
