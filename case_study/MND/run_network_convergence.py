"""Network (edge) bootstrap-convergence analysis (MND, cell_type2).

Treats N=100 bootstrap as ground truth. Runs 100 bootstrap iterations of GTra
Step1+2, recording the cell-state transition edges recovered in EACH iteration.
Then, for N = 10,20,...,100, the "network at N" = edges reproduced in >= static_th%
of the first N iterations. We report how the number of connections and the
coverage / Jaccard vs the N=100 network converge with N.

Saves per-iteration edge records (edge_iterations.pkl) so curves can be recomputed
without re-running the (slow) bootstrap.
"""
import sys, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gtra
from gtra.cluster_func import _extract_dat, Run_step1_and_2
import ged_utils as ge

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "NETWORK_CONVERGENCE_figs"; OUT.mkdir(exist_ok=True)
NBOOT, THR = 100, 0.90


def build():
    ad = sc.read_h5ad("/data3/projects/2025_GTRA/data/1_MND/CCTSD_preproc_hvg.h5ad")
    if "highly_variable" in ad.var:
        ad = ad[:, ad.var.highly_variable].copy()
    obj = gtra.GTraObject(); obj.params.cell_type_label = "cell_type2"
    for t in sorted(ad.obs["timepoints"].unique()):
        dat = ad[ad.obs["timepoints"] == t]
        obj.upload_time_scRNA(dat.to_df(layer="counts"), dat.obs[["cell_type2"]])
    obj.select_genes()
    return obj


def edge_set(est_edges):
    """est_edges: {interval: {edge_name: 1}} -> set of (interval, edge_name)."""
    return {(it, e) for it, sub in est_edges.items() for e in sub}


def main():
    ge.patch_gtra()
    obj = build()
    dat = _extract_dat(obj)
    print(f"running {NBOOT} bootstrap iterations (Step1+2)...", flush=True)
    results = Parallel(n_jobs=16, backend="loky")(
        delayed(Run_step1_and_2)(dat) for _ in range(NBOOT))
    # res[i] = ((est_edges, score_edges), gcinfo)
    per_iter = [edge_set(r[0][0]) for r in results]
    pickle.dump(per_iter, open(OUT / "edge_iterations.pkl", "wb"))

    # cumulative edge occurrence; network(N) = edges in >= THR*N of first N iters
    def network_at(N):
        cnt = {}
        for s in per_iter[:N]:
            for e in s:
                cnt[e] = cnt.get(e, 0) + 1
        return {e for e, c in cnt.items() if c >= THR * N}

    truth = network_at(NBOOT)
    Ns = list(range(10, NBOOT + 1, 10))
    rows = []
    for N in Ns:
        net = network_at(N)
        inter = len(net & truth)
        rows.append({"N": N, "n_connections": len(net),
                     "coverage_of_100": inter / len(truth) if truth else np.nan,
                     "jaccard_vs_100": inter / len(net | truth) if (net | truth) else np.nan})
    df = pd.DataFrame(rows); df.to_csv(OUT / "network_convergence.csv", index=False)
    print(df.round(3).to_string(index=False), flush=True)
    print(f"\nN=100 network size = {len(truth)} connections", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df["N"], df["n_connections"], "-o"); ax[0].axhline(len(truth), ls="--", c="grey", label="N=100")
    ax[0].set_xlabel("bootstrap iterations N"); ax[0].set_ylabel("# connections (>=90% reproducible)")
    ax[0].set_title("Network size vs bootstrap N"); ax[0].legend()
    ax[1].plot(df["N"], df["coverage_of_100"], "-o", label="coverage of N=100")
    ax[1].plot(df["N"], df["jaccard_vs_100"], "-s", label="Jaccard vs N=100")
    ax[1].set_xlabel("bootstrap iterations N"); ax[1].set_ylabel("fraction"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title("Convergence to N=100 network"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "NC1_network_convergence.pdf", dpi=200)
    print("saved NETWORK_CONVERGENCE_figs/NC1_network_convergence.pdf\nDONE")


if __name__ == "__main__":
    main()
