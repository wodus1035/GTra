"""Run CellRank and emit a cell-state transition graph in the common format.

For physical-time series we use CellRank's designed approach: optimal-transport
couplings between consecutive timepoints (moscot TemporalProblem) wrapped in a
RealTimeKernel, giving a time-directed cell-cell transition matrix. We then
aggregate to a STATE x STATE matrix by the predefined annotation, row-normalize,
and keep edges with P >= THRESHOLD (same recipe as the WOT runner).

Run in py310:  python run_cellrank_edges.py MND   (or HSPC)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import cellrank as cr

warnings.filterwarnings("ignore")
EDGE_DIR = Path(__file__).resolve().parent.parent / "edges"
THRESHOLD = 0.20
MND_COLLAPSE = {"1-Neurons": "Neurons", "2-Young neurons": "Young neurons",
                "6-Young neurons": "Young neurons", "3-APs/RPs": "APs/RPs",
                "5-APs/RPs": "APs/RPs", "4-IPs": "IPs", "7-IPs": "IPs"}
CFG = {
    "MND": dict(h5ad="/data3/projects/2025_GTRA/data/1_MND/CCTSD_preproc_hvg.h5ad",
                time_col="timepoints", annot="cell_type2", hvg=True,
                counts_layer="counts", day_map=None),
    "HSPC": dict(h5ad="/data3/projects/2025_GTRA/data/2_HSPC/HSPC_proc.h5ad",
                 time_col="time", annot="celltype", hvg=False,
                 counts_layer=None, day_map={"control": 0., "3h": 3., "24h": 24., "72h": 72.}),
}


def main(dataset):
    cfg = CFG[dataset]
    ad = sc.read_h5ad(cfg["h5ad"])
    if cfg["hvg"] and "highly_variable" in ad.var:
        ad = ad[:, ad.var.highly_variable].copy()
    if cfg["counts_layer"]:
        ad.X = ad.layers[cfg["counts_layer"]].copy()
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
    sc.pp.pca(ad, n_comps=30); sc.pp.neighbors(ad, n_neighbors=15)
    ad.obs["day"] = (ad.obs[cfg["time_col"]].map(cfg["day_map"]).astype(float)
                     if cfg["day_map"] else ad.obs[cfg["time_col"]].astype(float))

    # CellRank's designed time-series approach: OT couplings (moscot) -> RealTimeKernel
    from moscot.problems.time import TemporalProblem
    ad.obs["day"] = ad.obs["day"].astype("category")
    tp = TemporalProblem(ad)
    tp = tp.prepare(time_key="day")
    tp = tp.solve(epsilon=1e-2, scale_cost="mean")
    rtk = cr.kernels.RealTimeKernel.from_moscot(tp)
    rtk.compute_transition_matrix(self_transitions="all", conn_weight=0.2,
                                  threshold="auto")
    ad.obs["day"] = ad.obs["day"].astype(float)
    T = rtk.transition_matrix
    T = T.toarray() if sp.issparse(T) else np.asarray(T)

    states = ad.obs[cfg["annot"]].astype(str).values
    days = ad.obs["day"].values
    uniq_days = sorted(np.unique(days))
    edge_prob = {}
    for d0, d1 in zip(uniq_days[:-1], uniq_days[1:]):
        si = np.where(days == d0)[0]
        ti = np.where(days == d1)[0]
        for s in np.unique(states[si]):
            rows = si[states[si] == s]
            mass = {tt: T[np.ix_(rows, ti[states[ti] == tt])].sum()
                    for tt in np.unique(states[ti])}
            tot = sum(mass.values()) + 1e-12
            for tt, v in mass.items():
                p = v / tot
                if p >= THRESHOLD:
                    edge_prob[(s, tt)] = max(edge_prob.get((s, tt), 0.0), p)

    rows = [(s, t, p) for (s, t), p in edge_prob.items()]  # cell_type2, no collapse
    EDGE_DIR.mkdir(exist_ok=True)
    out = EDGE_DIR / f"CellRank_{dataset}.csv"
    pd.DataFrame(rows, columns=["source", "target", "score"]).to_csv(out, index=False)
    print(f"saved {out}  ({len(rows)} edges)")


if __name__ == "__main__":
    main(sys.argv[1])
