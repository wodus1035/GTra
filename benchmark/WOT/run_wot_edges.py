"""Run WaddingtonOT and emit a cell-state transition graph in the common format.

WOT gives cell x cell transport maps between adjacent timepoints. We aggregate
each map to a STATE x STATE transition matrix using the predefined annotation,
row-normalize to P(target_state | source_state), and keep edges with
P >= THRESHOLD (answer-blind operating point). Output: edges/WOT_{dataset}.csv.

Run in gtra_test:  python run_wot_edges.py MND  (or HSPC)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import wot

warnings.filterwarnings("ignore")

EDGE_DIR = Path(__file__).resolve().parent.parent / "edges"
THRESHOLD = 0.20  # keep target states capturing >=20% of a source state's mass

CFG = {
    "MND": dict(h5ad="/data3/projects/2025_GTRA/data/1_MND/CCTSD_preproc_hvg.h5ad",
                time_col="timepoints", annot="cell_type2", hvg=True,
                day_map=None, counts_layer="counts"),
    "HSPC": dict(h5ad="/data3/projects/2025_GTRA/data/2_HSPC/HSPC_proc.h5ad",
                 time_col="time", annot="celltype", hvg=False,
                 day_map={"control": 0.0, "3h": 3.0, "24h": 24.0, "72h": 72.0},
                 counts_layer=None),
}
MND_COLLAPSE = {"1-Neurons": "Neurons", "2-Young neurons": "Young neurons",
                "6-Young neurons": "Young neurons", "3-APs/RPs": "APs/RPs",
                "5-APs/RPs": "APs/RPs", "4-IPs": "IPs", "7-IPs": "IPs"}


def main(dataset):
    cfg = CFG[dataset]
    ad = sc.read_h5ad(cfg["h5ad"])
    if cfg["hvg"] and "highly_variable" in ad.var:
        ad = ad[:, ad.var.highly_variable].copy()
    # normalized log expression for OT (CP10k + log1p)
    if cfg["counts_layer"]:
        ad.X = ad.layers[cfg["counts_layer"]].copy()
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)

    if cfg["day_map"]:
        ad.obs["day"] = ad.obs[cfg["time_col"]].map(cfg["day_map"]).astype(float)
    else:
        ad.obs["day"] = ad.obs[cfg["time_col"]].astype(float)
    ad = ad[np.argsort(ad.obs["day"].values)].copy()
    states = ad.obs[cfg["annot"]].astype(str).values

    ot_model = wot.ot.OTModel(ad, day_field="day", lambda1=1.0, lambda2=50.0,
                              epsilon=0.05, growth_iter=3)
    days = sorted(pd.unique(ad.obs["day"]))
    edge_prob = {}
    for t0, t1 in zip(days[:-1], days[1:]):
        tm = ot_model.compute_transport_map(t0, t1)
        M = tm.X.toarray() if sp.issparse(tm.X) else np.asarray(tm.X)
        src = ad.obs.loc[tm.obs.index, cfg["annot"]].astype(str).values
        tgt = ad.obs.loc[tm.var.index, cfg["annot"]].astype(str).values
        s_states = sorted(set(src)); t_states = sorted(set(tgt))
        for s in s_states:
            si = np.where(src == s)[0]
            row = {tt: M[np.ix_(si, np.where(tgt == tt)[0])].sum() for tt in t_states}
            tot = sum(row.values()) + 1e-12
            for tt, v in row.items():
                p = v / tot
                if p >= THRESHOLD:
                    key = (s, tt)
                    edge_prob[key] = max(edge_prob.get(key, 0.0), p)
        print(f"  {t0}->{t1} done", flush=True)

    rows = [(s, t, p) for (s, t), p in edge_prob.items()]  # cell_type2, no collapse
    EDGE_DIR.mkdir(exist_ok=True)
    out = EDGE_DIR / f"WOT_{dataset}.csv"
    pd.DataFrame(rows, columns=["source", "target", "score"]).to_csv(out, index=False)
    print(f"saved {out}  ({len(rows)} edges)")


if __name__ == "__main__":
    main(sys.argv[1])
