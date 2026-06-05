"""Run scEGOT and emit its cell-state transition graph in the common format.

scEGOT fits a per-timepoint GMM and builds a weighted cell-state graph
(make_cell_state_graph, threshold=0.5). We map each GMM cluster to a cell-type by
majority annotation, aggregate edges to cell-type space, and write
edges/scEGOT_{dataset}.csv (source,target,score=edge weight).

Run in gtra_bench:  python run_scegot_edges.py MND   (or HSPC)
"""
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from scegot import scEGOT

warnings.filterwarnings("ignore")
EDGE_DIR = Path(__file__).resolve().parent.parent / "edges"
MND_COLLAPSE = {"1-Neurons": "Neurons", "2-Young neurons": "Young neurons",
                "6-Young neurons": "Young neurons", "3-APs/RPs": "APs/RPs",
                "5-APs/RPs": "APs/RPs", "4-IPs": "IPs", "7-IPs": "IPs"}

CFG = {
    "MND": dict(h5ad="/data3/projects/2025_GTRA/data/1_MND/CCTSD_preproc_hvg.h5ad",
                day_key="Time_points", annot="cell_type2", gmm=[7, 7, 7, 7],
                days=["E11", "E13", "E15", "E17"], time_col="timepoints",
                time_order=None),
    "HSPC": dict(h5ad="/data3/projects/2025_GTRA/data/2_HSPC/HSPC_proc.h5ad",
                 day_key="time", annot="celltype", gmm=[6, 6, 6, 6],
                 days=["control", "3h", "24h", "72h"], time_col="time",
                 time_order=["control", "3h", "24h", "72h"]),
}


def main(dataset):
    cfg = CFG[dataset]
    ad = anndata.read_h5ad(cfg["h5ad"])
    if "highly_variable" in ad.var:
        ad = ad[:, ad.var.highly_variable].copy()
    if cfg["time_order"]:
        ad.obs[cfg["time_col"]] = pd.Categorical(ad.obs[cfg["time_col"]],
                                                  categories=cfg["time_order"], ordered=True)
        ad = ad[ad.obs[cfg["time_col"]].cat.codes.argsort()].copy()
    if cfg["day_key"] not in ad.obs:
        ad.obs[cfg["day_key"]] = ad.obs[cfg["time_col"]].astype(str).values

    seg = scEGOT(ad, verbose=False, adata_day_key=cfg["day_key"])
    seg.preprocess(30, recode_params={}, umi_target_sum=1e5, pca_random_state=2023,
                   apply_recode=True, apply_normalization_log1p=True,
                   apply_normalization_umi=True, select_genes=False)
    _, gmm_labels = seg.fit_predict_gmm(n_components_list=cfg["gmm"],
                                        covariance_type="full", max_iter=2000,
                                        n_init=10, random_state=2023)

    # (day_index, cluster) -> majority cell-type
    maj = {}
    for di, (X_t, labels) in enumerate(zip(seg.X_selected, gmm_labels)):
        cell_ids = list(X_t.index)
        ann = ad.obs.loc[cell_ids, cfg["annot"]].astype(str).values
        for c in np.unique(labels):
            maj[(di, int(c))] = Counter(ann[labels == c]).most_common(1)[0][0]

    cluster_names = seg.generate_cluster_names_with_day()
    G = seg.make_cell_state_graph(cluster_names, mode="pca", threshold=0.5)

    def node_ct(node):
        # node like 'e11-3' / 'control-2' -> (day_index, cluster)
        m = re.match(r"^(.*)-(\d+)$", str(node))
        daypart, c = m.group(1), int(m.group(2))
        di = next(i for i, d in enumerate(cfg["days"]) if d.lower() == daypart.lower())
        return maj[(di, c)]  # cell_type2, no collapse

    edge_w = {}
    for u, v, d in G.edges(data=True):
        s, t = node_ct(u), node_ct(v)
        w = float(d.get("weight", d.get("edge_weights", 1.0)))
        edge_w[(s, t)] = max(edge_w.get((s, t), 0.0), w)

    EDGE_DIR.mkdir(exist_ok=True)
    out = EDGE_DIR / f"scEGOT_{dataset}.csv"
    pd.DataFrame([(s, t, w) for (s, t), w in edge_w.items()],
                 columns=["source", "target", "score"]).to_csv(out, index=False)
    print(f"saved {out}  ({len(edge_w)} edges)")


if __name__ == "__main__":
    main(sys.argv[1])
