"""
Run GTra end-to-end (labeled + label-free), answer-UNCONSTRAINED, and score the
resulting cell-state trajectory against the answer graph via GED + edge-F1.

Usage:
    python run_ged.py --dataset MND  --N 50
    python run_ged.py --dataset HSPC --N 50

labeled   : cells fixed to the predefined annotation (label_flag=True).
unlabeled : cells re-clustered with Leiden (label_flag=False); clusters mapped
            to cell-types by majority annotation only for scoring (not inference).

Both runs set answer_path_dir="" so the trajectory is NOT filtered by the answer.
"""
import argparse, pickle, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx

import gtra
import ged_utils as ge

warnings.filterwarnings("ignore")

# MND uses the FINE cell_type2 annotation (7 states, as in the manuscript). The
# 1-cell subtype (7-IPs @ tp17) is handled by patch_gtra() (robust stat-testing).
# The answer is given in cell_type2 names; we only fix a typo ("enurons").
MND_COLLAPSE = {"2-Young enurons": "2-Young neurons"}

CONFIG = {
    "MND": dict(h5ad="/data3/projects/2025_GTRA/data/1_MND/CCTSD_preproc_hvg.h5ad",
                time_col="timepoints", annot="cell_type2",
                answer="../../answer_paths/MND_answer.csv",
                counts_layer="counts", organism="Mouse", collapse=MND_COLLAPSE),
    "HSPC": dict(h5ad="/data3/projects/2025_GTRA/data/2_HSPC/HSPC_proc.h5ad",
                 time_col="time", annot="celltype",
                 answer="../../answer_paths/HSPC_answer.csv",
                 counts_layer=None, organism="Human", collapse=None),
}


def load_answer(path, collapse=None):
    df = pd.read_csv(path)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        s, t = str(r["source"]).strip(), str(r["target"]).strip()
        if collapse:
            s, t = collapse.get(s, s), collapse.get(t, t)
        G.add_node(s); G.add_node(t); G.add_edge(s, t)
    return G


def build_obj(cfg, mode, N):
    ad = sc.read_h5ad(cfg["h5ad"])
    if "highly_variable" in ad.var and cfg["counts_layer"] == "counts":
        ad = ad[:, ad.var.highly_variable].copy()
    obj = gtra.GTraObject()
    obj.params.cell_type_label = cfg["annot"]
    for t in sorted(ad.obs[cfg["time_col"]].unique()):
        dat = ad[ad.obs[cfg["time_col"]] == t]
        cnt = dat.to_df(layer=cfg["counts_layer"]) if cfg["counts_layer"] else dat.to_df()
        meta = dat.obs[[cfg["annot"]]]
        obj.upload_time_scRNA(cnt, meta)          # 2-arg -> obs keeps annotation
    obj.select_genes()
    if mode == "unlabeled":
        obj.params.label_flag = False             # force Leiden cell clustering
    obj.params.answer_path_dir = ""               # no answer constraint
    obj.find_gclusters(N=N)
    obj.construct_trajectories()
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CONFIG))
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--modes", nargs="+", default=["labeled", "unlabeled"])
    args = ap.parse_args()
    ge.patch_gtra()   # robust _score_distribution (avoid bootstrap-relabel KeyError)
    cfg = CONFIG[args.dataset]
    out = Path(f"{args.dataset}_ged_out"); out.mkdir(exist_ok=True)
    Gref = load_answer(cfg["answer"], cfg.get("collapse"))

    results = {}
    for mode in args.modes:
        t0 = time.time()
        obj = build_obj(cfg, mode, args.N)
        G = ge.state_graph(obj, cfg["annot"], "cluster_label")
        prf = ge.edge_prf(G, Gref)                     # all edges
        prfT = ge.edge_prf(G, Gref, ignore_selfloops=True)  # transitions only
        ged = ge.graph_edit_distance(G, Gref, timeout=60)
        rand_tf1 = ge.random_transition_f1(G, Gref)
        results[mode] = dict(
            GED=ged, norm_GED=ge.normalized_ged(ged, G, Gref),
            f1=prf["f1"], precision=prf["precision"], recall=prf["recall"],
            trans_f1=prfT["f1"], trans_precision=prfT["precision"],
            trans_recall=prfT["recall"], rand_trans_f1=rand_tf1,
            n_pred=prf["n_pred"], n_ref=prf["n_ref"],
            edges=sorted(G.edges()))
        with open(out / f"{mode}_graph.pkl", "wb") as f:
            pickle.dump({"edges": list(G.edges()), "nodes": list(G.nodes())}, f)
        print(f"[{args.dataset}/{mode}] GED={ged:.0f} F1={prf['f1']:.3f} | "
              f"transF1={prfT['f1']:.3f} (rand {rand_tf1:.3f}) "
              f"transP={prfT['precision']:.2f} transR={prfT['recall']:.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        del obj

    pd.DataFrame(results).T.to_csv(out / "ged_summary.csv")
    print(f"saved -> {out/'ged_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
