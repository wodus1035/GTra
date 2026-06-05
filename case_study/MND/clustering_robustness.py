"""Clustering / metric robustness for the GTra revision (R3.4, R3.5, R2.3).

R3.4: does the community-detection algorithm matter? Compare Leiden vs Louvain
      per-timepoint cell clustering — agreement with the annotation (ARI/purity)
      and Leiden<->Louvain agreement. Robust if the two algorithms agree.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score as ari

import cellstate_utils as cu

warnings.filterwarnings("ignore")

CFG = {
    "MND": dict(h5ad="/data3/projects/2025_GTRA/data/1_MND/CCTSD_preproc_hvg.h5ad",
                time_col="timepoints", annot="cell_type2", counts_layer="counts"),
    "HSPC": dict(h5ad="/data3/projects/2025_GTRA/data/2_HSPC/HSPC_proc.h5ad",
                 time_col="time", annot="celltype", counts_layer=None),
}


def _cluster(sub, algo, resolution=0.5, n_neighbors=15, seed=0):
    sub = sub.copy()
    sub.layers["norm"] = sub.X.copy()
    sc.pp.normalize_total(sub, layer="norm"); sc.pp.log1p(sub, layer="norm")
    sub.layers["scaled"] = sub.layers["norm"].copy()
    sc.pp.scale(sub, max_value=10, layer="scaled")
    sub.X = sub.layers["scaled"]
    sc.tl.pca(sub, svd_solver="arpack")
    sc.pp.neighbors(sub, n_neighbors=n_neighbors, use_rep="X_pca")
    if algo == "leiden":
        sc.tl.leiden(sub, resolution=resolution, random_state=seed)
        return sub.obs["leiden"].astype(str).values
    sc.tl.louvain(sub, resolution=resolution, random_state=seed)
    return sub.obs["louvain"].astype(str).values


def main():
    rows = []
    for ds, cfg in CFG.items():
        ad = sc.read_h5ad(cfg["h5ad"])
        if cfg["counts_layer"]:
            ad.X = ad.layers[cfg["counts_layer"]].copy()
        for tp in sorted(ad.obs[cfg["time_col"]].unique()):
            sub = ad[ad.obs[cfg["time_col"]] == tp]
            ann = sub.obs[cfg["annot"]].astype(str).values
            lei = _cluster(sub, "leiden"); lou = _cluster(sub, "louvain")
            rows.append({
                "dataset": ds, "timepoint": str(tp),
                "ARI_leiden_vs_annot": ari(ann, lei),
                "ARI_louvain_vs_annot": ari(ann, lou),
                "purity_leiden": cu.purity(ann, lei),
                "purity_louvain": cu.purity(ann, lou),
                "ARI_leiden_vs_louvain": ari(lei, lou),
            })
            print(f"[{ds} {tp}] Leiden-vs-annot ARI={rows[-1]['ARI_leiden_vs_annot']:.3f} "
                  f"Louvain-vs-annot ARI={rows[-1]['ARI_louvain_vs_annot']:.3f} "
                  f"Leiden-vs-Louvain ARI={rows[-1]['ARI_leiden_vs_louvain']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    out = Path("ROBUSTNESS_figs"); out.mkdir(exist_ok=True)
    df.to_csv(out / "leiden_vs_louvain.csv", index=False)
    print("\n=== mean by dataset ===")
    print(df.groupby("dataset")[["ARI_leiden_vs_annot", "ARI_louvain_vs_annot",
          "ARI_leiden_vs_louvain"]].mean().round(3).to_string())
    print("DONE")


if __name__ == "__main__":
    main()
