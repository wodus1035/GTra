"""
Run gene-clustering bootstrap stability for MND and cache per-run labels.

Usage:
    python run_stability.py --N 50 --out stability_out

Caches, per (regime, timepoint, cell-cluster), the list of per-run gene-cluster
label vectors plus the full-data reference, so metrics/figures can be recomputed
cheaply in the notebook without re-running the (expensive) bootstrap.
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import stability_utils as su

DATA = "/data3/projects/2025_GTRA/data/1_MND/CCTSD_preproc_hvg.h5ad"
ANNOT = "cell_type2"
TP_COL = "timepoints"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--out", default="stability_out")
    ap.add_argument("--regimes", nargs="+", default=["annotation", "leiden"])
    ap.add_argument("--res", nargs="+", type=float, default=None,
                    help="override gene-clustering Leiden resolutions, e.g. --res 0.2")
    args = ap.parse_args()
    res_list = tuple(args.res) if args.res else None

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    ad = sc.read_h5ad(DATA)
    ad.X = ad.layers["counts"].copy()
    gene_names = ad.var_names.tolist()
    timepoints = sorted(ad.obs[TP_COL].unique())

    results = {
        "gene_names": gene_names,
        "timepoints": timepoints,
        "annotation": ANNOT,
        "N": args.N,
        "res_list": res_list,
        "regimes": {},
    }

    for regime in args.regimes:
        print(f"\n########## regime = {regime} ##########", flush=True)
        reg = {}
        for tp in timepoints:
            t0 = time.time()
            sub = ad[ad.obs[TP_COL] == tp].copy()

            # reference labels per cell-cluster
            if regime == "annotation":
                codes, _ = pd.factorize(sub.obs[ANNOT], sort=True)
                sub.obs["_ref_label"] = codes
                ref = su.reference_gene_clusters(sub, gene_names, "_ref_label",
                                                 res_list=res_list)
            else:
                # leiden reference computed inside bootstrap; recompute here too
                ref = None  # filled below from the bootstrap's reference pass

            runs = su.bootstrap_timepoint(
                sub, gene_names, ANNOT, regime=regime, N=args.N, seed0=0,
                res_list=res_list,
            )

            # for leiden, build reference from a full-data run for ARI/jaccard
            if regime == "leiden":
                ref = _leiden_reference(sub, gene_names, res_list=res_list)

            reg[int(tp)] = {"runs": runs, "ref": ref}
            print(f"  tp{tp}: {sub.n_obs} cells, "
                  f"{len(runs)} clusters, N={args.N}  ({time.time()-t0:.0f}s)",
                  flush=True)
        results["regimes"][regime] = reg

    with open(outdir / "stability_runs.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved -> {outdir/'stability_runs.pkl'}", flush=True)


def _leiden_reference(sub, gene_names, res_list=None):
    """Full-data Leiden + gene clustering, for ARI/Jaccard reference (regime B)."""
    sub = sub.copy()
    sub.layers["raw"] = sub.X.copy()
    sub.layers["norm"] = sub.X.copy()
    sc.pp.normalize_total(sub, layer="norm")
    sc.pp.log1p(sub, layer="norm")
    sub.layers["scaled"] = sub.layers["norm"].copy()
    sc.pp.scale(sub, max_value=10, layer="scaled")
    sub.X = sub.layers["scaled"]
    sc.tl.pca(sub, svd_solver="arpack")
    sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca")
    sc.tl.leiden(sub, resolution=0.5, random_state=0)
    sub.X = sub.layers["raw"]
    sub.obs["_ref_label"] = sub.obs["leiden"].astype(int).values
    return su.reference_gene_clusters(sub, gene_names, "_ref_label", res_list=res_list)


if __name__ == "__main__":
    main()
