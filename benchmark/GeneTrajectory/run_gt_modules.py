"""Run GeneTrajectory (Nat Biotech 2025) on MND and export its gene modules.

GeneTrajectory is a gene-level method: it embeds genes and extracts gene
trajectories. We export the gene set of each inferred trajectory so it can be
compared to GTra's gene modules on the SAME gene-module axis (functional
coherence / known-program recovery) — not topology.

Run in py310:  python run_gt_modules.py   ->  gt_mnd_modules.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc
from pyALRA import alra, choose_k
from gene_trajectory.coarse_grain import select_top_genes, coarse_grain_adata
from gene_trajectory.extract_gene_trajectory import get_gene_embedding, extract_gene_trajectory
from gene_trajectory.get_graph_distance import get_graph_distance
from gene_trajectory.gene_distance_shared import cal_ot_mat
from gene_trajectory.run_dm import run_dm

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def main():
    ad = sc.read_h5ad("/data3/projects/2025_GTRA/data/1_MND/CCTSD_preproc_hvg.h5ad")
    if "highly_variable" in ad.var:
        ad = ad[:, ad.var.highly_variable].copy()
    # ALRA imputation (as in the benchmark notebook)
    k = choose_k(ad.X)
    ad.layers["alra"] = alra(ad.X, k["k"])["A_norm_rank_k_cor_sc"]
    genes = select_top_genes(ad, layer="counts", n_variable_genes=500)
    run_dm(ad)
    cell_graph_dist = get_graph_distance(ad, k=10)
    gexpr, gdist = coarse_grain_adata(ad, graph_dist=cell_graph_dist, features=genes, n=500)
    gene_dist_mat = cal_ot_mat(gene_expr=gexpr, ot_cost=gdist, show_progress_bar=False)
    gene_embedding, _ = get_gene_embedding(gene_dist_mat, k=5)
    gt = extract_gene_trajectory(gene_embedding, gene_dist_mat,
                                 t_list=[4, 8, 7], gene_names=genes, k=5)

    # gt is a DataFrame indexed by gene with a 'selected' column = trajectory id
    col = "selected" if "selected" in gt.columns else gt.columns[-1]
    modules = {}
    for traj, sub in gt.groupby(col):
        modules[str(traj)] = list(map(str, sub.index.tolist()))
    modules = {k2: v for k2, v in modules.items()
               if str(k2).lower() not in ("-1", "none", "nan") and len(v) >= 5}
    json.dump(modules, open(HERE / "gt_mnd_modules.json", "w"))
    print("GeneTrajectory MND modules:", {k2: len(v) for k2, v in modules.items()})
    print("saved gt_mnd_modules.json")


if __name__ == "__main__":
    main()
