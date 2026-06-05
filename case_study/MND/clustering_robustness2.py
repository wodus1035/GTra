"""R3.5 (gene-module clustering-method robustness) + R2.3 (magnitude-bias diagnostic).

R3.5: cluster the same gene temporal profiles with spherical k-means (GTra's
      choice), k-means, agglomerative, and GMM; compare functional coherence
      (GO enrichment). Robust if the algorithm choice does not change coherence.

R2.3: does the gene-gene graph cluster genes by EXPRESSION MAGNITUDE rather than
      co-expression? GTra z-scores each gene (across cells) before PCA/kNN. We
      quantify how much gene clusters separate by mean expression (eta^2) WITH
      (GTra) vs WITHOUT the z-score. Low eta^2 with z-score => magnitude is not
      the driver.
"""
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg

import gene_module_baseline as gm

warnings.filterwarnings("ignore")
DS = "MND"; ORG = "Mouse"


# ---------------- R3.5 ----------------
def r35():
    gtra = gm.gtra_modules(DS, min_genes=10)
    K = len(gtra)
    universe = sorted(set().union(*[set(m) for m in gtra]))
    from run_ged import CONFIG
    cfg = CONFIG[DS]
    ad = sc.read_h5ad(cfg["h5ad"])
    if "highly_variable" in ad.var:
        ad = ad[:, ad.var.highly_variable].copy()
    universe = [g for g in universe if g in set(ad.var_names)]
    gmat = gm._gene_temporal_matrix(ad, cfg["time_col"], cfg["annot"], None, cfg["counts_layer"])
    P = np.hstack([gmat[ct].loc[universe].values for ct in gmat])
    Z = np.nan_to_num((P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9))
    algos = {
        "kmeans": lambda: KMeans(K, random_state=0, n_init=10).fit_predict(Z),
        "spherical_kmeans": lambda: KMeans(K, random_state=0, n_init=10).fit_predict(
            Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)),
        "agglomerative": lambda: AgglomerativeClustering(K).fit_predict(Z),
        "gmm": lambda: GaussianMixture(K, random_state=0).fit_predict(Z),
    }
    print("=== R3.5 gene-module clustering-method robustness (MND, functional coherence) ===")
    rows = []
    for name, fn in algos.items():
        lab = fn()
        mods = [list(np.array(universe)[lab == c]) for c in np.unique(lab)]
        mods = [m for m in mods if len(m) >= 10]
        s, _ = gm.functional_coherence(mods, ORG, max_modules=30)
        rows.append({"algorithm": name, **s})
        print(f"  {name:16s} frac_sig={s['frac_sig']:.3f} mean_logp={s['mean_logp']:.2f} "
              f"nmod={s['n_modules']}", flush=True)
    pd.DataFrame(rows).to_csv("ROBUSTNESS_figs/r35_clustering_methods.csv", index=False)


# ---------------- R2.3 ----------------
def _gene_clusters(Xsub, zscore):
    """genes x cells -> leiden gene clusters; optionally gene-wise z-score (GTra)."""
    G = np.log2(Xsub + 1.0)
    if zscore:
        G = (G - G.mean(1, keepdims=True)) / (G.std(1, keepdims=True) + 1e-8)
    G = np.nan_to_num(G)
    n_comp = min(30, G.shape[1] - 1, G.shape[0] - 1)
    Gp = PCA(n_components=n_comp, random_state=0).fit_transform(G)
    k = min(15, G.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Gp)
    knn = nn.kneighbors_graph(Gp, mode="connectivity"); knn = knn.maximum(knn.T)
    src, tar = knn.nonzero()
    g = ig.Graph(n=G.shape[0], edges=list(zip(src.tolist(), tar.tolist())), directed=False)
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=0.4, seed=0)
    return np.array(part.membership)


def _eta2(values, labels):
    """fraction of variance in `values` explained by `labels` (between/total)."""
    grand = values.mean()
    ss_tot = ((values - grand) ** 2).sum()
    ss_bet = sum(len(values[labels == c]) * (values[labels == c].mean() - grand) ** 2
                 for c in np.unique(labels))
    return float(ss_bet / (ss_tot + 1e-12))


def r23():
    from run_ged import CONFIG
    cfg = CONFIG[DS]
    ad = sc.read_h5ad(cfg["h5ad"])
    if "highly_variable" in ad.var:
        ad = ad[:, ad.var.highly_variable].copy()
    ad.X = ad.layers[cfg["counts_layer"]].copy()
    print("\n=== R2.3 magnitude-bias: eta^2 of gene mean-expression across clusters ===")
    print("    (lower = clusters NOT driven by expression magnitude)")
    rows = []
    # representative cell types at one timepoint
    tp = sorted(ad.obs[cfg["time_col"]].unique())[1]
    sub_tp = ad[ad.obs[cfg["time_col"]] == tp]
    for ct in pd.unique(sub_tp.obs[cfg["annot"]].astype(str))[:4]:
        m = sub_tp.obs[cfg["annot"]].astype(str).values == ct
        if m.sum() < 30:
            continue
        X = sub_tp[m].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        Xg = X.T  # genes x cells
        mean_expr = np.log2(Xg + 1).mean(1)  # per-gene magnitude
        for zs in (True, False):
            lab = _gene_clusters(Xg, zscore=zs)
            rows.append({"cell_type": ct, "zscore(GTra)": zs,
                         "eta2_magnitude": _eta2(mean_expr, lab), "n_clusters": len(np.unique(lab))})
        z = rows[-2]["eta2_magnitude"]; nz = rows[-1]["eta2_magnitude"]
        print(f"  {ct:16s} eta2 WITH z-score(GTra)={z:.3f}  WITHOUT={nz:.3f}", flush=True)
    pd.DataFrame(rows).to_csv("ROBUSTNESS_figs/r23_magnitude_bias.csv", index=False)


if __name__ == "__main__":
    from pathlib import Path
    Path("ROBUSTNESS_figs").mkdir(exist_ok=True)
    r35()
    r23()
    print("DONE")
