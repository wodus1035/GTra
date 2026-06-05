"""
Gene-module benchmark: GTra modules vs a simpler baseline (ISM review R2.5/R4.3/R4.4).

GTra's distinctive output is not the cell-state topology (≈ a naive pseudobulk
baseline) but the *dynamic gene-expression programs* along trajectories. We test
whether the full machinery yields more biologically coherent modules than a
simpler alternative:

  BASELINE (R2.5): cluster genes directly by their cell-type-specific pseudobulk
  temporal-expression vectors (k-means on per-cell-type gene x timepoint
  profiles) — i.e. skip GTra's inter-temporal Jaccard module linking and
  trajectory construction entirely.

Functional coherence (R4.3, quantitative): for every module we run GO Biological
Process enrichment (Enrichr) and summarize:
  * frac_sig    — fraction of modules with ≥1 term at adj-p < 0.05
  * mean_logp   — mean -log10(adj-p) of each module's top term
  * mean_nsig   — mean number of significant terms per module
A more coherent set of modules scores higher.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

from run_ged import CONFIG  # data paths / annot / collapse

ORGANISM = {"MND": "Mouse", "HSPC": "Human", "COVID": "Human"}
GTRA_MODULE_CSV = {
    "MND": "MND_out/Mouse_pattern_genes.csv",
    "HSPC": "../HSPC/HSPC_out/hspc_pattern_genes.csv",
}


# --------------------------------------------------------------------------- #
# modules
# --------------------------------------------------------------------------- #
def gtra_modules(dataset, min_genes=10):
    """GTra gene modules (gene sets) from pattern_genes.csv (MND/HSPC) or dills (COVID)."""
    if dataset in GTRA_MODULE_CSV:
        df = pd.read_csv(GTRA_MODULE_CSV[dataset], index_col=0)
        mods = []
        for col in df.columns:
            genes = [g for g in df[col].dropna().tolist() if isinstance(g, str) and g]
            if len(set(genes)) >= min_genes:
                mods.append(sorted(set(genes)))
        return mods
    raise ValueError(f"use covid path for {dataset}")


def _gene_temporal_matrix(ad, time_col, annot, collapse, counts_layer):
    """Per cell type: genes x timepoints pseudobulk (CP10k+log1p). Returns
    {celltype: (DataFrame genes x timepoints)} and the gene list."""
    X = ad.layers[counts_layer] if counts_layer else ad.X
    Xd = X.toarray() if sp.issparse(X) else np.asarray(X)
    Xln = np.log1p(Xd / (Xd.sum(1, keepdims=True) + 1e-12) * 1e4)
    genes = ad.var_names.tolist()
    states = ad.obs[annot].astype(str).values
    if collapse:
        states = np.array([collapse.get(s, s) for s in states])
    tps = sorted(ad.obs[time_col].unique())
    out = {}
    for ct in pd.unique(states):
        cols = []
        for t in tps:
            m = (states == ct) & (ad.obs[time_col].values == t)
            cols.append(Xln[m].mean(0) if m.sum() else np.zeros(Xln.shape[1]))
        out[ct] = pd.DataFrame(np.array(cols).T, index=genes,
                               columns=[str(t) for t in tps])
    return out


def _profiles_to_modules(profiles, genes, K, min_genes):
    """KMeans gene temporal profiles -> K modules (gene lists)."""
    Z = np.nan_to_num((profiles - profiles.mean(1, keepdims=True))
                      / (profiles.std(1, keepdims=True) + 1e-9))
    K = max(2, min(K, len(Z)))
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(Z)
    genes = np.asarray(genes)
    mods = [sorted(genes[km.labels_ == c].tolist()) for c in range(K)]
    return [m for m in mods if len(m) >= min_genes]


def baseline_modules(dataset, gtra_mods, min_genes=10):
    """FAIR R2.5 baseline: same gene universe (genes GTra placed in modules) and
    same #modules, but assigned by plain k-means on cell-type-specific temporal
    profiles instead of GTra's trajectory-linking. Isolates the clustering step."""
    cfg = CONFIG[dataset]
    universe = sorted(set().union(*[set(m) for m in gtra_mods]))
    ad = sc.read_h5ad(cfg["h5ad"])
    if "highly_variable" in ad.var and cfg["counts_layer"] == "counts":
        ad = ad[:, ad.var.highly_variable].copy()
    universe = [g for g in universe if g in set(ad.var_names)]
    gmat = _gene_temporal_matrix(ad, cfg["time_col"], cfg["annot"],
                                 cfg.get("collapse"), cfg["counts_layer"])
    # concatenate per-cell-type temporal profiles -> one vector per gene
    profiles = np.hstack([gmat[ct].loc[universe].values for ct in gmat])
    return _profiles_to_modules(profiles, universe, len(gtra_mods), min_genes)


def covid_modules_gtra(min_genes=10):
    """GTra COVID modules pooled across patients (from merge_pattern_dict gene sets)."""
    import glob, dill
    mods = []
    for p in sorted(glob.glob("../../../../covid_obj/*.dill")):
        o = dill.load(open(p, "rb"))
        mpd = getattr(o, "merge_pattern_dict", {}) or {}
        for _, df in mpd.items():
            genes = [g for g in list(getattr(df, "index", [])) if isinstance(g, str)]
            if len(set(genes)) >= min_genes:
                mods.append(sorted(set(genes)))
        del o
    return mods


def covid_baseline_modules(n_modules, min_genes=10):
    """FAIR COVID baseline: pool per-patient cell-type temporal profiles over the
    1082-gene universe, k-means into n_modules (matched to GTra's module count)."""
    import glob, dill
    paths = sorted(glob.glob("../../../../covid_obj/*.dill"))
    # gene universe = intersection across patients (var_names differ per patient)
    gsets = []
    for p in paths:
        o = dill.load(open(p, "rb")); gsets.append(set(o.tp_data_dict[0].var_names)); del o
    genes_ref = sorted(set.intersection(*gsets))
    cols_all = []
    for p in paths:
        o = dill.load(open(p, "rb"))
        tps = list(range(o.tp_data_num))
        states = sorted(set().union(*[set(o.tp_data_dict[t].obs["mye_sub"].astype(str)) for t in tps]))
        for ct in states:
            for t in tps:
                ad = o.tp_data_dict[t][:, genes_ref]
                m = (ad.obs["mye_sub"].astype(str).values == ct)
                X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
                Xln = np.log1p(X / (X.sum(1, keepdims=True) + 1e-12) * 1e4)
                cols_all.append(Xln[m].mean(0) if m.sum() else np.zeros(len(genes_ref)))
        del o
    profiles = np.array(cols_all).T  # genes x (patient*celltype*tp)
    return _profiles_to_modules(profiles, genes_ref, n_modules, min_genes)


# --------------------------------------------------------------------------- #
# functional coherence (Enrichr GO BP)
# --------------------------------------------------------------------------- #
def functional_coherence(modules, organism, gene_sets="GO_Biological_Process_2021",
                         max_modules=40, sig=0.05, max_genes=60):
    """Per-module GO enrichment; returns (summary dict, per-module DataFrame).

    Each module is truncated to its first `max_genes` genes so module SIZE is
    comparable across methods (otherwise large sets enrich trivially)."""
    import gseapy as gp
    rows = []
    for i, genes in enumerate(modules[:max_modules]):
        genes = list(genes)[:max_genes]
        try:
            enr = gp.enrichr(gene_list=list(genes), gene_sets=[gene_sets],
                             organism=organism, outdir=None)
            res = enr.res2d
            adjp = pd.to_numeric(res["Adjusted P-value"], errors="coerce")
            best = float(adjp.min()) if len(adjp) else 1.0
            nsig = int((adjp < sig).sum())
        except Exception:
            best, nsig = 1.0, 0
        rows.append({"module": i, "n_genes": len(genes),
                     "best_adjp": best, "neglogp": -np.log10(best + 1e-300),
                     "n_sig": nsig, "is_sig": best < sig})
    df = pd.DataFrame(rows)
    summary = {"n_modules": len(df),
               "frac_sig": float(df["is_sig"].mean()) if len(df) else 0.0,
               "mean_logp": float(df["neglogp"].mean()) if len(df) else 0.0,
               "mean_nsig": float(df["n_sig"].mean()) if len(df) else 0.0}
    return summary, df
