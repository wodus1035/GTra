"""
Gene-clustering bootstrap stability utilities (GTra revision).

Addresses the reviewer concern: "gene-module composition will change across
bootstrap runs — show that it is stable over many runs."

Two regimes are supported so the result is defensible:

  Regime A ("annotation"):  cell identity is FIXED to the predefined cell-type
      annotation (label_flag=True style). Only the *cells* are subsampled, so
      the measured variability reflects the gene-clustering step alone. This
      isolates and shows the intrinsic stability of the gene modules.

  Regime B ("leiden"):      reproduces GTra's default behaviour (Leiden cell
      clustering is re-run every bootstrap). Each run's cell clusters are
      MATCHED back to a full-data reference (Hungarian assignment on cell
      overlap) before stability is measured. This reflects the stability of
      the full pipeline.

For every (timepoint, cell-cluster) we store the per-run gene-cluster label
vector (aligned to a single fixed gene list). From these we compute:

  * consensus matrix  -> PAC (proportion of ambiguous clustering) + heatmaps
  * ARI / AMI         -> each run's gene partition vs the reference partition
  * module Jaccard    -> best-matched module overlap + "core gene" fraction

The GTra pipeline itself only keeps the consensus matrix (and compares
mismatched Leiden cluster indices across runs), which inflates apparent
instability; this module recomputes everything with correct alignment.
"""

import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from gtra.cluster_func import knn_based_gene_clustering


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_dense(X):
    return X if isinstance(X, np.ndarray) else X.toarray()


def _labels_from_modules(modules, gene_names):
    """list-of-gene-lists -> integer label vector aligned to `gene_names`.

    Genes not present in any module get label -1 (treated as 'unassigned',
    excluded from consensus/ARI rather than lumped into cluster 0).
    """
    g2c = {g: ci for ci, mod in enumerate(modules) for g in mod}
    return np.array([g2c.get(g, -1) for g in gene_names], dtype=int)


def _stratified_subsample(obs, group_col, frac, rng, min_keep):
    """Subsample cell ids within each group (mirrors _create_random_cells)."""
    keep = []
    for _, idx in obs.groupby(group_col, observed=True).groups.items():
        ids = list(idx)
        n = len(ids)
        if n == 0:
            continue
        N = min(max(int(n * frac), min_keep), n)
        if N >= n:
            keep.extend(ids)
        else:
            keep.extend(rng.choice(ids, size=N, replace=False).tolist())
    return keep


def _match_clusters_to_reference(boot_labels, ref_labels):
    """Hungarian match of bootstrap cell-cluster ids to reference ids.

    boot_labels / ref_labels: int arrays over the SAME cells (intersection).
    Returns dict {boot_id -> ref_id}. Unmatched boot ids map to -1.
    """
    b_ids = np.unique(boot_labels)
    r_ids = np.unique(ref_labels)
    # overlap (contingency) matrix
    M = np.zeros((len(b_ids), len(r_ids)), dtype=float)
    for i, b in enumerate(b_ids):
        bm = boot_labels == b
        for j, r in enumerate(r_ids):
            M[i, j] = np.sum(bm & (ref_labels == r))
    # maximize overlap -> minimize -overlap
    ri, ci = linear_sum_assignment(-M)
    mapping = {int(b_ids[i]): int(r_ids[j]) for i, j in zip(ri, ci)}
    for b in b_ids:
        mapping.setdefault(int(b), -1)
    return mapping


# --------------------------------------------------------------------------- #
# reference (full-data) gene clustering
# --------------------------------------------------------------------------- #
def _gene_cluster(X, obs, cid, gene_names, cell_label_col, seed, res_list):
    """Wrapper around knn_based_gene_clustering that optionally overrides res_list."""
    kw = dict(cell_label_col=cell_label_col, seed=seed)
    if res_list is not None:
        kw["res_list"] = tuple(res_list)
    return knn_based_gene_clustering(X, obs, cid, gene_names, **kw)


def reference_gene_clusters(adata, gene_names, cell_label_col, seed=1234, res_list=None):
    """Gene clustering on the FULL data, per cell cluster.

    res_list: override Leiden resolutions (None -> GTra default 0.2/0.4/0.6).
    Returns {cell_cluster_id: label_vector(len=n_genes)}.
    """
    X = _to_dense(adata.X)
    obs = adata.obs
    out = {}
    for cid in sorted(obs[cell_label_col].unique()):
        mods = _gene_cluster(X, obs, cid, gene_names, cell_label_col, seed, res_list)
        if len(mods) == 0:
            continue
        out[int(cid)] = _labels_from_modules(mods, gene_names)
    return out


# --------------------------------------------------------------------------- #
# bootstrap (one timepoint)
# --------------------------------------------------------------------------- #
def bootstrap_timepoint(
    adata,
    gene_names,
    annotation_col,
    regime="annotation",
    N=50,
    frac=0.8,
    min_keep=11,
    cn_neighbors=15,
    cn_resolution=0.5,
    seed0=0,
    res_list=None,
):
    """Run N bootstrap gene-clustering iterations for one timepoint.

    res_list: override gene-clustering Leiden resolutions (None -> GTra default).

    Returns dict keyed by reference cell-cluster id:
        { ref_cid: list_of_label_vectors (one per run, len=n_genes, -1=unassigned) }
    Under regime 'leiden', runs whose cluster cannot be matched contribute -1.
    """
    import scanpy as sc

    runs = {}  # ref_cid -> list of label vectors

    if regime == "annotation":
        # fixed integer cell labels from the annotation
        codes, uniques = pd.factorize(adata.obs[annotation_col], sort=True)
        adata = adata.copy()
        adata.obs["_ref_label"] = codes
        ref_col = "_ref_label"
        ref_ids = sorted(np.unique(codes))
    elif regime == "leiden":
        # full-data Leiden reference (inline; cell_graph_clustering needs a GTraObject)
        ref_ad = adata.copy()
        ref_ad.layers["raw"] = ref_ad.X.copy()
        ref_ad.layers["norm"] = ref_ad.X.copy()
        sc.pp.normalize_total(ref_ad, layer="norm")
        sc.pp.log1p(ref_ad, layer="norm")
        ref_ad.layers["scaled"] = ref_ad.layers["norm"].copy()
        sc.pp.scale(ref_ad, max_value=10, layer="scaled")
        ref_ad.X = ref_ad.layers["scaled"]
        sc.tl.pca(ref_ad, svd_solver="arpack")
        sc.pp.neighbors(ref_ad, n_neighbors=cn_neighbors, use_rep="X_pca")
        sc.tl.leiden(ref_ad, resolution=cn_resolution, random_state=seed0)
        ref_ad.X = ref_ad.layers["raw"]
        ref_codes = ref_ad.obs["leiden"].astype(int).values
        ref_series = pd.Series(ref_codes, index=ref_ad.obs_names)
        ref_ids = sorted(np.unique(ref_codes))
        adata = adata.copy()
        adata.obs["_ref_label"] = ref_series.reindex(adata.obs_names).values
        annotation_col = "_ref_label"  # subsample stratified on reference
    else:
        raise ValueError(regime)

    for cid in ref_ids:
        runs[int(cid)] = []

    rng = np.random.default_rng(seed0)
    for run in range(N):
        keep = _stratified_subsample(adata.obs, annotation_col, frac, rng, min_keep)
        sub = adata[keep].copy()
        Xs = _to_dense(sub.X)

        if regime == "annotation":
            cl_col = "_ref_label"
            sub_ref = sub.obs["_ref_label"].values
        else:
            # re-run Leiden on the subsample
            sub.layers["raw"] = sub.X.copy()
            sub.layers["norm"] = sub.X.copy()
            sc.pp.normalize_total(sub, layer="norm")
            sc.pp.log1p(sub, layer="norm")
            sub.layers["scaled"] = sub.layers["norm"].copy()
            sc.pp.scale(sub, max_value=10, layer="scaled")
            sub.X = sub.layers["scaled"]
            sc.tl.pca(sub, svd_solver="arpack")
            sc.pp.neighbors(sub, n_neighbors=cn_neighbors, use_rep="X_pca")
            sc.tl.leiden(sub, resolution=cn_resolution, random_state=run)
            sub.X = sub.layers["raw"]
            boot_lab = sub.obs["leiden"].astype(int).values
            ref_lab = sub.obs["_ref_label"].values.astype(float)
            ok = ~np.isnan(ref_lab)
            mapping = _match_clusters_to_reference(
                boot_lab[ok], ref_lab[ok].astype(int)
            )
            sub.obs["_match"] = [mapping.get(int(b), -1) for b in boot_lab]
            cl_col = "_match"
            sub_ref = sub.obs["_match"].values
            Xs = _to_dense(sub.X)

        gnames = sub.var_names.tolist()
        for cid in ref_ids:
            mods = _gene_cluster(Xs, sub.obs, cid, gnames, cl_col, run, res_list)
            if len(mods) == 0:
                runs[int(cid)].append(None)
            else:
                runs[int(cid)].append(_labels_from_modules(mods, gene_names))
    return runs


# --------------------------------------------------------------------------- #
# stability metrics from stored per-run labels
# --------------------------------------------------------------------------- #
def consensus_matrix(run_labels):
    """run_labels: list of int label vectors (len n_genes, -1=unassigned, or None).

    Returns (C, counts) where C[i,j] = fraction of runs in which genes i,j were
    BOTH assigned and co-clustered, among runs where both were assigned.
    """
    valid = [r for r in run_labels if r is not None]
    if len(valid) == 0:
        return None, None
    n = len(valid[0])
    co = np.zeros((n, n), dtype=float)
    both = np.zeros((n, n), dtype=float)
    for lab in valid:
        assigned = (lab >= 0)
        a = assigned[:, None] & assigned[None, :]
        same = (lab[:, None] == lab[None, :]) & a
        co += same
        both += a
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.where(both > 0, co / both, np.nan)
    return C, both


def pac_score(C, lo=0.1, hi=0.9):
    """Proportion of ambiguous clustering (lower = more stable)."""
    iu = np.triu_indices_from(C, k=1)
    v = C[iu]
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan
    return float(((v > lo) & (v < hi)).mean())


def ari_vs_reference(run_labels, ref_labels):
    """ARI / AMI of each run's partition vs the reference, over genes assigned
    in both. Returns DataFrame with per-run ari, ami."""
    rows = []
    for k, lab in enumerate(run_labels):
        if lab is None:
            continue
        mask = (lab >= 0) & (ref_labels >= 0)
        if mask.sum() < 3 or len(np.unique(lab[mask])) < 1:
            continue
        rows.append({
            "run": k,
            "ari": adjusted_rand_score(ref_labels[mask], lab[mask]),
            "ami": adjusted_mutual_info_score(ref_labels[mask], lab[mask]),
            "n_genes": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def consensus_modules(C, K, ref_labels=None):
    """Derive consensus modules by hierarchical clustering on (1 - consensus).

    Restricted to genes assigned in the reference (ref_labels >= 0) if given.
    Returns (labels_full, keep_idx) where labels_full has -1 for excluded genes.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n = C.shape[0]
    keep = np.arange(n) if ref_labels is None else np.where(ref_labels >= 0)[0]
    Csub = np.nan_to_num(C[np.ix_(keep, keep)])
    D = 1.0 - Csub
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    Z = linkage(squareform(D, checks=False), method="average")
    sub_lab = fcluster(Z, t=K, criterion="maxclust")
    labels_full = np.full(n, -1, dtype=int)
    labels_full[keep] = sub_lab
    return labels_full, keep


def module_reproducibility(C, module_labels):
    """For each consensus module, mean within-module consensus (m_k, higher =
    more reproducible). Returns DataFrame[module, size, mean_consensus]."""
    rows = []
    for m in np.unique(module_labels):
        if m < 0:
            continue
        idx = np.where(module_labels == m)[0]
        if len(idx) < 2:
            rows.append({"module": int(m), "size": len(idx), "mean_consensus": np.nan})
            continue
        sub = C[np.ix_(idx, idx)]
        iu = np.triu_indices_from(sub, 1)
        v = sub[iu]
        v = v[~np.isnan(v)]
        rows.append({"module": int(m), "size": len(idx),
                     "mean_consensus": float(v.mean()) if v.size else np.nan})
    return pd.DataFrame(rows)


def module_jaccard(run_labels, ref_labels):
    """For each run, best-match its modules to reference modules (Hungarian on
    Jaccard) and report the mean matched Jaccard. Also returns per-gene 'core'
    score = fraction of runs the gene stays in its reference module's best match.
    """
    ref_ids = [c for c in np.unique(ref_labels) if c >= 0]
    ref_sets = {c: set(np.where(ref_labels == c)[0]) for c in ref_ids}

    matched = []
    # per-gene: how often it lands in the run-module matched to its ref module
    n = len(ref_labels)
    core_hit = np.zeros(n)
    core_tot = np.zeros(n)

    for lab in run_labels:
        if lab is None:
            continue
        b_ids = [c for c in np.unique(lab) if c >= 0]
        if not b_ids:
            continue
        b_sets = {c: set(np.where(lab == c)[0]) for c in b_ids}
        J = np.zeros((len(ref_ids), len(b_ids)))
        for i, rc in enumerate(ref_ids):
            for j, bc in enumerate(b_ids):
                inter = len(ref_sets[rc] & b_sets[bc])
                union = len(ref_sets[rc] | b_sets[bc])
                J[i, j] = inter / union if union else 0.0
        ri, ci = linear_sum_assignment(-J)
        run_match = J[ri, ci].mean()
        matched.append(run_match)
        # core scoring
        ref_to_boot = {ref_ids[i]: b_ids[j] for i, j in zip(ri, ci)}
        for g in range(n):
            rc = ref_labels[g]
            if rc < 0:
                continue
            core_tot[g] += 1
            if lab[g] == ref_to_boot.get(rc, -999):
                core_hit[g] += 1
    with np.errstate(invalid="ignore"):
        core = np.where(core_tot > 0, core_hit / core_tot, np.nan)
    return np.array(matched), core
