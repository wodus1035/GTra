"""
Cell-state defence utilities (GTra revision).

Reviewer concern: GTra relies on (predefined) cell states. Show that the cell
states are data-supported — i.e. an UNSUPERVISED per-timepoint clustering
(GTra's own `cell_graph_clustering`, Leiden) recovers the annotation.

Strategy: for each timepoint, cluster cells without labels and compare to the
predefined annotation (coarse `cell_type` = major states, and fine `cell_type2`).
We report several complementary agreement metrics, because ARI alone is
conservative when the data has continuous developmental structure that Leiden
splits more finely than a discrete annotation:

  * ARI / AMI / NMI            — partition agreement
  * homogeneity / completeness / V-measure
  * purity                     — each Leiden cluster's majority-label fraction
                                 (high purity = clusters are biologically coherent)
  * inverse purity             — each annotated state's majority-cluster fraction

The Leiden pipeline here is identical to gtra.cluster_func.cell_graph_clustering
(normalize_total -> log1p -> scale -> PCA -> neighbors -> Leiden) so the result
reflects what GTra actually does when run label-free.
"""

import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)


def leiden_clustering(sub, resolution=0.5, n_neighbors=15, seed=0):
    """Unsupervised per-timepoint Leiden, matching cell_graph_clustering."""
    sub = sub.copy()
    sub.layers["norm"] = sub.X.copy()
    sc.pp.normalize_total(sub, layer="norm")
    sc.pp.log1p(sub, layer="norm")
    sub.layers["scaled"] = sub.layers["norm"].copy()
    sc.pp.scale(sub, max_value=10, layer="scaled")
    sub.X = sub.layers["scaled"]
    sc.tl.pca(sub, svd_solver="arpack")
    sc.pp.neighbors(sub, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.leiden(sub, resolution=resolution, random_state=seed)
    return sub.obs["leiden"].astype(str).values


def purity(true, pred):
    """Each predicted cluster -> majority true label; fraction correct."""
    ct = pd.crosstab(pd.Series(pred), pd.Series(true))
    return float(ct.max(axis=1).sum() / ct.values.sum())


def inverse_purity(true, pred):
    """Each true label -> majority predicted cluster; fraction correct."""
    ct = pd.crosstab(pd.Series(true), pd.Series(pred))
    return float(ct.max(axis=1).sum() / ct.values.sum())


def agreement(true, pred):
    h, c, v = homogeneity_completeness_v_measure(true, pred)
    return {
        "ARI": adjusted_rand_score(true, pred),
        "AMI": adjusted_mutual_info_score(true, pred),
        "NMI": normalized_mutual_info_score(true, pred),
        "homogeneity": h,
        "completeness": c,
        "V_measure": v,
        "purity": purity(true, pred),
        "inv_purity": inverse_purity(true, pred),
        "n_pred": len(np.unique(pred)),
        "n_true": len(np.unique(true)),
    }


def evaluate_timepoints(adata, tp_col, annot_cols, resolution=0.5,
                        n_neighbors=15, seed=0, use_counts_layer="counts"):
    """Per-timepoint unsupervised clustering vs each annotation column.

    Returns (summary_df, labels_dict) where labels_dict[tp] = leiden labels and
    summary_df has one row per (timepoint, annotation column).
    """
    ad = adata.copy()
    if use_counts_layer and use_counts_layer in ad.layers:
        ad.X = ad.layers[use_counts_layer].copy()

    rows, labels = [], {}
    for tp in sorted(ad.obs[tp_col].unique()):
        sub = ad[ad.obs[tp_col] == tp]
        lab = leiden_clustering(sub, resolution, n_neighbors, seed)
        labels[tp] = pd.Series(lab, index=sub.obs_names)
        for ac in annot_cols:
            m = agreement(sub.obs[ac].values, lab)
            m.update({"timepoint": tp, "annotation": ac, "n_cells": sub.n_obs,
                      "resolution": resolution})
            rows.append(m)
    return pd.DataFrame(rows), labels


def evaluate_covid_objects(obj_paths, annot_col="mye_sub", resolution=0.5,
                           n_neighbors=15, seed=0):
    """Cell-state agreement for COVID, iterating the per-patient GTra dills.

    Each dill's tp_data_dict[tp] is an AnnData (raw counts in X) annotated with
    `annot_col`. Returns (summary_df, labels) where summary has patient+timepoint
    rows and labels[(patient, tp)] = leiden labels.
    """
    import dill

    rows, labels = [], {}
    for p in obj_paths:
        pid = p.split("/")[-1].split("_")[0]
        with open(p, "rb") as f:
            obj = dill.load(f)
        for tp in range(obj.tp_data_num):
            sub = obj.tp_data_dict[tp]
            if annot_col not in sub.obs:
                continue
            lab = leiden_clustering(sub, resolution, n_neighbors, seed)
            labels[(pid, tp)] = pd.Series(lab, index=sub.obs_names)
            m = agreement(sub.obs[annot_col].values, lab)
            m.update({"patient": pid, "timepoint": tp, "n_cells": sub.n_obs,
                      "annotation": annot_col, "resolution": resolution})
            rows.append(m)
        del obj
    return pd.DataFrame(rows), labels


def confusion_from_labels(true, pred, normalize="index"):
    """Contingency table (true x pred), optionally row-normalized."""
    ct = pd.crosstab(pd.Series(true), pd.Series(pred))
    if normalize == "index":
        ct = ct.div(ct.sum(axis=1), axis=0)
    return ct


def confusion(adata, tp_col, tp, annot_col, labels, normalize="index"):
    """Contingency table (annotation x leiden) for one timepoint, row-normalized."""
    sub = adata[adata.obs[tp_col] == tp]
    lab = labels[tp].reindex(sub.obs_names).values
    return confusion_from_labels(sub.obs[annot_col].values, lab, normalize)


def resolution_scan(adata, tp_col, annot_col, resolutions, tp=None,
                    n_neighbors=15, seed=0, use_counts_layer="counts"):
    """Agreement vs Leiden resolution. If tp is None, averages over all tps."""
    ad = adata.copy()
    if use_counts_layer and use_counts_layer in ad.layers:
        ad.X = ad.layers[use_counts_layer].copy()
    tps = [tp] if tp is not None else sorted(ad.obs[tp_col].unique())
    rows = []
    for res in resolutions:
        for t in tps:
            sub = ad[ad.obs[tp_col] == t]
            lab = leiden_clustering(sub, res, n_neighbors, seed)
            m = agreement(sub.obs[annot_col].values, lab)
            m.update({"resolution": res, "timepoint": t})
            rows.append(m)
    return pd.DataFrame(rows)
