"""
Figures + summary tables for gene-clustering bootstrap stability (MND).

Reads the cache produced by run_stability.py and computes:
  * summary table: per (regime, timepoint, cell-cluster) PAC / ARI / AMI /
    module-Jaccard / core-gene fraction
  * Supplementary figures:
      F1  consensus heatmap (one representative cell cluster)
      F2  PAC across timepoints x clusters, annotation vs leiden
      F3  ARI / AMI distribution across runs (violin)
      F4  module Jaccard + core-gene fraction
      F5  naive (mis-aligned ccmatrix) vs corrected PAC  [the key defence]
"""
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

import stability_utils as su


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
def build_summary(res):
    """Long-form DataFrame of stability metrics."""
    rows = []
    for regime, reg in res["regimes"].items():
        for tp, d in reg.items():
            ref = d["ref"]
            for cid, run_labels in d["runs"].items():
                if ref is None or cid not in ref:
                    continue
                C, _ = su.consensus_matrix(run_labels)
                if C is None:
                    continue
                pac = su.pac_score(C)
                ari_df = su.ari_vs_reference(run_labels, ref[cid])
                jac, core = su.module_jaccard(run_labels, ref[cid])
                core_assigned = core[~np.isnan(core)]
                refK = len(np.unique(ref[cid][ref[cid] >= 0]))
                rows.append({
                    "regime": regime, "timepoint": tp, "cluster": cid,
                    "refK": refK,
                    "PAC": pac,
                    "ARI": ari_df["ari"].mean() if len(ari_df) else np.nan,
                    "ARI_std": ari_df["ari"].std() if len(ari_df) else np.nan,
                    "AMI": ari_df["ami"].mean() if len(ari_df) else np.nan,
                    "module_jaccard": np.nanmean(jac) if len(jac) else np.nan,
                    "core_frac": float((core_assigned >= 0.8).mean()) if core_assigned.size else np.nan,
                    "n_valid_runs": int(sum(r is not None for r in run_labels)),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def fig_consensus_heatmap(res, regime, tp, cid, ax=None):
    d = res["regimes"][regime][tp]
    C, _ = su.consensus_matrix(d["runs"][cid])
    ref = d["ref"][cid]
    keep = np.where(ref >= 0)[0]
    Csub = C[np.ix_(keep, keep)]
    Csub = np.nan_to_num(Csub)
    # order genes by hierarchical clustering of (1-consensus)
    dist = 1 - Csub
    np.fill_diagonal(dist, 0)
    Z = linkage(dist[np.triu_indices_from(dist, 1)], method="average")
    order = leaves_list(Z)
    Cord = Csub[np.ix_(order, order)]
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(Cord, cmap="rocket_r", vmin=0, vmax=1, square=True,
                xticklabels=False, yticklabels=False, cbar_kws={"label": "consensus"},
                rasterized=True, ax=ax)
    ax.set_title(f"{regime} | tp{tp} | cluster {cid}\nPAC={su.pac_score(C):.3f}")
    return ax


def fig_pac_bars(summary, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=summary, x="timepoint", y="PAC", hue="regime",
                errorbar="sd", ax=ax)
    ax.set_ylabel("PAC (lower = more stable)")
    ax.set_title("Gene-module ambiguity across bootstraps")
    return ax


def fig_ari_violin(res, regime, summary, ax=None):
    rows = []
    reg = res["regimes"][regime]
    for tp, d in reg.items():
        ref = d["ref"]
        for cid, run_labels in d["runs"].items():
            if ref is None or cid not in ref:
                continue
            adf = su.ari_vs_reference(run_labels, ref[cid])
            for _, r in adf.iterrows():
                rows.append({"timepoint": tp, "cluster": cid, "ARI": r["ari"]})
    df = pd.DataFrame(rows)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df, x="timepoint", y="ARI", inner="box", cut=0, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"Run-vs-reference ARI ({regime})")
    return ax, df


def module_reproducibility_table(res, regime):
    """Per consensus-module within-module mean consensus (m_k), all cell-states."""
    rows = []
    reg = res["regimes"][regime]
    for tp, d in reg.items():
        ref = d["ref"]
        for cid, run_labels in d["runs"].items():
            if ref is None or cid not in ref:
                continue
            C, _ = su.consensus_matrix(run_labels)
            if C is None:
                continue
            K = len(np.unique(ref[cid][ref[cid] >= 0]))
            mods, _ = su.consensus_modules(C, K, ref_labels=ref[cid])
            rep = su.module_reproducibility(C, mods)
            rep["timepoint"] = tp
            rep["cluster"] = cid
            rows.append(rep)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fig_module_reproducibility(res, regime, ax=None):
    rep = module_reproducibility_table(res, regime)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=rep, x="mean_consensus", weights="size", bins=20,
                 color="#4c72b0", ax=ax)
    med = rep["mean_consensus"].median()
    ax.axvline(med, ls="--", c="k", lw=1, label=f"median = {med:.2f}")
    ax.set_xlabel("within-module mean consensus  (higher = reproducible)")
    ax.set_ylabel("genes (weighted)")
    ax.set_title(f"Consensus-module reproducibility ({regime})")
    ax.legend()
    return ax, rep


def fig_jaccard_core(summary, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    m = summary.melt(id_vars=["regime", "timepoint", "cluster"],
                     value_vars=["module_jaccard", "core_frac"],
                     var_name="metric", value_name="value")
    sns.boxplot(data=m, x="metric", y="value", hue="regime", ax=ax)
    ax.set_title("Module overlap & core-gene fraction")
    return ax
