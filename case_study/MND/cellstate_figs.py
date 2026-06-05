"""Figures for the cell-state defence (MND)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cellstate_utils as cu


def fig_agreement_bars(summary, annotation, metrics=("ARI", "AMI", "V_measure", "purity"),
                       order=None, ax=None):
    df = summary[summary["annotation"] == annotation]
    m = df.melt(id_vars=["timepoint"], value_vars=list(metrics),
                var_name="metric", value_name="value")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=m, x="timepoint", y="value", hue="metric", order=order, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"Unsupervised clustering vs '{annotation}'")
    ax.set_ylabel("agreement")
    return ax


def fig_confusion_grid(adata, tp_col, annotation, labels, tps, ax=None):
    if ax is None:
        _, axes = plt.subplots(1, len(tps), figsize=(4 * len(tps), 3.5))
    else:
        axes = ax
    axes = np.atleast_1d(axes)
    for a, tp in zip(axes, tps):
        ct = cu.confusion(adata, tp_col, tp, annotation, labels, normalize="index")
        sns.heatmap(ct, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f",
                    cbar=False, ax=a)
        a.set_title(f"tp{tp}")
        a.set_xlabel("Leiden cluster")
        a.set_ylabel(annotation if tp == tps[0] else "")
    return axes


def _get_coords(sub, coords):
    """coords: obsm key (str) or pair of obs column names (tuple)."""
    if isinstance(coords, str):
        return np.asarray(sub.obsm[coords])[:, :2]
    return sub.obs[list(coords)].values


def fig_tsne_compare(adata, tp_col, annotation, labels, tps,
                     coords=("tSNE_1", "tSNE_2")):
    fig, axes = plt.subplots(2, len(tps), figsize=(3.6 * len(tps), 7))
    for j, tp in enumerate(tps):
        sub = adata[adata.obs[tp_col] == tp]
        xy = _get_coords(sub, coords)
        ann = sub.obs[annotation].astype(str).values
        lab = labels[tp].reindex(sub.obs_names).astype(str).values
        for i, (vals, title) in enumerate([(ann, f"tp{tp} — annotation"),
                                           (lab, f"tp{tp} — Leiden")]):
            a = axes[i, j] if len(tps) > 1 else axes[i]
            for k, g in enumerate(sorted(pd.unique(vals))):
                msk = vals == g
                a.scatter(xy[msk, 0], xy[msk, 1], s=4, alpha=.7,
                          label=g, color=sns.color_palette("tab20")[k % 20])
            a.set_title(title, fontsize=9)
            a.set_xticks([]); a.set_yticks([])
            a.legend(markerscale=2, fontsize=6, loc="best", frameon=False)
    fig.tight_layout()
    return fig


def fig_resolution(res_df, metrics=("ARI", "AMI", "purity", "V_measure"), ax=None):
    g = res_df.groupby("resolution")[list(metrics)].mean().reset_index()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for met in metrics:
        ax.plot(g["resolution"], g[met], marker="o", label=met)
    ax.axvline(0.5, ls="--", c="grey", lw=1, label="GTra default")
    ax.set_xlabel("Leiden resolution"); ax.set_ylabel("agreement (mean over tps)")
    ax.set_ylim(0, 1); ax.legend(fontsize=8)
    ax.set_title("Agreement vs clustering resolution")
    return ax
