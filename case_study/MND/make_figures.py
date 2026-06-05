"""Generate all stability supplementary figures + summary tables (MND)."""
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import stability_utils as su
import stability_figs as sf

FIG = Path("MND_stability_figs"); FIG.mkdir(exist_ok=True)
DPI = 200


def load_merged():
    resA = sf.load("stability_out_A/stability_runs.pkl")
    resB = sf.load("stability_out_B/stability_runs.pkl")
    res = dict(resA)
    res["regimes"] = {**resA["regimes"], **resB["regimes"]}
    return res


def covid_naive_pac():
    def _pac(M, lo=.1, hi=.9):
        iu = np.triu_indices_from(M, 1); v = M[iu]
        return float(((v > lo) & (v < hi)).mean())
    rows = []
    for f in sorted(glob.glob("../../../../covid_obj/*.dill")):
        import dill
        o = dill.load(open(f, "rb"))
        for tp in o.ccmatrix:
            for ct in o.ccmatrix[tp]:
                rows.append(_pac(o.ccmatrix[tp][ct]))
    return np.array(rows)


def main():
    res = load_merged()
    summary = sf.build_summary(res)
    summary.to_csv(FIG / "stability_summary.csv", index=False)

    # ---- F5 defence (mis-aligned vs corrected PAC) ----
    cov = covid_naive_pac()
    cov_df = pd.DataFrame({"source": "COVID stored\n(mis-aligned)", "PAC": cov})
    cor = summary[["regime", "PAC"]].copy()
    cor["source"] = cor["regime"].map({
        "annotation": "MND corrected\n(annotation)",
        "leiden": "MND corrected\n(leiden+match)"})
    pdf = pd.concat([cov_df, cor[["source", "PAC"]]], ignore_index=True)
    order = ["COVID stored\n(mis-aligned)", "MND corrected\n(annotation)",
             "MND corrected\n(leiden+match)"]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=pdf, x="source", y="PAC", order=order,
                palette=["#d9534f", "#5cb85c", "#5bc0de"], ax=ax)
    sns.stripplot(data=pdf, x="source", y="PAC", order=order, color="0.2",
                  size=2.5, alpha=.5, ax=ax)
    ax.set_ylabel("PAC  (lower = more stable)"); ax.set_xlabel("")
    ax.axhline(0.1, ls="--", c="grey", lw=1)
    ax.set_title("Apparent instability is a cluster-alignment artifact")
    fig.tight_layout(); fig.savefig(FIG / "F5_pac_defence.pdf", dpi=DPI); plt.close(fig)

    # ---- F1 consensus heatmaps (rasterized) ----
    tps = res["timepoints"]
    fig, axes = plt.subplots(1, len(tps), figsize=(4 * len(tps), 4))
    for ax, tp in zip(np.atleast_1d(axes), tps):
        cid = sorted(res["regimes"]["annotation"][tp]["runs"])[0]
        sf.fig_consensus_heatmap(res, "annotation", tp, cid, ax=ax)
    fig.tight_layout(); fig.savefig(FIG / "F1_consensus_heatmaps.pdf", dpi=DPI); plt.close(fig)

    # ---- F2 PAC bars ----
    fig, ax = plt.subplots(figsize=(6, 4)); sf.fig_pac_bars(summary, ax=ax)
    fig.tight_layout(); fig.savefig(FIG / "F2_pac_bars.pdf", dpi=DPI); plt.close(fig)

    # ---- F3 ARI violin ----
    fig, ax = plt.subplots(figsize=(6, 4))
    _, ari_df = sf.fig_ari_violin(res, "annotation", summary, ax=ax)
    fig.tight_layout(); fig.savefig(FIG / "F3_ari_violin.pdf", dpi=DPI); plt.close(fig)

    # ---- F4 module Jaccard & core ----
    fig, ax = plt.subplots(figsize=(6, 4)); sf.fig_jaccard_core(summary, ax=ax)
    fig.tight_layout(); fig.savefig(FIG / "F4_jaccard_core.pdf", dpi=DPI); plt.close(fig)

    # ---- F6 consensus-module reproducibility (B) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    _, rep = sf.fig_module_reproducibility(res, "annotation", ax=ax)
    rep.to_csv(FIG / "module_reproducibility.csv", index=False)
    fig.tight_layout(); fig.savefig(FIG / "F6_module_reproducibility.pdf", dpi=DPI); plt.close(fig)

    print("=== summary (mean by regime) ===")
    print(summary.groupby("regime")[["refK", "PAC", "ARI", "AMI",
          "module_jaccard", "core_frac"]].mean().round(3))
    print("COVID naive PAC: %.3f (n=%d)" % (cov.mean(), len(cov)))
    sw = np.average(rep.dropna()["mean_consensus"], weights=rep.dropna()["size"])
    print("B: size-weighted within-module consensus = %.3f" % sw)
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
