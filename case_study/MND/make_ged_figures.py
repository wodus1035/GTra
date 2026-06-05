"""Combine GED results (MND/HSPC/COVID) into the labeled-vs-unlabeled figure.

Reads:
  MND_ged_out/ged_summary.csv, HSPC_ged_out/ged_summary.csv   (run_ged.py)
  ../COVID/COVID_ged_figs/covid_ged_full.csv                  (covid_ged_full.py)
"""
import warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
FIG = Path("GED_figs"); FIG.mkdir(exist_ok=True)
METRICS = ["GED", "f1", "trans_f1", "trans_recall", "trans_precision", "rand_trans_f1"]


def load_all():
    rows = []
    for ds, path in [("MND", "MND_ged_out/ged_summary.csv"),
                     ("HSPC", "HSPC_ged_out/ged_summary.csv")]:
        df = pd.read_csv(path, index_col=0)  # index = mode
        for mode, r in df.iterrows():
            rows.append({"dataset": ds, "mode": mode,
                         **{m: r[m] for m in METRICS if m in r}})
    cov = pd.read_csv("../COVID/COVID_ged_figs/covid_ged_full.csv")
    for mode, g in cov.groupby("mode"):
        rows.append({"dataset": "COVID", "mode": mode,
                     **{m: g[m].mean() for m in METRICS if m in g}})
    return pd.DataFrame(rows)


def main():
    df = load_all()
    df.to_csv(FIG / "ged_combined.csv", index=False)
    print(df.round(3).to_string(index=False))

    # transition-F1: labeled vs unlabeled, with random baseline markers
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, met, ttl in [(axes[0], "trans_f1", "Transition-edge F1 (off-diagonal)"),
                         (axes[1], "GED", "Graph edit distance (lower=better)")]:
        sns.barplot(data=df, x="dataset", y=met, hue="mode",
                    order=["MND", "HSPC", "COVID"], hue_order=["labeled", "unlabeled"], ax=ax)
        ax.set_title(ttl); ax.set_xlabel("")
        if met == "trans_f1":
            # overlay random baseline per dataset (mean of the two modes)
            rb = df.groupby("dataset")["rand_trans_f1"].mean()
            for i, ds in enumerate(["MND", "HSPC", "COVID"]):
                if ds in rb:
                    ax.hlines(rb[ds], i - 0.4, i + 0.4, colors="red", ls="--", lw=1.5)
            ax.plot([], [], "r--", label="random baseline")
            ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "GED_labeled_vs_unlabeled.pdf", dpi=200); plt.close(fig)
    print("\nsaved -> GED_figs/GED_labeled_vs_unlabeled.pdf")


if __name__ == "__main__":
    main()
