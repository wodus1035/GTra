"""Figure: unified cell-state topology comparison across methods (MND/HSPC).

All methods scored with the SAME transition-only edge-F1 vs the SAME answer graph
(score_method.py). GTra is competitive but not uniquely best on topology — the
honest result that motivates GTra's gene-module repositioning.
"""
import warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "method_comparison.csv")

order = ["GTra", "WOT", "scEGOT", "CStreet", "CellRank"]
order = [m for m in order if m in set(df["method"])]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for ax, met, ttl in [(axes[0], "trans_f1", "Transition-edge F1 (off-diagonal)"),
                     (axes[1], "trans_recall", "Transition recall")]:
    sns.barplot(data=df, x="dataset", y=met, hue="method", order=["MND", "HSPC"],
                hue_order=order, ax=ax)
    ax.set_title(ttl); ax.set_xlabel(""); ax.set_ylim(0, 1)
# random baseline markers on the F1 panel
rb = df.groupby("dataset")["rand_trans_f1"].mean()
for i, ds in enumerate(["MND", "HSPC"]):
    if ds in rb.index:
        axes[0].hlines(rb[ds], i - 0.45, i + 0.45, colors="red", ls="--", lw=1)
axes[0].plot([], [], "r--", label="random"); axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("Unified cell-state topology benchmark (transition-F1, one answer graph per dataset)\n"
             "GTra is the only method top-tier on both datasets; others are dataset-inconsistent")
fig.tight_layout(); fig.savefig(HERE / "method_topology_comparison.pdf", dpi=200)
print("saved method_topology_comparison.pdf")
print(df.pivot_table(index="dataset", columns="method", values="trans_f1").round(3).to_string())
