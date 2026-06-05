"""
COVID concordance v3 — robust small-n statistics (reinforces workstream 07).

n=7 patients makes the Mann-Whitney DP-vs-RP slope test underpowered and unstable.
This recomputes the evidence from the already-saved per-patient slopes
(CONCORDANCE_figs/concordance_v2.csv) with tests that use ALL patients and are
exact for small n:

  (1) direction-concordance: each patient's module-score trend should match its
      phenotype (DP -> increasing, beta>0; RP -> decreasing, beta<0). Exact
      one-sided binomial test (H0: p=0.5).
  (2) exact permutation of the DP-vs-RP slope difference (all label splits).

Both the GTra-pattern module and the DEG baseline are scored identically, so the
comparison is apples-to-apples (no circular construction advantage).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "CONCORDANCE_figs")
CSV = os.path.join(FIGDIR, "concordance_v2.csv")


def exact_perm_greater(dp, rp):
    allv = np.array(list(dp) + list(rp)); n = len(allv); k = len(dp)
    obs = np.median(dp) - np.median(rp)
    cnt = tot = 0
    for idx in combinations(range(n), k):
        m = np.zeros(n, bool); m[list(idx)] = True
        if np.median(allv[m]) - np.median(allv[~m]) >= obs - 1e-12:
            cnt += 1
        tot += 1
    return obs, cnt / tot


def main():
    df = pd.read_csv(CSV)
    stats = []
    for mod in ["GTra", "DEG"]:
        d = df[df.module == mod]
        dp = d[d.pheno == "DP"]["beta"].values
        rp = d[d.pheno == "RP"]["beta"].values
        conc = int(sum(b > 0 for b in dp) + sum(b < 0 for b in rp))
        ntot = len(dp) + len(rp)
        bt = binomtest(conc, ntot, 0.5, alternative="greater").pvalue
        obs, pp = exact_perm_greater(dp, rp)
        stats.append({"module": mod, "n_patients": ntot,
                      "direction_concordant": f"{conc}/{ntot}",
                      "binom_p": round(bt, 4),
                      "median_DP_minus_RP": round(obs, 4),
                      "exact_perm_p": round(pp, 4)})
        print(f"[{mod}] {conc}/{ntot} concordant  binom p={bt:.3f} | "
              f"median DP-RP={obs:.3f}  exact-perm p={pp:.3f}")
    sdf = pd.DataFrame(stats)
    sdf.to_csv(os.path.join(FIGDIR, "concordance_v3_stats.csv"), index=False)

    # ---- figure: per-patient slope by phenotype, GTra vs DEG ----
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=False)
    for ax, mod in zip(axes, ["GTra", "DEG"]):
        d = df[df.module == mod]
        for _, r in d.iterrows():
            x = 0 if r.pheno == "DP" else 1
            mk = "^" if r.pheno == "DP" else "v"
            col = "#E15759" if r.pheno == "DP" else "#4E79A7"
            ax.scatter(x + np.random.uniform(-0.06, 0.06), r.beta,
                       marker=mk, s=90, color=col, edgecolor="black", lw=0.5, zorder=3)
        ax.axhline(0, color="gray", lw=1, ls="--")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["DP (expect ↑)", "RP (expect ↓)"])
        srow = sdf[sdf.module == mod].iloc[0]
        ax.set_title(f"{mod}\n{srow.direction_concordant} concordant "
                     f"(binom p={srow.binom_p})", fontsize=10)
        ax.set_ylabel("module-score slope β (per patient)")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-patient module trend vs phenotype (n=7, underpowered)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "CONC3_direction_concordance.pdf"))
    print("Saved -> CONC3_direction_concordance.pdf, concordance_v3_stats.csv")


if __name__ == "__main__":
    np.random.seed(0)
    main()
