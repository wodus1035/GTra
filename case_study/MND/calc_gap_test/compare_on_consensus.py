"""
Quantify the impact of the `_calc_gap` off-by-one fix (B3) on MND gene-module
counts, using the saved per-cell_type consensus matrices --- no bootstrap
re-run, no GTra env required (numpy + scipy only).

For each (cell type, timepoint) it reproduces exactly what cc_clustering does:
    linked = linkage(C, "ward");  K = _calc_gap(linked);  fcluster(..., K)
under both the current and the fixed _calc_gap, and reports K and the resulting
module sizes side by side.

Run:  python compare_on_consensus.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

from calc_gap_fixed import calc_gap_current, calc_gap_fixed

CC_DIR = os.path.join(
    os.path.dirname(__file__), "..", "MND_stability_figs", "consensus_matrices"
)
OUT_CSV = os.path.join(os.path.dirname(__file__), "calc_gap_comparison.csv")


def module_sizes(linked, K):
    labels = fcluster(linked, K, criterion="maxclust")
    return sorted(np.bincount(labels)[1:].tolist(), reverse=True)


def main():
    files = sorted(glob.glob(os.path.join(CC_DIR, "*.npz")))
    if not files:
        raise SystemExit(f"No consensus npz found under {CC_DIR}")

    rows = []
    for f in files:
        cell_type = os.path.basename(f).replace("consensus_", "").replace(".npz", "")
        d = np.load(f, allow_pickle=True)
        tps = sorted(int(k[4:]) for k in d.keys() if k.startswith("C_tp"))
        for tp in tps:
            C = d[f"C_tp{tp}"].astype(float)
            # cc_clustering treats consensus-matrix rows as feature vectors
            linked = linkage(C, "ward")

            k_old = calc_gap_current(linked)
            k_new = calc_gap_fixed(linked)

            rows.append({
                "cell_type": cell_type,
                "tp": tp,
                "n_genes": C.shape[0],
                "K_current": k_old,
                "K_fixed": k_new,
                "changed": k_old != k_new,
                "sizes_current": module_sizes(linked, k_old),
                "sizes_fixed": module_sizes(linked, k_new),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    n_changed = int(df["changed"].sum())
    print(f"\n{len(df)} (cell_type, timepoint) cases; {n_changed} change under the fix\n")
    show = df[[
        "cell_type", "tp", "n_genes", "K_current", "K_fixed", "changed"
    ]]
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(show.to_string(index=False))

    if n_changed:
        print("\nCases that change (module sizes):")
        for _, r in df[df["changed"]].iterrows():
            print(f"  {r['cell_type']} tp{r['tp']}: "
                  f"K {r['K_current']}->{r['K_fixed']}  "
                  f"sizes {r['sizes_current']} -> {r['sizes_fixed']}")
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
