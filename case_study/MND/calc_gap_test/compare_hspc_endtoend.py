"""
End-to-end impact of the B3 `_calc_gap` fix on the REAL HSPC object
(case_study/HSPC/hspc_full.dill), using the consensus matrices it already
carries --- so no bootstrap re-run.

For both the current and the fixed `_calc_gap` we re-run `cc_clustering`
(which is `linkage -> _calc_gap -> fcluster` per time x cell-cluster) on a
deepcopy of the loaded object and count the resulting gene modules.
The fix is applied ONLY via monkey-patch (apply_fix); the source is untouched.

Run with the gtra_test env:
  /data1/home/jyj/miniconda3/envs/gtra_test/bin/python compare_hspc_endtoend.py
"""

import os
import sys
import copy

import dill

sys.path.insert(0, os.path.dirname(__file__))
from calc_gap_fixed import apply_fix

DILL = os.path.join(os.path.dirname(__file__), "..", "..", "HSPC", "hspc_full.dill")
OUT = os.path.join(os.path.dirname(__file__), "hspc_endtoend_result.txt")


def module_counts(obj):
    """Return (total_modules, {time: [modules-per-cellcluster]})."""
    from gtra.cluster_func import cc_clustering
    cc_clustering(obj)  # writes obj.gene_label_info
    per_time = {}
    total = 0
    for t, clusters in obj.gene_label_info.items():
        sizes = [len(mods) for mods in clusters]  # n gene-modules per cell cluster
        per_time[t] = sizes
        total += sum(sizes)
    return total, per_time


def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    log(f"Loading {os.path.abspath(DILL)} ...")
    with open(DILL, "rb") as f:
        base = dill.load(f)
    log(f"tp_data_num={base.tp_data_num}, genes={len(base.genes)}")

    # stored (paper) result already in the object
    stored_total = sum(
        sum(len(m) for m in clusters)
        for clusters in base.gene_label_info.values()
    )
    log(f"stored gene_label_info total modules = {stored_total}")

    # --- current (recomputed from stored ccmatrix) ---
    obj_cur = copy.deepcopy(base)
    cur_total, cur_pt = module_counts(obj_cur)
    log(f"\n[current _calc_gap] total modules = {cur_total}")
    log(f"  (sanity vs stored: {'MATCH' if cur_total == stored_total else 'DIFF'})")

    # --- fixed (monkey-patched) ---
    obj_fix = copy.deepcopy(base)
    restore = apply_fix()
    try:
        fix_total, fix_pt = module_counts(obj_fix)
    finally:
        restore()
    log(f"[fixed   _calc_gap] total modules = {fix_total}")

    log(f"\nDELTA total modules: {cur_total} -> {fix_total} "
        f"({fix_total - cur_total:+d}, "
        f"{100*(fix_total-cur_total)/cur_total:+.1f}%)")

    log("\nPer-timepoint module counts (sum over cell clusters): current -> fixed")
    for t in sorted(cur_pt):
        c = sum(cur_pt[t]); fx = sum(fix_pt.get(t, []))
        flag = "" if c == fx else "  <-- changed"
        log(f"  t{t}: {c} -> {fx}{flag}")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
