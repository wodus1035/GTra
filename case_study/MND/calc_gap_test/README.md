# B3 — `_calc_gap` off-by-one: isolated test (NOT applied to source)

The production `_calc_gap` (src/gtra/cluster_func.py) returns `n - jump_idx`
clusters at the largest dendrogram gap. Cutting there actually leaves
`n - jump_idx - 1` clusters, so production over-counts by one (a clean
3-cluster dataset returns 4). This directory tests the corrected version
**without touching the source tree**, because `_calc_gap` drives gene-module
counts in `cc_clustering` and therefore every downstream result.

## Files
- `calc_gap_fixed.py` — `calc_gap_current` (verbatim production), `calc_gap_fixed`
  (the `-1` correction; min_k/max_k clamp identical), and `apply_fix()` which
  monkey-patches the fix across gtra.cluster_func / utils / visualize for a full
  pipeline re-run.
- `compare_on_consensus.py` — env-light (numpy+scipy). Replays
  `linkage -> _calc_gap -> fcluster` on the 7 saved MND per-cell_type consensus
  matrices (`../MND_stability_figs/consensus_matrices/*.npz`) under both versions.
- `calc_gap_comparison.csv` — full output.

## Result (MND, 27 cell_type x timepoint cases)

**All 27 cases change — every K drops by exactly 1** (3->2 or 4->3). The
min_k=2 clamp does NOT absorb it here: the largest gap is not at the final
merge, so `optimal_k > min_k` and the `-1` always bites. Concretely the fix
merges the smallest module into the largest, e.g.:

```
1-Neurons tp11: K 3->2  [1070, 474, 456] -> [1544, 456]
3-APs_RPs tp13: K 3->2  [1204, 458, 338] -> [1662,  338]
2-Young   tp13: K 4->3  [929, 436, 365, 270] -> [1365, 365, 270]
```

## Result (HSPC, end-to-end on the real object)

`compare_hspc_endtoend.py` loads `case_study/HSPC/hspc_full.dill` and re-runs
`cc_clustering` on its stored consensus matrices under both versions
(gtra_test env). Sanity: the current recompute reproduces the stored
gene_label_info exactly (81 modules), confirming the test is faithful.

```
total gene modules: 81 (current)  ->  57 (fixed)   = -24  (-29.6%)
per timepoint: t0 22->16  t1 21->15  t2 20->14  t3 18->12   (every tp loses one module per cell cluster)
```

So the "fix" deletes ~30% of HSPC gene modules globally — consistent with the
MND finding (one fewer module per cell cluster everywhere).

## Interpretation / recommendation

This is an off-by-one in the *formula*, but in effect it is a **granularity
change**, not a silent correctness bug with one obvious answer:

- The current (finer) modules are what the manuscript reports AND what the
  bootstrap-stability rebuttal (REVISION/01_gene_stability) validated as
  reproducible. Switching to the coarser cut would invalidate/require re-running
  all cell-state / GED / benchmarking / stability results.
- "Textbook-correct gap cut" (fixed) is not necessarily better biology; it just
  collapses the smallest module each time.

=> Keep production as-is for the submission. Only adopt `calc_gap_fixed` if we
deliberately decide to re-run the entire pipeline at the coarser resolution and
re-validate. To trial it end-to-end (HSPC):

```python
from calc_gap_fixed import apply_fix
restore = apply_fix()      # patch before running find_gclusters()/cc_clustering
# ... run pipeline ...
restore()                  # revert
```
