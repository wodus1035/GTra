"""
B3 fix candidate for GTra's `_calc_gap` (cluster-count off-by-one).

The production version (src/gtra/cluster_func.py) computes:
    optimal_k = n - jump_idx
where `jump_idx = argmax(diff(merge_distances))`. Cutting the dendrogram at
that largest gap leaves `n - (jump_idx + 1)` clusters, so the production value
over-counts by one (verified: a clean 3-cluster dataset returns 4).

`calc_gap_current` reproduces the production logic verbatim; `calc_gap_fixed`
applies the `-1` correction. Everything else (min_k / max_k clamping) is kept
identical so the ONLY difference under test is the off-by-one.

`apply_fix()` monkey-patches the fixed version into the live gtra package
(cluster_func + the copies imported into utils and visualize) so a full
pipeline can be re-run with the fix WITHOUT editing the source tree.
"""

import numpy as np


def calc_gap_current(linked, min_k=2, max_k=None):
    """Verbatim copy of the current production _calc_gap."""
    d = linked[:, 2]
    delta_d = np.diff(d)
    jump_idx = np.argmax(delta_d)

    n = linked.shape[0] + 1
    optimal_k = n - jump_idx

    if max_k is None:
        max_k = n // 2
    optimal_k = max(min_k, min(optimal_k, max_k))
    return optimal_k


def calc_gap_fixed(linked, min_k=2, max_k=None):
    """Off-by-one corrected: cutting at the largest gap leaves n-jump_idx-1 clusters."""
    d = linked[:, 2]
    delta_d = np.diff(d)
    jump_idx = np.argmax(delta_d)

    n = linked.shape[0] + 1
    optimal_k = n - jump_idx - 1          # <-- the only change vs current

    if max_k is None:
        max_k = n // 2
    optimal_k = max(min_k, min(optimal_k, max_k))
    return optimal_k


def apply_fix():
    """
    Patch calc_gap_fixed over every module that holds a reference to _calc_gap.

    `_calc_gap` is imported by name into gtra.utils and gtra.visualize
    (`from .cluster_func import _calc_gap`), so each module keeps its own
    binding; patching only gtra.cluster_func would miss those. Returns a
    restore() callable.
    """
    import gtra.cluster_func as cf
    import gtra.utils as gu
    import gtra.visualize as gv

    originals = {}
    for mod in (cf, gu, gv):
        if hasattr(mod, "_calc_gap"):
            originals[mod] = mod._calc_gap
            mod._calc_gap = calc_gap_fixed

    def restore():
        for mod, fn in originals.items():
            mod._calc_gap = fn

    return restore
