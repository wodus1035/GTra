"""
Bootstrap-N sweep on SIMULATION data with a known ground-truth answer graph.

How does GTra's recovered cell-state trajectory (transition-F1 / GED vs the true
graph) change with the bootstrap count N in find_gclusters(N)?

Data: one FIXED bifurcation dataset (sim.simulate_timeseries_topology), labeled
run (true cell_type). Ground truth: Prog->FateA, Prog->FateB (+ self-loops).
For each N we run several independent trials (bootstrap is stochastic) and
aggregate. Runs are answer-UNCONSTRAINED (answer_path_dir="") so scoring is fair.

Usage:
  python sim_nsweep.py            # full sweep
  python sim_nsweep.py --smoke    # quick check (N=[3], 1 trial)
"""

import os
import sys
import argparse
import random

import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "simulation"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "case_study", "MND"))

import gtra
import ged_utils as G
from sim import simulate_timeseries_topology

# ---- experiment config -------------------------------------------------- #
DATA_SEED = 0
N_TIMEPOINTS = 6
REPS_PER_TIME = 3
CELLS_PER_SAMPLE = 150
N_GRID = [1, 2, 3, 5, 10, 20, 30, 50]
N_TRIALS = 5
CELL_TYPES = ["Prog", "FateA", "FateB"]
# ground-truth answer graph (self-loops + the two true transitions)
ANSWER_EDGES = [("Prog", "Prog"), ("FateA", "FateA"), ("FateB", "FateB"),
                ("Prog", "FateA"), ("Prog", "FateB")]


def build_answer_graph():
    g = nx.DiGraph()
    for ct in CELL_TYPES:
        g.add_node(ct)
    for s, t in ANSWER_EDGES:
        g.add_edge(s, t)
    return g


def make_sim():
    """One fixed bifurcation dataset -> {time_idx: (counts_cellxgene, cell_type meta)}."""
    counts_df, cell_meta, _sm, _gm, _pb, _comp = simulate_timeseries_topology(
        topology="bifurcation",
        n_timepoints=N_TIMEPOINTS,
        reps_per_time=REPS_PER_TIME,
        cells_per_sample=CELLS_PER_SAMPLE,
        seed=DATA_SEED,
        bifurcation_time=0.4,
    )
    cm = cell_meta.set_index("cell_id")
    X = counts_df.T  # cell x gene
    X = X.loc[cm.index]
    per_tp = {}
    for tp in sorted(cm["time_idx"].unique()):
        ids = cm.index[cm["time_idx"] == tp]
        meta = cm.loc[ids, ["cell_type"]].copy()
        per_tp[int(tp)] = (X.loc[ids].copy(), meta)
    return per_tp


def run_once(per_tp, N, trial):
    """Build a fresh labeled GTra object, run to trajectories, score vs answer."""
    # vary the (host-side) RNG between trials; loky workers also reseed themselves
    random.seed(1000 * trial + N)
    np.random.seed(1000 * trial + N)

    obj = gtra.GTraObject()
    obj.params.cell_type_label = "cell_type"
    obj.params.output_dir = os.path.join(HERE, "_scratch")
    obj.params.output_name = f"sim_N{N}_t{trial}"
    obj.params.answer_path_dir = ""          # answer-UNCONSTRAINED (fair scoring)
    for tp in sorted(per_tp):
        cnt, meta = per_tp[tp]
        obj.upload_time_scRNA(cnt, meta)

    obj.find_gclusters(N=N)
    obj.construct_trajectories()

    G_pred = G.state_graph(obj, annot_col="cell_type")
    G_ref = build_answer_graph()
    ov = G.edge_prf(G_pred, G_ref)
    tr = G.edge_prf(G_pred, G_ref, ignore_selfloops=True)
    ged = G.graph_edit_distance(G_pred, G_ref)
    nged = G.normalized_ged(ged, G_pred, G_ref)
    return {
        "N": N, "trial": trial,
        "n_pred_edges": G_pred.number_of_edges(),
        "overall_f1": ov["f1"], "overall_prec": ov["precision"], "overall_rec": ov["recall"],
        "trans_f1": tr["f1"], "trans_prec": tr["precision"], "trans_rec": tr["recall"],
        "ged": ged, "norm_ged": nged,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    n_grid = [3] if args.smoke else N_GRID
    n_trials = 1 if args.smoke else N_TRIALS

    per_tp = make_sim()
    print(f"sim: {len(per_tp)} timepoints; "
          f"cells/tp ~ {[v[0].shape[0] for v in per_tp.values()]}; "
          f"genes={list(per_tp.values())[0][0].shape[1]}")
    # random baseline (depends only on the answer graph + a typical pred size)
    rand_tr = G.random_transition_f1(build_answer_graph(), build_answer_graph(), n=2000)

    rows = []
    for N in n_grid:
        for t in range(n_trials):
            try:
                r = run_once(per_tp, N, t)
                rows.append(r)
                print(f"  N={N:3d} trial={t}: trans_f1={r['trans_f1']:.3f} "
                      f"overall_f1={r['overall_f1']:.3f} ged={r['ged']:.0f}")
            except Exception as e:
                print(f"  N={N:3d} trial={t}: FAILED {type(e).__name__}: {str(e)[:90]}")
                rows.append({"N": N, "trial": t, "overall_f1": np.nan,
                             "trans_f1": np.nan, "ged": np.nan, "norm_ged": np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "nsweep_results.csv"), index=False)

    agg = df.groupby("N").agg(
        trans_f1_mean=("trans_f1", "mean"), trans_f1_std=("trans_f1", "std"),
        overall_f1_mean=("overall_f1", "mean"), overall_f1_std=("overall_f1", "std"),
        ged_mean=("ged", "mean"), ged_std=("ged", "std"),
        norm_ged_mean=("norm_ged", "mean"), norm_ged_std=("norm_ged", "std"),
    ).reset_index()
    agg["random_trans_f1"] = rand_tr
    agg.to_csv(os.path.join(HERE, "nsweep_summary.csv"), index=False)
    print("\n=== summary (mean over trials) ===")
    print(agg.to_string(index=False))
    print(f"\nrandom transition-F1 baseline: {rand_tr:.3f}")

    _plot(agg, rand_tr)
    print(f"\nSaved -> nsweep_results.csv, nsweep_summary.csv, nsweep_curves.pdf")


def _plot(agg, rand_tr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = agg["N"].values

    ax = axes[0]
    for col, lab, c in [("trans_f1", "transition F1", "#E15759"),
                        ("overall_f1", "overall F1", "#4E79A7")]:
        m = agg[f"{col}_mean"].values
        s = agg[f"{col}_std"].fillna(0).values
        ax.plot(x, m, "-o", color=c, label=lab)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.18)
    ax.axhline(rand_tr, ls="--", color="gray", lw=1, label=f"random trans-F1 ({rand_tr:.2f})")
    ax.set_xlabel("bootstrap N"); ax.set_ylabel("F1"); ax.set_title("F1 vs bootstrap N")
    ax.set_ylim(0, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for col, lab, c in [("ged", "GED", "#59A14F"), ("norm_ged", "normalized GED", "#B07AA1")]:
        m = agg[f"{col}_mean"].values
        s = agg[f"{col}_std"].fillna(0).values
        ax.plot(x, m, "-o", color=c, label=lab)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.18)
    ax.set_xlabel("bootstrap N"); ax.set_ylabel("graph edit distance")
    ax.set_title("GED vs bootstrap N (lower = better)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "nsweep_curves.pdf"))


if __name__ == "__main__":
    main()
