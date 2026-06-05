"""
Full bootstrap-N sweep on SIMULATION data, paper-figure grade.

Grid: topology {linear, bifurcation, cyclic} x mode {labeled, label-free}
      x N {1,2,3,5,10,20,30,50,75,100} x trials (default 10).

Each cell has a KNOWN ground-truth answer graph (self-loops + true transitions).
- labeled  : GTra uses the true cell_type as the cell label.
- label-free: GTra clusters cells with Leiden (label_flag=False); the data-driven
  states are mapped to true cell types by majority vote ONLY at scoring time
  (the true annotation is injected into obs post-hoc, never used by GTra).
All runs are answer-UNCONSTRAINED (answer_path_dir="").

Results are checkpointed row-by-row to nsweep_full_results.csv (resumable-ish:
already-completed (topology,mode,N,trial) rows are skipped on restart).

Usage:
  python sim_nsweep_full.py --probe   # 2 timing runs, no full sweep
  python sim_nsweep_full.py           # full grid
  python sim_nsweep_full.py --trials 5
"""

import os
import sys
import time
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

# ---- config ------------------------------------------------------------- #
DATA_SEED = 0
N_TIMEPOINTS = 6
REPS_PER_TIME = 3
CELLS_PER_SAMPLE = 150
N_GRID = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
DEFAULT_TRIALS = 10
MODES = ["labeled"]   # label-free dropped: degenerate on simulation data (Leiden
                       # over/under-clusters the few simulated types). The label-value
                       # argument is carried by the REAL-data GED workstream instead.
LABELFREE_RES = 0.3   # cn_cluster_resolution for label-free Leiden (matches sim notebooks)

# topology -> (cell_types, transition edges); answer = self-loops + transitions
TOPOS = {
    "linear":      (["CT0", "CT1", "CT2"],
                    [("CT0", "CT1"), ("CT1", "CT2")]),
    "bifurcation": (["Prog", "FateA", "FateB"],
                    [("Prog", "FateA"), ("Prog", "FateB")]),
    "cyclic":      (["Phase0", "Phase1", "Phase2"],
                    [("Phase0", "Phase1"), ("Phase1", "Phase2"), ("Phase2", "Phase0")]),
}
RESULTS_CSV = os.path.join(HERE, "nsweep_full_results.csv")


def answer_graph(topology):
    cts, trans = TOPOS[topology]
    g = nx.DiGraph()
    for ct in cts:
        g.add_node(ct); g.add_edge(ct, ct)        # self-loop
    for s, t in trans:
        g.add_edge(s, t)
    return g


def make_sim(topology):
    """Fixed dataset -> ({tp: counts_cellxgene}, full cell_id->cell_type meta)."""
    counts_df, cell_meta, *_ = simulate_timeseries_topology(
        topology=topology, n_timepoints=N_TIMEPOINTS, reps_per_time=REPS_PER_TIME,
        cells_per_sample=CELLS_PER_SAMPLE, seed=DATA_SEED, bifurcation_time=0.4,
    )
    cm = cell_meta.set_index("cell_id")
    X = counts_df.T.loc[cm.index]
    per_tp = {}
    for tp in sorted(cm["time_idx"].unique()):
        ids = cm.index[cm["time_idx"] == tp]
        per_tp[int(tp)] = X.loc[ids].copy()
    full_ct = cm["cell_type"]
    return per_tp, full_ct


def run_once(per_tp, full_ct, topology, mode, N, trial):
    random.seed(1000 * trial + N)
    np.random.seed(1000 * trial + N)

    obj = gtra.GTraObject()
    obj.params.output_dir = os.path.join(HERE, "_scratch")
    obj.params.output_name = f"{topology}_{mode}_N{N}_t{trial}"
    obj.params.answer_path_dir = ""

    if mode == "labeled":
        obj.params.cell_type_label = "cell_type"
        for tp in sorted(per_tp):
            meta = full_ct.reindex(per_tp[tp].index).to_frame("cell_type")
            obj.upload_time_scRNA(per_tp[tp], meta)
    else:  # label-free
        obj.params.cn_cluster_resolution = LABELFREE_RES
        for tp in sorted(per_tp):
            obj.upload_time_scRNA(per_tp[tp])

    obj.find_gclusters(N=N)
    obj.construct_trajectories()

    # for label-free, inject the true annotation for majority-mapped scoring
    if mode != "labeled":
        for tp in range(obj.tp_data_num):
            obs = obj.tp_data_dict[tp].obs
            obs["cell_type"] = full_ct.reindex(obs.index).values

    G_pred = G.state_graph(obj, annot_col="cell_type")
    G_ref = answer_graph(topology)
    ov = G.edge_prf(G_pred, G_ref)
    tr = G.edge_prf(G_pred, G_ref, ignore_selfloops=True)
    ged = G.graph_edit_distance(G_pred, G_ref)
    nged = G.normalized_ged(ged, G_pred, G_ref)
    return {
        "topology": topology, "mode": mode, "N": N, "trial": trial,
        "n_pred_edges": G_pred.number_of_edges(),
        "overall_f1": ov["f1"], "trans_f1": tr["f1"],
        "trans_prec": tr["precision"], "trans_rec": tr["recall"],
        "ged": ged, "norm_ged": nged,
    }


def _done_set():
    if not os.path.isfile(RESULTS_CSV):
        return set(), []
    df = pd.read_csv(RESULTS_CSV)
    done = set(zip(df["topology"], df["mode"], df["N"], df["trial"]))
    return done, df.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    args = ap.parse_args()

    sims = {topo: make_sim(topo) for topo in TOPOS}

    if args.probe:
        for mode in MODES:
            t0 = time.time()
            r = run_once(*sims["bifurcation"], "bifurcation", mode, 100, 0)
            print(f"PROBE {mode:10s} N=100: {time.time()-t0:5.1f}s  "
                  f"trans_f1={r['trans_f1']:.3f} edges={r['n_pred_edges']}")
        return

    done, rows = _done_set()
    cols = ["topology", "mode", "N", "trial", "n_pred_edges",
            "overall_f1", "trans_f1", "trans_prec", "trans_rec", "ged", "norm_ged"]
    total = len(TOPOS) * len(MODES) * len(N_GRID) * args.trials
    i = len(done)
    t_start = time.time()
    for topo in TOPOS:
        for mode in MODES:
            for N in N_GRID:
                for trial in range(args.trials):
                    if (topo, mode, N, trial) in done:
                        continue
                    i += 1
                    try:
                        r = run_once(*sims[topo], topo, mode, N, trial)
                    except Exception as e:
                        r = {"topology": topo, "mode": mode, "N": N, "trial": trial,
                             "n_pred_edges": np.nan, "overall_f1": np.nan, "trans_f1": np.nan,
                             "trans_prec": np.nan, "trans_rec": np.nan, "ged": np.nan, "norm_ged": np.nan}
                        print(f"[{i}/{total}] {topo}/{mode} N={N} t={trial} FAILED {type(e).__name__}: {str(e)[:70]}")
                    rows.append(r)
                    pd.DataFrame(rows)[cols].to_csv(RESULTS_CSV, index=False)  # checkpoint
                    el = time.time() - t_start
                    print(f"[{i}/{total}] {topo:11s}/{mode:10s} N={N:3d} t={trial} "
                          f"trans_f1={r.get('trans_f1', float('nan')):.3f} "
                          f"({el/60:.1f}min elapsed)")

    summarize_and_plot()


def summarize_and_plot():
    df = pd.read_csv(RESULTS_CSV)
    agg = df.groupby(["topology", "mode", "N"]).agg(
        trans_f1_mean=("trans_f1", "mean"), trans_f1_std=("trans_f1", "std"),
        overall_f1_mean=("overall_f1", "mean"), overall_f1_std=("overall_f1", "std"),
        ged_mean=("ged", "mean"), ged_std=("ged", "std"),
        norm_ged_mean=("norm_ged", "mean"), norm_ged_std=("norm_ged", "std"),
    ).reset_index()
    agg.to_csv(os.path.join(HERE, "nsweep_full_summary.csv"), index=False)
    print("\n=== summary ===")
    print(agg.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    topos = list(TOPOS)
    metrics = [("trans_f1", "transition F1", (0, 1.02)),
               ("overall_f1", "overall F1", (0, 1.02)),
               ("norm_ged", "normalized GED (lower better)", None)]
    mode_style = {"labeled": ("#E15759", "-o"), "label_free": ("#4E79A7", "-s")}

    fig, axes = plt.subplots(len(metrics), len(topos),
                             figsize=(4.2 * len(topos), 3.4 * len(metrics)), squeeze=False)
    for ci, topo in enumerate(topos):
        rand = G.random_transition_f1(answer_graph(topo), answer_graph(topo), n=2000)
        for ri, (col, lab, ylim) in enumerate(metrics):
            ax = axes[ri][ci]
            for mode, (c, ls) in mode_style.items():
                sub = agg[(agg.topology == topo) & (agg["mode"] == mode)].sort_values("N")
                if sub.empty:
                    continue
                m = sub[f"{col}_mean"].values; s = sub[f"{col}_std"].fillna(0).values
                ax.plot(sub["N"], m, ls, color=c, label=mode, ms=4)
                ax.fill_between(sub["N"], m - s, m + s, color=c, alpha=0.15)
            if col == "trans_f1":
                ax.axhline(rand, ls="--", color="gray", lw=1, label=f"random ({rand:.2f})")
            if ylim:
                ax.set_ylim(*ylim)
            if ri == 0:
                ax.set_title(topo, fontsize=12, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(lab, fontsize=9)
            if ri == len(metrics) - 1:
                ax.set_xlabel("bootstrap N")
            ax.grid(alpha=0.3)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "nsweep_full_curves.pdf"))
    print("\nSaved -> nsweep_full_results.csv, nsweep_full_summary.csv, nsweep_full_curves.pdf")


if __name__ == "__main__":
    main()
