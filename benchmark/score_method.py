"""
Unified scoring harness for the GTra method benchmark (ISMB review meta/R1.1/R3.3).

Every method (GTra, scEGOT, CStreet, WOT, CellRank, GeneTrajectory, ...) is asked
to emit its predicted CELL-STATE TRANSITION GRAPH as a common CSV:

    edges/{method}_{dataset}.csv   with columns:  source, target, score

where `source`/`target` are CELL-TYPE names in the dataset's answer-graph space
(map data-driven clusters to cell-types by majority annotation before writing).

This module scores any such file the SAME way GTra is scored — transition-only
(off-diagonal) edge precision/recall/F1 plus GED — against the dataset's answer
graph, so all methods sit on one consistent axis.

Run:  python score_method.py            # scores everything under ./edges/
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

# reuse the GED metric utilities from the case study
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "case_study" / "MND"))
import ged_utils as ge  # noqa: E402

ANSWERS = {
    "MND": "../answer_paths/MND_answer.csv",
    "HSPC": "../answer_paths/HSPC_answer.csv",
    "COVID": "../answer_paths/COVID_answer.csv",
}
# MND uses the FINE cell_type2 annotation (7 states); only fix a typo in the answer.
MND_ANSWER_FIX = {"2-Young enurons": "2-Young neurons"}
EDGE_DIR = Path(__file__).resolve().parent / "edges"


def _answer(dataset):
    base = Path(__file__).resolve().parent
    df = pd.read_csv(base / ANSWERS[dataset])
    fix = MND_ANSWER_FIX if dataset == "MND" else {}
    G = nx.DiGraph()
    for _, r in df.iterrows():
        s, t = str(r["source"]).strip(), str(r["target"]).strip()
        s, t = fix.get(s, s), fix.get(t, t)
        G.add_node(s); G.add_node(t); G.add_edge(s, t)
    return G


def _pred_graph(csv, dataset=None):
    df = pd.read_csv(csv)
    fix = MND_ANSWER_FIX if dataset == "MND" else {}
    G = nx.DiGraph()
    for _, r in df.iterrows():
        s, t = str(r["source"]).strip(), str(r["target"]).strip()
        s, t = fix.get(s, s), fix.get(t, t)
        G.add_node(s); G.add_node(t); G.add_edge(s, t)
    return G


def score_file(csv):
    """csv name must be {method}_{dataset}.csv. Returns metrics dict."""
    name = Path(csv).stem
    method, dataset = name.rsplit("_", 1)
    Gref = _answer(dataset)
    G = _pred_graph(csv, dataset)
    prf = ge.edge_prf(G, Gref)
    prfT = ge.edge_prf(G, Gref, ignore_selfloops=True)
    return {
        "method": method, "dataset": dataset,
        "trans_f1": prfT["f1"], "trans_recall": prfT["recall"],
        "trans_precision": prfT["precision"],
        "rand_trans_f1": ge.random_transition_f1(G, Gref),
        "f1": prf["f1"], "GED": ge.graph_edit_distance(G, Gref, timeout=30),
        "n_pred": prf["n_pred"], "n_ref": prf["n_ref"],
    }


def main():
    EDGE_DIR.mkdir(exist_ok=True)
    files = sorted(glob.glob(str(EDGE_DIR / "*.csv")))
    if not files:
        print(f"No edge files in {EDGE_DIR}. Each method should write "
              f"{{method}}_{{dataset}}.csv with columns source,target,score.")
        return
    rows = [score_file(f) for f in files]
    df = pd.DataFrame(rows).sort_values(["dataset", "trans_f1"], ascending=[True, False])
    out = Path(__file__).resolve().parent / "method_comparison.csv"
    df.to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
