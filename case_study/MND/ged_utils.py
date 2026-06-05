"""
Trajectory-accuracy via Graph Edit Distance (GTra revision, workstream 3).

Goal: show GTra recovers the correct cell-state trajectory WITHOUT predefined
labels. We run GTra label-free (Leiden states) with NO answer-path constraint,
build the cell-state transition graph, map each data-driven state to a cell-type
by majority annotation, and compare to the reference (answer_path) graph via
Graph Edit Distance (and an interpretable edge precision/recall/F1).

A GTra trajectory node is named  t{timepoint}_{cellcluster}_{genecluster}.
The cell-state is (timepoint, cellcluster); we collapse gene-clusters and
aggregate transitions to the cell-type level so the prediction lives in the
same label space as the answer graph.

IMPORTANT: the answer-path constraint (`check_standard_path`) filters candidate
edges to the answer when `answer_path_dir` points at a file — using such a run
for GED would be circular. Always evaluate runs produced with answer_path_dir="".
"""

import re
import copy
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx


# --------------------------------------------------------------------------- #
# robustness patch for GTra's _score_distribution
# --------------------------------------------------------------------------- #
def patch_gtra():
    """Make GTra's internal `_score_distribution` robust.

    Under bootstrap subsampling, `add_annotation` filters rare cell types and
    re-indexes the survivors, so integer cluster ids are not stable across runs.
    The stock `_score_distribution` then KeyErrors when a score_dict cluster id
    is absent from the full-data re-clustering. We skip such (unresolvable)
    edges instead of crashing; since we run answer-UNCONSTRAINED, this only
    drops a few edges from the significance distribution.
    """
    import gtra.cluster_func as gcf

    def _robust_score_distribution(obj):
        ct_label_dict = {}
        for tp in range(obj.tp_data_num):
            gcf.cell_clustering(obj, tp)
            idx = obj.tp_data_dict[tp].obs.value_counts().index
            ct_label_dict[tp] = {str(i[1]): i[0] for i in idx}
        x = copy.deepcopy(obj.score_dict)
        rows = []
        for it in x.keys():
            for st, vals in x[it].items():
                tok = st.split("_")
                if obj.params.label_flag:
                    source = ct_label_dict.get(it, {}).get(tok[0])
                    target = ct_label_dict.get(it + 1, {}).get(tok[1])
                else:
                    source, target = f"t{it}_{tok[0]}", f"t{it+1}_{tok[1]}"
                if source is None or target is None:
                    continue
                for v in vals:
                    rows.append([it, source, target, v])
        return pd.DataFrame(rows, columns=["Interval", "source", "target", "score"])

    gcf._score_distribution = _robust_score_distribution


# --------------------------------------------------------------------------- #
# cluster -> cell-type mapping (per timepoint, by majority annotation)
# --------------------------------------------------------------------------- #
def cluster_to_celltype(obj, annot_col, cluster_col="cluster_label"):
    """Return {(tp, cluster_id): celltype} by majority vote of annotation.

    Works for both label-free runs (cluster_col holds Leiden ids) and labeled
    runs (cluster_col holds annotation-derived ids).
    """
    mapping = {}
    for tp in range(obj.tp_data_num):
        obs = obj.tp_data_dict[tp].obs
        if cluster_col not in obs or annot_col not in obs:
            continue
        for cc, grp in obs.groupby(cluster_col, observed=True):
            maj = Counter(grp[annot_col].astype(str)).most_common(1)[0][0]
            mapping[(tp, int(cc))] = maj
    return mapping


def _parse_node(name):
    """t{tp}_{cc}_{gc} -> (tp, cc)."""
    tp, cc, _gc = map(int, re.findall(r"\d+", name))
    return tp, cc


# --------------------------------------------------------------------------- #
# build the predicted cell-type transition graph
# --------------------------------------------------------------------------- #
def state_graph(obj, annot_col, cluster_col="cluster_label"):
    """Cell-type transition DiGraph from obj.node_info.

    Edges (tp,cc)->(tp+1,cc') are mapped to celltype->celltype and aggregated
    across timepoints (self-loops kept; the answer graph contains them too).
    Edge weight = number of supporting (timepoint) edges.
    """
    ni = obj.node_info
    c2t = cluster_to_celltype(obj, annot_col, cluster_col)
    G = nx.DiGraph()
    # ensure all observed cell-types are nodes (so isolated states count)
    for ct in set(c2t.values()):
        G.add_node(ct)
    if ni is None or len(ni) == 0:
        return G
    for _, row in ni.iterrows():
        ts, cs = _parse_node(row["from"])
        tt, ct = _parse_node(row["to"])
        s = c2t.get((ts, cs)); t = c2t.get((tt, ct))
        if s is None or t is None:
            continue
        if G.has_edge(s, t):
            G[s][t]["weight"] += 1
        else:
            G.add_edge(s, t, weight=1)
    return G


def answer_graph(csv_path):
    """Reference DiGraph from an answer_path CSV (source,target)."""
    df = pd.read_csv(csv_path)
    G = nx.DiGraph()
    for _, r in df.iterrows():
        s, t = str(r["source"]).strip(), str(r["target"]).strip()
        G.add_node(s); G.add_node(t)
        G.add_edge(s, t)
    return G


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _node_match(a, b):
    return a.get("label", a) == b.get("label", b)


def graph_edit_distance(G_pred, G_ref, node_labeled=True, timeout=30):
    """Graph edit distance with cell-type node labels (closed form).

    Nodes are identified by their cell-type name, so a node may only match the
    same-named node (substitutions between different cell-types are disallowed,
    i.e. infinite cost). Under that constraint the optimal GED is simply the
    number of node + (directed) edge insertions/deletions, i.e. the size of the
    symmetric differences. This is exact and deterministic — unlike
    nx.graph_edit_distance, which on small DiGraphs with a timeout can return a
    spurious 0. (`timeout` kept for call compatibility.)
    """
    node_sd = set(G_pred.nodes()) ^ set(G_ref.nodes())
    edge_sd = set(G_pred.edges()) ^ set(G_ref.edges())
    return float(len(node_sd) + len(edge_sd))


def edge_prf(G_pred, G_ref, ignore_selfloops=False):
    """Edge-level precision/recall/F1 on the shared cell-type node space."""
    def edges(G):
        e = set(G.edges())
        if ignore_selfloops:
            e = {(a, b) for a, b in e if a != b}
        return e
    P, R = edges(G_pred), edges(G_ref)
    tp = len(P & R)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(R) if R else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "n_pred": len(P), "n_ref": len(R), "n_correct": tp}


def random_transition_f1(G_pred, G_ref, n=500, seed=0):
    """Mean transition-only (off-diagonal) edge-F1 under random node relabeling.

    Self-loops are preserved by any permutation, so the meaningful baseline is
    computed on transition edges only. Returns mean F1 over n permutations.
    """
    def trans(G):
        return {(a, b) for a, b in G.edges() if a != b}
    refT = trans(G_ref)
    predT = list(trans(G_pred))
    nodes = list(G_ref.nodes())
    rng = np.random.default_rng(seed)
    f1s = []
    for _ in range(n):
        perm = dict(zip(nodes, rng.permutation(nodes)))
        sT = {(perm[a], perm[b]) for a, b in predT if a in perm and b in perm}
        tp = len(sT & refT)
        p = tp / len(sT) if sT else 0.0
        r = tp / len(refT) if refT else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def normalized_ged(ged, G_pred, G_ref):
    """GED normalized by the worst-case (build ref from empty + delete pred)."""
    denom = G_pred.number_of_nodes() + G_pred.number_of_edges() \
        + G_ref.number_of_nodes() + G_ref.number_of_edges()
    return ged / denom if denom else 0.0
