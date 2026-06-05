"""
Simpler baselines for the GTra trajectory (ISMB review: meta + R2.5/R4.2/R4.4).

Reviewers ask whether the full gene-module machinery is necessary, or whether a
simple baseline recovers the same cell-state trajectory. We implement the most
direct one:

  PSEUDOBULK-CORRELATION-ONLY (R4.2): for each pair of adjacent timepoints, score
  every (source state -> target state) edge by the correlation of the two states'
  pseudobulk expression vectors — no gene modules, no Jaccard/cosine module
  scoring. Build the cell-state transition graph by thresholding, and evaluate it
  with the SAME metric used for GTra (transition-only edge-F1 / GED vs the answer
  graph). We sweep the threshold and report the baseline's BEST F1 (a generous
  upper bound): if GTra still beats the baseline's best, the module machinery adds
  value.

States are the predefined annotation (same cell states GTra uses in labeled mode),
so this isolates "module machinery vs plain pseudobulk correlation".
"""
import warnings
from itertools import product

import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx

import ged_utils as ge

warnings.filterwarnings("ignore")

# reuse the GED dataset config (paths, annot, answer, collapse)
from run_ged import CONFIG, MND_COLLAPSE, load_answer


def _lognorm_pseudobulk(X, states):
    """X: cells x genes (raw counts). Returns {state: mean lognorm vector}."""
    import scipy.sparse as sp
    Xd = X.toarray() if sp.issparse(X) else np.asarray(X)
    Xln = np.log1p(Xd / (Xd.sum(1, keepdims=True) + 1e-12) * 1e4)  # CP10k + log1p
    pb = {}
    for s in pd.unique(states):
        m = states == s
        if m.sum() == 0:
            continue
        pb[s] = Xln[m].mean(0)
    return pb


def _timepoint_pseudobulks(dataset):
    """Return ordered list of {state: pseudobulk vector} per timepoint, using the
    predefined annotation as the state, plus the collapse map (for MND)."""
    cfg = CONFIG[dataset]
    collapse = cfg.get("collapse")
    pbs = []
    if dataset == "COVID":
        import glob, dill
        # average is per-patient; here we pool by concatenating patients is wrong,
        # so COVID is handled separately (per patient) in run_baselines.
        raise ValueError("use run_covid_baseline for COVID")
    ad = sc.read_h5ad(cfg["h5ad"])
    if "highly_variable" in ad.var and cfg["counts_layer"] == "counts":
        ad = ad[:, ad.var.highly_variable].copy()
    X_all = ad.layers[cfg["counts_layer"]] if cfg["counts_layer"] else ad.X
    for t in sorted(ad.obs[cfg["time_col"]].unique()):
        m = (ad.obs[cfg["time_col"]] == t).values
        states = ad.obs[cfg["annot"]].astype(str).values[m]
        if collapse:
            states = np.array([collapse.get(s, s) for s in states])
        pbs.append(_lognorm_pseudobulk(X_all[m], states))
    return pbs


def correlation_edges(pbs, method="pearson"):
    """Score every source(t)->target(t+1) edge by pseudobulk correlation.
    Returns list of (src, tgt, score) aggregated across adjacent timepoints
    (max score kept per cell-type edge)."""
    best = {}
    for t in range(len(pbs) - 1):
        for s, sv in pbs[t].items():
            for tt, tv in pbs[t + 1].items():
                if method == "pearson":
                    r = np.corrcoef(sv, tv)[0, 1]
                else:  # cosine
                    r = float(sv @ tv / (np.linalg.norm(sv) * np.linalg.norm(tv) + 1e-12))
                key = (s, tt)
                best[key] = max(best.get(key, -np.inf), r)
    return [(s, t, v) for (s, t), v in best.items()]


def graph_at_threshold(edges, theta):
    G = nx.DiGraph()
    for s, t, v in edges:
        G.add_node(s); G.add_node(t)
        if v >= theta:
            G.add_edge(s, t)
    return G


def best_baseline(edges, Gref):
    """Sweep threshold; return BEST transition-F1 (OPTIMISTIC ORACLE — the
    threshold is chosen against the answer, so this upper-bounds the baseline)."""
    scores = sorted({round(v, 4) for _, _, v in edges})
    best = {"trans_f1": -1}
    for theta in scores:
        G = graph_at_threshold(edges, theta)
        prfT = ge.edge_prf(G, Gref, ignore_selfloops=True)
        if prfT["f1"] > best["trans_f1"]:
            ged = ge.graph_edit_distance(G, Gref, timeout=30)
            best = {"theta": theta, "trans_f1": prfT["f1"],
                    "trans_recall": prfT["recall"], "trans_precision": prfT["precision"],
                    "GED": ged, "n_pred_edges": G.number_of_edges()}
    best["rand_trans_f1"] = ge.random_transition_f1(
        graph_at_threshold(edges, best["theta"]), Gref)
    return best


def fair_baseline(edges, Gref, k=1):
    """ANSWER-BLIND operating point: connect each source to its top-k
    correlated targets (self-loops excluded). No access to the answer — directly
    comparable to GTra's single operating point."""
    by_src = {}
    for s, t, v in edges:
        if s == t:
            continue
        by_src.setdefault(s, []).append((v, t))
    G = nx.DiGraph()
    for s, _, _ in edges:
        G.add_node(s)
    for _, t, _ in edges:
        G.add_node(t)
    for s, lst in by_src.items():
        for v, t in sorted(lst, reverse=True)[:k]:
            G.add_edge(s, t)
    prfT = ge.edge_prf(G, Gref, ignore_selfloops=True)
    return {"trans_f1": prfT["f1"], "trans_recall": prfT["recall"],
            "trans_precision": prfT["precision"],
            "GED": ge.graph_edit_distance(G, Gref, timeout=30),
            "rand_trans_f1": ge.random_transition_f1(G, Gref),
            "n_pred_edges": G.number_of_edges()}
