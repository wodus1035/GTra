"""COVID trajectory accuracy via GED — label-free, answer-unconstrained.

Reuses the per-patient GTra dills (gene clustering / statistical testing already
done) and only re-constructs the trajectory with answer_path_dir="" so the
result is NOT filtered by the answer. Compares to the COVID answer graph.
A label-shuffled baseline shows the recovery is well above chance.
"""
import sys, glob, copy, warnings
from pathlib import Path
sys.path.insert(0, str(Path("../MND").resolve()))

import numpy as np
import pandas as pd
import networkx as nx
import dill

import ged_utils as ge

warnings.filterwarnings("ignore")
ANS = "../../answer_paths/COVID_answer.csv"
OBJ = sorted(glob.glob("../../../../covid_obj/*.dill"))
FIG = Path("COVID_ged_figs"); FIG.mkdir(exist_ok=True)


def shuffled_baseline(G_pred, G_ref, n=200, seed=0):
    """Relabel predicted-graph nodes randomly; mean GED/F1 over n shuffles."""
    rng = np.random.default_rng(seed)
    nodes = list(G_ref.nodes())
    geds, f1s = [], []
    pred_edges = list(G_pred.edges())
    for _ in range(n):
        perm = rng.permutation(nodes)
        relabel = dict(zip(nodes, perm))
        Gs = nx.DiGraph(); Gs.add_nodes_from(nodes)
        for a, b in pred_edges:
            if a in relabel and b in relabel:
                Gs.add_edge(relabel[a], relabel[b])
        geds.append(ge.graph_edit_distance(Gs, G_ref))
        f1s.append(ge.edge_prf(Gs, G_ref)["f1"])
    return float(np.mean(geds)), float(np.mean(f1s))


def main():
    Gref = ge.answer_graph(ANS)
    rows = []
    for p in OBJ:
        pid = Path(p).name.split("_")[0]
        o = dill.load(open(p, "rb"))
        o.params.answer_path_dir = ""
        o.construct_trajectories()
        G = ge.state_graph(o, "mye_sub", "cluster_label")
        prf = ge.edge_prf(G, Gref)
        ged = ge.graph_edit_distance(G, Gref)
        bged, bf1 = shuffled_baseline(G, Gref)
        rows.append({"patient": pid, "GED": ged, "norm_GED": ge.normalized_ged(ged, G, Gref),
                     "f1": prf["f1"], "precision": prf["precision"], "recall": prf["recall"],
                     "n_pred": prf["n_pred"], "n_ref": prf["n_ref"], "n_correct": prf["n_correct"],
                     "rand_GED": bged, "rand_f1": bf1})
        print(f"{pid}: GED={ged:.0f} (rand {bged:.1f})  F1={prf['f1']:.3f} (rand {bf1:.3f})  "
              f"P={prf['precision']:.2f} R={prf['recall']:.2f}", flush=True)
        del o
    df = pd.DataFrame(rows)
    df.to_csv(FIG / "covid_ged.csv", index=False)
    print("\n=== mean ===")
    print(df[["GED", "rand_GED", "f1", "rand_f1", "precision", "recall"]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
