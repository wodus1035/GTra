"""CellRank (RealTimeKernel via moscot) on COVID, per patient — mean transition-F1."""
import sys, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata as ad
import networkx as nx
import dill
import cellrank as cr
from moscot.problems.time import TemporalProblem

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "case_study" / "MND"))
import ged_utils as ge

warnings.filterwarnings("ignore")
ANSWER = "/data1/home/jyj/PROJECT/2025/submission/NatureCom/GTra/answer_paths/COVID_answer.csv"
THR = 0.20


def patient_graph(A):
    A = A.copy()
    sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)
    sc.pp.pca(A, n_comps=30); sc.pp.neighbors(A, n_neighbors=15)
    A.obs["day"] = A.obs["day"].astype("category")
    tp = TemporalProblem(A).prepare(time_key="day").solve(epsilon=1e-2, scale_cost="mean")
    rtk = cr.kernels.RealTimeKernel.from_moscot(tp)
    rtk.compute_transition_matrix(self_transitions="all", conn_weight=0.2, threshold="auto")
    T = rtk.transition_matrix
    T = T.toarray() if sp.issparse(T) else np.asarray(T)
    states = A.obs["mye_sub"].values
    days = A.obs["day"].astype(float).values
    ud = sorted(np.unique(days))
    edge = {}
    for d0, d1 in zip(ud[:-1], ud[1:]):
        si = np.where(days == d0)[0]; ti = np.where(days == d1)[0]
        for s in np.unique(states[si]):
            rows = si[states[si] == s]
            mass = {tt: T[np.ix_(rows, ti[states[ti] == tt])].sum() for tt in np.unique(states[ti])}
            tot = sum(mass.values()) + 1e-12
            for tt, v in mass.items():
                if v / tot >= THR:
                    edge[(s, tt)] = max(edge.get((s, tt), 0.0), v / tot)
    G = nx.DiGraph()
    for s, t in edge:
        G.add_node(s); G.add_node(t); G.add_edge(s, t)
    return G


def main():
    Gref = ge.answer_graph(ANSWER)
    H5 = Path(__file__).resolve().parents[1] / "covid_h5ad"
    rows = []
    for p in sorted(glob.glob(str(H5 / "*.h5ad"))):
        pid = Path(p).stem
        try:
            G = patient_graph(sc.read_h5ad(p))
            prfT = ge.edge_prf(G, Gref, ignore_selfloops=True)
            rows.append({"patient": pid, "trans_f1": prfT["f1"], "trans_recall": prfT["recall"],
                         "trans_precision": prfT["precision"]})
            print(f"{pid}: transF1={prfT['f1']:.3f} R={prfT['recall']:.2f}", flush=True)
        except Exception as e:
            print(f"{pid}: FAILED {str(e)[:80]}", flush=True)
    df = pd.DataFrame(rows)
    print("\nCellRank COVID mean:", df[["trans_f1", "trans_recall", "trans_precision"]].mean().round(3).to_dict())
    df.to_csv(Path(__file__).resolve().parents[1] / "edges" / "_covid_CellRank_perpatient.csv", index=False)


if __name__ == "__main__":
    main()
