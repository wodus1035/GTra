"""WOT on COVID, per patient — mean transition-F1 vs the COVID answer.

Reuses covid_obj/*.dill (per-patient tp adatas with mye_sub + raw counts).
For each patient: build a 3-timepoint AnnData, run WOT OT, aggregate transport
to mye_sub state transitions, threshold, score with the unified metric. Report
the per-patient mean (matching how GTra COVID was scored).
"""
import sys, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata as ad
import dill
import wot

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "case_study" / "MND"))
import ged_utils as ge

warnings.filterwarnings("ignore")
ANSWER = "/data1/home/jyj/PROJECT/2025/submission/NatureCom/GTra/answer_paths/COVID_answer.csv"
THR = 0.20


def patient_graph(obj):
    # build one AnnData across the 3 timepoints with raw counts + mye_sub + day
    mats, obs = [], []
    common = None
    for tp in range(obj.tp_data_num):
        a = obj.tp_data_dict[tp]
        common = set(a.var_names) if common is None else (common & set(a.var_names))
    common = sorted(common)
    for tp in range(obj.tp_data_num):
        a = obj.tp_data_dict[tp][:, common]
        X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        mats.append(X)
        o = pd.DataFrame({"mye_sub": a.obs["mye_sub"].astype(str).values,
                          "day": float(tp)}, index=[f"{tp}_{i}" for i in range(a.n_obs)])
        obs.append(o)
    A = ad.AnnData(np.vstack(mats), obs=pd.concat(obs), var=pd.DataFrame(index=common))
    sc.pp.normalize_total(A, target_sum=1e4); sc.pp.log1p(A)
    otm = wot.ot.OTModel(A, day_field="day", lambda1=1.0, lambda2=50.0,
                         epsilon=0.05, growth_iter=3)
    days = sorted(A.obs["day"].unique())
    edge = {}
    for d0, d1 in zip(days[:-1], days[1:]):
        tm = otm.compute_transport_map(d0, d1)
        M = tm.X.toarray() if sp.issparse(tm.X) else np.asarray(tm.X)
        src = A.obs.loc[tm.obs.index, "mye_sub"].values
        tgt = A.obs.loc[tm.var.index, "mye_sub"].values
        for s in np.unique(src):
            si = np.where(src == s)[0]
            row = {tt: M[np.ix_(si, np.where(tgt == tt)[0])].sum() for tt in np.unique(tgt)}
            tot = sum(row.values()) + 1e-12
            for tt, v in row.items():
                if v / tot >= THR:
                    edge[(s, tt)] = max(edge.get((s, tt), 0.0), v / tot)
    import networkx as nx
    G = nx.DiGraph()
    for s, t in edge:
        G.add_node(s); G.add_node(t); G.add_edge(s, t)
    return G


def main():
    Gref = ge.answer_graph(ANSWER)
    rows = []
    for p in sorted(glob.glob("/data1/home/jyj/PROJECT/2025/submission/covid_obj/*.dill")):
        pid = Path(p).name.split("_")[0]
        obj = dill.load(open(p, "rb"))
        G = patient_graph(obj)
        prfT = ge.edge_prf(G, Gref, ignore_selfloops=True)
        rows.append({"patient": pid, "trans_f1": prfT["f1"], "trans_recall": prfT["recall"],
                     "trans_precision": prfT["precision"]})
        print(f"{pid}: transF1={prfT['f1']:.3f} R={prfT['recall']:.2f}", flush=True)
        del obj
    df = pd.DataFrame(rows)
    print("\nWOT COVID mean:", df[["trans_f1", "trans_recall", "trans_precision"]].mean().round(3).to_dict())
    df.to_csv(Path(__file__).resolve().parents[1] / "edges" / "_covid_WOT_perpatient.csv", index=False)


if __name__ == "__main__":
    main()
