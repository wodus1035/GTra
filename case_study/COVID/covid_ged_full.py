"""
COVID trajectory accuracy via GED — labeled vs label-free, answer-unconstrained.

Per patient (7 dills):
  * unlabeled : reuse the existing dill (Leiden states, bootstrap already done);
                only re-construct the trajectory with answer_path_dir="".
  * labeled   : rebuild a GTra object from the dill's per-timepoint AnnData with
                the mye_sub annotation fixed (label_flag=True), re-bootstrap, and
                construct the trajectory unconstrained.

Both are scored against the COVID answer graph with full GED + transition-only F1
(self-loops excluded) and a transition-only random-relabel baseline.
"""
import sys, glob, copy, warnings, time
from pathlib import Path
sys.path.insert(0, str(Path("../MND").resolve()))

import numpy as np
import pandas as pd
import dill
import scanpy as sc

import gtra
import ged_utils as ge

warnings.filterwarnings("ignore")
ANS = "../../answer_paths/COVID_answer.csv"
OBJ = sorted(glob.glob("../../../../covid_obj/*.dill"))
ANNOT = "mye_sub"
N = 50
FIG = Path("COVID_ged_figs"); FIG.mkdir(exist_ok=True)


def score(obj, Gref):
    G = ge.state_graph(obj, ANNOT, "cluster_label")
    prf = ge.edge_prf(G, Gref)
    prfT = ge.edge_prf(G, Gref, ignore_selfloops=True)
    ged = ge.graph_edit_distance(G, Gref, timeout=60)
    return dict(GED=ged, f1=prf["f1"], precision=prf["precision"], recall=prf["recall"],
                trans_f1=prfT["f1"], trans_precision=prfT["precision"],
                trans_recall=prfT["recall"],
                rand_trans_f1=ge.random_transition_f1(G, Gref),
                n_pred=prf["n_pred"], n_ref=prf["n_ref"])


def build_labeled(src_obj):
    obj = gtra.GTraObject()
    obj.params.cell_type_label = ANNOT
    for tp in range(src_obj.tp_data_num):
        ad = src_obj.tp_data_dict[tp]
        cnt = ad.to_df()                       # X is raw counts
        meta = ad.obs[[ANNOT]]
        obj.upload_time_scRNA(cnt, meta)       # 2-arg -> label_flag=True
    obj.select_genes()
    obj.params.answer_path_dir = ""
    obj.find_gclusters(N=N)
    obj.construct_trajectories()
    return obj


def main():
    ge.patch_gtra()   # robust _score_distribution for the labeled re-bootstrap
    Gref = ge.answer_graph(ANS)
    rows = []
    for p in OBJ:
        pid = Path(p).name.split("_")[0]
        t0 = time.time()
        src = dill.load(open(p, "rb"))

        # unlabeled: reuse, just re-construct trajectory unconstrained
        u = copy.deepcopy(src)
        u.params.answer_path_dir = ""
        u.construct_trajectories()
        su = score(u, Gref); su.update(patient=pid, mode="unlabeled")
        rows.append(su)

        # labeled: rebuild + bootstrap with mye_sub fixed
        lab = build_labeled(src)
        sl = score(lab, Gref); sl.update(patient=pid, mode="labeled")
        rows.append(sl)

        print(f"{pid}: unlab transF1={su['trans_f1']:.3f} R={su['trans_recall']:.2f} | "
              f"lab transF1={sl['trans_f1']:.3f} R={sl['trans_recall']:.2f} | "
              f"rand {su['rand_trans_f1']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        del src, u, lab

    df = pd.DataFrame(rows)
    df.to_csv(FIG / "covid_ged_full.csv", index=False)
    print("\n=== mean by mode ===")
    print(df.groupby("mode")[["GED", "f1", "trans_f1", "trans_recall",
          "trans_precision", "rand_trans_f1"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
