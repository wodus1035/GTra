"""Run CStreet and emit its cell-state transition graph in the common format.

CStreet takes per-timepoint expression + state labels and outputs a cell-state
connection graph (CellStatesConnCytoscape.txt) whose node names embed the state
(e.g. 'timepoint1_(1)3-APs/RPs'). We strip the prefix to recover the state,
filter by ConnectionProbabilities, collapse to cell-type space, and write
edges/CStreet_{dataset}.csv.

Run in py_cstreet:  python run_cstreet_edges.py MND   (or HSPC)
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from cstreet import CStreetData

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
EDGE_DIR = HERE.parent / "edges"
MND_COLLAPSE = {"1-Neurons": "Neurons", "2-Young neurons": "Young neurons",
                "6-Young neurons": "Young neurons", "3-APs/RPs": "APs/RPs",
                "5-APs/RPs": "APs/RPs", "4-IPs": "IPs", "7-IPs": "IPs"}
MIN_PROB = 0.09


def load_mnd():
    ad = sc.read_h5ad("/data3/projects/2025_GTRA/data/1_MND/raw/MouseCortex.h5ad")
    ad.obs = ad.obs.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
    ad.var_names = ad.var_names.map(lambda x: x.decode() if isinstance(x, bytes) else str(x))
    tps = ["e11", "e13", "e15", "e17"]
    data = [(ad[ad.obs["Time_points"] == t].to_df(),
             ad[ad.obs["Time_points"] == t].obs["cell_type2"].tolist()) for t in tps]
    return data


def load_hspc():
    DP = Path("/data3/projects/2025_GTRA/data/2_HSPC")
    ad = sc.read_h5ad(DP / "HSPC_preproc.h5ad")
    ad.obs = ad.obs.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
    ad.var_names = ad.var_names.map(lambda x: x.decode() if isinstance(x, bytes) else str(x))
    top_genes = pd.read_csv(DP / "HVG_genes.txt").values.reshape(-1).tolist()
    mapping = {"myel. prog. #1": "MYP", "myel. prog. #2": "MYP", "myel. prog. #3": "MYP",
               "ery. prog. #1": "EryP", "ery. prog. #2": "EryP", "ery. prog. #3": "EryP",
               "HSCs #1": "HSCs", "HSCs #2": "HSCs", "LMPPs #1": "LMPPs", "LMPPs #2": "LMPPs",
               "MK prog.": "MKP", "eosinophil prog.": "eosiP"}
    ad.obs["celltype"] = ad.obs["clusters"].map(lambda x: mapping.get(x, x)).astype(str)
    ad = ad[:, [g for g in top_genes if g in set(ad.var_names)]]
    tps = ["control", "3h", "24h", "72h"]
    data = [(ad[ad.obs["time"] == t].to_df(),
             ad[ad.obs["time"] == t].obs["celltype"].tolist()) for t in tps]
    return data


def main(dataset):
    data = load_mnd() if dataset == "MND" else load_hspc()
    cdata = CStreetData()
    for df, st in data:
        cdata.add_new_timepoint_scdata(df, st)
    cdata.params.Output_Dir = str(HERE)
    cdata.params.Output_Name = f"CS_{dataset}"
    cdata.params.Switch_Normalize = False
    cdata.params.Switch_LogTransform = False
    cdata.params.ProbParam_SamplingSize = 50
    cdata.params.Threshold_MinCellNumofStates = 20
    cdata.params.Threshold_MinProbability = MIN_PROB
    cdata.params.WithinTimePointParam_k = 25
    cdata.params.BetweenTimePointParam_k = 25
    cdata.run_cstreet()

    conn = HERE / f"CS_{dataset}" / f"CS_{dataset}_CellStatesConnCytoscape.txt"
    res = pd.read_csv(conn, sep="\t")
    res = res[res["ConnectionProbabilities"] > MIN_PROB]
    strip = lambda s: re.sub(r"^timepoint\d+_(\(\d+\))?", "", str(s))
    edge = {}
    for _, r in res.iterrows():
        s = strip(r["SourceNode"]); t = strip(r["TargetNode"])  # cell_type2, no collapse
        edge[(s, t)] = max(edge.get((s, t), 0.0), float(r["ConnectionProbabilities"]))
    EDGE_DIR.mkdir(exist_ok=True)
    out = EDGE_DIR / f"CStreet_{dataset}.csv"
    pd.DataFrame([(s, t, w) for (s, t), w in edge.items()],
                 columns=["source", "target", "score"]).to_csv(out, index=False)
    print(f"saved {out}  ({len(edge)} edges)")


if __name__ == "__main__":
    main(sys.argv[1])
