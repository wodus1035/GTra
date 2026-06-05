"""Full GTra HSPC case-study run WITH the answer-path constraint (as in the paper).

Unlike the benchmark/GED runs (answer_path_dir="" to avoid circularity), the
biological case study uses the biologically-motivated answer path, which keeps
curated trajectories such as the HSC self-transition. We run Steps 1-3
(find_gclusters -> construct_trajectories -> pattern_clustering) and dill the
object so pattern scores can use merge_pattern_dict (temporal profiles) directly.
"""
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc
import dill

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MND"))
import gtra
import ged_utils as ge

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def main():
    ge.patch_gtra()  # robust stat-testing
    ad = sc.read_h5ad("/data3/projects/2025_GTRA/data/2_HSPC/HSPC_proc.h5ad")
    obj = gtra.GTraObject()
    obj.params.cell_type_label = "celltype"
    obj.params.answer_path_type = "HSPC"
    obj.params.answer_path_dir = str(HERE.parent / "answer_paths" / "HSPC_answer.csv")
    obj.params.output_dir = str(HERE / "HSPC_full_out")
    obj.params.output_name = "hspc_full"
    Path(obj.params.output_dir).mkdir(exist_ok=True)
    order = ["control", "3h", "24h", "72h"]
    for t in order:
        dat = ad[ad.obs["time"] == t]
        obj.upload_time_scRNA(dat.to_df(), dat.obs[["celltype"]])
    obj.select_genes()
    obj.find_gclusters(N=50)
    obj.construct_trajectories()
    obj.pattern_clustering()
    with open(HERE / "hspc_full.dill", "wb") as f:
        dill.dump(obj, f)
    n = len(getattr(obj, "merge_pattern_dict", {}) or {})
    print(f"saved hspc_full.dill  (merge_pattern_dict modules={n})", flush=True)
    # report whether an HSC self-transition trajectory exists
    keys = list((getattr(obj, "merge_pattern_dict", {}) or {}).keys())
    print("DONE")


if __name__ == "__main__":
    main()
