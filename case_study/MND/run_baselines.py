"""Run the pseudobulk-correlation baseline on all 3 datasets and compare to GTra.

Output: BASELINE_figs/baseline_vs_gtra.csv  (method x dataset x transition metrics)
GTra numbers are read from the GED workstream (ged_combined.csv).
"""
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import dill

import ged_utils as ge
import benchmark_baselines as bb
from run_ged import CONFIG, load_answer

warnings.filterwarnings("ignore")
FIG = Path("BASELINE_figs"); FIG.mkdir(exist_ok=True)


def covid_baseline():
    """Per-patient pseudobulk-correlation baseline for COVID (mye_sub states)."""
    Gref = ge.answer_graph("../../answer_paths/COVID_answer.csv")
    rows = []
    for p in sorted(glob.glob("../../../../covid_obj/*.dill")):
        pid = Path(p).name.split("_")[0]
        o = dill.load(open(p, "rb"))
        pbs = []
        for tp in range(o.tp_data_num):
            ad = o.tp_data_dict[tp]
            states = ad.obs["mye_sub"].astype(str).values
            pbs.append(bb._lognorm_pseudobulk(ad.X, states))
        edges = bb.correlation_edges(pbs)
        b = bb.best_baseline(edges, Gref); f = bb.fair_baseline(edges, Gref, k=1)
        rows.append({"patient": pid,
                     "oracle_trans_f1": b["trans_f1"], "oracle_trans_recall": b["trans_recall"],
                     "fair_trans_f1": f["trans_f1"], "fair_trans_recall": f["trans_recall"],
                     "fair_GED": f["GED"], "rand_trans_f1": f["rand_trans_f1"]})
        del o
    df = pd.DataFrame(rows)
    return df.drop(columns="patient").mean().to_dict()


def main():
    results = []
    for ds in ["MND", "HSPC"]:
        Gref = load_answer(CONFIG[ds]["answer"], CONFIG[ds].get("collapse"))
        pbs = bb._timepoint_pseudobulks(ds)
        edges = bb.correlation_edges(pbs)
        b = bb.best_baseline(edges, Gref); f = bb.fair_baseline(edges, Gref, k=1)
        results.append({"dataset": ds, "method": "pseudobulk-corr",
                        "oracle_trans_f1": b["trans_f1"], "oracle_trans_recall": b["trans_recall"],
                        "fair_trans_f1": f["trans_f1"], "fair_trans_recall": f["trans_recall"],
                        "fair_GED": f["GED"], "rand_trans_f1": f["rand_trans_f1"]})
        print(f"[{ds}] pseudobulk-corr: ORACLE transF1={b['trans_f1']:.3f}(R{b['trans_recall']:.2f}) | "
              f"FAIR(top1) transF1={f['trans_f1']:.3f}(R{f['trans_recall']:.2f}) rand {f['rand_trans_f1']:.3f}",
              flush=True)

    cb = covid_baseline(); cb.update(dataset="COVID", method="pseudobulk-corr")
    results.append(cb)
    print(f"[COVID] pseudobulk-corr: ORACLE transF1={cb['oracle_trans_f1']:.3f} | "
          f"FAIR transF1={cb['fair_trans_f1']:.3f}(R{cb['fair_trans_recall']:.2f}) rand {cb['rand_trans_f1']:.3f}",
          flush=True)

    base = pd.DataFrame(results)
    base.to_csv(FIG / "baseline_results.csv", index=False)

    # merge with GTra (labeled + unlabeled) from the GED workstream
    gtra = pd.read_csv("GED_figs/ged_combined.csv")
    gtra = gtra[["dataset", "mode", "trans_f1", "trans_recall"]].rename(columns={"mode": "method"})
    gtra["method"] = "GTra-" + gtra["method"]
    # the FAIR (answer-blind) baseline is the one comparable to GTra's operating point
    bl_fair = base[["dataset"]].copy()
    bl_fair["method"] = "pseudobulk-corr(fair)"
    bl_fair["trans_f1"] = base["fair_trans_f1"]; bl_fair["trans_recall"] = base["fair_trans_recall"]
    bl_or = base[["dataset"]].copy()
    bl_or["method"] = "pseudobulk-corr(oracle)"
    bl_or["trans_f1"] = base["oracle_trans_f1"]; bl_or["trans_recall"] = base["oracle_trans_recall"]
    comp = pd.concat([gtra, bl_fair, bl_or], ignore_index=True)
    comp.to_csv(FIG / "baseline_vs_gtra.csv", index=False)
    print("\n=== transition-F1 by dataset x method ===")
    print(comp.pivot_table(index="dataset", columns="method", values="trans_f1").round(3).to_string())


if __name__ == "__main__":
    main()
