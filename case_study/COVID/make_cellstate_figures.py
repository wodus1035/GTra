"""Generate COVID cell-state defence figures (imports shared utils from ../MND)."""
import sys, glob, warnings
from pathlib import Path
sys.path.insert(0, str(Path("../MND").resolve()))

import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dill

import cellstate_utils as cu

warnings.filterwarnings("ignore")
FIG = Path("COVID_cellstate_figs"); FIG.mkdir(exist_ok=True)
OBJ = sorted(glob.glob("../../../../covid_obj/*.dill"))
DPI = 200


def main():
    summary, labels = cu.evaluate_covid_objects(OBJ, annot_col="mye_sub", resolution=0.5)
    summary.to_csv(FIG / "cellstate_agreement.csv", index=False)
    print("COVID overall mean vs mye_sub:",
          summary[["ARI", "AMI", "V_measure", "purity", "inv_purity"]].mean().round(3).to_dict())

    # CS-F1 per-patient agreement
    g = summary.groupby("patient")[["ARI", "AMI", "V_measure", "purity"]].mean().reset_index()
    m = g.melt(id_vars="patient", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=m, x="patient", y="value", hue="metric", ax=ax)
    ax.set_ylim(0, 1); ax.set_title("Unsupervised clustering vs mye_sub (per patient)")
    fig.tight_layout(); fig.savefig(FIG / "CSF1_agreement_per_patient.pdf", dpi=DPI); plt.close(fig)

    # CS-F2 confusion for representative patient
    pid = "P6"
    obj = dill.load(open([p for p in OBJ if f"{pid}_" in p][0], "rb"))
    tps = list(range(obj.tp_data_num))
    fig, axes = plt.subplots(1, len(tps), figsize=(4 * len(tps), 3.5))
    for a, tp in zip(np.atleast_1d(axes), tps):
        sub = obj.tp_data_dict[tp]
        lab = labels[(pid, tp)].reindex(sub.obs_names).values
        ct = cu.confusion_from_labels(sub.obs["mye_sub"].values, lab, normalize="index")
        sns.heatmap(ct, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f", cbar=False, ax=a)
        a.set_title(f"{pid} tp{tp}"); a.set_xlabel("Leiden cluster")
        a.set_ylabel("mye_sub" if tp == 0 else "")
    fig.tight_layout(); fig.savefig(FIG / f"CSF2_confusion_{pid}.pdf", dpi=DPI); plt.close(fig)

    # CS-F3 distribution
    m = summary.melt(id_vars=["patient", "timepoint"],
                     value_vars=["ARI", "AMI", "purity", "V_measure"],
                     var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=m, x="metric", y="value", ax=ax)
    sns.stripplot(data=m, x="metric", y="value", color="0.2", size=3, alpha=.6, ax=ax)
    ax.set_ylim(0, 1); ax.set_title("Cell-state recovery across 7 patients × 3 timepoints")
    fig.tight_layout(); fig.savefig(FIG / "CSF3_agreement_distribution.pdf", dpi=DPI); plt.close(fig)

    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
