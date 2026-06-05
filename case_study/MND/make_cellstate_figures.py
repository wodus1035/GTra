"""Generate cell-state defence figures + tables (MND)."""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

import cellstate_utils as cu
import cellstate_figs as cf

warnings.filterwarnings("ignore")

DATA = "/data3/projects/2025_GTRA/data/1_MND/CCTSD_preproc_hvg.h5ad"
TP = "timepoints"
ANNOTS = ["cell_type", "cell_type2"]   # coarse (major states), fine
FIG = Path("MND_cellstate_figs"); FIG.mkdir(exist_ok=True)
DPI = 200


def main():
    ad = sc.read_h5ad(DATA)
    summary, labels = cu.evaluate_timepoints(ad, TP, ANNOTS, resolution=0.5)
    summary.to_csv(FIG / "cellstate_agreement.csv", index=False)

    print("=== mean agreement by annotation (Leiden res=0.5) ===")
    print(summary.groupby("annotation")[["ARI", "AMI", "V_measure",
          "purity", "inv_purity", "homogeneity", "completeness"]].mean().round(3))

    tps = sorted(ad.obs[TP].unique())

    # CS-F1 agreement bars (coarse)
    fig, ax = plt.subplots(figsize=(7, 4))
    cf.fig_agreement_bars(summary, "cell_type", ax=ax)
    fig.tight_layout(); fig.savefig(FIG / "CSF1_agreement_coarse.pdf", dpi=DPI); plt.close(fig)

    # CS-F2 confusion grid (coarse)
    fig, axes = plt.subplots(1, len(tps), figsize=(4 * len(tps), 3.5))
    cf.fig_confusion_grid(ad, TP, "cell_type", labels, tps, ax=axes)
    fig.tight_layout(); fig.savefig(FIG / "CSF2_confusion_coarse.pdf", dpi=DPI); plt.close(fig)

    # CS-F3 tSNE annotation vs Leiden
    fig = cf.fig_tsne_compare(ad, TP, "cell_type", labels, tps)
    fig.savefig(FIG / "CSF3_tsne_compare.pdf", dpi=DPI); plt.close(fig)

    # CS-F4 resolution sensitivity (coarse, all tps)
    res_df = cu.resolution_scan(ad, TP, "cell_type",
                                resolutions=[0.2, 0.3, 0.5, 0.8, 1.0])
    res_df.to_csv(FIG / "cellstate_resolution.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); cf.fig_resolution(res_df, ax=ax)
    fig.tight_layout(); fig.savefig(FIG / "CSF4_resolution.pdf", dpi=DPI); plt.close(fig)

    print("\nfigures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
