"""Generate HSPC cell-state defence figures (imports shared utils from ../MND)."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path("../MND").resolve()))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc

import cellstate_utils as cu
import cellstate_figs as cf

warnings.filterwarnings("ignore")
FIG = Path("HSPC_cellstate_figs"); FIG.mkdir(exist_ok=True)
TP, ANNOT = "time", "celltype"
TP_ORDER = ["control", "3h", "24h", "72h"]
DPI = 200


def main():
    ad = sc.read_h5ad("/data3/projects/2025_GTRA/data/2_HSPC/HSPC_proc.h5ad")
    summary, labels = cu.evaluate_timepoints(ad, TP, [ANNOT], resolution=0.5,
                                             use_counts_layer=None)
    summary.to_csv(FIG / "cellstate_agreement.csv", index=False)
    print("HSPC mean vs celltype:",
          summary[["ARI", "AMI", "V_measure", "purity", "inv_purity"]].mean().round(3).to_dict())

    fig, ax = plt.subplots(figsize=(7, 4))
    cf.fig_agreement_bars(summary, ANNOT, order=TP_ORDER, ax=ax)
    fig.tight_layout(); fig.savefig(FIG / "CSF1_agreement.pdf", dpi=DPI); plt.close(fig)

    fig, axes = plt.subplots(1, len(TP_ORDER), figsize=(4 * len(TP_ORDER), 3.5))
    cf.fig_confusion_grid(ad, TP, ANNOT, labels, TP_ORDER, ax=axes)
    fig.tight_layout(); fig.savefig(FIG / "CSF2_confusion.pdf", dpi=DPI); plt.close(fig)

    fig = cf.fig_tsne_compare(ad, TP, ANNOT, labels, TP_ORDER, coords="X_umap")
    fig.savefig(FIG / "CSF3_umap_compare.pdf", dpi=DPI); plt.close(fig)

    res_df = cu.resolution_scan(ad, TP, ANNOT, resolutions=[0.2, 0.3, 0.5, 0.8, 1.0],
                                use_counts_layer=None)
    res_df.to_csv(FIG / "cellstate_resolution.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); cf.fig_resolution(res_df, ax=ax)
    fig.tight_layout(); fig.savefig(FIG / "CSF4_resolution.pdf", dpi=DPI); plt.close(fig)

    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
