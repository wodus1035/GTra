"""Per-cell-type consensus heatmaps (one PDF per cell_type2) + saved matrices.

For each of the 7 MND cell_type2 states, write a separate PDF showing the
bootstrap gene-clustering consensus heatmap across all 4 timepoints. Also persist
the consensus matrices (+ gene order + PAC) so the figures can be regenerated
instantly without re-running the (slow) bootstrap.
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import stability_utils as su, stability_figs as sf
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "MND_stability_figs" / "per_celltype"; FIGDIR.mkdir(parents=True, exist_ok=True)
MATDIR = HERE / "MND_stability_figs" / "consensus_matrices"; MATDIR.mkdir(parents=True, exist_ok=True)
CID2NAME = {0: "1-Neurons", 1: "2-Young neurons", 2: "3-APs_RPs", 3: "4-IPs",
            4: "5-APs_RPs", 5: "6-Young neurons", 6: "7-IPs"}  # '/' -> '_' for filenames


def ordered(C, ref):
    keep = np.where(ref >= 0)[0] if ref is not None else np.arange(C.shape[0])
    Csub = np.nan_to_num(C[np.ix_(keep, keep)])
    D = 1 - Csub; np.fill_diagonal(D, 0); D = (D + D.T) / 2
    order = leaves_list(linkage(squareform(D, checks=False), "average"))
    return Csub[np.ix_(order, order)], keep[order]


def main():
    res = sf.load("stability_out_A/stability_runs.pkl")
    reg = res["regimes"]["annotation"]; tps = res["timepoints"]
    genes = np.array(res["gene_names"])
    pac_rows = []
    for cid, name in CID2NAME.items():
        fig, axes = plt.subplots(1, len(tps), figsize=(4 * len(tps), 4))
        axes = np.atleast_1d(axes)
        saved = {}
        for ax, tp in zip(axes, tps):
            runs = reg[tp]["runs"].get(cid)
            C, _ = su.consensus_matrix(runs) if runs is not None else (None, None)
            if C is None:
                ax.set_visible(False); continue
            pac = su.pac_score(C)
            ref = reg[tp]["ref"].get(cid)
            Cord, gorder = ordered(C, ref)
            sns.heatmap(Cord, cmap="rocket_r", vmin=0, vmax=1, square=True,
                        xticklabels=False, yticklabels=False, rasterized=True,
                        cbar_kws={"label": "consensus"}, ax=ax)
            ax.set_title(f"tp{tp}  (PAC={pac:.2f}, {Cord.shape[0]} genes)", fontsize=11)
            saved[f"C_tp{tp}"] = C.astype(np.float32)
            saved[f"genes_tp{tp}"] = genes[gorder] if ref is not None else genes
            pac_rows.append({"cell_type": name.replace("_", "/"), "timepoint": tp,
                             "PAC": round(pac, 3)})
        disp = name.replace("_", "/")
        fig.suptitle(f"MND gene-clustering bootstrap consensus — {disp} (N={res['N']})", y=1.02)
        fig.tight_layout()
        fig.savefig(FIGDIR / f"consensus_{name}.pdf", dpi=180); plt.close(fig)
        # persist matrices for this cell type
        np.savez_compressed(MATDIR / f"consensus_{name}.npz", **saved)
        print(f"saved consensus_{name}.pdf + .npz", flush=True)
    pd.DataFrame(pac_rows).to_csv(FIGDIR / "PAC_per_celltype.csv", index=False)
    print(f"\n{len(CID2NAME)} per-cell-type PDFs + matrices saved.")
    print(f"  figures: {FIGDIR}")
    print(f"  matrices (reproducible re-plot): {MATDIR}")
    print("DONE")


if __name__ == "__main__":
    main()
