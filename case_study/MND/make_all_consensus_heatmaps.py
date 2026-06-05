"""All-cell-state x all-timepoint gene-clustering consensus heatmaps (MND, cell_type2).

The original F1 showed one cell cluster per timepoint. Here we render the bootstrap
consensus heatmap for EVERY (cell_type2 state x timepoint) and tabulate PAC for all,
so stability is shown for all high-resolution cell types (1-Neurons, 3-APs/RPs,
5-APs/RPs, 4-IPs, 2/6-Young neurons, 7-IPs) at every time point.
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import stability_utils as su, stability_figs as sf
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "MND_stability_figs"; OUT.mkdir(exist_ok=True)
# factorize(cell_type2, sort=True) order
CID2NAME = {0: "1-Neurons", 1: "2-Young neurons", 2: "3-APs/RPs", 3: "4-IPs",
            4: "5-APs/RPs", 5: "6-Young neurons", 6: "7-IPs"}


def main():
    res = sf.load("stability_out_A/stability_runs.pkl")
    reg = res["regimes"]["annotation"]
    tps = res["timepoints"]
    cids = sorted(reg[tps[0]]["runs"].keys())
    nrow, ncol = len(cids), len(tps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow))
    pac_rows = []
    for i, cid in enumerate(cids):
        for j, tp in enumerate(tps):
            ax = axes[i, j]
            runs = reg[tp]["runs"].get(cid)
            C, _ = su.consensus_matrix(runs) if runs is not None else (None, None)
            name = CID2NAME.get(cid, str(cid))
            if C is None:
                ax.set_visible(False); continue
            pac = su.pac_score(C)
            pac_rows.append({"cell_type": name, "timepoint": tp, "PAC": round(pac, 3),
                             "n_valid_runs": int(sum(r is not None for r in runs))})
            ref = reg[tp]["ref"].get(cid)
            keep = np.where(ref >= 0)[0] if ref is not None else np.arange(C.shape[0])
            Csub = np.nan_to_num(C[np.ix_(keep, keep)])
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            D = 1 - Csub; np.fill_diagonal(D, 0); D = (D + D.T) / 2
            order = leaves_list(linkage(squareform(D, checks=False), "average"))
            sns.heatmap(Csub[np.ix_(order, order)], cmap="rocket_r", vmin=0, vmax=1,
                        square=True, xticklabels=False, yticklabels=False,
                        cbar=False, rasterized=True, ax=ax)
            ax.set_title(f"{name} | tp{tp}\nPAC={pac:.2f}", fontsize=9)
    fig.suptitle("MND gene-clustering bootstrap consensus — all cell_type2 states x timepoints", y=1.001)
    fig.tight_layout()
    fig.savefig(OUT / "F1b_all_consensus_heatmaps.pdf", dpi=180); plt.close(fig)

    df = pd.DataFrame(pac_rows)
    df.to_csv(OUT / "PAC_all_states_timepoints.csv", index=False)
    print("=== PAC by cell_type2 x timepoint (lower=more stable) ===")
    print(df.pivot_table(index="cell_type", columns="timepoint", values="PAC").round(3).to_string())
    print(f"\noverall PAC mean={df['PAC'].mean():.3f}")
    print("saved F1b_all_consensus_heatmaps.pdf + PAC_all_states_timepoints.csv\nDONE")


if __name__ == "__main__":
    main()
