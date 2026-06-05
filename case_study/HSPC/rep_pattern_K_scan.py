"""Does a 3h-interferon REPRESENTATIVE pattern emerge at finer K?

module_evaluation's auto-K (_calc_gap) chose K=3, merging the transient 3h-IFN
into a 24h super-cluster. We re-cluster the same pattern-distance matrix at
K=3..10 and, for each representative cluster, report its temporal peak and
interferon recovery (organism=Mouse). Goal: find the K at which GTra surfaces a
3h-peaked, IFN-enriched representative pattern (the published C1 module).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, dill, gseapy as gp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf  # noqa: F401
from scipy.cluster.hierarchy import linkage, fcluster
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MND"))
from gtra.utils import l2norm
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
TPS = ["control", "3h", "24h", "72h"]


def ifn_p(genes):
    try:
        h = gp.enrichr(gene_list=list(map(str, genes)), gene_sets=["MSigDB_Hallmark_2020"],
                       organism="Mouse", outdir=None).res2d
    except Exception:
        return 0.0
    best = 1.0
    for t in ["Interferon Gamma Response", "Interferon Alpha Response"]:
        s = h[h["Term"].str.contains(t, case=False, na=False)]
        if len(s):
            best = min(best, float(pd.to_numeric(s["Adjusted P-value"], errors="coerce").min()))
    return -np.log10(best + 1e-300)


def main():
    obj = dill.load(open(HERE / "hspc_full.dill", "rb"))
    obj.plot_patterns(); plt.close("all")
    obj.module_evaluation()
    sp = obj.sig_patterns.reset_index(drop=True)
    pdist = obj.pattern_dist.values
    linked = linkage(pdist, "ward")
    best_per_k = []
    for K in range(3, 11):
        lab = fcluster(linked, K, criterion="maxclust")
        rows = []
        for c in np.unique(lab):
            pids = sp.loc[lab == c, "Pattern_ID"].tolist()
            genes, cents = set(), []
            for pi in pids:
                e = l2norm(obj.merge_pattern_dict[pi]); genes |= set(map(str, e.index))
                cents.append(e.mean(axis=0).values)
            z = np.mean(cents, 0); z = (z - z.mean()) / (z.std() + 1e-9)
            rows.append((TPS[int(np.argmax(z))], len(genes), ifn_p(genes)))
        # representative cluster that is 3h-peaked with best IFN
        h3 = [r for r in rows if r[0] == "3h"]
        best3 = max(h3, key=lambda r: r[2]) if h3 else None
        bestany = max(rows, key=lambda r: r[2])
        best_per_k.append({"K": K, "n_3h_clusters": len(h3),
                           "best_3h_IFN": round(best3[2], 2) if best3 else None,
                           "best_3h_ngenes": best3[1] if best3 else None,
                           "best_IFN_any": round(bestany[2], 2), "best_IFN_peak": bestany[0]})
        b3 = round(best3[2], 1) if best3 else None
        n3 = best3[1] if best3 else None
        print(f"K={K}: #3h-peak-rep={len(h3)} best3hIFN={b3} (n={n3}) "
              f"| bestIFNany={bestany[2]:.1f}@{bestany[0]}", flush=True)
    pd.DataFrame(best_per_k).to_csv(HERE / "REP_PATTERN_figs" / "rep_K_scan.csv", index=False)
    print("DONE")


if __name__ == "__main__":
    main()
