"""Lever B test: does finer Step-3 granularity let GTra isolate a tight IFN module?

GTra's IFN-rich trajectory modules (e.g. 28_0, 309 genes, peak 3h) dilute the
interferon program to ~10% of genes -> moderate enrichment (28.74). Step-3
(spherical k-means on each trajectory's gene temporal profiles) used too few
clusters. Here we re-cluster the SAME trajectory genes at increasing K and check
whether the best sub-cluster's interferon enrichment rises toward the baseline
(57.84), i.e. whether finer granularity makes GTra competitive.
"""
import warnings
import numpy as np, pandas as pd, dill, gseapy as gp
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
from pathlib import Path
HERE = Path(__file__).resolve().parent


def ifn_p(genes):
    try:
        h = gp.enrichr(gene_list=list(map(str, genes)), gene_sets=["MSigDB_Hallmark_2020"],
                       organism="Mouse", outdir=None).res2d
    except Exception:
        return 0.0, 0
    best = 1.0
    for t in ["Interferon Gamma Response", "Interferon Alpha Response"]:
        s = h[h["Term"].str.contains(t, case=False, na=False)]
        if len(s):
            best = min(best, float(pd.to_numeric(s["Adjusted P-value"], errors="coerce").min()))
    return -np.log10(best + 1e-300), len(genes)


def main():
    o = dill.load(open(HERE / "hspc_full.dill", "rb"))
    # IFN-rich, 3h-peak trajectory modules from earlier diagnosis
    for key in ["28_0", "16_0"]:
        df = o.merge_pattern_dict[key]
        genes = np.array(list(map(str, df.index)))
        P = np.asarray(df.values, float)                  # genes x timepoints
        Z = np.nan_to_num((P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9))
        Zs = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)  # spherical
        base_p, base_n = ifn_p(genes)
        print(f"\n=== trajectory module {key}: {len(genes)} genes, whole-module IFN -log10p={base_p:.1f} ===", flush=True)
        for K in [2, 4, 6, 8, 12]:
            lab = KMeans(K, random_state=0, n_init=10).fit_predict(Zs)
            best = (0.0, 0, None)
            for c in range(K):
                gl = genes[lab == c]
                if len(gl) < 10:
                    continue
                p, n = ifn_p(gl)
                if p > best[0]:
                    best = (p, n, c)
            print(f"  K={K:2d}: best sub-module IFN -log10p={best[0]:.1f} (n={best[1]} genes)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
