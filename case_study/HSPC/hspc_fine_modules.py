"""GTra fine-resolution modules (lever B) and re-evaluation.

Diagnosis: GTra's coarse final modules come from merge_sim_patterns recombining
Step-3 sub-clusters. The finer temporal sub-structure exists (granularity_test:
splitting a 309-gene module raises best IFN from 28.7 to ~41). Here we refine
EVERY final module by spherical k-means (elbow K) on its gene x timepoint
profile, yielding a fine-resolution module set, and re-evaluate interferon
recovery vs the coarse modules and the unconstrained temporal-clustering baseline.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, dill, gseapy as gp
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def ifn_best(modules, max_modules=80):
    best = 0.0
    for genes in sorted(modules, key=len, reverse=True)[:max_modules]:
        gl = list(map(str, genes))
        if len(gl) < 10:
            continue
        try:
            h = gp.enrichr(gene_list=gl, gene_sets=["MSigDB_Hallmark_2020"], organism="Mouse", outdir=None).res2d
        except Exception:
            continue
        for t in ["Interferon Gamma Response", "Interferon Alpha Response"]:
            s = h[h["Term"].str.contains(t, case=False, na=False)]
            if len(s):
                best = max(best, -np.log10(float(pd.to_numeric(s["Adjusted P-value"], errors="coerce").min()) + 1e-300))
    return best


def refine(df, max_per_cluster=80):
    """Split a module's genes by temporal-profile spherical k-means; K ~ size/target."""
    genes = np.array(list(map(str, df.index)))
    if len(genes) <= max_per_cluster:
        return [list(genes)]
    P = np.asarray(df.values, float)
    Z = np.nan_to_num((P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9))
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    K = int(np.ceil(len(genes) / max_per_cluster)) + 2
    lab = KMeans(K, random_state=0, n_init=10).fit_predict(Z)
    return [list(genes[lab == c]) for c in np.unique(lab) if (lab == c).sum() >= 10]


def main():
    o = dill.load(open(HERE / "hspc_full.dill", "rb"))
    coarse = [list(map(str, df.index)) for df in o.merge_pattern_dict.values() if len(df.index) >= 10]
    fine = []
    for df in o.merge_pattern_dict.values():
        fine += refine(df)
    fine = [m for m in fine if len(m) >= 10]
    print(f"coarse modules: {len(coarse)} (median {int(np.median([len(m) for m in coarse]))} genes)")
    print(f"fine   modules: {len(fine)} (median {int(np.median([len(m) for m in fine]))} genes)")
    cb = ifn_best(coarse); fb = ifn_best(fine)
    print(f"\nHSPC interferon recovery (best module, -log10 adj p):")
    print(f"  GTra coarse (current):       {cb:.1f}")
    print(f"  GTra fine-resolution (B):    {fb:.1f}")
    print(f"  (baseline temporal k-means:  57.8  — from fig3_hsc_specific)")
    pd.DataFrame([{"variant": "GTra_coarse", "n_modules": len(coarse), "IFN_recovery": cb},
                  {"variant": "GTra_fine", "n_modules": len(fine), "IFN_recovery": fb},
                  {"variant": "baseline_kmeans", "n_modules": None, "IFN_recovery": 57.8}]
                 ).to_csv(HERE / "FIG3_figs" / "fine_vs_coarse_ifn.csv", index=False)
    print("\nsaved FIG3_figs/fine_vs_coarse_ifn.csv\nDONE")


if __name__ == "__main__":
    main()
