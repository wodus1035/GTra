"""Corrected, quantitative Fig 3 (R4.3): interferon-program recovery,
GTra (answer-path modules) vs simpler time-series clustering baselines.

Reviewer R4.3: (i) the text claims interferon enrichment that the old Fig 3 did
not clearly show, and (ii) GTra ~ STEM. We replace the qualitative heatmap with a
quantitative comparison on the SAME gene universe: for each method we cluster the
genes by their cell-type pseudobulk temporal profiles (control/3h/24h/72h) and
score each method by the best-module interferon recovery (MSigDB Hallmark
Interferon, -log10 adj p) and overall GO coherence. Methods:
  GTra (answer-path trajectory modules), k-means, spherical k-means,
  STEM-like (assign genes to canonical temporal templates).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import dill
import gseapy as gp
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MND"))

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ORG = "Human"
TPS = ["control", "3h", "24h", "72h"]
IFN = ["Interferon Alpha Response", "Interferon Gamma Response"]


def ifn_and_go(modules, max_modules=30, max_genes=60):
    """best IFN recovery (-log10 adjp) and best GO coherence over modules."""
    best_ifn, best_go = 0.0, 0.0
    for genes in sorted(modules, key=len, reverse=True)[:max_modules]:
        gl = list(dict.fromkeys(genes))[:max_genes]
        if len(gl) < 10:
            continue
        try:
            h = gp.enrichr(gene_list=gl, gene_sets=["MSigDB_Hallmark_2020"], organism=ORG, outdir=None).res2d
            for t in IFN:
                s = h[h["Term"].str.contains(t, case=False, na=False)]
                if len(s):
                    p = float(pd.to_numeric(s["Adjusted P-value"], errors="coerce").min())
                    best_ifn = max(best_ifn, -np.log10(p + 1e-300))
        except Exception:
            pass
        try:
            e = gp.enrichr(gene_list=gl, gene_sets=["GO_Biological_Process_2021"], organism=ORG, outdir=None).res2d
            p = float(pd.to_numeric(e["Adjusted P-value"], errors="coerce").min())
            best_go = max(best_go, -np.log10(p + 1e-300))
        except Exception:
            pass
    return best_ifn, best_go


def main():
    # GTra answer-path modules
    obj = dill.load(open(HERE / "hspc_full.dill", "rb"))
    gtra = [[g for g in list(df.index) if isinstance(g, str)] for df in obj.merge_pattern_dict.values()]
    gtra = [m for m in gtra if len(set(m)) >= 10]
    K = len(gtra)
    universe = sorted(set().union(*[set(m) for m in gtra]))

    # gene x timepoint pseudobulk profiles over the SAME universe
    ad = sc.read_h5ad("/data3/projects/2025_GTRA/data/2_HSPC/HSPC_proc.h5ad")
    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    Xln = np.log1p(X / (X.sum(1, keepdims=True) + 1e-12) * 1e4)
    gidx = {g: i for i, g in enumerate(ad.var_names)}
    universe = [g for g in universe if g in gidx]
    ui = [gidx[g] for g in universe]
    tcol = ad.obs["time"].astype(str).values
    prof = np.zeros((len(ui), len(TPS)))
    for j, tp in enumerate(TPS):
        m = tcol == tp
        prof[:, j] = Xln[np.ix_(m, ui)].mean(0)
    Z = np.nan_to_num((prof - prof.mean(1, keepdims=True)) / (prof.std(1, keepdims=True) + 1e-9))

    def km(zscored):
        lab = KMeans(K, random_state=0, n_init=10).fit_predict(zscored)
        return [list(np.array(universe)[lab == c]) for c in np.unique(lab)]

    # STEM-like canonical templates over 4 timepoints (monotonic, transient, etc.)
    T = np.array([[0,1,2,3],[3,2,1,0],[0,3,1,0],[0,1,3,1],[3,0,1,2],[0,2,3,3],
                  [3,1,0,0],[0,3,3,3],[3,3,1,0],[0,0,1,3]], float)
    T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-9)
    assign = np.argmax(Z @ T.T, axis=1)
    stem = [list(np.array(universe)[assign == c]) for c in np.unique(assign)]

    methods = {
        "GTra": gtra,
        "k-means": km(Z),
        "spherical k-means": km(Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)),
        "STEM-like": [m for m in stem if len(m) >= 10],
    }
    rows = []
    for name, mods in methods.items():
        ifn, go = ifn_and_go(mods)
        rows.append({"method": name, "n_modules": len(mods), "IFN_recovery": ifn, "GO_coherence": go})
        print(f"{name:18s} nmod={len(mods):3d}  IFN -log10p={ifn:.2f}  GO -log10p={go:.2f}", flush=True)
    df = pd.DataFrame(rows)
    out = HERE / "FIG3_figs"; out.mkdir(exist_ok=True)
    df.to_csv(out / "fig3_ifn_recovery.csv", index=False)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(data=df, x="method", y="IFN_recovery", color="#c0504d", ax=ax)
    ax.axhline(-np.log10(0.05), ls="--", c="grey", lw=1, label="adj p=0.05")
    ax.set_title("HSPC interferon-program recovery (best module)\nGTra vs time-series clustering baselines")
    ax.set_ylabel("-log10(adj p), Hallmark Interferon"); ax.set_xlabel(""); ax.legend()
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(out / "FIG3_ifn_recovery.pdf", dpi=200)
    print("saved FIG3_figs/FIG3_ifn_recovery.pdf\nDONE")


if __name__ == "__main__":
    main()
