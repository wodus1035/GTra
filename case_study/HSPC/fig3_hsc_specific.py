"""HSC-specific Fig 3 (R4.3): interferon recovery on the HSC self-transition,
matching the paper's Case-Study-1 comparison unit.

Fair setup: BOTH GTra and the baselines operate on HSCs only.
- Baseline: cluster genes by their HSC-only pseudobulk temporal profile
  (control/3h/24h/72h) with k-means / spherical k-means / STEM-like templates.
- GTra: gene modules whose trajectory is HSC-dominant (via convert_path_name on
  the answer-path object).
Score each method by best-module interferon recovery (Hallmark, -log10 adj p).
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
from gtra.utils import convert_path_name

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ORG = "Human"
TPS = ["control", "3h", "24h", "72h"]
IFN = ["Interferon Alpha Response", "Interferon Gamma Response"]


def best_ifn(modules, max_modules=40, max_genes=60):
    best, who = 0.0, None
    for genes in sorted(modules, key=len, reverse=True)[:max_modules]:
        gl = list(dict.fromkeys(genes))[:max_genes]
        if len(gl) < 10:
            continue
        try:
            h = gp.enrichr(gene_list=gl, gene_sets=["MSigDB_Hallmark_2020"], organism=ORG, outdir=None).res2d
        except Exception:
            continue
        for t in IFN:
            s = h[h["Term"].str.contains(t, case=False, na=False)]
            if len(s):
                v = -np.log10(float(pd.to_numeric(s["Adjusted P-value"], errors="coerce").min()) + 1e-300)
                if v > best:
                    best, who = v, (t, len(gl))
    return best, who


def main():
    obj = dill.load(open(HERE / "hspc_full.dill", "rb"))
    # GTra HSC-dominant modules
    gtra_hsc = []
    for key, df in obj.merge_pattern_dict.items():
        try:
            path = convert_path_name(obj, key)
        except Exception:
            continue
        cts = path.split("->")
        if cts.count("HSCs") >= 2:  # HSC-dominant trajectory
            genes = [g for g in list(df.index) if isinstance(g, str)]
            if len(set(genes)) >= 10:
                gtra_hsc.append(genes)
    print(f"GTra HSC-dominant modules: {len(gtra_hsc)} (sizes {[len(m) for m in gtra_hsc][:8]})", flush=True)

    # HSC-only pseudobulk temporal profiles
    ad = sc.read_h5ad("/data3/projects/2025_GTRA/data/2_HSPC/HSPC_proc.h5ad")
    hsc = ad[ad.obs["celltype"].astype(str) == "HSCs"]
    X = hsc.X.toarray() if sp.issparse(hsc.X) else np.asarray(hsc.X)
    Xln = np.log1p(X / (X.sum(1, keepdims=True) + 1e-12) * 1e4)
    tcol = hsc.obs["time"].astype(str).values
    prof = np.vstack([Xln[tcol == tp].mean(0) for tp in TPS]).T  # genes x 4
    genes = np.array(ad.var_names)
    keep = np.argsort(prof.var(1))[::-1][:1500]  # HSC-variable genes
    g_keep = genes[keep]
    P = prof[keep]
    Z = np.nan_to_num((P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9))
    K = max(8, len(gtra_hsc))

    def km(zz):
        lab = KMeans(K, random_state=0, n_init=10).fit_predict(zz)
        return [list(g_keep[lab == c]) for c in np.unique(lab)]
    T = np.array([[0,1,2,3],[3,2,1,0],[0,3,1,0],[0,1,3,1],[3,0,1,2],[0,2,3,3],
                  [3,1,0,0],[0,3,3,3],[3,3,1,0],[0,0,1,3]], float)
    T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-9)
    stem = [list(g_keep[np.argmax(Z @ T.T, 1) == c]) for c in range(len(T))]

    methods = {"GTra (HSC modules)": gtra_hsc, "k-means (HSC)": km(Z),
               "spherical k-means (HSC)": km(Z / (np.linalg.norm(Z, 1, keepdims=True) + 1e-9)),
               "STEM-like (HSC)": [m for m in stem if len(m) >= 10]}
    rows = []
    for name, mods in methods.items():
        v, who = best_ifn(mods)
        rows.append({"method": name, "n_modules": len(mods), "IFN_recovery": v})
        print(f"{name:24s} nmod={len(mods):3d}  best IFN -log10p={v:.2f}  ({who})", flush=True)
    df = pd.DataFrame(rows)
    out = HERE / "FIG3_figs"; out.mkdir(exist_ok=True)
    df.to_csv(out / "fig3_hsc_ifn.csv", index=False)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(data=df, x="method", y="IFN_recovery", color="#4c72b0", ax=ax)
    ax.axhline(-np.log10(0.05), ls="--", c="grey", lw=1, label="adj p=0.05")
    ax.set_title("HSC self-transition: interferon recovery\n(HSC-restricted; GTra vs time-series clustering)")
    ax.set_ylabel("-log10(adj p), Hallmark Interferon"); ax.set_xlabel(""); ax.legend()
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(out / "FIG3_hsc_ifn_recovery.pdf", dpi=200)
    print("saved FIG3_figs/FIG3_hsc_ifn_recovery.pdf\nDONE")


if __name__ == "__main__":
    main()
