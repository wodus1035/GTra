"""Trajectory/pattern importance scoring for GTra (R4.3 + pattern-score workstream).

Two PATTERN scores (intrinsic, no external DB) and two BIOLOGICAL scores per
GTra trajectory module:

  PATTERN
    P1 coherence            mean pairwise Pearson r of member genes' temporal
                            z-profiles (how tight the temporal program is)
    P2 modulation_strength  std of the module's z-scored centroid across time
                            (how strongly it is modulated over time)
  BIOLOGICAL
    B1 go_coherence         best GO-BP term -log10(adj p) (Enrichr)
    B2 program_recovery     best -log10(adj p) for the system's canonical program
                            (HSPC: interferon, via MSigDB Hallmark)

Descriptors: n_genes, tr_length (path length). A composite IMPORTANCE ranks
trajectories by the mean of the min-max-scaled four scores. Run in gtra_test.
"""
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import gseapy as gp

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
H5AD = "/data3/projects/2025_GTRA/data/2_HSPC/HSPC_proc.h5ad"
PATTERN_CSV = HERE / "HSPC_out" / "hspc_pattern_genes.csv"
TP_ORDER = ["control", "3h", "24h", "72h"]
ORG = "Human"
IFN_TERMS = ["Interferon Alpha Response", "Interferon Gamma Response"]


def parse_col(col):
    """'HSCs->HSCs->MYP->MYP[3_0]' -> (path list, id)."""
    m = re.match(r"^(.*)\[(.+)\]$", col)
    path = m.group(1).split("->")
    return [p.strip() for p in path], m.group(2)


def lognorm(ad):
    X = ad.layers["counts"] if "counts" in ad.layers else ad.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return np.log1p(X / (X.sum(1, keepdims=True) + 1e-12) * 1e4)


def main():
    ad = sc.read_h5ad(H5AD)
    Xln = lognorm(ad)
    genes = list(ad.var_names)
    gidx = {g: i for i, g in enumerate(genes)}
    annot = ad.obs["celltype"].astype(str).values
    tcol = ad.obs["time"].astype(str).values
    df = pd.read_csv(PATTERN_CSV, index_col=0)

    rows = []
    for col in df.columns:
        path, mid = parse_col(col)
        gl = [g for g in df[col].dropna().tolist() if isinstance(g, str) and g in gidx]
        if len(set(gl)) < 10:
            continue
        gi = [gidx[g] for g in dict.fromkeys(gl)]
        # per-gene temporal profile along the path's cell types
        prof = np.zeros((len(gi), len(TP_ORDER)))
        for ti, (ct, tp) in enumerate(zip(path, TP_ORDER)):
            m = (annot == ct) & (tcol == tp)
            if m.sum() == 0:
                m = (tcol == tp)  # fallback: any cell at that tp
            prof[:, ti] = Xln[np.ix_(m, gi)].mean(0)
        Z = (prof - prof.mean(1, keepdims=True)) / (prof.std(1, keepdims=True) + 1e-9)
        Z = np.nan_to_num(Z)
        # P1 coherence
        if len(Z) > 1:
            C = np.corrcoef(Z)
            coh = float(C[np.triu_indices_from(C, 1)].mean())
        else:
            coh = np.nan
        # P2 modulation strength
        cent = Z.mean(0)
        modu = float(cent.std())
        rows.append({"trajectory": col, "path": "->".join(path), "n_genes": len(gi),
                     "tr_length": len(set(path)), "coherence": coh,
                     "modulation": modu, "_genes": list(dict.fromkeys(gl))})
    res = pd.DataFrame(rows)

    # biological scores (cap to top-30 by n_genes to bound Enrichr calls)
    res = res.sort_values("n_genes", ascending=False).head(30).reset_index(drop=True)
    go_coh, prog = [], []
    for _, r in res.iterrows():
        gl = list(r["_genes"])[:60]
        try:
            e = gp.enrichr(gene_list=gl, gene_sets=["GO_Biological_Process_2021"],
                           organism=ORG, outdir=None).res2d
            go_coh.append(-np.log10(float(pd.to_numeric(e["Adjusted P-value"], errors="coerce").min()) + 1e-300))
        except Exception:
            go_coh.append(0.0)
        try:
            h = gp.enrichr(gene_list=gl, gene_sets=["MSigDB_Hallmark_2020"],
                           organism=ORG, outdir=None).res2d
            best = 1.0
            for t in IFN_TERMS:
                s = h[h["Term"].str.contains(t, case=False, na=False)]
                if len(s):
                    best = min(best, float(pd.to_numeric(s["Adjusted P-value"], errors="coerce").min()))
            prog.append(-np.log10(best + 1e-300))
        except Exception:
            prog.append(0.0)
    res["go_coherence"] = go_coh
    res["program_recovery_IFN"] = prog

    def scale(x):
        x = np.asarray(x, float)
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)
    res["importance"] = np.nanmean(np.vstack([
        scale(res["coherence"]), scale(res["modulation"]),
        scale(res["go_coherence"]), scale(res["program_recovery_IFN"])]), axis=0)
    res = res.sort_values("importance", ascending=False)
    out = HERE / "PATTERN_SCORE_figs"; out.mkdir(exist_ok=True)
    res.drop(columns="_genes").to_csv(out / "hspc_trajectory_scores.csv", index=False)
    print("=== Top trajectories by importance (HSPC) ===")
    print(res[["path", "n_genes", "coherence", "modulation", "go_coherence",
               "program_recovery_IFN", "importance"]].head(8).round(3).to_string(index=False))
    top_ifn = res.sort_values("program_recovery_IFN", ascending=False).iloc[0]
    print(f"\nTop IFN-recovery trajectory: {top_ifn['path']} "
          f"(IFN -log10p={top_ifn['program_recovery_IFN']:.2f}, n={top_ifn['n_genes']})")
    print("DONE")


if __name__ == "__main__":
    main()
