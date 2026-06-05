"""Pattern/biological trajectory scores from a saved GTra object (answer-path run).

Reads merge_pattern_dict (per-module gene x timepoint profiles) directly, so the
scores use GTra's own curated, answer-path-constrained modules (incl. HSC
self-transition). Same 2 pattern + 2 biological scores as pattern_scores.py.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import dill
import gseapy as gp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MND"))

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ORG = "Human"
IFN_TERMS = ["Interferon Alpha Response", "Interferon Gamma Response"]


def main():
    obj = dill.load(open(HERE / "hspc_full.dill", "rb"))
    mpd = obj.merge_pattern_dict
    # trajectory name per pattern key, if available
    traj = {}
    sp = getattr(obj, "sig_patterns", None)
    if sp is not None and "trajectory" in getattr(sp, "columns", []):
        for _, r in sp.iterrows():
            traj[str(r["Pattern_ID"])] = str(r["trajectory"])

    rows = []
    for key, df in mpd.items():
        genes = [g for g in list(df.index) if isinstance(g, str)]
        if len(set(genes)) < 10:
            continue
        P = np.asarray(df.values, float)              # genes x timepoints
        Z = np.nan_to_num((P - P.mean(1, keepdims=True)) / (P.std(1, keepdims=True) + 1e-9))
        coh = float(np.corrcoef(Z)[np.triu_indices(len(Z), 1)].mean()) if len(Z) > 1 else np.nan
        modu = float(Z.mean(0).std())
        rows.append({"key": str(key), "trajectory": traj.get(str(key), str(key)),
                     "n_genes": len(genes), "tr_length": P.shape[1],
                     "coherence": coh, "modulation": modu, "_genes": list(dict.fromkeys(genes))})
    res = pd.DataFrame(rows).sort_values("n_genes", ascending=False).head(30).reset_index(drop=True)

    goc, prog = [], []
    for _, r in res.iterrows():
        gl = list(r["_genes"])[:60]
        try:
            e = gp.enrichr(gene_list=gl, gene_sets=["GO_Biological_Process_2021"], organism=ORG, outdir=None).res2d
            goc.append(-np.log10(float(pd.to_numeric(e["Adjusted P-value"], errors="coerce").min()) + 1e-300))
        except Exception:
            goc.append(0.0)
        try:
            h = gp.enrichr(gene_list=gl, gene_sets=["MSigDB_Hallmark_2020"], organism=ORG, outdir=None).res2d
            best = 1.0
            for t in IFN_TERMS:
                s = h[h["Term"].str.contains(t, case=False, na=False)]
                if len(s):
                    best = min(best, float(pd.to_numeric(s["Adjusted P-value"], errors="coerce").min()))
            prog.append(-np.log10(best + 1e-300))
        except Exception:
            prog.append(0.0)
    res["go_coherence"] = goc; res["program_recovery_IFN"] = prog

    def scale(x):
        x = np.asarray(x, float); return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)
    res["importance"] = np.nanmean(np.vstack([scale(res["coherence"]), scale(res["modulation"]),
                                              scale(res["go_coherence"]), scale(res["program_recovery_IFN"])]), axis=0)
    res = res.sort_values("importance", ascending=False)
    out = HERE / "PATTERN_SCORE_figs"; out.mkdir(exist_ok=True)
    res.drop(columns="_genes").to_csv(out / "hspc_answerpath_trajectory_scores.csv", index=False)
    print("=== Top trajectories by importance (HSPC, answer-path) ===")
    print(res[["trajectory", "n_genes", "coherence", "modulation", "go_coherence",
               "program_recovery_IFN", "importance"]].head(8).round(3).to_string(index=False))
    ti = res.sort_values("program_recovery_IFN", ascending=False).iloc[0]
    print(f"\nTop IFN trajectory: {ti['trajectory']} (IFN -log10p={ti['program_recovery_IFN']:.2f}, n={ti['n_genes']})")
    print("DONE")


if __name__ == "__main__":
    main()
