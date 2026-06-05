"""Identify GTra's 3h-interferon REPRESENTATIVE pattern (module_evaluation output).

Uses the proper representative patterns (sig_patterns clustered into module_df),
not raw merge_pattern_dict. For each representative cluster: gene set (union of
member patterns), temporal centroid, and interferon enrichment (organism=Mouse).
Reports which representative pattern is the transient 3h-IFN program.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, dill, gseapy as gp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MND"))
from gtra.utils import l2norm
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "REP_PATTERN_figs"; OUT.mkdir(exist_ok=True)
TPS = ["control", "3h(T2)", "24h", "72h"]


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
    obj.plot_patterns(); plt.close("all")            # writes pattern_genes.csv
    obj.module_evaluation()                           # sig_patterns + module_df
    md = obj.module_df
    print(f"representative clusters: {md['cluster'].nunique()}  (from {len(md)} sig patterns)\n")
    rows = []
    fig, axes = plt.subplots(md["cluster"].nunique(), 1, figsize=(5, 2.4 * md["cluster"].nunique()))
    axes = np.atleast_1d(axes)
    for ax, (c, d) in zip(axes, md.groupby("cluster")):
        genes, cents = set(), []
        for pi in d["Pattern_ID"]:
            expr = l2norm(obj.merge_pattern_dict[pi])
            genes |= set(map(str, expr.index))
            cents.append(expr.mean(axis=0).values)
        cent = np.mean(cents, axis=0)
        z = (cent - cent.mean()) / (cent.std() + 1e-9)
        peak = TPS[int(np.argmax(z))]
        ip = ifn_p(genes)
        trajs = "; ".join(sorted({str(t) for t in d["trajectory"]}))[:70]
        rows.append({"rep_cluster": int(c), "n_patterns": len(d), "n_genes": len(genes),
                     "peak": peak, "IFN_recovery": round(ip, 2), "trajectories": trajs})
        ax.plot(range(4), z, "-o", lw=2)
        ax.set_xticks(range(4)); ax.set_xticklabels(TPS)
        ax.set_title(f"Rep pattern {c}: peak={peak}, IFN -log10p={ip:.1f}, n={len(genes)}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "rep_patterns_IFN.pdf", dpi=200); plt.close(fig)
    res = pd.DataFrame(rows).sort_values("IFN_recovery", ascending=False)
    res.to_csv(OUT / "rep_patterns_IFN.csv", index=False)
    print(res.to_string(index=False))
    top = res.iloc[0]
    print(f"\n>>> GTra's interferon representative pattern: cluster {top['rep_cluster']} "
          f"(peak={top['peak']}, IFN -log10p={top['IFN_recovery']}, {top['n_genes']} genes)")
    print("DONE")


if __name__ == "__main__":
    main()
