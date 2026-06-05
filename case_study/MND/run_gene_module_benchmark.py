"""GTra modules vs fair pseudobulk-temporal baseline — functional coherence (R2.5/R4.3).

Same gene universe + same #modules per dataset; the only difference is GTra's
trajectory-linking vs plain k-means on temporal profiles. GO-BP enrichment
(Enrichr) summarizes functional coherence. Capped to N_CAP modules/method to
bound API calls.
"""
import warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gene_module_baseline as gm

warnings.filterwarnings("ignore")
FIG = Path("GENEMODULE_figs"); FIG.mkdir(exist_ok=True)
N_CAP = 30   # modules per method per dataset (largest first)


def cap(mods, n=N_CAP):
    return sorted(mods, key=len, reverse=True)[:n]


def run_dataset(ds):
    if ds == "COVID":
        gtra = cap(gm.covid_modules_gtra(min_genes=10))
        base = cap(gm.covid_baseline_modules(n_modules=max(2, len(gtra))))
    else:
        gtra = cap(gm.gtra_modules(ds, min_genes=10))
        base = cap(gm.baseline_modules(ds, gtra_mods=gtra))
    org = gm.ORGANISM[ds]
    sg, dg = gm.functional_coherence(gtra, org, max_modules=N_CAP)
    sb, db = gm.functional_coherence(base, org, max_modules=N_CAP)
    dg["method"] = "GTra"; dg["dataset"] = ds
    db["method"] = "baseline"; db["dataset"] = ds
    print(f"[{ds}] GTra:     n={sg['n_modules']} frac_sig={sg['frac_sig']:.3f} "
          f"mean_logp={sg['mean_logp']:.2f} mean_nsig={sg['mean_nsig']:.1f} "
          f"(median size {int(np.median([len(m) for m in gtra]))})", flush=True)
    print(f"[{ds}] baseline: n={sb['n_modules']} frac_sig={sb['frac_sig']:.3f} "
          f"mean_logp={sb['mean_logp']:.2f} mean_nsig={sb['mean_nsig']:.1f} "
          f"(median size {int(np.median([len(m) for m in base]))})", flush=True)
    return ({"dataset": ds, "method": "GTra", **sg},
            {"dataset": ds, "method": "baseline", **sb},
            pd.concat([dg, db], ignore_index=True))


def main():
    summ, per_mod = [], []
    for ds in ["MND", "HSPC", "COVID"]:
        g, b, dfm = run_dataset(ds)
        summ += [g, b]; per_mod.append(dfm)
    S = pd.DataFrame(summ); S.to_csv(FIG / "genemodule_summary.csv", index=False)
    pd.concat(per_mod, ignore_index=True).to_csv(FIG / "genemodule_permodule.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, met, ttl in [(axes[0], "frac_sig", "Frac. modules GO-significant"),
                         (axes[1], "mean_logp", "Mean top-term -log10(adj p)"),
                         (axes[2], "mean_nsig", "Mean # significant terms")]:
        sns.barplot(data=S, x="dataset", y=met, hue="method",
                    order=["MND", "HSPC", "COVID"], hue_order=["GTra", "baseline"], ax=ax)
        ax.set_title(ttl); ax.set_xlabel("")
    fig.suptitle("Gene-module functional coherence: GTra vs pseudobulk-temporal baseline (matched universe & #modules)")
    fig.tight_layout(); fig.savefig(FIG / "GM1_functional_coherence.pdf", dpi=200); plt.close(fig)
    print("\n", S.round(3).to_string(index=False))
    print("saved -> GENEMODULE_figs/")


if __name__ == "__main__":
    main()
