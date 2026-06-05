"""Extract GTra's REPRESENTATIVE patterns via the built-in pipeline functions.

Order required by the code:
  plot_patterns()      -> writes {output}_pattern_genes.csv (visualize.py:547)
  module_evaluation()  -> get_sig_patterns (Friedman p<1e-3) + cluster -> sig_patterns, module_df
  plot_module_cluster(), plot_rep_patterns()  -> representative-pattern figures

We then export the representative pattern groups (gene sets + centroids) and
check whether the interferon (ISG) program is among them.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf  # noqa: F401 (draw_patterns needs this)
import dill

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MND"))

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "REP_PATTERN_figs"; OUT.mkdir(exist_ok=True)
ISG = set("STAT1 STAT2 IRF1 IRF7 IRF9 ISG15 IFIT1 IFIT2 IFIT3 MX1 MX2 OAS1 OAS2 OAS3 OASL "
          "RSAD2 IFI44 IFI44L IFI6 IFITM1 IFITM3 USP18 GBP1 GBP2 BST2 XAF1 DDX58 IFIH1 CXCL10 "
          "CMPK2 HERC5 SAMD9 SAMD9L LY6E EIF2AK2 PARP9 DTX3L".split())


def main():
    obj = dill.load(open(HERE / "hspc_full.dill", "rb"))
    obj.params.time_point_label = ["control", "3h", "24h", "72h"]
    obj.plot_patterns()                      # writes pattern_genes.csv
    plt.savefig(OUT / "rep_patterns_all.pdf", dpi=150); plt.close("all")
    obj.module_evaluation()                  # sig_patterns + module_df
    obj.plot_module_cluster(); plt.savefig(OUT / "module_cluster.pdf", dpi=150); plt.close("all")
    obj.plot_rep_patterns(); plt.savefig(OUT / "rep_patterns.pdf", dpi=150); plt.close("all")

    sp = obj.sig_patterns; md = obj.module_df
    print(f"sig patterns: {len(sp)}  representative clusters: {md['cluster'].nunique()}")
    print(sp[["Pattern_ID", "trajectory", "Trend", "nGenes"]].to_string(index=False)
          if "trajectory" in sp.columns else sp.head().to_string())

    # representative cluster gene sets + ISG check
    rows = []
    for c, d in md.groupby("cluster"):
        genes = set()
        for pi in d["Pattern_ID"]:
            genes |= set(map(str, obj.merge_pattern_dict[pi].index))
        rows.append({"rep_cluster": int(c), "n_patterns": len(d), "n_genes": len(genes),
                     "ISG_overlap": len(genes & ISG),
                     "trajectories": "; ".join(map(str, d["trajectory"].tolist()))[:80]})
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "representative_clusters.csv", index=False)
    print("\n=== representative pattern clusters ===")
    print(rep.to_string(index=False))
    print(f"\nISG genes anywhere in representative patterns: "
          f"{sum(r['ISG_overlap'] for _, r in rep.iterrows())}")
    print("DONE")


if __name__ == "__main__":
    main()
