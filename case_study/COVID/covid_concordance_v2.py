"""COVID concordance v2: GTra-pattern-derived module (no circularity) + DEG baseline.

(1) GTra module: from each patient's CD8+T/NK trajectory patterns (merge_pattern_dict),
    classify increasing/decreasing by centroid slope. DP+ = genes of INCREASING T/NK
    modules pooled over DP patients; RP- = DECREASING over RP patients; module = DP+ ∩ RP-.
    The gene set is GTra's trajectory output (its linking machinery), so the slope test
    is no longer the same quantity that defined the genes.
(2) DEG baseline: DP-up = genes higher at last vs first timepoint across DP patients
    (paired Wilcoxon); RP-down = lower across RP; module_DEG = DP-up ∩ RP-down.

Both modules scored identically: per-patient slope beta (DP vs RP, Mann-Whitney) and
per-patient Spearman concordance with clinical WHO/NEWS severity.
"""
import sys, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, dill
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu, linregress
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, seaborn as sns
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MND"))
from gtra.utils import convert_path_name, l2norm

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
SEV = "/data2/NIH_COVID_19/clinical/severity_score"
OUT = HERE / "CONCORDANCE_figs"; OUT.mkdir(exist_ok=True)
TNK = {"CD8+T", "NK"}


def lognorm(ad):
    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    return np.log1p(X / (X.sum(1, keepdims=True) + 1e-12) * 1e4)


def main():
    cl = pd.read_csv(HERE / "clinical.csv", index_col=0)
    pname2pid = dict(cl[["pname", "pid"]].values); pheno = dict(cl[["pname", "pheno"]].values)
    dates = {r["pname"]: [int(d.replace("Day", "")) for d in r["dates"].split("|")] for _, r in cl.iterrows()}

    pb, common, objs = {}, None, {}
    gtra_inc, gtra_dec = {}, {}   # per patient: genes in increasing / decreasing T/NK modules
    for p in sorted(glob.glob("../../../../covid_obj/*.dill")):
        pn = Path(p).stem.split("_")[0]; o = dill.load(open(p, "rb"))
        # T/NK pseudobulk per timepoint
        cols = []
        for tp in range(o.tp_data_num):
            ad = o.tp_data_dict[tp]; m = ad.obs["mye_sub"].astype(str).isin(TNK).values
            Xln = lognorm(ad); cols.append(Xln[m].mean(0) if m.sum() else np.full(ad.n_vars, np.nan))
        g = list(map(str, o.tp_data_dict[0].var_names))
        pb[pn] = pd.DataFrame(np.array(cols).T, index=g)
        common = set(g) if common is None else (common & set(g))
        # GTra T/NK trajectory modules -> increasing / decreasing by centroid slope
        inc, dec = set(), set()
        for k, df in o.merge_pattern_dict.items():
            try: path = convert_path_name(o, k)
            except Exception: continue
            cts = path.split("->")
            if sum(c in TNK for c in cts) < 2:   # T/NK-dominant trajectory
                continue
            cent = l2norm(df).mean(axis=0).values
            sl = linregress(np.arange(len(cent)), cent).slope
            genes = set(map(str, df.index))
            (inc if sl > 0 else dec).update(genes)
        gtra_inc[pn] = inc; gtra_dec[pn] = dec
        del o
    common = sorted(common)
    DP = [pn for pn in pb if pheno[pn] == "DP"]; RP = [pn for pn in pb if pheno[pn] == "RP"]

    # ---- module 1: GTra patterns ----
    DPplus = set.union(*[gtra_inc[pn] for pn in DP]) & set(common)
    RPminus = set.union(*[gtra_dec[pn] for pn in RP]) & set(common)
    mod_gtra = sorted(DPplus & RPminus)

    # ---- module 2: DEG baseline (last vs first tp, paired across group) ----
    def deg(group, up=True):
        first = np.array([pb[pn].loc[common].values[:, 0] for pn in group])
        last = np.array([pb[pn].loc[common].values[:, -1] for pn in group])
        diff = (last - first).mean(0)
        thr = np.percentile(diff, 90 if up else 10)
        return set(np.array(common)[diff > thr]) if up else set(np.array(common)[diff < thr])
    mod_deg = sorted(deg(DP, True) & deg(RP, False))
    print(f"GTra module = {len(mod_gtra)} genes; DEG-baseline module = {len(mod_deg)} genes", flush=True)

    # ---- score + metrics for a given module ----
    def evaluate(module, label):
        rows = []
        for pn in pb:
            pid = pname2pid[pn]
            try: sv = pd.read_csv(f"{SEV}/{pid}.csv", index_col=0)
            except Exception: sv = None
            mg = [g for g in module if g in pb[pn].index]
            score = np.nanmean(pb[pn].loc[mg].values, 0) if mg else np.full(pb[pn].shape[1], np.nan)
            x = np.arange(len(score)); beta = linregress(x, score).slope
            who = [sv.iloc[d-1]["WHO"] if (sv is not None and d-1 < len(sv)) else np.nan for d in dates[pn]]
            def rho(sev):
                sev = np.array(sev, float); ok = ~np.isnan(sev)
                return spearmanr(score[ok], sev[ok])[0] if ok.sum() >= 3 else np.nan
            rows.append({"patient": pn, "pheno": pheno[pn], "module": label,
                         "beta": beta, "rho_WHO": rho(who)})
        return pd.DataFrame(rows)

    res = pd.concat([evaluate(mod_gtra, "GTra"), evaluate(mod_deg, "DEG")], ignore_index=True)
    res.to_csv(OUT / "concordance_v2.csv", index=False)
    for lab in ["GTra", "DEG"]:
        d = res[res.module == lab]
        dpb, rpb = d[d.pheno == "DP"]["beta"], d[d.pheno == "RP"]["beta"]
        pmw = mannwhitneyu(dpb, rpb, alternative="greater")[1]
        rW = d["rho_WHO"].dropna()
        wp = wilcoxon(rW, alternative="greater")[1] if len(rW) >= 3 else np.nan
        print(f"[{lab}] slope DP {dpb.median():.3f} vs RP {rpb.median():.3f} MWU p={pmw:.3f} | "
              f"rho_WHO median={rW.median():.3f} (n={len(rW)}) Wilcoxon p={wp if np.isnan(wp) else round(wp,3)}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(data=res, x="module", y="beta", hue="pheno", hue_order=["DP", "RP"], ax=ax[0])
    ax[0].axhline(0, ls="--", c="grey"); ax[0].set_title("Module-score slope: DP vs RP\n(GTra vs DEG baseline)")
    m = res.dropna(subset=["rho_WHO"])
    sns.stripplot(data=m, x="module", y="rho_WHO", hue="pheno", dodge=True, size=8, ax=ax[1])
    ax[1].axhline(0, ls="--", c="grey"); ax[1].set_ylim(-1, 1); ax[1].set_title("Score–severity concordance (WHO)")
    fig.tight_layout(); fig.savefig(OUT / "CONC2_gtra_vs_deg.pdf", dpi=200)
    print("saved CONCORDANCE_figs/CONC2_gtra_vs_deg.pdf\nDONE")


if __name__ == "__main__":
    main()
